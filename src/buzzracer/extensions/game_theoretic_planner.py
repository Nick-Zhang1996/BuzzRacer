# pylint: disable=logging-fstring-interpolation
""" Game Theoretic Planner, responsible for generating ref traj for multiple cars """
from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass, replace
import logging
import multiprocessing as mp
import os
import queue
import sys
from time import perf_counter
from types import SimpleNamespace

import pickle
import ctypes
import numpy as np
from scipy.interpolate import splev

import rd3g
from rd3g.games.car_racing_casadi import CarRacingCasadiConfig, CarRacingCasadi
from rd3g.core.base_solver import Solution
from rd3g.solvers.interior_point_game import InteriorPointGame, InteriorPointGameConfig

from buzzracer.common import BASEDIR, LoggingFilter
from buzzracer.utilities.execution_timer import ExecutionTimer
from buzzracer.types import CartesianState, CurvilinearState, Control
from buzzracer.extensions.extension import Extension, ExtensionConfig, ExtensionState
from buzzracer.controllers.stanley_controller import StanleyController, StanleyControllerConfig, StanleyControllerState
from buzzracer.sysid.kinematic_bicycle_model import KinematicBicycleModelCartesian
from buzzracer.tracks.track import LocalTrajOutput

if TYPE_CHECKING:
    from buzzracer.cars.car import Car, CarParam
    from buzzracer.main import MainState
    from buzzracer.tracks.curvilinear_track import CurvilinearTrack

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addFilter(LoggingFilter(interval=1.0))


class ShiftedRacelineTrack:
    """Lightweight track-like wrapper exposing local_trajectory on a shifted raceline."""

    def __init__(self,
                 name: str,
                 r_vec: np.ndarray,
                 s_vec: np.ndarray,
                 phi_vec: np.ndarray,
                 curvature_vec: np.ndarray,
                 left_width_vec: np.ndarray,
                 right_width_vec: np.ndarray,
                 speed_vec: np.ndarray):
        self.name = name
        self.r_vec = r_vec
        self.s_vec = s_vec
        self.phi_vec = phi_vec
        self.curvature_vec = curvature_vec
        self.left_width_vec = left_width_vec
        self.right_width_vec = right_width_vec
        self.speed_vec = speed_vec

    @classmethod
    def from_track(cls,
                   name: str,
                   track: CurvilinearTrack,
                   requested_shift: float,
                   boundary_buffer: float):
        """Build a left/right shifted copy of the reference raceline."""
        data = track.data
        applied_shift = np.full_like(data.left_width_vec, requested_shift, dtype=float)
        if requested_shift >= 0:
            max_shift = np.maximum(data.left_width_vec - boundary_buffer, 0.0)
            applied_shift = np.minimum(applied_shift, max_shift)
        else:
            max_shift = np.maximum(data.right_width_vec - boundary_buffer, 0.0)
            applied_shift = -np.minimum(-applied_shift, max_shift)

        lateral = np.column_stack((
            np.cos(data.phi_vec + np.pi / 2),
            np.sin(data.phi_vec + np.pi / 2)))
        shifted_points = data.r_vec + lateral * applied_shift[:, np.newaxis]
        tangent = np.roll(shifted_points, -1, axis=0) - shifted_points
        shifted_heading = np.arctan2(tangent[:, 1], tangent[:, 0])
        left_width = data.left_width_vec - applied_shift
        right_width = data.right_width_vec + applied_shift

        return cls(name=name,
                   r_vec=shifted_points,
                   s_vec=data.s_vec,
                   phi_vec=shifted_heading,
                   curvature_vec=data.curvature_vec,
                   left_width_vec=left_width,
                   right_width_vec=right_width,
                   speed_vec=data.speed_vec)

    def local_trajectory(self, state: CartesianState) -> LocalTrajOutput:
        """Mimic CurvilinearTrack.local_trajectory on shifted geometry."""
        x = state[0]
        y = state[1]

        dxx = self.r_vec[:, 0] - x
        dyy = self.r_vec[:, 1] - y
        index = int(np.argmin(dxx**2 + dyy**2))
        raceline_point = self.r_vec[index]
        point_count = len(self.r_vec)

        dr = self.r_vec[(index + 1) % point_count] - self.r_vec[index]
        dr_norm = np.linalg.norm(dr)
        if dr_norm < 1e-9:
            track_tangent = np.array([np.cos(self.phi_vec[index]), np.sin(self.phi_vec[index])])
        else:
            track_tangent = dr / dr_norm
        track_to_car = (x - self.r_vec[index, 0], y - self.r_vec[index, 1])
        offset = track_tangent[0] * track_to_car[1] - track_tangent[1] * track_to_car[0]
        left_margin = self.left_width_vec[index] - offset
        right_margin = self.right_width_vec[index] + offset
        return LocalTrajOutput(ref_point=raceline_point,
                               lateral_err=offset,
                               raceline_dir=self.phi_vec[index],
                               curvature=self.curvature_vec[index],
                               v_target=self.speed_vec[index],
                               progress=self.s_vec[index],
                               left_margin=left_margin,
                               right_margin=right_margin)


@dataclass(frozen=True)
class PlannerWorkerSpec:
    """Static worker configuration for one solver process."""

    name: str
    use_zero_guess: bool


@dataclass(frozen=True)
class PlannerSolveRequest:
    """One planning request dispatched to a solver worker."""

    job_id: int
    cart_x0: np.ndarray
    curv_x0: np.ndarray


@dataclass
class PlannerSolveCandidate:
    """Serializable solve candidate returned by a worker."""

    curv_trajs: np.ndarray
    sol: Solution
    side_summary: str
    u_ref: np.ndarray
    x0: np.ndarray
    target_x_ref: np.ndarray
    debug_dict: dict


@dataclass(frozen=True)
class PlannerSolveInputs:
    """Serializable inputs needed to replay or triage one solve."""

    side_summary: str
    u_ref: np.ndarray
    x0: np.ndarray
    target_x_ref: np.ndarray


@dataclass(frozen=True)
class PlannerSolveResult:
    """Result envelope passed back from a worker to the planner."""

    job_id: int
    worker_name: str
    candidate: PlannerSolveCandidate | None = None
    solver_inputs: PlannerSolveInputs | None = None
    error: str | None = None


class GameTheoreticPlannerConfig(ExtensionConfig):
    """ Read-only config """

    def __init__(self, main_config):
        super().__init__(main_config)
        self.horizon: int = 10
        """ Game horizon """
        self.car_count: int = 4
        """ Number of cars, N """
        self.stitching_steps: int = 10
        """ Number of steps to keep in previous trajectory in next iteration """
        self.multiprocess: bool = True
        """ Run planner in a separate process, necessary for realtime operation"""
        if self.multiprocess:
            assert main_config.multiprocess, 'MainConfig.multiprocess must be also true'
        self.dt: float = 0.05
        self.use_stanley_control_guess: bool = True
        self.stanley_config = StanleyControllerConfig(main_config, None)
        self.initial_guess_shift_margin: float = 0.04
        """ Lateral shift for left/right Stanley reference lines. """
        self.initial_guess_boundary_buffer: float = 1e-3
        """ Keep shifted racelines slightly inside the track boundary. """
        self.max_traj_len: int = 100
        """ Maximum length of trajectory. Defines buffer size for mp.Array"""
        self.residual_threshold: int = 1.0
        """ Maxmimum residual of accepted solutions. """
        self.use_cpp_solver: bool = True


class GameTheoreticPlannerState(ExtensionState):
    """ Planner State, mutable"""

    def __init__(self, config):
        super().__init__(config)
        self.cars: Car = None
        """ (N, ) Cars under this planner """
        self.car_count: int = config.car_count
        self.car_params: list[CarParam] = None
        self.traj_ts: np.ndarray = None
        """ (traj_len), time stamp from time() for traj points. Non-process safe"""
        self.cart_traj: np.ndarray = None
        """ (n=6, N, traj_len) non-process safe Cartesian State Trajectory.
            state: (x,y,heading,vf,vs,omega) """
        self.curv_traj: np.ndarray = None
        """ (n=5, N, traj_len) non-process safe Curvilinear State Trajectory.
            state: (s, n, heading_err, vf, vs)"""

        self.traj_sync_lock: mp.synchronize.Lock = mp.Lock()
        """ Keeps cart/curv trajectory buffers and their length consistent across processes. """
        self.cart_traj_len: int = mp.Value(ctypes.c_int)
        """ cart_traj_sync.shape[2] Length of cart_traj """
        self.traj_ts_sync = mp.Array(ctypes.c_double, config.max_traj_len, lock=False)
        """ (traj_len), time stamp from time() for traj points. Process safe"""
        self.cart_traj_sync = mp.Array(
            ctypes.c_double, 6*config.car_count*config.max_traj_len, lock=False)
        """ (n=6, N, cart_traj_len) Process safe Cartesian State Trajectory"""
        self.curv_traj_sync: np.ndarray = mp.Array(
            ctypes.c_double, 6*config.car_count*config.max_traj_len, lock=False)
        """ (n=5, N, traj_len) process safe Curvilinear State Trajectory.
            state: (s, n, heading_err, vf, vs)"""
        self.split_point_sync = mp.Value(ctypes.c_double)
        """ Split point for breaking track progress so the discontinuity is away from planned traj"""
        # self.ctrl_traj: np.ndarray = None
        # """ (m*N, T) of ctrl trajectory """
        self.child_process: mp.Process = None
        self.planner_ready: mp.synchronize.Event = mp.Event()
        self.solver = None
        self.initial_guess_tracks = None
        self.max_speeds: np.ndarray = None
        self.worker_processes: dict[str, mp.Process] = {}
        self.worker_request_queues: dict[str, mp.Queue] = {}
        self.worker_result_queue: mp.Queue | None = None
        self.next_solver_job_id: int = 0


@Extension.register('planner', GameTheoreticPlannerConfig, GameTheoreticPlannerState)
class GameTheoreticPlanner(Extension):
    """ Central planner that relies on a dynamic game solver. """

    WORKER_SPECS = (
        PlannerWorkerSpec(name='stanley_seeded', use_zero_guess=False),
        PlannerWorkerSpec(name='zero_guess', use_zero_guess=True),
    )

    def __init__(self, config, state):
        super().__init__(config, state)
        self.state: GameTheoreticPlannerState
        self.config: GameTheoreticPlannerConfig

    def init(self):
        # Load game theoretic solver
        self.state.cars = Extension.main.cars
        if len(self.state.cars) != self.config.car_count:
            logger.error(f"Cars in xml config != car_count in GameTheoreticPlannerConfig")
            raise RuntimeError
        self.state.car_params = [car.param for car in self.state.cars]
        self.state.max_speeds = np.asarray(
            [car.controller.config.max_speed for car in self.state.cars[:self.config.car_count]],
            dtype=float,
        )
        # DEBUG
        # self.state.solver = GameTheoreticPlanner.make_solver(self.config, self.main.track)
        if self.config.multiprocess:
            p = mp.Process(target=GameTheoreticPlanner.process_fun,
                           args=(self.state, self.main.state,
                                 self.main.track, self.config))
            p.start()
            self.state.child_process = p
        else:
            self.state.solver = GameTheoreticPlanner.make_solver(self.config, self.main.track)
            self.state.initial_guess_tracks = GameTheoreticPlanner.make_initial_guess_tracks(
                self.main.track, self.config)

    def final(self):
        child = self.state.child_process
        if child is not None and child.is_alive():
            child.join(timeout=1.0)

    @staticmethod
    def make_solver(config, track):
        """ Setup casadi solver """
        # x = [s, n, phi, v_forward, v_sideway]
        c = config
        N = c.car_count  # pylint: disable=invalid-name
        default = CarRacingCasadiConfig
        x0 = np.zeros((default.n, N), dtype=float, order='F')
        x_ref = np.zeros((default.n, N), dtype=float, order='F')
        x_ref[3, :] = 1.0  # dummy target speed

        game_config = GameTheoreticPlanner.make_game_config(config, x0, x_ref)
        game = CarRacingCasadi(game_config, track)
        solver_config = InteriorPointGameConfig(
            inertia_correction=False,
            iterations=20,
            variational_gne=game_config.variational_gne,
        )
        solver = InteriorPointGame(solver_config, game, cpp_only=config.use_cpp_solver)
        if config.use_cpp_solver:
            GameTheoreticPlanner.ensure_rd3g_repo_on_pythonpath()
            solver.init_cpp_backend()
        return solver

    @staticmethod
    def ensure_rd3g_repo_on_pythonpath():
        """Make rd3g's top-level build package importable in solver workers."""
        rd3g_repo = os.path.dirname(os.path.dirname(os.path.abspath(rd3g.__file__)))
        if rd3g_repo not in sys.path:
            sys.path.insert(0, rd3g_repo)

    @staticmethod
    def make_game_config(config, x0: np.ndarray, x_ref: np.ndarray):
        """Create a fresh game config for a given initial state and reference."""
        c = config
        N = c.car_count  # pylint: disable=invalid-name
        default = CarRacingCasadiConfig
        J_Qr = np.diag([0, 5.0, 1.0, 1.0, 0.1])  # pylint: disable=invalid-name
        J_R = np.eye(default.m) * 1.0  # pylint: disable=invalid-name
        rows_per_collision = 4 if default.double_circle_h else 1
        n_h = (rows_per_collision * (N * (N - 1) // 2) + 2 * N) * c.horizon
        return CarRacingCasadiConfig(
            T=c.horizon,
            dt=config.dt,
            N=N,
            n=default.n,
            m=default.m,
            n_h=n_h,
            collision_radius=default.collision_radius,
            x0=np.array(x0, dtype=float, order='F', copy=True),
            target_x_ref=np.array(x_ref, dtype=float, order='F', copy=True),
            J_Qr=J_Qr.copy(order='F'),
            J_R=J_R.copy(order='F'))

    @staticmethod
    def make_initial_guess_tracks(track: CurvilinearTrack,
                                  config: GameTheoreticPlannerConfig):
        """Create shifted Stanley reference tracks used for the single mixed guess."""
        shift = config.initial_guess_shift_margin
        return {
            'left_raceline': ShiftedRacelineTrack.from_track(
                'left_raceline', track, shift, config.initial_guess_boundary_buffer),
            'right_raceline': ShiftedRacelineTrack.from_track(
                'right_raceline', track, -shift, config.initial_guess_boundary_buffer),
        }

    @staticmethod
    def get_cart_traj_sync(state):
        """ Retrieve planned trajectory, in sync. Call from another process
        Args:
            state: planner state """
        # When the planner runs in a different process, state.cart_traj in
        # the main process is not updated, instead, new cart_traj are placed in cart_traj_sync
        with state.traj_sync_lock:
            shape = (6, state.car_count, state.cart_traj_len.value)
            cart_traj = np.frombuffer(state.cart_traj_sync,
                                      dtype=np.float64,
                                      count=shape[0]*shape[1]*shape[2]
                                      ).reshape(shape, order='F').copy()
        return cart_traj

    def update(self):
        state = self.state
        if self.config.multiprocess:
            state.cart_traj = GameTheoreticPlanner.get_cart_traj_sync(state)
        else:
            timer = ExecutionTimer(enable=False, clock=perf_counter, clock_name='planner')
            GameTheoreticPlanner.update_fun(self.state, self.main.state,
                                            self.main.track, self.config, timer)
        cart_traj = self.state.cart_traj
        if cart_traj.shape[2] > 0:
            for i in range(self.config.car_count):
                points = cart_traj[:2, i, :].T
                self.main.visualization.draw_polyline(points)

    @staticmethod
    def process_fun(state: GameTheoreticPlannerState,
                    main_state: MainState,
                    track: CurvilinearTrack,
                    config: GameTheoreticPlannerConfig):
        """ Process function to run update_fun in a loop """
        t = ExecutionTimer(enable=True, clock=perf_counter, clock_name='planner')
        try:
            GameTheoreticPlanner.start_solver_workers(state, main_state, track, config)
            while not main_state.exit_request.is_set():
                GameTheoreticPlanner.update_fun(state, main_state, track, config, t)
        finally:
            main_state.exit_request.set()
            GameTheoreticPlanner.stop_solver_workers(state)
            t.summary()

    @staticmethod
    def start_solver_workers(state: GameTheoreticPlannerState,
                             main_state: MainState,
                             track: CurvilinearTrack,
                             config: GameTheoreticPlannerConfig):
        """Start persistent solver workers owned by the planner process."""
        state.worker_result_queue = mp.Queue()
        state.worker_request_queues = {}
        state.worker_processes = {}
        for spec in GameTheoreticPlanner.WORKER_SPECS:
            request_queue = mp.Queue(maxsize=1)
            proc = mp.Process(
                target=GameTheoreticPlanner.solver_worker_fun,
                args=(
                    spec,
                    request_queue,
                    state.worker_result_queue,
                    main_state.exit_request,
                    track,
                    config,
                    state.car_params,
                    state.max_speeds,
                ),
                name=f'planner_solver_{spec.name}',
            )
            proc.daemon = True
            proc.start()
            state.worker_request_queues[spec.name] = request_queue
            state.worker_processes[spec.name] = proc

    @staticmethod
    def stop_solver_workers(state: GameTheoreticPlannerState):
        """Stop solver workers and release their queues."""
        for request_queue in state.worker_request_queues.values():
            try:
                request_queue.put_nowait(None)
            except queue.Full:
                pass
            request_queue.close()
        for proc in state.worker_processes.values():
            proc.join(timeout=1.0)
        if state.worker_result_queue is not None:
            state.worker_result_queue.close()
        state.worker_processes = {}
        state.worker_request_queues = {}
        state.worker_result_queue = None

    @staticmethod
    def solver_worker_fun(spec: PlannerWorkerSpec,
                          request_queue: mp.Queue,
                          result_queue: mp.Queue,
                          exit_request: mp.synchronize.Event,
                          track: CurvilinearTrack,
                          config: GameTheoreticPlannerConfig,
                          car_params,
                          max_speeds: np.ndarray):
        """Solve one stream of planning jobs with a dedicated seed strategy."""
        solver = GameTheoreticPlanner.make_solver(config, track)
        initial_guess_tracks = GameTheoreticPlanner.make_initial_guess_tracks(track, config)
        while not exit_request.is_set():
            try:
                request = request_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if request is None:
                return
            try:
                candidate, solver_inputs = GameTheoreticPlanner.solve_candidate_request(
                    request,
                    spec,
                    solver,
                    initial_guess_tracks,
                    track,
                    config,
                    car_params,
                    max_speeds,
                )
                result_queue.put(PlannerSolveResult(
                    job_id=request.job_id,
                    worker_name=spec.name,
                    candidate=candidate,
                    solver_inputs=solver_inputs,
                ))
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.exception('Solver worker %s failed on job %d', spec.name, request.job_id)
                result_queue.put(PlannerSolveResult(
                    job_id=request.job_id,
                    worker_name=spec.name,
                    error=str(exc),
                ))

    @staticmethod
    def update_fun(state: GameTheoreticPlannerState,
                   main_state: MainState,
                   track: CurvilinearTrack,
                   config: GameTheoreticPlannerConfig,
                   t: ExecutionTimer | None = None):
        """ Update planner with a new plan, stitch to current plan """
        default = CarRacingCasadiConfig
        n = default.n
        N = config.car_count  # pylint: disable=invalid-name
        T = config.horizon  # pylint: disable=invalid-name

        # Wait till main_state.car_states are available
        if not main_state.car_states_first_available.wait(0.1):
            logger.debug('Waiting for main_state.car_states...')
            return

        if t is not None:
            t.s()
        cart_x = (CartesianState * N).from_buffer(main_state.car_states)
        curv_x = np.empty((n, N), dtype=float, order='F')
        for i in range(N):
            curv_x[:, i] = track.cart_to_curv(cart_x[i]).to_tuple()[:5]

        # Curvilinear state has a discontinuity around progress=0
        # Loosely assume care are close together, reposition the discontinuity in Frenet progress
        # from finishing line to the opposite of car mean position
        # s domain: [ split_point, curv_len + split_point]
        curv_len = track.data.raceline_len_m
        current_s = np.sort(curv_x[0, :])
        current_s = np.hstack([current_s[-1]-curv_len, current_s])
        gaps = np.diff(current_s) % curv_len
        max_gap_s = current_s[np.argmax(gaps)+1]  # s with biggest gap BEFORE it
        split_point = (max_gap_s - 0.1*curv_len) % curv_len
        curv_x[0, :] = (curv_x[0, :] - split_point) % curv_len + split_point

        cart_x = np.asarray([val.to_tuple() for val in cart_x], order='F').T
        if not state.planner_ready.is_set():
            # Initialize curv_traj
            state.curv_traj = curv_x.reshape((n, N, 1), order='F')  # s, n, heading_err, vf, vs
            state.cart_traj = cart_x.reshape((6, N, 1), order='F')  # x,y,heading,vf,vs,omega
            state.traj_ts = np.array(main_state.time)  # Dummy, will overwrite

        state.curv_traj[0, :, :] = (state.curv_traj[0, :, :] - split_point) % curv_len + split_point
        if not np.all(np.diff(state.curv_traj[0, :, :]) > 0):
            logger.warning("Planned traj progress is not monotonic")

        # Find stitching point on state.curv_traj
        # After all cars current progress. We cannot plan behind the cars' current pos
        realized_idx_car = np.fromiter(
            (np.searchsorted(state.curv_traj[0, i, :], curv_x[0, i]) for i in range(N)),
            dtype=np.intp,
            count=N)
        max_realized_idx = int(np.max(realized_idx_car))

        current_plan_horizon = state.curv_traj.shape[-1] - 1
        min_margin = current_plan_horizon - (max_realized_idx + config.stitching_steps)
        if min_margin <= 0:
            logger.warning('Planner cannot maintain sufficient margin to future, %d', min_margin)
        next_plan_idx_car = np.clip(realized_idx_car + config.stitching_steps,
                                    a_min=0, a_max=state.curv_traj.shape[-1]-1)
        margin = min_margin
        curv_x0 = np.empty((n, N), dtype=state.curv_traj.dtype, order='F')
        cart_x0 = np.empty((6, N), dtype=state.cart_traj.dtype, order='F')
        for i in range(N):
            next_plan_idx = next_plan_idx_car[i]
            curv_x0[:, i] = state.curv_traj[:, i, next_plan_idx]
            cart_x0[:, i] = state.cart_traj[:, i, next_plan_idx]
        if t is not None:
            t.s('plan_from_x0')
        retval = GameTheoreticPlanner.plan_from_x0(
            cart_x0, curv_x0, track, config, state, main_state, t)
        if t is not None:
            t.e('plan_from_x0')
        if retval is None:
            logger.debug('No valid output from planner')
            if t is not None:
                t.e()
            return
        new_curv_traj, sol, side_summary, debug_dict = retval
        reject = False
        msg = 'Accepted'
        if sol.residual > config.residual_threshold:
            if margin > 0:
                # Reject high residual solutions if existing plan has enough margin to future
                reject = True
                msg = "[Rejected]"
            else:
                # logger.warning('Forced to accept sol with residual %.4f, because margin=%d', sol.residual, margin)
                msg = "[Force Accept, low margin]"

        logger.info('dt=%.3f s, res=%.3f, margin=%d, candidate=%s, %s',
                    sol.elapsed_time, sol.residual, margin, side_summary, msg)

        if reject:
            if t is not None:
                t.e()
            return

        new_cart_traj = np.empty((6, N, T), dtype=float, order='F')
        for i in range(N):
            for k in range(T):
                curv_state = CurvilinearState(*new_curv_traj[:, i, k])
                new_cart_traj[:, i, k] = track.curv_to_cart(curv_state).to_tuple()

        # Remove realized traj
        # Stitch new plan onto state.curv_traj
        stitch_len = int(min(config.stitching_steps, np.min(next_plan_idx_car)))
        stitch_start_idx_car = next_plan_idx_car - stitch_len
        stitched_curv = np.empty((n, N, stitch_len+T),
                                 dtype=state.curv_traj.dtype, order='F')
        stitched_cart = np.empty((6, N, stitch_len+T),
                                 dtype=state.cart_traj.dtype, order='F')
        for i in range(N):
            next_plan_idx = next_plan_idx_car[i]
            stitch_start_idx = stitch_start_idx_car[i]
            if stitch_len > 0:
                stitched_curv[:, i, :stitch_len+1] = state.curv_traj[:,
                                                                     i, stitch_start_idx:next_plan_idx+1]
                stitched_cart[:, i, :stitch_len+1] = state.cart_traj[:,
                                                                     i, stitch_start_idx:next_plan_idx+1]
                stitched_curv[:, i, stitch_len+1:stitch_len+T] = new_curv_traj[:, i, :T-1]
                stitched_cart[:, i, stitch_len+1:stitch_len+T] = new_cart_traj[:, i, :T-1]
            else:
                stitched_curv[:, i, stitch_len:stitch_len+T] = new_curv_traj[:, i, :]
                stitched_cart[:, i, stitch_len:stitch_len+T] = new_cart_traj[:, i, :]

        state.curv_traj = stitched_curv
        state.cart_traj = stitched_cart

        # DEBUG: Check speed against finite difference on s (s, v, n)
        # state: (s, n, heading_err, vf, vs)
        """
        dsdt = np.diff(stitched_curv[0, 0, :])/config.dt
        v = stitched_curv[3, 0, :]
        gc = state.solver.game.config
        context = np.asarray(debug_dict.get("full_context")).reshape(gc.n_c, gc.N, gc.T)
        curvature = context[0, 0, :]
        print(f'{dsdt=}, {v=}, {curvature=}')
        """

        # Clip traj when created trajectory is too long
        traj_len = state.cart_traj.shape[2]
        if traj_len > config.max_traj_len:
            traj_len = config.max_traj_len
            logger.warning('Planned trajectory exceeds buffer length.'
                           f'{traj_len=}, {config.max_traj_len=}'
                           'Consider increasing buffer or reducing stitching steps')

        if not state.planner_ready.is_set():
            # If first plan, re-align time
            ts = main_state.time
        else:
            ts = state.traj_ts[stitch_start_idx]
        end_ts = ts + traj_len * config.dt
        state.traj_ts = np.linspace(ts, end_ts, traj_len, endpoint=False).flatten(order='F')

        flat_cart_traj = state.cart_traj[:, :, :traj_len].flatten(order='F')
        flat_cart_traj_len = len(flat_cart_traj)
        flat_curv_traj = state.curv_traj[:, :, :traj_len].flatten(order='F')
        flat_curv_traj_len = len(flat_curv_traj)
        with state.traj_sync_lock:
            state.cart_traj_sync[:flat_cart_traj_len] = flat_cart_traj
            state.curv_traj_sync[:flat_curv_traj_len] = flat_curv_traj
            state.traj_ts_sync[:traj_len] = state.traj_ts
            state.cart_traj_len.value = traj_len
            state.split_point_sync.value = split_point
        state.planner_ready.set()
        if t is not None:
            t.e()

    @staticmethod
    def get_initial_guess_track_name(curv_state: np.ndarray):
        """Pick left/right seed based on current offset from the center raceline."""
        return 'left_raceline' if curv_state[1] >= 0 else 'right_raceline'

    @staticmethod
    def plan_from_x0(cart_x0: np.ndarray,
                     curv_x0: np.ndarray,
                     track: CurvilinearTrack,
                     config: GameTheoreticPlannerConfig,
                     state: GameTheoreticPlannerState,
                     main_state: MainState,
                     t: ExecutionTimer | None):
        """Solve the same planning problem from two seeds and pick the best candidate."""
        if state.worker_processes:
            results = GameTheoreticPlanner.collect_worker_candidates(
                state, cart_x0, curv_x0, main_state)
        else:
            results = GameTheoreticPlanner.solve_candidates_locally(
                cart_x0, curv_x0, track, config, state, t)

        best_result = GameTheoreticPlanner.select_best_candidate(results)

        if best_result is None:
            GameTheoreticPlanner.maybe_save_solver_inputs(results, config, main_state)
            return None

        candidate = best_result.candidate
        assert candidate is not None
        return (
            candidate.curv_trajs,
            candidate.sol,
            f'{best_result.worker_name}:{candidate.side_summary}',
            candidate.debug_dict,
        )

    @staticmethod
    def solve_candidate_request(request: PlannerSolveRequest,
                                spec: PlannerWorkerSpec,
                                solver: InteriorPointGame,
                                initial_guess_tracks,
                                track: CurvilinearTrack,
                                config: GameTheoreticPlannerConfig,
                                car_params,
                                max_speeds: np.ndarray):
        """Solve one queued worker request."""
        x0, x_ref, u_ref, side_summary = GameTheoreticPlanner.build_candidate_inputs(
            request.cart_x0,
            request.curv_x0,
            track,
            config,
            car_params,
            max_speeds,
            initial_guess_tracks,
            spec.use_zero_guess,
        )
        solver_inputs = PlannerSolveInputs(
            side_summary=side_summary,
            u_ref=u_ref,
            x0=x0,
            target_x_ref=x_ref,
        )
        return GameTheoreticPlanner.solve_candidate(
            solver,
            x0,
            x_ref,
            u_ref,
            side_summary,
        ), solver_inputs

    @staticmethod
    def solve_candidates_locally(cart_x0: np.ndarray,
                                 curv_x0: np.ndarray,
                                 track: CurvilinearTrack,
                                 config: GameTheoreticPlannerConfig,
                                 state: GameTheoreticPlannerState,
                                 t: ExecutionTimer | None):
        """Fallback path when the planner itself is not running in a separate process."""
        results = []
        for spec in GameTheoreticPlanner.WORKER_SPECS:
            if t is not None:
                t.s(f'prep_{spec.name}')
            x0, x_ref, u_ref, side_summary = GameTheoreticPlanner.build_candidate_inputs(
                cart_x0,
                curv_x0,
                track,
                config,
                state.car_params,
                state.max_speeds,
                state.initial_guess_tracks,
                spec.use_zero_guess,
            )
            if t is not None:
                t.e(f'prep_{spec.name}')
                t.s(f'solve_{spec.name}')
            candidate = GameTheoreticPlanner.solve_candidate(
                state.solver,
                x0,
                x_ref,
                u_ref,
                side_summary,
            )
            solver_inputs = PlannerSolveInputs(
                side_summary=side_summary,
                u_ref=u_ref,
                x0=x0,
                target_x_ref=x_ref,
            )
            if t is not None:
                t.e(f'solve_{spec.name}')
            results.append(PlannerSolveResult(
                job_id=-1,
                worker_name=spec.name,
                candidate=candidate,
                solver_inputs=solver_inputs,
            ))
        return results

    @staticmethod
    def collect_worker_candidates(state: GameTheoreticPlannerState,
                                  cart_x0: np.ndarray,
                                  curv_x0: np.ndarray,
                                  main_state: MainState):
        """Dispatch the current planning problem to all worker processes."""
        state.next_solver_job_id += 1
        job_id = state.next_solver_job_id
        request = PlannerSolveRequest(
            job_id=job_id,
            cart_x0=np.array(cart_x0, dtype=float, order='F', copy=True),
            curv_x0=np.array(curv_x0, dtype=float, order='F', copy=True),
        )
        expected_workers = set()
        for spec in GameTheoreticPlanner.WORKER_SPECS:
            request_queue = state.worker_request_queues.get(spec.name)
            if request_queue is None:
                logger.warning('Solver worker %s is unavailable', spec.name)
                continue
            try:
                request_queue.put(request, timeout=0.1)
            except queue.Full:
                proc = state.worker_processes.get(spec.name)
                if proc is None or not proc.is_alive():
                    logger.error('Solver worker %s is no longer alive', spec.name)
                else:
                    logger.error('Solver worker %s queue is unexpectedly full', spec.name)
                continue
            expected_workers.add(spec.name)

        results = []
        while expected_workers and not main_state.exit_request.is_set():
            try:
                result = state.worker_result_queue.get(timeout=0.1)
            except queue.Empty:
                dead_workers = {
                    name for name in expected_workers
                    if not state.worker_processes[name].is_alive()
                }
                if dead_workers:
                    logger.error('Solver workers died while planning: %s',
                                 ', '.join(sorted(dead_workers)))
                    break
                continue
            if result.job_id != job_id:
                logger.debug('Ignoring stale solver result for job %d from %s',
                             result.job_id, result.worker_name)
                continue
            if result.worker_name not in expected_workers:
                logger.debug('Ignoring duplicate solver result from %s', result.worker_name)
                continue
            results.append(result)
            expected_workers.remove(result.worker_name)
        return results

    @staticmethod
    def select_best_candidate(results: list[PlannerSolveResult]):
        """Pick the lowest-residual candidate among successful worker results."""
        best = None
        for result in results:
            if result.error is not None:
                logger.warning('Solver candidate %s failed: %s',
                               result.worker_name, result.error)
                continue
            candidate = result.candidate
            if candidate is None or not np.isfinite(candidate.sol.residual):
                continue
            best_candidate = best.candidate if best is not None else None
            if best_candidate is None or candidate.sol.residual < best_candidate.sol.residual:
                best = result
        return best

    @staticmethod
    def build_candidate_inputs(cart_x0: np.ndarray,
                               curv_x0: np.ndarray,
                               track: CurvilinearTrack,
                               config: GameTheoreticPlannerConfig,
                               car_params,
                               max_speeds: np.ndarray,
                               initial_guess_tracks,
                               use_zero_guess: bool):
        """Build the solver initial condition, target state, and control seed."""
        default = CarRacingCasadiConfig
        n = default.n
        m = default.m
        N = config.car_count  # pylint: disable=invalid-name
        T = config.horizon  # pylint: disable=invalid-name

        x0 = np.array(curv_x0, dtype=float, order='F', copy=True)
        cart_x0 = np.array(cart_x0, dtype=float, order='F', copy=True)
        x0[3, :] = np.clip(x0[3, :], a_min=0.5, a_max=None)
        cart_x0[3, :] = np.clip(cart_x0[3, :], a_min=0.5, a_max=None)

        x_ref = np.zeros((n, N), dtype=float, order='F')
        progress_mod = np.mod(x0[0, :], track.data.raceline_len_m)
        target_speed = np.asarray(splev(progress_mod, track.data.speed_s), dtype=float)
        x_ref[3, :] = np.clip(target_speed, a_min=0.2, a_max=max_speeds)

        u_ref = np.zeros((m * N, T), dtype=float, order='F')
        if use_zero_guess:
            return x0, x_ref, u_ref, 'zero_ref'

        u_ref_3d = u_ref.reshape((m, N, T), order='F')
        side_names = []
        dummy_main_state = SimpleNamespace(car_target_v=np.zeros(N, dtype=float))
        for i in range(N):
            x = CartesianState(*cart_x0[:, i])
            stanley_state = StanleyControllerState(config.stanley_config)
            ref_name = GameTheoreticPlanner.get_initial_guess_track_name(curv_x0[:, i])
            side_names.append(ref_name)
            ref_track = initial_guess_tracks[ref_name]
            for k in range(T):
                if config.use_stanley_control_guess:
                    u, _, _, _ = StanleyController.control(x,
                                                           car_params[i],
                                                           ref_track,
                                                           config.stanley_config,
                                                           stanley_state,
                                                           dummy_main_state, i)
                    u = Control(u.steering, 0)
                    u_ref_3d[:, i, k] = u.to_tuple()
                else:
                    u = Control(0, 0)
                x = KinematicBicycleModelCartesian.advance_dynamics(
                    x, u, car_params[i], config.dt, simple_throttle=True)

        return x0, x_ref, u_ref, f'stanley_ref'

    @staticmethod
    def solve_candidate(solver: InteriorPointGame,
                        x0: np.ndarray,
                        x_ref: np.ndarray,
                        u_ref: np.ndarray,
                        side_summary: str):
        """Run one solve and package its trajectory and debug data."""
        assert not np.any(np.isnan(u_ref))
        assert not np.any(np.isnan(x0))
        solver.game.config = replace(solver.game.config, x0=x0, target_x_ref=x_ref)
        if solver.cpp_solver is not None:
            sol: Solution = solver.solve_cpp_backend(u_ref)
        else:
            sol: Solution = solver.solve(u_ref)
        if np.isnan(sol.residual):
            return None

        gc = solver.game.config
        params_np = [gc.get_int_param_np(), gc.get_double_param_np()]
        u = sol.u.reshape((gc.m * gc.N, gc.T), order='F')
        if solver.cpp_solver is None:
            rollout = solver.rollout_casadi(x0, u, *params_np).full()
        else:
            rollout = solver.cpp_solver.rollout(x0, u, *params_np)
        curv_trajs = rollout.reshape((gc.n, gc.N, gc.T), order='F')
        debug_dict = {}
        if hasattr(solver.cpp_solver, 'get_full_context'):
            debug_dict['full_context'] = np.array(
                solver.cpp_solver.get_full_context(rollout),
                dtype=float,
                order='F',
                copy=True,
            )
        return PlannerSolveCandidate(
            curv_trajs=curv_trajs,
            sol=sol,
            side_summary=side_summary,
            u_ref=u_ref.copy(order='F'),
            x0=x0.copy(order='F'),
            target_x_ref=x_ref.copy(order='F'),
            debug_dict=debug_dict,
        )

    @staticmethod
    def maybe_save_solver_inputs(results: list[PlannerSolveResult],
                                 config: GameTheoreticPlannerConfig,
                                 main_state: MainState):
        """Keep the existing triage/debug input dump behavior."""

        if main_state.breakpoint.is_set():
            main_state.breakpoint.clear()
            if not results or results[0].solver_inputs is None:
                logger.warning('Unable to save triage state: no solver inputs available')
                return

            solver_inputs = results[0].solver_inputs
            game_config = GameTheoreticPlanner.make_game_config(
                config, solver_inputs.x0, solver_inputs.target_x_ref)
            filename = save_game(game_config, solver_inputs.u_ref)
            logger.warning('Saved triage state to %s', filename)


def save_game(gc, u_ref):
    """ Save a game for debugging offline"""
    delattr(gc, '_int_param_sx')
    delattr(gc, '_int_param_np')
    delattr(gc, '_double_param_sx')
    delattr(gc, '_double_param_np')
    delattr(gc, '_param_dict')
    save = {'u_ref': u_ref, 'gc': gc}
    output_dir = os.path.join(BASEDIR, 'outputs')
    triage_idx = 1
    while True:
        filename = os.path.join(output_dir, f'triage_{triage_idx}.p')
        if not os.path.exists(filename):
            break
        triage_idx += 1
    with open(filename, 'wb') as f:
        pickle.dump(save, f)
    logger.info('Saved to %s', filename)
    return filename
