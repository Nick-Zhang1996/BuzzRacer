# pylint: disable=logging-fstring-interpolation
""" Game Theoretic Planner, responsible for generating ref traj for multiple cars """
from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import replace
import logging
import multiprocessing as mp
import os
import copy

import pickle
import ctypes
import numpy as np

from rd3g.games.car_racing_casadi import CarRacingCasadiConfig, CarRacingCasadi
from rd3g.solvers.rd3g_casadi import RD3GCasadi, RD3GCasadiConfig, Solution

from buzzracer.common import BASEDIR
from buzzracer.types import CartesianState, CurvilinearState, Control
from buzzracer.extensions.extension import Extension, ExtensionConfig, ExtensionState
from buzzracer.controllers.stanley_controller import StanleyController, StanleyControllerConfig, StanleyControllerState
from buzzracer.sysid.kinematic_bicycle_model import KinematicBicycleModelCartesian

if TYPE_CHECKING:
    from buzzracer.cars.car import Car, CarParam
    from buzzracer.main import MainState
    from buzzracer.tracks.curvilinear_track import CurvilinearTrack

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class GameTheoreticPlannerConfig(ExtensionConfig):
    """ Read-only config """

    def __init__(self, main_config):
        super().__init__(main_config)
        self.horizon: int = 20
        """ Game horizon """
        self.car_count: int = 8
        """ Number of cars, N """
        self.stitching_steps: int = 10
        """ Number of steps to keep in previous trajectory in next iteration """
        self.multiprocess: bool = True
        """ Run planner in a separate process, necessary for realtime operation"""
        if self.multiprocess:
            assert main_config.multiprocess, 'MainConfig.multiprocess must be also true'
        self.dt: float = 0.02
        # TODO use multiple work processes to evaluate multiple initial guess simultaneously
        # e.g. zero control, stanley (following left/middle/right raceline)
        self.use_stanley_control_guess: bool = True
        self.stanley_config = StanleyControllerConfig(main_config, None)
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
        self.cart_traj: np.ndarray = None
        """ (n=6, N, traj_len) non-process safe Cartesian State Trajectory """
        self.curv_traj: np.ndarray = None
        """ (n=5, N, traj_len) non-process safe Curvilinear State Trajectory """
        self.cart_traj_len: int = mp.Value(ctypes.c_int)
        """ cart_traj_sync.shape[2] Length of cart_traj """
        self.cart_traj_sync = mp.Array(ctypes.c_double, 6*config.car_count*config.max_traj_len)
        """ (n=6, N, cart_traj_len) Process safe Cartesian State Trajectory"""
        # self.ctrl_traj: np.ndarray = None
        # """ (m*N, T) of ctrl trajectory """
        self.stanley_state: StanleyControllerState = StanleyControllerState(config)
        self.child_process: mp.Process = None
        self.planner_ready: mp.synchronize.Event = mp.Event()


@Extension.register('planner', GameTheoreticPlannerConfig, GameTheoreticPlannerState)
class GameTheoreticPlanner(Extension):
    """ Central planner that relies on a dynamic game solver. """

    def __init__(self, config, state):
        super().__init__(config, state)
        self.state: GameTheoreticPlannerState
        self.config: GameTheoreticPlannerConfig

    def init(self):
        # Load game theoretic solver
        self.state.cars = Extension.main.cars
        self.state.car_params = [car.param for car in self.state.cars]
        if self.config.multiprocess:
            p = mp.Process(target=GameTheoreticPlanner.process_fun,
                           args=(self.state, self.main.state,
                                 self.main.track, self.config))
            p.start()
            self.state.child_process = p
        else:
            self.state.solver = GameTheoreticPlanner.make_solver(self.config, self.main.track)

    @staticmethod
    def make_solver(config, track):
        """ Setup casadi solver """
        # x = [s, n, phi, v_forward, v_sideway]
        c = config
        N = c.car_count  # pylint: disable=invalid-name
        default = CarRacingCasadiConfig
        J_Qr = np.diag([0, 5.0, 1.0, 1.0, 0.1])  # pylint: disable=invalid-name
        J_R = np.eye(default.m) * 1.0  # pylint: disable=invalid-name

        x0 = np.zeros((default.n, N))
        x_ref = np.zeros((default.n, N))
        x_ref[3, :] = 1.0  # dummy target speed
        rows_per_collision = 4 if default.double_circle_h else 1
        n_h = (rows_per_collision * (N * (N - 1) // 2) + 2 * N) * c.horizon

        game_config = CarRacingCasadiConfig(
            T=c.horizon,
            dt=config.dt,
            N=N,
            n=default.n,
            m=default.m,
            n_h=n_h,
            collision_radius=default.collision_radius,
            x0=x0.copy(order='F'),
            target_x_ref=x_ref.copy(order='F'),
            J_Qr=J_Qr.copy(order='F'),
            J_R=J_R.copy(order='F'))
        game = CarRacingCasadi(game_config, track)
        solver_config = RD3GCasadiConfig(inertia_correction=False, iterations=20)
        solver = RD3GCasadi(solver_config, game, cpp_only=config.use_cpp_solver)
        if config.use_cpp_solver:
            solver.init_cpp_backend()
        return solver

    @staticmethod
    def get_cart_traj_sync(state):
        """ Retrieve planned trajectory, in sync. Call from another process
        Args:
            state: planner state """
        # When the planner runs in a different process, state.cart_traj in
        # the main process is not updated, instead, new cart_traj are placed in cart_traj_sync
        shape = (6, state.car_count, state.cart_traj_len.value)
        cart_traj = np.frombuffer(state.cart_traj_sync.get_obj(),
                                  dtype=np.float64,
                                  count=shape[0]*shape[1]*shape[2]
                                  ).reshape(shape, order='F').copy()
        return cart_traj

    def update(self):
        state = self.state
        if self.config.multiprocess:
            state.cart_traj = GameTheoreticPlanner.get_cart_traj_sync(state)
        else:
            GameTheoreticPlanner.update_fun(self.state, self.main.state,
                                            self.main.track, self.config)
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
        try:
            state.solver = GameTheoreticPlanner.make_solver(config, track)
            while not main_state.exit_request.is_set():
                GameTheoreticPlanner.update_fun(state, main_state, track, config)
        finally:
            main_state.exit_request.set()

    @staticmethod
    def update_fun(state: GameTheoreticPlannerState,
                   main_state: MainState,
                   track: CurvilinearTrack,
                   config: GameTheoreticPlannerConfig):
        """ Update planner with a new plan, stitch to current plan """
        # t = Extension.main.timer
        default = CarRacingCasadiConfig
        solver = state.solver
        n = default.n
        N = config.car_count  # pylint: disable=invalid-name
        T = config.horizon  # pylint: disable=invalid-name

        # Wait till main_state.car_states are available
        if not main_state.car_states_first_available.wait(0.1):
            logger.debug('Waiting for main_state.car_states...')
            return

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
        if state.curv_traj is None:
            # First plan
            state.curv_traj = curv_x.reshape((n, N, 1), order='F')  # s, n, heading_err, vf, vs
            state.cart_traj = cart_x.reshape((6, N, 1), order='F')  # x,y,heading,vf,vs,omega

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
        retval = GameTheoreticPlanner.plan_from_x0(
            solver, cart_x0, curv_x0, track, config, state, main_state)
        if retval is None:
            logger.debug('No valid output from planner')
            return
        new_curv_traj, sol = retval
        logger.info(f'{sol.elapsed_time=:.6f}, {sol.residual=:.6f}, {margin=}')
        if sol.residual > config.residual_threshold:
            if margin > 0:
                # Reject high residual solutions if existing plan has enough margin to future
                logger.warning('Rejected solution with residual %.4f, margin=%d',
                               sol.residual, margin)
                return
            else:
                logger.warning('Forced to accept solution with residual %.4f, '
                               'because margin=%d', sol.residual, margin)

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
            stitched_curv[:, i, :stitch_len] = state.curv_traj[:, i, stitch_start_idx:next_plan_idx]
            stitched_cart[:, i, :stitch_len] = state.cart_traj[:, i, stitch_start_idx:next_plan_idx]
            stitched_curv[:, i, stitch_len:stitch_len+T] = new_curv_traj[:, i, :]
            stitched_cart[:, i, stitch_len:stitch_len+T] = new_cart_traj[:, i, :]

        state.curv_traj = stitched_curv
        state.cart_traj = stitched_cart

        # Clip traj when created trajectory is too long
        traj_len = state.cart_traj.shape[2]
        if traj_len > config.max_traj_len:
            traj_len = config.max_traj_len
            logger.warning('Planned trajectory exceeds buffer length.'
                           f'{traj_len=}, {config.max_traj_len=}'
                           'Consider increasing buffer or reducing stitching steps')

        flat_cart_traj = state.cart_traj[:, :, :traj_len].flatten(order='F')
        flat_cart_traj_len = len(flat_cart_traj)

        with state.cart_traj_sync.get_lock():
            state.cart_traj_sync[:flat_cart_traj_len] = flat_cart_traj
        state.cart_traj_len.value = traj_len
        state.planner_ready.set()

    @staticmethod
    def plan_from_x0(solver,
                     cart_x0: np.ndarray,
                     curv_x0: np.ndarray,
                     track: CurvilinearTrack,
                     config: GameTheoreticPlannerConfig,
                     state: GameTheoreticPlannerState,
                     main_state: MainState):
        """ Call game solver to build a plan. 
        Return:
            curv_traj: (n,N,T) Planned trajectory. From rolling out control solution.
            residual: 
        """
        default = CarRacingCasadiConfig
        n = default.n
        m = default.m
        N = config.car_count  # pylint: disable=invalid-name
        T = config.horizon  # pylint: disable=invalid-name
        # Initial conditions for all cars
        x0 = curv_x0
        x0[3, :] = np.clip(x0[3, :], a_min=1.0, a_max=None)  # override speed
        cart_x0[3, :] = np.clip(cart_x0[3, :], a_min=1.0, a_max=None)  # override speed
        x_ref = np.zeros((n, N), order='F')
        for i in range(N):
            x_ref[3, i] = x0[3, i]  # target speed

        new_config = replace(solver.game.config, x0=x0, target_x_ref=x_ref)
        solver.game.config = new_config
        u_ref = np.zeros((m * N, T), order='F')

        # Initial guess for control sequence from stanley controller
        u_ref_3d = u_ref.reshape((m, N, T), order='F')
        debug_stanley_states = []
        for i in range(N):
            x = CartesianState(*cart_x0[:, i])
            debug_states = []
            for k in range(T):
                debug_states.append(x)
                if config.use_stanley_control_guess:
                    u, _, _, _ = StanleyController.control(x,
                                                           state.car_params[i],
                                                           track,
                                                           config.stanley_config,
                                                           state.stanley_state,
                                                           main_state, i)
                    # only use steering, keep throttle 0
                    # print(f'{i=}, {k=}, {u=}')
                    u = Control(u.steering, 0)
                    u_ref_3d[:, i, k] = u.to_tuple()
                else:
                    u = Control(0, 0)
                # logger.debug(f'{x=}, {u=}')
                x = KinematicBicycleModelCartesian.advance_dynamics(
                    x, u, state.car_params[i], config.dt, simple_throttle=True)
                debug_states.append(x)
            debug_stanley_states.append(debug_states)

        # Call solver
        assert not np.any(np.isnan(u_ref))
        assert not np.any(np.isnan(x0))
        if config.use_cpp_solver:
            sol: Solution = solver.solve_cpp_backend(u_ref)
        else:
            sol: Solution = solver.solve(u_ref)
        if np.isnan(sol.residual):
            return None

        # Clip solution to reasonable number
        # clip_u = np.clip(sol.u, -radians(27), radians(27), order='F')

        # Save inputs to solver when user press 'b' for triaging
        if np.isnan(sol.residual) or main_state.breakpoint.is_set():
            main_state.breakpoint.clear()
            # main_state.exit_request.set()
            gc = copy.copy(solver.game.config)
            delattr(gc, '_int_param_sx')
            delattr(gc, '_int_param_np')
            delattr(gc, '_double_param_sx')
            delattr(gc, '_double_param_np')
            delattr(gc, '_param_dict')
            save = {'x0': x0, 'target_x_ref': x_ref, 'u_ref': u_ref, 'gc': gc}
            filename = os.path.join(BASEDIR, 'outputs', 'input.p')
            with open(filename, 'wb') as f:
                pickle.dump(save, f)
            logger.info('Saved to %s', filename)
            # sol: Solution = solver.solve(u_ref)
            # DEBUG
            # solver_traj = solver._rollout_full_x(sol.u.reshape((m, N, T), order='F'))
            # ax = solver.visualize(u_ref, show=False)
            # py_sol = solver.solve(u_ref)
            # py_solver_traj = solver._rollout_full_x(py_sol.u.reshape((m, N, T), order='F'))
            # for i in range(N):
            #     # kinbike_traj = np.array([(v.x, v.y) for v in debug_stanley_states[i]])
            #     solver_curv_state = [CurvilinearState(*solver_traj[:, i, k]) for k in range(T)]
            #     solver_cart_traj = np.asarray(
            #         [track.curv_to_cart(val).to_tuple()[:2] for val in solver_curv_state])
            #     py_solver_curv_state = [CurvilinearState(
            #         *py_solver_traj[:, i, k]) for k in range(T)]
            #     py_solver_cart_traj = np.asarray(
            #         [track.curv_to_cart(val).to_tuple()[:2] for val in py_solver_curv_state])

            #     # print(f'{i=}, {kinbike_traj.shape=}, {solver_traj.shape=}')
            #     # print(u_ref.reshape((m, N, T), order='F')[0, i, :])
            #     # print(f'{kinbike_traj=}')
            #     # print(f'{solver_cart_traj=}')
            #     # fig, ax = plt.subplots()
            #     # ax.plot(kinbike_traj[:, 0], kinbike_traj[:, 1], 'o-', label='kin')
            #     ax.plot(solver_cart_traj[:, 0], solver_cart_traj[:, 1], '--', label='cpp')
            #     ax.plot(py_solver_cart_traj[:, 0], py_solver_cart_traj[:, 1], 'o', label='py')
            # ax.legend()
            # plt.show()

        #  Visualize planned trajectory for all agents
        # NOTE use solution or re-do rollout (n*N,T)
        # x_ref = sol.x
        gc = solver.game.config
        params_np = [gc.get_int_param_np(), gc.get_double_param_np()]
        u = sol.u.reshape((gc.m*gc.N, gc.T), order='F')
        if solver.cpp_solver is None:
            x_ref = solver.rollout_casadi(x0, u, *params_np).full()
        else:
            x_ref = solver.cpp_solver.rollout(x0, u, *params_np)

        curv_trajs = x_ref.reshape((n, N, T), order='F')
        return curv_trajs, sol
