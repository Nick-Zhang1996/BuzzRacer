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
from buzzracer.sysid.dynamic_bicycle_model import DynamicBicycleModelCartesian

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
        self.car_count: int = 4
        """ Number of cars, N """
        self.stitching_steps: int = 5
        """ Number of steps to keep in previous trajectory in next iteration """
        self.multiprocess: bool = True
        """ Run planner in a separate process, necessary for realtime operation"""
        if self.multiprocess:
            assert main_config.multiprocess, 'MainConfig.multiprocess must be also true'
        self.dt: float = main_config.dt
        self.use_stanley_control_guess: bool = False
        self.stanley_config = StanleyControllerConfig(main_config, None)
        self.max_traj_len: int = 100
        """ Maximum length of trajectory. Defines buffer size for mp.Array"""


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
            # TODO wait till first plan is available

    @staticmethod
    def make_solver(config, track):
        # Setup casadi solver
        # x = [s, n, phi, v_forward, v_sideway]
        # J_Qr = np.diag([0, 5.0, 0.1, 1.0, 0.1])
        # J_R = np.eye(m) * 1.0
        c = config
        N = c.car_count
        default = CarRacingCasadiConfig
        J_Qr = np.diag([0, 5.0, 1.0, 1.0, 0.1])
        J_R = np.eye(default.m) * 1.0

        # TODO resample if cars collide
        # s_vec = np.random.uniform(low=0.0, high=0.1, size=N)
        # v_vec = np.random.uniform(low=1.0, high=1.0, size=N)
        # phi_vec = np.random.uniform(low=radians(-5), high=radians(5), size=N)
        # n_vec = np.random.uniform(low=-0.1, high=0.1, size=N)
        # vs_vec = np.zeros(N)

        # n, N
        # x0 = np.vstack([s_vec, n_vec, phi_vec, v_vec, vs_vec])
        x0 = np.zeros((default.n, N))
        x_ref = np.zeros((default.n, N))
        x_ref[3, :] = 1.0  # target initial speed

        game_config = CarRacingCasadiConfig(
            T=c.horizon,
            dt=default.dt,
            N=N,
            n=default.n,
            m=default.m,
            n_hi=(4 * N+2) * c.horizon if default.double_circle_h else (2+N) * c.horizon,
            collision_radius=default.collision_radius,
            x0=x0.copy(order='F'),
            target_x_ref=x_ref.copy(order='F'),
            J_Qr=J_Qr.copy(order='F'),
            J_R=J_R.copy(order='F'))
        game = CarRacingCasadi(game_config, track)
        solver_config = RD3GCasadiConfig(inertia_correction=False, iterations=20)
        solver = RD3GCasadi(solver_config, game, cpp_only=False)
        # solver.init_cpp_backend() # TODO set after cpp done
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

        # Dirty hack to get some initial states
        # idx = [10, 200, 400, 800]
        # for i in idx:
        #     val = np.hstack([track.data.r_vec[i], track.data.phi_vec[i]])
        #     print(val)
        t = Extension.main.timer
        default = CarRacingCasadiConfig
        solver = state.solver
        n = default.n
        m = default.m
        N = config.car_count
        T = config.horizon

        # Wait till main_state.car_states are available
        if not main_state.car_states_first_available.wait(0.1):
            logger.debug('Waiting for main_state.car_states...')
            return

        cart_x = (CartesianState * N).from_buffer(main_state.car_states)
        curv_x = np.empty((n, N), dtype=float, order='F')
        for i in range(N):
            curv_x[:, i] = track.cart_to_curv(cart_x[i]).to_tuple()[:5]
        cart_x = np.asarray([val.to_tuple() for val in cart_x])
        if state.curv_traj is None:
            state.curv_traj = curv_x.reshape((n, N, 1), order='F')  # s, n, heading_err, vf, vs
            state.cart_traj = cart_x.reshape((6, N, 1), order='F')  # x,y,heading,vf,vs,omega

        # Find stitching point on state.curv_traj[sti_idx]
        # This must be AFTER all cars max progress. We cannot plan behind the cars
        # THis must be BEFORE end of last traj
        # TODO what if cars are not following planned path perfectly?
        # do we continue planning from car position or from previous plan?
        # TODO validate trajectory, discard implausible ones
        # TODO assert monotonicity in progress
        # TODO wrap raceline_len_m
        realized_idx = np.max(
            [np.searchsorted(state.curv_traj[0, i, :], curv_x[0, i]) for i in range(N)])
        margin = config.horizon - realized_idx + config.stitching_steps
        if margin <= 0:
            logger.warning(f'Planner cannot maintain sufficient margin to future, {margin=}')
        next_plan_idx = np.clip(realized_idx + config.stitching_steps,
                                a_min=None, a_max=state.curv_traj.shape[-1]-1)
        curv_x0 = state.curv_traj[:, :, next_plan_idx]
        cart_x0 = state.cart_traj[:, :, next_plan_idx]
        new_curv_traj = GameTheoreticPlanner.plan_from_x0(
            solver, cart_x0, curv_x0, track, config, state, main_state)
        if new_curv_traj is None:
            logger.debug('No valid output from planner')
            return
        new_cart_traj = np.empty((6, N, T), dtype=float, order='F')
        for i in range(N):
            for k in range(T):
                curv_state = CurvilinearState(*new_curv_traj[:, i, k])
                new_cart_traj[:, i, k] = track.curv_to_cart(curv_state).to_tuple()

        # Remove realized traj
        # Stitch new plan onto state.curv_traj
        state.curv_traj = np.dstack(
            [state.curv_traj[:, :, realized_idx:next_plan_idx], new_curv_traj])
        state.cart_traj = np.dstack(
            [state.cart_traj[:, :, realized_idx:next_plan_idx], new_cart_traj])

        flat_cart_traj = state.cart_traj.flatten(order='F')
        flat_cart_traj_len = len(flat_cart_traj)
        # TODO handle when created trajectory is too long
        with state.cart_traj_sync.get_lock():
            state.cart_traj_sync[:flat_cart_traj_len] = flat_cart_traj
        state.cart_traj_len.value = state.cart_traj.shape[2]
        state.planner_ready.set()

    @staticmethod
    def plan_from_x0(solver, cart_x0: np.ndarray, curv_x0: np.ndarray, track, config, state, main_state):
        default = CarRacingCasadiConfig
        n = default.n
        m = default.m
        N = config.car_count
        T = config.horizon
        # Initial conditions for all cars
        x0 = curv_x0
        x0[3, :] = 1.0  # override speed
        x_ref = np.zeros((n, N), order='F')
        for i in range(N):
            x_ref[3, i] = 1.0  # target speed

        new_config = replace(solver.game.config, x0=curv_x0, target_x_ref=x_ref, dt=0.02)
        solver.game.config = new_config
        # TODO: Initial guess for control sequence
        # From previous step, stitch with stanley controller
        u_ref = np.zeros((m * N, T), order='F')

        u_ref_3d = u_ref.reshape((m, N, T), order='F')
        for i in range(N):
            x = CartesianState(*cart_x0[i])
            for k in range(T):
                if config.use_stanley_control_guess:
                    u, _, _, _ = StanleyController.control(x,
                                                           state.car_params[i],
                                                           track,
                                                           config.stanley_config,
                                                           state.stanley_state,
                                                           main_state, i)
                    u_ref_3d[:, i, k] = u.to_tuple()
                else:
                    u = Control(0, 0)
                x = DynamicBicycleModelCartesian.advance_dynamics(
                    x, u, state.car_params[i], solver.game.config.dt)

        # Call solver
        assert not np.any(np.isnan(u_ref))
        assert not np.any(np.isnan(curv_x0))
        # sol: Solution = solver.solve_cpp_backend(u_ref)
        sol: Solution = solver.solve(u_ref)
        print(f'{sol.elapsed_time=}, {sol.residual=}')
        if np.isnan(sol.residual):
            return None

        # Clip solution to reasonable number
        # TODO clip steering
        # clip_u = np.clip(sol.u, -radians(27), radians(27), order='F')
        clip_u = sol.u

        # Save inputs to solver when user press 'b'
        if np.isnan(sol.residual) or main_state.breakpoint.is_set():
            gc = copy.copy(solver.game.config)
            print(dir(gc))
            delattr(gc, '_int_param_sx')
            delattr(gc, '_int_param_np')
            delattr(gc, '_double_param_sx')
            delattr(gc, '_double_param_np')
            delattr(gc, '_param_dict')
            save = {'x0': curv_x0, 'target_x_ref': x_ref, 'u_ref': u_ref, 'gc': gc}
            filename = os.path.join(BASEDIR, 'outputs', 'input.p')

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            solver.game.config = data['gc']
            solver.game.config.__post_init__()
            u_ref = data['u_ref']

            solver.cpp_solver = None
            sol: Solution = solver.solve(u_ref)
            solver.visualize(sol.u)  # this gives incorrect results

            new_solver = RD3GCasadi(solver.config, solver.game)
            sol: Solution = new_solver.solve(u_ref)
            new_solver.visualize(sol.u)  # this works

            # with open(filename, 'wb') as f:
            #     pickle.dump(save, f)
            # logger.info('Saved to %s', filename)
            main_state.breakpoint.clear()
            main_state.exit_request.set()

        # DEBUG: Visualize planned trajectory for all agents
        # (n*N,T)
        # x_ref = sol.x
        gc = solver.game.config
        params_np = [gc.get_int_param_np(), gc.get_double_param_np()]
        # TODO
        # x_ref = solver.cpp_solver.rollout(solver.x0, clip_u, *params_np)
        clip_u = clip_u.reshape((gc.m*gc.N, gc.T), order='F')
        x_ref = solver.rollout_casadi(x0, clip_u, *params_np).full()

        curv_trajs = x_ref.reshape((n, N, T), order='F')
        return curv_trajs
