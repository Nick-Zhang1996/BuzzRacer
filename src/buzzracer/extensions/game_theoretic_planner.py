""" Game Theoretic Planner, responsible for generating ref traj for multiple cars """
from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import replace
import os
import logging
from math import radians

import pickle
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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GameTheoreticPlannerConfig(ExtensionConfig):
    """ Read-only config """

    def __init__(self, main_config):
        super().__init__(main_config)
        self.horizon: int = 20
        """ Game horizon """
        self.car_count: int = 4
        """ Number of cars, N """
        self.stitching_steps: int = 10
        """ Number of steps to keep in previous trajectory in next iteration """
        self.multiprocess: bool = False
        self.dt: float = main_config.dt
        self.use_stanley_control_guess: bool = False
        self.stanley_config = StanleyControllerConfig(main_config, None)


class GameTheoreticPlannerState(ExtensionState):
    """ Planner State, mutable"""

    def __init__(self, config):
        super().__init__(config)
        self.cars: Car = None
        """ (N, ) Cars under this planner """
        self.car_params: list[CarParam] = None
        self.cart_traj: np.ndarray = None
        """ (n=6, N, T) of Cartesian State Trajectory """
        self.curv_traj: np.ndarray = None
        """ (n=5, N, T) of Curvilinear State Trajectory """
        # self.ctrl_traj: np.ndarray = None
        """ (m*N, T) of ctrl trajectory """
        self.stanley_state: StanleyControllerState = StanleyControllerState(config)


@Extension.register('planner', GameTheoreticPlannerConfig,
                    GameTheoreticPlannerState)
class GameTheoreticPlanner(Extension):

    def __init__(self, config, state):
        super().__init__(config, state)
        self.state: GameTheoreticPlannerState
        self.config: GameTheoreticPlannerConfig

        # Setup casadi solver
        # x = [s, n, phi, v_forward, v_sideway]
        # J_Qr = np.diag([0, 5.0, 0.1, 1.0, 0.1])
        # J_R = np.eye(m) * 1.0
        c = self.config
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
        game = CarRacingCasadi(game_config, self.main.track)
        solver_config = RD3GCasadiConfig(inertia_correction=False, iterations=20)
        solver = RD3GCasadi(solver_config, game, cpp_only=False)
        solver.init_cpp_backend()

        self.state.solver = solver

    def init(self):
        # Load game theoretic solver
        self.state.cars = Extension.main.cars
        self.state.car_params = [car.param for car in self.state.cars]
        if self.config.multiprocess:
            pass
        else:
            pass

    def update(self):
        if self.config.multiprocess:
            pass
        else:
            GameTheoreticPlanner.update_fun(self.state, self.main.state,
                                            self.main.track, self.config,
                                            self.main.visualization)

    @staticmethod
    def update_fun(state, main_state, track, config: GameTheoreticPlannerConfig, visualization):

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

        # Get curvilinear states
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

        for i in range(N):
            points = state.cart_traj[:2, i, :].T
            visualization.draw_polyline(points)

    @staticmethod
    def plan_from_x0(solver, cart_x0: np.ndarray, curv_x0: np.ndarray, track, config, state, main_state):
        default = CarRacingCasadiConfig
        n = default.n
        m = default.m
        N = config.car_count
        T = config.horizon
        # Initial conditions for all cars
        solver.x0 = curv_x0
        x_ref = np.zeros((n, N), order='F')
        for i in range(N):
            x_ref[3, i] = cart_x0[3, i]  # target speed

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
                    u, _, _ = StanleyController.control(x,
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
        sol: Solution = solver.solve_cpp_backend(u_ref)
        # sol: Solution = solver.solve(u_ref)
        print(f'{sol.elapsed_time}, {sol.residual=}')
        # Clip solution to reasonable number
        clip_u = np.clip(sol.u, -radians(27), radians(27), order='F')
        if main_state.breakpoint.is_set():
            save = {'x0': curv_x0, 'target_x_ref': x_ref, 'u_ref': u_ref}
            filename = os.path.join(BASEDIR, 'outputs', 'input.p')
            with open(filename, 'wb') as f:
                pickle.dump(save, f)
            logger.info(f'Saved to {filename}')
            sol: Solution = solver.solve(u_ref)
            solver.visualize(sol.u)
            main_state.breakpoint.clear()
            breakpoint()

        # DEBUG: Visualize planned trajectory for all agents
        # (n*N,T)
        # x_ref = sol.x
        gc = solver.game.config
        params_np = [gc.get_int_param_np(), gc.get_double_param_np()]
        x_ref = solver.cpp_solver.rollout(solver.x0, clip_u, *params_np)
        curv_trajs = x_ref.reshape((n, N, T), order='F')
        return curv_trajs
