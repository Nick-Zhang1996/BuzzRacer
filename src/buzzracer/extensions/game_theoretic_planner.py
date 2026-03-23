""" Game Theoretic Planner, responsible for generating ref traj for multiple cars """
from __future__ import annotations
from typing import TYPE_CHECKING
from math import radians

import numpy as np
from rd3g.games.car_racing_casadi import CarRacingCasadiConfig, CarRacingCasadi
from rd3g.solvers.rd3g_casadi import RD3GCasadi, RD3GCasadiConfig

from buzzracer.types import CartesianState
from buzzracer.extensions.extension import Extension, ExtensionConfig, ExtensionState

if TYPE_CHECKING:
    from buzzracer.cars.car import Car


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


class GameTheoreticPlannerState(ExtensionState):
    """ Planner State, mutable"""

    def __init__(self, config):
        super().__init__(config)
        self.cars: Car = None
        """ (N, ) Cars under this planner """
        self.cart_traj: np.ndarray = None
        """ (n=6, N, T) of Cartesian State Trajectory """
        self.curv_traj: np.ndarray = None
        """ (n=6, N, T) of Curvilinear State Trajectory """
        self.ctrl_traj: np.ndarray = None
        """ (m, N, T) of ctrl trajectory """


@Extension.register('planner', GameTheoreticPlannerConfig, GameTheoreticPlannerState)
class GameTheoreticPlanner(Extension):

    def __init__(self, config, state):
        super().__init__(config, state)
        Extension.extensions.append(self)
        self.state: GameTheoreticPlannerState
        self.config: GameTheoreticPlannerConfig

        # Setup casadi solver
        # x = [s, n, phi, v_forward, v_sideway]
        # J_Qr = np.diag([0, 5.0, 0.1, 1.0, 0.1])
        # J_R = np.eye(m) * 1.0
        c = self.config
        N = c.car_count
        default = CarRacingCasadiConfig
        J_Qr = np.diag([0, 5.0, 0.5, 2.0, 0.1])
        J_R = np.eye(default.m) * 1.0

        # TODO resample if cars collide
        s_vec = np.random.uniform(low=2.0, high=3.0, size=N)
        v_vec = np.random.uniform(low=0.5, high=1.5, size=N)
        phi_vec = np.random.uniform(low=radians(-5), high=radians(5), size=N)
        n_vec = np.random.uniform(low=-0.1, high=0.1, size=N)
        vs_vec = np.zeros(N)

        # n, N
        x0 = np.vstack([s_vec, n_vec, phi_vec, v_vec, vs_vec])
        x_ref = np.zeros((default.n, N))
        x_ref[3, :] = v_vec  # target initial speed

        game_config = CarRacingCasadiConfig(
            T=c.horizon,
            dt=default.dt,
            N=N,
            n=default.n,
            m=default.m,
            n_hi=4 * N * c.horizon if default.double_circle_h else N * c.horizon,
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
        self.cars = Extension.main.cars
        if self.config.multiprocess:
            pass
        else:
            pass

    def update(self):
        if self.config.multiprocess:
            pass
        else:
            GameTheoreticPlanner.update_fun(
                self.state, self.main.state, self.main.track, self.config)

    @staticmethod
    def update_fun(state, main_state, track, config):
        t = Extension.main.timer
        solver = state.solver
        default = CarRacingCasadiConfig
        N = config.car_count

        t.s('cart 2 curv')
        cart_states = (CartesianState * N).from_buffer(main_state.car_states)
        curv_states = np.empty((default.n, N), dtype=float, order='F')
        for i in range(N):
            curv_states[:, i] = track.cart_to_curv(cart_states[i]).to_tuple()[:5]
        # (n,N)
        t.e('cart 2 curv')

        # Initial conditions for all cars
        # Initial guess for control sequence
        # From previous step
        # Stitch with stanley controller
        # Send to solver
        x0 = curv_states
        default = CarRacingCasadiConfig
        solver.guess = np.zeros((default.m, N, default.T), order='F')
        solver.x0 = x0
        CarRacingCasadiConfig.x0 = x0
        u_ref = np.zeros((default.m*N, default.T), order='F')
        print(x0)
        return
        sol = solver.solve_cpp_backend(u_ref)
        print(f'{sol.elapsed_time}, {sol.residual=}')
        # Stitch solution to previous traj (state, control)
