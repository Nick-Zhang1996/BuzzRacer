""" Game Theoretic Planner, responsible for generating ref traj for multiple cars """
from __future__ import annotations
from typing import TYPE_CHECKING
from math import radians

import numpy as np
from rd3g.games.car_racing_casadi import CarRacingCasadiConfig

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

    def init(self):
        # Load game theoretic solver
        self.cars = Extension.main.cars

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

        rd3g_config = CarRacingCasadiConfig(
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

    def update(self):
        state = self.state
        # Initial conditions for all cars
        # Initial guess for control sequence
        # From previous step
        # Stitch with stanley controller
        # Send to solver
        # Stitch solution to previous traj (state, control)
