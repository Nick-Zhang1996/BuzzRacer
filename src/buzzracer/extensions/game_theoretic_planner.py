""" Game Theoretic Planner, responsible for generating ref traj for multiple cars """
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

from buzzracer.extensions.extension import Extension, ExtensionConfig, ExtensionState

if TYPE_CHECKING:
    from buzzracer.cars.car import Car


class GameTheoreticPlannerConfig(ExtensionConfig):
    """ Read-only config """
    horizon: int  # Game horizon
    car_count: int  # Number of cars, N
    stitching_steps: int  # Number of steps to keep in previous trajectory in next iteration


class GameTheoreticPlannerState(ExtensionState):
    """ Planner State, mutable"""
    cars: Car  # (N, ) Cars under this planner
    cart_traj: np.ndarray  # (n=6, N, T) of Cartesian State Trajectory
    curv_traj: np.ndarray  # (n=6, N, T) of Curvilinear State Trajectory
    ctrl_traj: np.ndarray  # (m, N, T) of ctrl trajectory


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

    def update(self):
        state = self.state
        # Initial conditions for all cars
        # Initial guess for control sequence
        # From previous step
        # Stitch with stanley controller
        # Send to solver
        # Stitch solution to previous traj (state, control)
