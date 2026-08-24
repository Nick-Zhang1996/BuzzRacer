"""Simulator for the simplified drift dynamics model."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from buzzracer.extensions.extension import Extension, ExtensionState
from buzzracer.extensions.simulator import Simulator, SimulatorConfig
from buzzracer.sysid.drift_model import DriftModel
from buzzracer.types import CartesianState, Control

if TYPE_CHECKING:
    from buzzracer.cars.car import Car
    from buzzracer.cars.car_param import CarParam


@Extension.register('simulator', SimulatorConfig, ExtensionState)
class DriftSimulator(Simulator):
    """Simulator for the simplified cartesian drift model."""

    max_v = 3.0
    using_kinematics = False
    state_type = CartesianState

    def init(self):
        super().init()
        for car in self.main.cars:
            self.add_car(car)
        self.main.new_state_update.set()

    def add_car(self, car: Car):
        """Register a car using the simplified drift model."""
        car.sim_state = car.state
        car.state_dim = 6
        car.control_dim = 2

        noise = False
        car.noise = noise
        noise_cov = np.diag([0.01] * 6)
        if noise:
            car.noise_cov = noise_cov
            assert np.array(noise_cov).shape == (6, 6)

        car.local_states_hist = []
        car.norm = []
        super().add_car(car)

    @staticmethod
    def advance_dynamics(state: CartesianState,
                         control: Control,
                         car_param: CarParam,
                         dt: float,
                         curvature: float = None) -> CartesianState:
        """Advance the simplified drift dynamics by one simulation step."""
        del curvature
        x, y, heading, vx, vy, omega = state
        steering, throttle = control
        return DriftModel.advance_dynamics(
            CartesianState(
                x=x,
                y=y,
                heading=heading,
                v_forward=vx,
                v_sideway=vy,
                omega=omega,
            ),
            Control(steering=steering, throttle=throttle),
            car_param,
            dt,
        )
