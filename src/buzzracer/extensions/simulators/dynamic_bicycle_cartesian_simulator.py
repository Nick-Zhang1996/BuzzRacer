''' Simulator for an Ackermann steering vehicle with dynamic bicycle model'''
# page 30 of book Vehicle Dynamics and Control
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from buzzracer.types import CartesianState, Control
from buzzracer.extensions.simulator import Simulator, SimulatorConfig
from buzzracer.extensions.extension import Extension, ExtensionState
from buzzracer.sysid.dynamic_bicycle_model import DynamicBicycleModelCartesian

if TYPE_CHECKING:
    from buzzracer.cars.car import Car


@Extension.register('simulator', SimulatorConfig, ExtensionState)
class DynamicBicycleCartesianSimulator(Simulator):
    ''' Simulator for an Ackermann steering vehicle with dynamic bicycle model'''
    max_v = 3.0
    ''' Maximum speed a car can achieve '''
    using_kinematics = False
    ''' Use Kinematics model instead'''
    state_type = CartesianState

    def init(self):
        super().init()
        for car in self.main.cars:
            self.add_car(car)
        self.main.new_state_update.set()

    def add_car(self, car: Car):
        '''Add a car to use DynamicBicycleCartesianSimulator for state updates
            car needs to (x,y,heading,v_forward,v_sideway,omega)
        '''
        car.sim_state = car.state

        car.state_dim = 6
        car.control_dim = 2

        # not implemented: support for artificially added noise
        noise = False
        car.noise = noise
        noise_cov = np.diag([0.01]*6)
        if noise:
            car.noise_cov = noise_cov
            assert np.array(noise_cov).shape == (6, 6)

        # car.state_hist = []
        car.local_states_hist = []
        car.norm = []
        super().add_car(car)

    @staticmethod
    def advance_dynamics(state: CartesianState,
                         control: Control,
                         car: Car,
                         dt: float,
                         curvature: float = None) -> CartesianState:
        """advance dynamics by dt.

        Args:
            state: state of the car, may be CartesianState or CurvilinearState
            control: (steering,throttle) steering in rad, left positive, throttle in [-1,1], 
                    positive indicates acceleration
            car: Car object, contains information about the car's kinematics, 
                also contains car.sim_state for simulators that do not use car.state for update
            dt: Time step to advance dynamics by, unit:seconds
            curvature: unused, only for curvilinear
        Return: 
            state at next time step.
        """
        x, y, heading, vx, vy, omega = state
        _state = CartesianState(x=x,
                                y=y,
                                heading=heading,
                                v_forward=vx,
                                v_sideway=vy,
                                omega=omega)
        steering, throttle = control
        _control = Control(steering=steering, throttle=throttle)
        next_car_state = DynamicBicycleModelCartesian.advance_dynamics(_state, _control, car, dt)
        return next_car_state
