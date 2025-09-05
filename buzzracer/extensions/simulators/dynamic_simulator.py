''' Simulator for an Ackermann steering vehicle with dynamic bicycle model'''
# page 30 of book Vehicle Dynamics and Control

from math import sin, cos

import numpy as np

from buzzracer.types import CartesianState, Control
from buzzracer.extensions.simulator import Simulator
from buzzracer.cars.car import Car
from buzzracer.sysid.dynamic_bicycle_model import DynamicBicycleModel

class DynamicSimulator(Simulator):
    ''' Simulator for an Ackermann steering vehicle with dynamic bicycle model'''
    max_v = 3.0
    ''' Maximum speed a car can achieve '''
    using_kinematics = False
    ''' Use Kinematics model instead'''

    def init(self):
        super().init()
        for car in self.main.cars:
            self.add_car(car)
        self.main.new_state_update.set()

    def add_car(self, car: Car):
        '''Add a car to use DynamicSimulator for state updates
            car needs to (x,y,heading,v_forward,v_sideway,omega)
        '''
        x, y, heading, v_forward, v_sideway, _ = car.states
        car.Vx = v_forward
        car.Vy = v_sideway

        car.x = x
        car.y = y
        car.psi = heading

        car.d_x = car.Vx*cos(car.psi)-car.Vy*sin(car.psi)
        car.d_y = car.Vx*sin(car.psi)+car.Vy*cos(car.psi)
        car.d_psi = 0
        car.sim_states = np.array(
            [car.x, car.d_x, car.y, car.d_y, car.psi, car.d_psi])

        car.state_dim = 6
        car.control_dim = 2

        # not implemented: support for artificially added noise
        noise = False
        car.noise = noise
        noise_cov = np.diag([0.01]*6)
        if noise:
            car.noise_cov = noise_cov
            assert np.array(noise_cov).shape == (6, 6)

        # car.states_hist = []
        car.local_states_hist = []
        car.norm = []
        super().add_car(car)

    @staticmethod
    def advance_dynamics(car_states, control, car, dt):
        """advance dynamics by self.dt.
        NOTE using car frame origined at CG with x pointing forward, y leftward

        Args:
            car_states: Cartesian state of the car, (x,y,heading,v_forward,v_sideway,omega)
            control: (steering,throttle) steering in rad, left positive, throttle in [-1,1], 
                    positive indicates acceleration
            car: Car object, contains information about the car's kinematics, 
                also contains car.sim_states for simulators that do not use car.states for update
            dt: Time step to advance dynamics by, unit:seconds
        Return: 
            state at next time step.
        """
        x, y, heading, vx, vy, omega = car_states
        _state = CartesianState(x=x,
                               y=y,
                               heading=heading,
                               v_forward=vx,
                               v_sideway=vy,
                               omega=omega)
        steering, throttle = control
        _control = Control(steering=steering, throttle=throttle)
        next_car_state = DynamicBicycleModel.advance_dynamics(_state, _control, car, dt)
        return np.array(next_car_state)
