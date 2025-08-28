''' Simulator for an Ackermann steering vehicle with dynamic bicycle model'''
# page 30 of book Vehicle Dynamics and Control

from math import sin, cos, tan, atan

import numpy as np

from buzzracer.extensions.simulators.kinematic_simulator import KinematicSimulator
from buzzracer.extensions.simulator import Simulator
from buzzracer.car.car import Car
from buzzracer.sysid.tire import tire_curve

class DynamicSimulator(Simulator):
    ''' Simulator for an Ackermann steering vehicle with dynamic bicycle model'''
    max_v = 3.0
    ''' Maximum speed a car can achieve '''
    using_kinematics = False
    ''' Use Kinematics model instead'''

    def init(self):
        super().init()
        DynamicSimulator.dt = self.main.dt
        KinematicSimulator.dt = DynamicSimulator.dt
        KinematicSimulator.max_v = DynamicSimulator.max_v
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
        lf = car.lf
        lr = car.lr
        L = car.L

        Iz = car.Iz
        m = car.m
        dt = DynamicSimulator.dt

        x, y, heading, vx, vy, omega = car_states
        steering, throttle = control

        # for small longitudinal velocity use kinematic model
        if (vx < 0.05):
            beta = atan(lr/L*tan(steering))
            def norm(a, b):
                return (a**2+b**2)**0.5
            # motor model
            d_vx = 6.17*(throttle - vx/15.2 - 0.333)
            vx = vx + d_vx * dt
            vy = norm(vx, vy)*sin(beta)
            d_omega = 0.0
            omega = vx/L*tan(steering)

            slip_f = 0
            slip_r = 0
            Ffy = 0
            Fry = 0

        else:
            slip_f = -np.arctan((omega*lf + vy)/vx) + steering
            slip_r = np.arctan((omega*lr - vy)/vx)

            # Ffy = Df * np.sin( C * np.arctan(B *slip_f)) * 9.8 * lr / (lr + lf) * m
            # Fry = Dr * np.sin( C * np.arctan(B *slip_r)) * 9.8 * lf / (lr + lf) * m
            Ffy = tire_curve(slip_f) * m * 9.8 * lr/(lr+lf)
            Fry = 1.15*tire_curve(slip_r) * m * 9.8 * lf/(lr+lf)

            # Dynamics
            # d_vx = 1.0/m * (Frx - Ffy * np.sin( steering ) + m * vy * omega)
            d_vx = 6.17*(throttle - vx/15.2 - 0.333)
            d_vy = 1.0/m * (Fry + Ffy * np.cos(steering) - m * vx * omega)
            d_omega = 1.0/Iz * (Ffy * lf * np.cos(steering) - Fry * lr)

            # discretization
            vx = vx + d_vx * dt
            vy = vy + d_vy * dt
            omega = omega + d_omega * dt

        # back to global frame
        vxg = vx*cos(heading)-vy*sin(heading)
        vyg = vx*sin(heading)+vy*cos(heading)

        # update x,y, heading
        x += vxg*dt
        y += vyg*dt
        heading += omega*dt + 0.5 * d_omega * dt * dt

        car_states = x, y, heading, vx, vy, omega
        return np.array(car_states)
