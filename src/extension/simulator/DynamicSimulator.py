# dynamic simulator of a passenger vehicle
# page 30 of book vehicle dynamics and control
from math import sin, cos, tan, radians, degrees, pi, atan
from threading import Event, Lock
from simulator.KinematicSimulator import KinematicSimulator
from common import *
from Simulator import Simulator
import matplotlib.pyplot as plt
import numpy as np
from sysid.tire import tire_curve
from extension import Simulator
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


class DynamicSimulator(Simulator):
    def __init__(self, main):
        super().__init__(main)
        DynamicSimulator.max_v = 3.0
        DynamicSimulator.using_kinematics = False

    def init(self):
        super().init()
        self.cars = self.main.cars
        DynamicSimulator.dt = self.main.dt
        KinematicSimulator.dt = DynamicSimulator.dt
        KinematicSimulator.max_v = 100
        for car in self.cars:
            self.add_car(car)
        self.main.new_state_update.set()

    # add a car to be DynamicSimu
    # car needs to (x,y,heading,v_forward,v_sideway,omega)
    def add_car(self, car):
        x, y, heading, v_forward, v_sideway, omega = car.states
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

    # advance vehicle dynamics
    # NOTE using car frame origined at CG with x pointing forward, y leftward
    # this method does NOT update car.sim_states, only returns a sim_state
    # this is to make itself useful for when update is not necessary
    #    x,y,psi,v_forward,v_sideway,d_psi = car_states
    @staticmethod
    def advance_dynamics(car_states, control, car):
        lf = car.lf
        lr = car.lr
        L = car.L

        Iz = car.Iz
        m = car.m
        dt = DynamicSimulator.dt

        # NOTE here vx = vf, vy = vs, different convention
        x, y, heading, vx, vy, omega = car_states
        steering, throttle = control

        # for small longitudinal velocity use kinematic model
        if (vx < 0.05):
            beta = atan(lr/L*tan(steering))
            def norm(a, b): return (a**2+b**2)**0.5
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
