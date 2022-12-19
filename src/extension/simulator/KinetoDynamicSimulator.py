# dynamic simulator of a passenger vehicle
# page 30 of book vehicle dynamics and control
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from extension import Simulator

from sysid.tire import tireCurve
import numpy as np
from math import sin, cos, tan, radians, degrees, pi, atan, fabs
import matplotlib.pyplot as plt
from extension.Simulator import Simulator
from common import *
from threading import Event, Lock
from extension.simulator.KinematicSimulator import KinematicSimulator


class KinetoDynamicSimulator(Simulator):
    def __init__(self, main):
        super().__init__(main)
        KinetoDynamicSimulator.max_v = 3.0
        KinetoDynamicSimulator.using_kinematics = False

    def init(self):
        super().init()
        self.cars = self.main.cars
        KinetoDynamicSimulator.dt = self.main.dt
        KinematicSimulator.dt = KinetoDynamicSimulator.dt
        KinematicSimulator.max_v = 100
        for car in self.cars:
            self.addCar(car)
        self.main.new_state_update.set()

    # add a car to be DynamicSimu  
    # car needs to (x,y,heading,v_forward,v_sideway,omega)
    def addCar(self, car):
        x, y, heading, v_forward, v_sideway, omega = car.states
        car.Vx = v_forward
        car.Vy = v_sideway

        car.x = x
        car.y = y
        car.psi = heading

        car.d_x = car.Vx * cos(car.psi) - car.Vy * sin(car.psi)
        car.d_y = car.Vx * sin(car.psi) + car.Vy * cos(car.psi)
        car.d_psi = 0
        car.sim_states = np.array([car.x, car.d_x, car.y, car.d_y, car.psi, car.d_psi])

        car.state_dim = 6
        car.control_dim = 2

        # not implemented: support for artificially added noise
        noise = False
        car.noise = noise
        if noise:
            car.noise_cov = noise_cov
            assert np.array(noise_cov).shape == (6, 6)

        # car.states_hist = []
        car.local_states_hist = []
        car.norm = []

    # advance vehicle dynamics
    # this method does NOT update car.sim_states, only returns a sim_state
    # this is to make itself useful for when update is not necessary
    @staticmethod
    def advanceDynamics(car_states, control, curvature, car):
        lf = car.lf
        lr = car.lr
        L = car.L

        Iz = car.Iz
        m = car.m
        dt = KinetoDynamicSimulator.dt

        K_us = 0.02852
        motor_A = 27.42298
        tau_a = 0.379826
        k_D = 0.0942299
        c_r = 4.49905

        tau_delta = 0.0613
        tau_omega = 0.12574

        a_x, delta, v_x, Omega, zeta, n, xi = car_states
        throttle, delta_0 = control
        a_x0 = motor_A * throttle

        # v_x = vxActual[j-1]
        Omegadot = 1 / tau_omega * (v_x / L * (delta + K_us) - Omega)
        v_xdot = a_x - k_D / m * v_x ** 2 - c_r * v_x
        a_xdot = 1 / tau_a * (a_x0 - a_x)
        deltadot = 1 / tau_delta * (delta_0 - delta)

        zetadot = - (v_x * np.cos(xi)) / (n * curvature - 1)
        # print("zeta dot: ", zetadot)
        ndot = v_x * np.sin(xi)
        xidot = Omega + (v_x * np.cos(xi) * curvature) / (n * curvature - 1)

        # Left Riemann Sum integrate
        Omega += Omegadot * dt
        v_x += v_xdot * dt
        a_x += a_xdot * dt
        delta += deltadot * dt
        zeta += zetadot * dt
        n += ndot * dt
        xi += xidot * dt

        car_states = (a_x, delta, v_x, Omega, zeta, n, xi)

        return np.array(car_states)

    def update(self):
        # print_ok(self.prefix() + "update")
        for car in self.cars:
            zeta = car.states[4]
            curvature = self.getCurvature(zeta, car)
            car.states = self.advanceDynamics(car.states, (car.throttle, car.steering), curvature, car)
            # print(self.prefix()+str(car.states))
            # print(self.prefix()+"T: %.1f, S:%.1f"%(car.throttle, degrees(car.steering)))
        self.main.new_state_update.set()
        self.main.sim_t += self.main.dt
        self.matchRealTime()

    # guess is a zeta value
    def getCurvature(self, zeta, car):
        idx = np.searchsorted(car.controller.ss, zeta, side="left")
        if idx > 0 and (idx == len(car.controller.ss) or fabs(zeta - self.ss[idx - 1]) < fabs(zeta - self.ss[idx])):
            return car.controller.k_signed_smooth[idx - 1]
        else:
            return car.controller.k_signed_smooth[idx]