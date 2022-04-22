# dynamic simulator of a passenger vehicle
# page 30 of book vehicle dynamics and control
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from extension import Simulator

from sysid.tire import tireCurve
import numpy as np
from math import sin, cos, tan, radians, degrees, pi, atan
import matplotlib.pyplot as plt
from extension.Simulator import Simulator
from common import *
from threading import Event, Lock
from extension.simulator.KinematicSimulator import KinematicSimulator


class CalebDynamicSimulator(Simulator):
    def __init__(self, main):
        super().__init__(main)
        CalebDynamicSimulator.max_v = 3.0
        CalebDynamicSimulator.using_kinematics = False

    def init(self):
        super().init()
        self.cars = self.main.cars
        CalebDynamicSimulator.dt = self.main.dt
        KinematicSimulator.dt = CalebDynamicSimulator.dt
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
    # NOTE using car frame origined at CG with x pointing forward, y leftward
    # this method does NOT update car.sim_states, only returns a sim_state
    # this is to make itself useful for when update is not necessary
    #    x,y,psi,v_forward,v_sideway,d_psi = car_states
    @staticmethod
    def advanceDynamics(car_states, control, car):
        lf = car.lf
        lr = car.lr
        L = car.L
        h = 0.01
        m = car.m
        I = car.Iz
        g = 9.81

        motor_A = 6.17
        motor_B = 15.2
        motor_C = 0.333

        Df = 1.1  # 3.93731
        Dr = 1.1  # 6.23597
        C = 1.6  # 2.80646
        B = 2.3  # 0.51943

        dt = CalebDynamicSimulator.dt

        # NOTE here vx = vf, vy = vs, different convention
        xG, yG, heading, vf, vs, omega = car_states
        throttle, steering = control

        # for small longitudinal velocity use kinematic model
        if vf < 0.05:
            beta = atan(lr / L * tan(steering))
            norm = lambda a, b: (a ** 2 + b ** 2) ** 0.5
            # motor model
            d_vx = motor_A * (throttle - vf / motor_C - motor_B)
            vf = vf + d_vx * dt
            vs = norm(vf, vs) * sin(beta)
            phiddot = 0.0
            omega = vs / L * tan(steering)

            slip_f = 0
            slip_r = 0
            Ffy = 0
            Fry = 0

        else:
            accelForce = motor_A * (throttle - vf / motor_B - motor_C)  # motor model
            # WEIGHT SHIFT
            frontWeight = (accelForce * h + m * g * lr) / L
            rearWeight = (-accelForce * h + m * g * lf) / L

            frontslip = -(np.arctan2(vs + lf * omega, vf) - steering)
            rearslip = -np.arctan2((vs - lr * omega), vf)
            tc = lambda slip, D, weight: D * np.sin(C * np.arctan(B * slip)) * weight

            Flf = 0
            Fcf = tc(frontslip, Df, frontWeight)
            Flr = 0.5 * motor_A * (throttle - vf / motor_B - motor_C)  # motor model
            Fcr = tc(rearslip, Dr, rearWeight)

            Fxf = -Fcf * sin(steering)
            Fxr = Flr
            Fyf = Fcf * cos(steering)
            Fyr = Fcr

            # xddot = vs * omega - 2 / m * Cf * frontslip * np.sin(st) + th / m
            # xddot = vs * omega + 2 / m * Fxf + 2 / m * Fxr
            xddot = 2 * Fxr
            # yddot = -vf * omega + 2 / m * Cf * frontslip * np.cos(st) + 2 / m * Cr * rearslip
            yddot = -vf * omega + 2 / m * Fyf + 2 / m * Fyr
            phiddot = 2 * lf / I * Fyf - 2 * lr / I * Fyr

            vf += xddot * dt
            vs += yddot * dt
            omega += phiddot * dt

        # convert back to global
        vxG = vf * np.cos(heading) - vs * np.sin(heading)
        vyG = vf * np.sin(heading) + vs * np.cos(heading)

        xG += vxG * dt
        yG += vyG * dt
        heading += omega * dt + 0.5 * phiddot * dt**2

        car_states = xG, yG, heading, vf, vs, omega
        return np.array(car_states)

    def update(self):
        # print_ok(self.prefix() + "update")
        for car in self.cars:
            car.states = self.advanceDynamics(car.states, (car.throttle, car.steering), car)
            # print(self.prefix()+str(car.states))
            # print(self.prefix()+"T: %.1f, S:%.1f"%(car.throttle, degrees(car.steering)))
        self.main.new_state_update.set()
        self.main.sim_t += self.main.dt
        self.matchRealTime()
