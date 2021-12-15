# This is a linearized version of DynamicSimulator.py

# Linear Time Invariant Car Model
# This model is a result of linearizing the Dynamic Bicycle Model

import numpy as np

from common import *
from threading import Event
from Simulator import Simulator
from math import degrees


class LTIDynamicSimulator(Simulator):

    def __init__(self, main):
        super().__init__(main)

        # Insert default parameters here
        # param1 =
        # param2 =
        LTIDynamicSimulator.v_fw_0 = 1.0  # need to provide a better value
        LTIDynamicSimulator.max_v = 3.0

    def init(self):
        super().init()
        self.cars = self.main.cars
        LTIDynamicSimulator.dt = self.main.dt
        for car in self.cars:
            self.addCar(car)
        self.main.new_state_update = Event()
        self.main.new_state_update.set()

    # THIS IS COPIED STRAIGHT FROM DYNAMIC SIMULATOR
    # add a car to be DynamicSim
    # car needs to (x,y,heading,v_forward,v_sideways,omega)
    def addCar(self, car):
        x, y, heading, v_forward, v_sideway, omega = car.states
        car.Vx = v_forward
        car.Vy = v_sideway

        car.x = x
        car.y = y
        car.psi = heading

        car.d_x = car.Vx * np.cos(car.psi) - car.Vy * np.sin(car.psi)
        car.d_y = car.Vx * np.sin(car.psi) + car.Vy * np.cos(car.psi)
        car.d_psi = 0
        car.sim_states = np.array([car.x, car.d_x, car.y, car.d_y, car.psi, car.d_psi])

        car.state_dim = 6
        car.control_dim = 2

        # not implemented: support for artificially added noise
        # noise = False
        # car.noise = noise
        # if noise:
        #     car.noise_cov = noise_cov
        #     assert np.array(noise_cov).shape == (6, 6)

        # car.states_hist = []
        car.local_states_hist = []
        car.norm = []

    def update(self):
        # print_ok(self.prefix() + "update")
        for car in self.cars:
            car.states = self.advanceDynamics(car.states, (car.throttle, car.steering), car)
            print(self.prefix() + str(car.states))
            print(self.prefix() + "T: %.1f, S:%.1f" % (car.throttle, degrees(car.steering)))
        self.main.new_state_update.set()
        self.main.sim_t += self.main.dt
        self.matchRealTime()

    # x,y,v,heading = sim_states
    @staticmethod
    def advanceDynamics(sim_states, control, car):
        lf = car.lf
        lr = car.lr
        # L = car.L

        # Df = car.Df
        # Dr = car.Dr
        # B = car.B
        # C = car.C
        Cf = car.C
        Cr = car.C
        # Cm1 = car.Cm1
        # Cm2 = car.Cm2
        # Cr = car.Cr
        # Cd = car.Cd
        Iz = car.Iz
        m = car.m

        multBy2 = False
        if multBy2:
            factor = 2
        else:
            factor = 1


        '''
        throttle = np.clip(throttle, -1.0, 1.0)
        steering = np.clip(throttle, -radians(27), radians(27))
        '''
        X, Y, heading, v_forward, v_sideways, omega = sim_states
        # slow down if car is in collision
        '''
        if (car.in_collision):
            v *= 0.9
        '''
        # throttle = control[0]
        # steering = control[1]

        dt = LTIDynamicSimulator.dt
        v_fw_0 = LTIDynamicSimulator.v_fw_0

        A11 = factor * (Cf + Cr) / (m * v_fw_0)
        A12 = factor * (Cf * lf - Cr * lr) / (m * v_fw_0) - v_fw_0  # factor intentionally not on the v_fw_0 on the tail end here
        A21 = factor * (Cf * lf - Cr * lr) / (Iz * v_fw_0)
        A22 = factor * (Cf * lf ** 2 + Cr * lr ** 2) / (Iz * v_fw_0)

        A = np.zeros((3, 3))  # generalize this to the length of the state vector
        A[1][1] = A11
        A[1][2] = A12
        A[2][1] = A21
        A[2][2] = A22

        state = sim_states[3:6]  # get just v_forward, v_sideways, and omega

        B00 = 1/m * car.Cm1
        B11 = -factor * Cf / m
        B21 = -factor * Cf * lf / Iz

        B = np.zeros((3, 2))  # generalize this to length of state vector, number of inputs
        B[0][0] = B00
        B[1][1] = B11
        B[2][1] = B21

        statedot = np.matmul(A, state) + np.matmul(B, control)

        newState = state + statedot * dt
        v_forward, v_sideways, omega = newState

        # Go back to reference frame. This is nonlinear!!
        dXdt = v_forward * np.cos(heading) - v_sideways * np.sin(heading)
        dYdt = v_forward * np.sin(heading) + v_sideways * np.cos(heading)

        X += dt * dXdt
        Y += dt * dYdt
        heading += dt * newState[2]  # newState[2] is omega

        return np.array([X, Y, heading, v_forward, v_sideways, omega])
