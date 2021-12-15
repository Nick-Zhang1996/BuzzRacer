# This is a linearized version of KinematicSimulator.py

# Also, the reference frame for these derivations is based at the middle of rear axle of the vehicle, unlike
# KinematicSimulator.py

# Linear Time Invariant Car Model
# This model creates LTI Kinematic Bicycle Models valid for a short period of time

# refer to paper
# The Kinematic Bicycle Model: a Consistent Model for Planning Feasible Trajectories for Autonomous Vehicles?

import numpy as np
from math import radians
from common import *
from threading import Event
from Simulator import Simulator


class LTIVarParamKinematicSimulator(Simulator):

    def __init__(self, main):
        super().__init__(main)

        # for when a specific car instance is not specified
        self.lr = 45e-3
        self.lf = 45e-3
        LTIVarParamKinematicSimulator.max_v = 3.0

    def init(self):
        super().init()
        self.cars = self.main.cars
        LTIVarParamKinematicSimulator.dt = self.main.dt
        for car in self.cars:
            self.addCar(car)
        self.main.new_state_update = Event()
        self.main.new_state_update.set()

    # add a car to be KinematicSimulator
    # car needs to have .lf, .lr, .L .states (x,y,heading,v_forward,v_sideways,omega)
    def addCar(self, car):
        x, y, heading, v_forward, v_sideways, omega = car.states
        car.sim_states = np.array([x, y, v_forward, heading])
        return

    def update(self):
        # print_ok("[KinematicSimulator]: update")
        for car in self.cars:
            car.states = self.advanceDynamics(car.states, (car.throttle, car.steering), car)
            # x, y, v, heading = car.sim_states
            # car.states = (x, y, heading, v, 0, 0)
            # print_info(self.prefix()+str(v))
        self.main.new_state_update.set()
        self.main.sim_t += self.main.dt
        self.matchRealTime()

    # x,y,v,heading = sim_states
    @staticmethod
    def advanceDynamics(sim_states, control, car):
        lr = car.lr
        lf = car.lf
        L = lr + lf
        m = car.m
        dt = LTIVarParamKinematicSimulator.dt

        x, y, heading, v_forward, v_sideways, omega = sim_states

        # Varying parameters
        v0 = v_forward
        psi0 = heading
        psidot0 = omega
        steer0 = control[1]

        # print("v = " + str(v_forward) + "     psidot0 = " + str(psidot0) + "      steer = " + str(steer0))

        A = np.zeros((4, 4))
        A[0][2] = np.cos(psi0)
        A[1][2] = np.sin(psi0)
        # A[2][2] = -np.tan(steer0) * (psidot0 + v0 / L * np.tan(steer0))
        A[3][2] = 1/L * np.tan(steer0)

        state = np.array([x, y, v_forward, heading])

        B = np.zeros((4, 2))
        B[2][0] = 1/m
        # B[2][1] = -v0 * (1 / (np.cos(steer0)) ** 2) * (psidot0 + v0 / L * np.tan(steer0))
        # print("B[2][1]:  " + str(B[2][1]))
        B[3][1] = (v0 / L) / (np.cos(steer0) ** 2)
        # print("B[3][1]:  " + str(B[3][1]))

        statedot = np.matmul(A, state) + np.matmul(B, control)  # xdot, ydot, vdot, psidot

        print('v0 = {0:.3f}    B[3][1] = {1:.3f}    head = {2:.3f}    steer = {3:.3f}    throttle = {4:.3f}'.format(v0, B[3][1], state[3], steer0, control[0]))
        # print("v = " + str(v_forward) + "     psidot0 = " + str(psidot0) + "    steer = " + str(steer0) + "    vdot = " + str(statedot[2]))
        # print("B[2][1]:  " + str(B[2][1]) + "B[3][1]:  " + str(B[3][1]) + "   throttle = " + str(control[0]))
        state += statedot * dt

        x = state[0]
        y = state[1]
        heading = state[3]
        v_forward = state[2]
        # v_sideways passes through
        omega = statedot[3]

        '''
        throttle = np.clip(throttle, -1.0, 1.0)
        steering = np.clip(throttle, -radians(27), radians(27))
        '''
        #x, y, v, heading = sim_states
        # slow down if car is in collision
        '''
        if (car.in_collision):
            v *= 0.9
        '''
        # throttle = control[0]
        # steering = control[1]

        # dt = LTIKinematicSimulator.dt


        # # Linear equations of motion
        # if (v > LTIKinematicSimulator.max_v):
        #     dvdt = -0.01
        # else:
        #     dvdt = throttle
        # dheadingdt = LTIKinematicSimulator.v_0 / (lf + lr) * steering
        #
        # v += dvdt * dt
        # heading += dheadingdt * dt
        #
        # # Non linear equations for coordinate transform
        # # dxdt and dydt are non-linear, but the small angle approx does not work for heading
        # dxdt = v * np.cos(heading)
        # dydt = v * np.sin(heading)
        #
        # x += dt * dxdt
        # y += dt * dydt
        return np.array([x, y, heading, v_forward, v_sideways, omega])
