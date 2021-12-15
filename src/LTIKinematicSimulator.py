# This is a (partially) linearized version of KinematicSimulator.py
# The small angle assumption cannot be made and therefore sin(heading) and cos(heading) remain nonlinear

# Also, the reference frame for these derivations is based at the middle of rear axle of the vehicle, unlike
# KinematicSimulator.py

# Linear Time Invariant Car Model
# This model is a result of linearizing the Kinematic Bicycle Model

# refer to paper
# The Kinematic Bicycle Model: a Consistent Model for Planning Feasible Trajectories for Autonomous Vehicles?

import numpy as np
from math import radians
from common import *
from threading import Event
from Simulator import Simulator


class LTIKinematicSimulator(Simulator):

    def __init__(self, main):
        super().__init__(main)

        # for when a specific car instance is not specified
        self.lr = 45e-3
        self.lf = 45e-3
        LTIKinematicSimulator.max_v = 3.0
        LTIKinematicSimulator.v_0 = 2.0  # need to provide a better value

    def init(self):
        super().init()
        self.cars = self.main.cars
        LTIKinematicSimulator.dt = self.main.dt
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
            car.sim_states = self.advanceDynamics(car.sim_states, (car.throttle, car.steering), car)
            x, y, v, heading = car.sim_states
            car.states = (x, y, heading, v, 0, 0)
            # print_info(self.prefix()+str(v))
        self.main.new_state_update.set()
        self.main.sim_t += self.main.dt
        self.matchRealTime()

    # x,y,v,heading = sim_states
    @staticmethod
    def advanceDynamics(sim_states, control, car):
        lr = car.lr
        lf = car.lf
        v_0 = LTIKinematicSimulator.v_0

        '''
        throttle = np.clip(throttle, -1.0, 1.0)
        steering = np.clip(throttle, -radians(27), radians(27))
        '''
        x, y, v, heading = sim_states
        # slow down if car is in collision
        '''
        if (car.in_collision):
            v *= 0.9
        '''
        throttle = control[0]
        steering = control[1]

        dt = LTIKinematicSimulator.dt


        # Linear equations of motion
        if (v > LTIKinematicSimulator.max_v):
            dvdt = -0.01
        else:
            dvdt = throttle
        dheadingdt = LTIKinematicSimulator.v_0 / (lf + lr) * steering

        v += dvdt * dt
        heading += dheadingdt * dt

        # Non linear equations for coordinate transform
        # dxdt and dydt are non-linear, but the small angle approx does not work for heading
        dxdt = v * np.cos(heading)
        dydt = v * np.sin(heading)

        x += dt * dxdt
        y += dt * dydt
        return np.array([x, y, v, heading])
