from CarController import CarController
from PidController import PidController

import numpy as np
import math

class PidCarController(CarController):
    def __init__(self, car, config):
        super().__init__(car, config)

        self.curvature_compensation = 0.2
        self.max_offset = 0.15
        self.v_target_factor = 0.97

        P = 13 # to be more aggressive use 15
        I = 0.0 #0.1
        D = 0.4
        dt = self.car.main.dt

        self.throttle_pid = PidController(P,I,D,dt,1,2)

        P = 2.5 # to be more aggressive use 15
        I = 0.0 #0.1
        D = 0.5

        self.heading_pid = PidController(P,I,D,dt,1,2)


        P = 0.2 # to be more aggressive use 15
        I = 0.0 #0.1
        D = 0.4

        self.curvature_pid = PidController(P,I,D,dt,1,2)

    def control(self):
        trajectory = self.track.localTrajectory(self.car.states)
        
        if trajectory is None:
            print("no trajectory")
            return (0,0)
        
        (local_ctrl_pnt, offset, orientation, curvature, v_target) = trajectory

        v_target *= self.v_target_factor

        (x, y, heading, v_forward, _, _) = self.car.states
        
        desiredOffset = np.copysign(curvature * curvature, curvature) * self.curvature_compensation / v_target

        throttle = max(min(self.throttle_pid.control(v_target, v_forward),self.car.max_throttle),-1)
        steering =  self.heading_pid.control(min(max(desiredOffset, -self.max_offset), self.max_offset), offset)

        self.car.throttle = throttle
        self.car.steering = steering

        return (throttle, steering)