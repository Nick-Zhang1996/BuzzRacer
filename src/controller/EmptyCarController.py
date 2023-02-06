from common import *
from math import isnan,pi,degrees,radians
from controller.CarController import CarController
from controller.PidController import PidController

class EmptyCarController(CarController):
    def __init__(self, car,config):
        super().__init__(car,config)

    def control(self):
        valid = True
        # may be needed to keep car steady
        throttle = self.calcThrottle(0)
        steering = 0
        self.car.throttle = throttle
        self.car.steering = steering

        return valid
    def calcThrottle(self,ax):
        vx = self.car.states[3]
        return ax/6.17+0.333 + vx/15.2

