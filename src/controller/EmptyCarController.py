from common import *
from math import isnan,pi,degrees,radians
from controller.CarController import CarController
from controller.PidController import PidController

class EmptyCarController(CarController):
    def __init__(self, car,config):
        super().__init__(car,config)

    def control(self):
        valid = True
        throttle = self.calcThrottle(0)
        steering = 0
        if valid:
            self.car.throttle = throttle
            self.car.steering = steering
        else:
            print_warning("[StanleyCarController]: car %d invalid results from ctrlCar", self.car.id)
            self.car.throttle = 0.0
            self.car.steering = 0.0

        return valid
    def calcThrottle(self,ax):
        vx = self.car.states[3]
        return ax/6.17+0.333 + vx/15.2

