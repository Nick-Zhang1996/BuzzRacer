from common import *
from math import isnan, pi, degrees, radians
from buzzracer.controller.car_controller import CarController
from buzzracer.controller.pid_controller import PidController


class EmptyCarController(CarController):
    def __init__(self, car, config):
        super().__init__(car, config)
        self.car.throttle = 0
        self.car.steering = 0

    def control(self):
        valid = True
        # may be needed to keep car steady
        # throttle = self.calc_throttle(0)
        # steering = 0
        # self.car.throttle = throttle
        # self.car.steering = steering

        return valid

    def calc_throttle(self, ax):
        vx = self.car.states[3]
        return ax/6.17+0.333 + vx/15.2
