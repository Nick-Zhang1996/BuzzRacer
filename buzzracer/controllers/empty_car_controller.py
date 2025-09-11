''' Empty Car contorller, doesn't do anything, mainly for testing'''
from buzzracer.controllers.car_controller import CarController


class EmptyCarController(CarController):
    def __init__(self, car, config):
        super().__init__(car, config)
        self.car.throttle = 0
        self.car.steering = 0

    def control(self):
        valid = True
        throttle = self.calc_throttle(0) + 0.1
        self.car.throttle = throttle
        self.car.steering = 0

        return valid

    def calc_throttle(self, ax):
        vx = self.car.state[3]
        return ax/6.17+0.333 + vx/15.2
