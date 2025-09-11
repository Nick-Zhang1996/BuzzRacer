from common import *
from math import isnan, pi, degrees, radians, sin, cos
from buzzracer.controllers.car_controller import CarController
from buzzracer.controllers.pid_controller import PidController


class PointMassMpcCarController(CarController):
    """for sanity check of point mass curvilinear model."""

    def __init__(self, car, config):
        super().__init__(car, config)

    def init(self):
        self.simulator = self.main.simulator
        assert (isinstance(self.simulator, KinematicBicycleCurvilinearSimulator))

    def control(self):
        for car in self.main.cars:
            # s,v,n,phi
            throttle = 1.0 if car.sim_state[1] < 1.0 else -1.0
            steering = -car.sim_state[3] - car.sim_state[2]
            print(car.sim_state)
            print(f'T = {throttle} S = {steering}')

            car.throttle = throttle
            car.steering = steering
        return
