from common import *
from math import isnan,pi,degrees,radians,sin,cos
from controller.CarController import CarController
from controller.PidController import PidController

class PointMassMpcCarController(CarController):
    ''' 
    for sanity check of point mass curvilinear model
    '''
    def __init__(self, car,config):
        super().__init__(car,config)

    def init(self):
        self.simulator = self.main.simulator
        assert(isinstance(self.simulator,CurvilinearSimulator))

    def control(self):
        for car in self.main.cars:
            # s,v,n,phi
            throttle = 1.0 if car.sim_states[1] < 1.0 else -1.0
            steering = -car.sim_states[3] - car.sim_states[2]
            print(car.sim_states)
            print(f'T = {throttle} S = {steering}')

            car.throttle = throttle
            car.steering = steering
        return

