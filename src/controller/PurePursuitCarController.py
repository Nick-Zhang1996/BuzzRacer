from common import *
from math import isnan,pi,degrees,radians
from controller.CarController import CarController
from controller.PidController import PidController
from planner import Planner

class PurePursuitCarController(CarController):
    def __init__(self, car,config):
        super().__init__(car,config)
        config_planner = config.getElementsByTagName('planner')[0]
        planner_class = eval(config_planner.firstChild.nodeValue)
        self.planner = planner_class(config_planner)
        self.planner.main = self.main
        self.planner.car = self.car
        self.print_ok("setting planner attributes")
        for key,value_text in config_planner.attributes.items():
            setattr(self.planner,key,eval(value_text))
            self.print_info(" main.",key,'=',value_text)
        self.planner.init()
        #self.planner.test()

    def control(self):
        self.planner.plan()
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

