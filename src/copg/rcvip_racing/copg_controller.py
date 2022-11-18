# CopgCarController

import sys
sys.path.insert(0,'../..') # inorder to run within the folder

from common import PrintObject
import numpy as np
from controller.CarController import CarController

class CopgCarController(CarController):
    def __init__(self, car,config):
        super().__init__(car,config)

    def control(self):
            self.car.throttle = throttle
            self.car.steering = steering
