import numpy as np
from common import *
from Extension import Extension

class CollisionChecker(Extension):
    def __init__(self,main):
        Extension.__init__(self,main)
        print_ok("[CollisionChecker]: in use")
        self.car = self.main.cars[0]
        self.collision_count = 0

    def update(self):
        if (self.car.controller.isInObstacle()):
            self.collision_count += 1
            print_ok(self.prefix(), "collision = %d"%(self.collision_count))



    def final(self):
        print_ok(self.prefix(), "total collision = %d"%(self.collision_count))

