import numpy as np
from common import *
from extension.Extension import Extension

class CollisionChecker(Extension):
    def __init__(self,main):
        Extension.__init__(self,main)
        self.collision_count = [0] * len(self.main.cars)
        self.cumsum_collision_by_lap_vec = [[] * len(self.main.cars)]

    def update(self):
        for i in range(len(self.main.cars)):
            car = self.main.cars[i]
            if (car.controller.isInObstacle()):
                self.collision_count[i] += 1
                car.in_collision = True
                #print_ok(self.prefix(), "collision = %d"%(self.collision_count))
            else:
                car.in_collision = False
            try:
                if (car.laptimer.new_lap.is_set()):
                    self.cumsum_collision_by_lap_vec[i].append(self.collision_count[i])
            except AttributeError:
                pass

    def final(self):
        for i in range(len(self.main.cars)):
            print_ok(self.prefix(), "car %d, total collision = %d"%(i,self.collision_count[i]))
            self.main.cars[i].debug_dict.update({'collision_vec':self.cumsum_collision_by_lap_vec[i]})
        self.main.car_total_collisions = self.collision_count

