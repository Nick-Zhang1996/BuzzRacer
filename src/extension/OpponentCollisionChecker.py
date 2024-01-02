import numpy as np
from common import *
from extension.Extension import Extension

# check collision with opponents, for iLQGameCarController
class OpponentCollisionChecker(Extension):
    def __init__(self,main):
        Extension.__init__(self,main)
        self.collision_count = 0
        self.last_collision_ts = -1e3
        self.lockout_timestep = 10
        self.in_collision = False
        self.timestep = 0

    def update(self):
        self.timestep += 1
        if (self.main.cars[1].controller.isInCollision()):
            if (not self.in_collision and self.timestep > self.lockout_timestep+self.last_collision_ts):
                self.in_collision = True
                self.collision_count += 1
        else:
            if (self.in_collision):
                self.in_collision = False
                self.last_collision_ts = self.timestep

    def final(self):
        self.print_info("total car-car collision = %d"%(self.collision_count))
        self.main.opponent_collision_count = self.collision_count

