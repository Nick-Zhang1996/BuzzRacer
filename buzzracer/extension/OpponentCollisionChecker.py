''' Check collision with opponents, for iLQGameCarController. '''
from extension.Extension import Extension


class OpponentCollisionChecker(Extension):
    ''' Check collision with opponents, for iLQGameCarController. 

    NOTE this only checks car 0
    '''
    def __init__(self):
        Extension.__init__(self, 'opponent_collision_checker')
        self.collision_count = 0
        self.last_collision_ts = -1e3
        self.lockout_timestep = 10
        self.in_collision = False
        self.timestep = 0

    def update(self):
        self.timestep += 1
        if (self.main.cars[0].controller.is_in_collision()):
            if (not self.in_collision and self.timestep > self.lockout_timestep+self.last_collision_ts):
                self.in_collision = True
                self.collision_count += 1
        else:
            if (self.in_collision):
                self.in_collision = False
                self.last_collision_ts = self.timestep

    def final(self):
        self.print_info('total car-car collision = %d' %
                        (self.collision_count))
        self.main.opponent_collision_count = self.collision_count
