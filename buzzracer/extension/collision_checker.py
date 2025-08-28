''' check collision with static obstacles '''
import numpy as np
from buzzracer.extension.extension import Extension
from buzzracer.car.car import Car


class CollisionChecker(Extension):

    def __init__(self):
        Extension.__init__(self, 'collision_checker')

        self.collision_count: dict[Car, int] = {
            car: 0
            for car in self.main.cars
        }
        ''' Running sum of collision count, resets every lap'''
        self.collision_by_lap_vec = {car: 0 for car in self.main.cars}
        ''' Collision count by lap'''
        self.car_is_in_collision: dict[Car, bool] = {
            car: False
            for car in self.main.cars
        }
        ''' Is car in collision at this time step'''

    def update(self):
        for car in self.main.cars:
            if (self.main.track.is_in_obstacle(car.states)[0]):
                self.print_info('collision with obstacle')
                self.collision_count[car] += 1
                self.car_is_in_collision[car] = True
            else:
                self.car_is_in_collision[car] = False
            try:
                if (car.laptimer.new_lap.is_set()):
                    self.collision_by_lap_vec[car].append(
                        self.collision_count[car])
                    self.collision_count[car] = 0
            except AttributeError:
                pass

    def final(self):
        total_vec = []
        mean_vec = []
        for i in range(len(self.main.cars)):
            total = np.sum(self.collision_by_lap_vec[i][1:])
            mean = np.mean(self.collision_by_lap_vec[i][1:])
            total_vec.append(total)
            mean_vec.append(mean)
            self.print_info(
                'car %d, total obstacle collision = %d, mean = %.2f', i, total,
                mean)
        self.main.car_total_collisions = total_vec
