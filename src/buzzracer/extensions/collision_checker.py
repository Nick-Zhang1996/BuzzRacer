''' check collision with static obstacles 
UNTESTED '''
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from buzzracer.extensions.extension import Extension, ExtensionConfig, ExtensionState
if TYPE_CHECKING:
    from buzzracer.cars.car import Car


class CollisionCheckerState(ExtensionState):
    def __init__(self, config):
        super().__init__(config)
        self.collision_count: dict[Car, int]
        ''' Running sum of collision count, resets every lap'''
        self.collision_by_lap_vec: dict[Car, list[int]]
        ''' Collision count by lap'''
        self.car_is_in_collision: dict[Car, bool]
        ''' Is car in collision at this time step'''


@Extension.register('collision_checker', ExtensionConfig, CollisionCheckerState)
class CollisionChecker(Extension):
    """ Check collisions with obstacles"""

    def __init__(self, config, state):
        super().__init__(config, state)

        self.state.collision_count = {
            car: 0
            for car in self.main.cars
        }
        self.state.collision_by_lap_vec = {car: [] for car in self.main.cars}
        self.state.car_is_in_collision = {
            car: False
            for car in self.main.cars
        }

    def update(self):
        for car in self.main.cars:
            if (self.main.track.is_in_obstacle(car.state)[0]):
                self.print_info('collision with obstacle')
                self.state.collision_count[car] += 1
                self.state.car_is_in_collision[car] = True
            else:
                self.state.car_is_in_collision[car] = False
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
        for i, car in enumerate(self.main.cars):
            total = np.sum(self.state.collision_by_lap_vec[car][1:])
            mean = np.mean(self.state.collision_by_lap_vec[car][1:])
            total_vec.append(total)
            mean_vec.append(mean)
            self.print_info('car %d, total obstacle collision = %d, mean = %.2f', i, total, mean)
        self.main.car_total_collisions = total_vec
