''' Extension to check boundary violations '''
import numpy as np

from buzzracer.extension.Extension import Extension
from buzzracer.car.Car import Car

# count number of times car is in collision with boundary


class BoundaryChecker(Extension):
    ''' Extension to check boundary violations. '''

    def __init__(self):
        Extension.__init__(self, 'boundary_checker')
        self.collision_count: dict[Car, int] = {
            car: 0
            for car in self.main.cars
        }
        ''' Running sum of collision count, resets every lap'''
        self.discretized_raceline = self.main.track.discretized_raceline
        ''' Local reference of discretized raceline'''
        self.car_is_in_collision: dict[Car, bool] = {
            car: False
            for car in self.main.cars
        }
        ''' Dict to indicate if a car is in collision'''

    def update(self):
        for car in self.main.cars:
            if self.is_out_of_boundary(car):
                if not car.in_collision:
                    car.in_collision = True
                    self.collision_count[car] += 1
                    self.print_ok(
                        self.prefix(), 'car %d collision = %d' %
                        (car.id, self.collision_count[car]))
            else:
                self.car_is_in_collision = False

    def final(self):
        for car in self.main.cars:
            self.print_info('car %d, total boundary violation = %d' %
                            (car.id, self.collision_count[car]))
            car.total_boundary_collision = self.collision_count[car.id]

    def is_out_of_boundary(self, car):
        car_coord = car.states[0:2]
        car_heading = car.states[2]
        left, right = self.main.track.precise_track_boundary(
            car_coord, car_heading)
        out = left < 0 or right < 0
        return out

    def is_out_of_boundary_discrete(self, car):
        #x, y, heading, vf, vs, omega = car.states
        x, y, _ = car.states
        ref_points = self.discretized_raceline[:, 0:2]
        ref_heading = self.discretized_raceline[:, 2]
        left_bdry = self.discretized_raceline[:, 3]
        right_bdry = self.discretized_raceline[:, 4]
        # self.discretized_raceline = np.vstack([
        # self.raceline_points,
        # self.raceline_headings,
        # vv,
        # self.raceline_left_boundary,
        # self.raceline_right_boundary]).T
        dx_vec = ref_points[:, 0] - x
        dy_vec = ref_points[:, 1] - y
        dist_vec = ((dx_vec)**2 + (dy_vec)**2)**0.5
        idx = np.argmin(dist_vec)
        dist = dist_vec[idx]
        dx = dx_vec[idx]
        dy = dy_vec[idx]

        raceline_to_point_angle = np.arctan2(dy, dx)
        heading_diff = np.mod(
            raceline_to_point_angle - ref_heading[idx] + np.pi,
            2 * np.pi) - np.pi
        margin = 0.05
        if heading_diff > 0:
            out = dist + margin > left_bdry[idx]
        else:
            out = dist + margin > right_bdry[idx]
        return out
