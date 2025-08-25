import numpy as np
from common import *
from extension.Extension import Extension

# count number of times car is in collision with boundary


class BoundaryChecker(Extension):
    def __init__(self, main):
        Extension.__init__(self, main)
        # running sum of collision count, reset every lap
        self.collision_count = [0] * len(self.main.cars)

    def post_init(self):
        self.discretized_raceline = self.main.track.discretized_raceline
        for car in self.main.cars:
            car.in_collision = False

    def update(self):
        for i in range(len(self.main.cars)):
            car = self.main.cars[i]
            if (self.is_out_of_boundary(car)):
                if (not car.in_collision):
                    car.in_collision = True
                    self.collision_count[i] += 1
                    print_ok(self.prefix(), 'car %d collision = %d' %
                             (i, self.collision_count[i]))
            else:
                if (car.in_collision):
                    car.in_collision = False

    def final(self):
        for i in range(len(self.main.cars)):
            self.print_info('car %d, total boundary violation = %d' %
                            (i, self.collision_count[i]))
            self.main.cars[i].total_boundary_collision = self.collision_count[i]

    def is_out_of_boundary(self, car):
        car_coord = car.states[0:2]
        car_heading = car.states[2]
        left, right = self.main.track.precise_track_boundary(
            car_coord, car_heading)
        out = left < 0 or right < 0
        return out

    def is_out_of_boundary_discrete(self, car):
        x, y, heading, vf, vs, omega = car.states
        ref_points = self.discretized_raceline[:, 0:2]
        ref_heading = self.discretized_raceline[:, 2]
        left_bdry = self.discretized_raceline[:, 3]
        right_bdry = self.discretized_raceline[:, 4]
        # self.discretized_raceline = np.vstack([self.raceline_points,self.raceline_headings,vv, self.raceline_left_boundary, self.raceline_right_boundary]).T
        dx_vec = ref_points[:, 0]-x
        dy_vec = ref_points[:, 1]-y
        dist_vec = ((dx_vec)**2 + (dy_vec)**2)**0.5
        idx = np.argmin(dist_vec)
        dist = dist_vec[idx]
        dx = dx_vec[idx]
        dy = dy_vec[idx]

        raceline_to_point_angle = np.arctan2(dy, dx)
        heading_diff = np.mod(raceline_to_point_angle -
                              ref_heading[idx] + np.pi, 2*np.pi) - np.pi
        margin = 0.05
        if (heading_diff > 0):
            out = dist + margin > left_bdry[idx]
        else:
            out = dist + margin > right_bdry[idx]
        return out
        '''
  // performance barrier FIXME
  find_closest_id(state,*u_estimate,  &idx,&dist);
  *u_estimate = idx;
  
  float tangent_angle = raceline[idx][RACELINE_HEADING];
  float raceline_to_point_angle = atan2f(raceline[idx][RACELINE_Y] - state[STATE_Y], raceline[idx][RACELINE_X] - state[STATE_X]) ;
  float angle_diff = fmodf(raceline_to_point_angle - tangent_angle + PI, 2*PI) - PI;

  float cost;

  if (angle_diff > 0.0){
    // point is to left of raceline
    cost = (dist +0.05> raceline[idx][RACELINE_LEFT_BOUNDARY])? 1:0;

  } else {
    cost = (dist +0.05> raceline[idx][RACELINE_RIGHT_BOUNDARY])? 1:0;
  }
        '''
