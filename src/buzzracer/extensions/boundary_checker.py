''' Extension to check boundary violations '''
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

from buzzracer.types import CartesianState, CurvilinearState
from buzzracer.common import get_logger
from buzzracer.extensions.extension import Extension, ExtensionConfig, ExtensionState
if TYPE_CHECKING:
    from buzzracer.cars.car import Car

logger = get_logger(__name__)


class BoundaryCheckerConfig(ExtensionConfig):
    def __init__(self, main_config):
        super().__init__(main_config)
        self.reset_pos = False
        """If True, snap cars back to their last in-boundary pose with zero velocity."""


class BoundaryCheckerState(ExtensionState):
    def __init__(self, config):
        super().__init__(config)
        self.discretized_raceline: np.ndarray
        self.collision_count: dict[Car, int]
        self.car_is_in_collision: dict[Car, bool]
        self.last_in_boundary_state: dict[Car, CartesianState]


@Extension.register('boundary_checker', BoundaryCheckerConfig, BoundaryCheckerState)
class BoundaryChecker(Extension):
    ''' Extension to check boundary violations.
    Count number of times car is in collision with boundary.
    Does not consider vehicle width, only counts when vehicle origin is out of track boundary.
      '''

    def __init__(self, config, state):
        super().__init__(config, state)
        self.state.collision_count = {
            car: 0
            for car in self.main.cars
        }
        ''' Running sum of collision count, resets every lap'''
        self.state.discretized_raceline = self.main.track.data.discretized_raceline
        ''' Local reference of discretized raceline'''
        self.state.car_is_in_collision = {
            car: False
            for car in self.main.cars
        }
        ''' Dict to indicate if a car is in collision'''
        self.state.last_in_boundary_state = {
            car: CartesianState(*car.state)
            for car in self.main.cars
        }
        for car in self.main.cars:
            car.in_collision = False

    def update(self):
        for car in self.main.cars:
            if self.is_out_of_boundary(car):
                self.state.car_is_in_collision[car] = True
                if not car.in_collision:
                    car.in_collision = True
                    self.state.collision_count[car] += 1
                    self.print_ok('car %s collision = %d' %
                                  (car.param.name, self.state.collision_count[car]))
                if self.config.reset_pos:
                    self._reset_car_to_last_in_boundary_state(car)
            else:
                car.in_collision = False
                self.state.car_is_in_collision[car] = False
                self.state.last_in_boundary_state[car] = CartesianState(*car.state)

    def final(self):
        for car in self.main.cars:
            logger.info('car %d, total boundary violation = %d' %
                        (car.id, self.state.collision_count[car]))
            car.total_boundary_collision = self.state.collision_count[car]

    def _reset_car_to_last_in_boundary_state(self, car):
        last_state = self.state.last_in_boundary_state.get(car)
        if last_state is None:
            return

        reset_state = CartesianState(
            x=last_state.x,
            y=last_state.y,
            heading=last_state.heading,
            v_forward=0.0,
            v_sideway=0.0,
            omega=0.0,
        )
        car.state = reset_state
        self.main.state.car_states[car.id] = reset_state

        if hasattr(car, 'sim_state'):
            if isinstance(car.sim_state, CurvilinearState):
                car.sim_state = self.main.track.cart_to_curv(reset_state)
            else:
                car.sim_state = CartesianState(*reset_state)

    def is_out_of_boundary(self, car):
        car_coord = car.state[0:2]
        return self.main.track.is_outside(car_coord)

    def is_out_of_boundary_discrete(self, car):
        # x, y, heading, vf, vs, omega = car.state
        x, y, _ = car.state
        ref = self.state.discretized_racelien
        ref_points = ref[:, 0:2]
        ref_heading = ref[:, 2]
        left_bdry = ref[:, 3]
        right_bdry = ref[:, 4]
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
