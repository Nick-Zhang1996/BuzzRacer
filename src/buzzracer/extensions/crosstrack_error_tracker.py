''' Extension to track crosstrack error (lateral error)'''
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from buzzracer.extensions.extension import Extension, ExtensionConfig, ExtensionState
if TYPE_CHECKING:
    from buzzracer.cars.car import Car


class CrosstrackErrorTrackerConfig(ExtensionConfig):
    def __init__(self, main_config):
        super().__init__(main_config)
        self.car_id = 0


class CrosstrackErrorTrackerState(ExtensionState):
    def __init__(self, config):
        super().__init__(config)
        self.car: Car
        self.crosstrack_error_vec = []


@Extension.register('crosstrack_error_tracker', ExtensionConfig, CrosstrackErrorTrackerState)
class CrosstrackErrorTracker(Extension):
    ''' Extension to track crosstrack error (lateral error)'''

    def __init__(self, config, state):
        super().__init__(config, state)
        self.state.car = self.main.cars[0]

    def update(self):
        state = self.state
        car = state.car
        retval = self.main.track.local_trajectory(car.state)
        if retval is None:
            self.print_warning('local_trajectory returned None')
        else:
            # parse return value from local_trajectory
            # (local_ctrl_pnt, offset, orientation, curvature, v_target,
            # u0) = retval
            err = np.abs(retval.lateral_err)
            # print_info("[CrosstrackErrorTracker]: new error %.4f"%err)
            state.crosstrack_error_vec.append(err)
        if (self.main.laptimer.laptimer_by_car[car].new_lap.is_set()):
            mean_err = np.mean(state.crosstrack_error_vec)
            self.print_info('current mean error = %.4f' % mean_err)

    def final(self):
        state = self.state
        if (len(state.crosstrack_error_vec) > 100):
            mean_err = np.mean(state.crosstrack_error_vec)
            self.print_ok('mean error = %.4f' % mean_err)
        else:
            self.print_warning('insufficient data')
