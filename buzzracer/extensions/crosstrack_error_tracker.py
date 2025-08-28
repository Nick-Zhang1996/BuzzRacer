''' Extension to track crosstrack error (lateral error)'''
import numpy as np
from buzzracer.extensions.extension import Extension


class CrosstrackErrorTracker(Extension):
    ''' Extension to track crosstrack error (lateral error)'''

    def __init__(self):
        Extension.__init__(self, 'crosstrack_error_tracker')
        self.print_ok('in use')
        self.crosstrack_error_vec = []
        self.car = self.main.cars[0]

    def update(self):
        if (self.car.critical_lap.is_set()):
            states = self.car.states
            retval = self.main.track.local_trajectory(states,
                                                      wheelbase=self.car.lr,
                                                      return_u=True)
            if retval is None:
                self.print_warning('local_trajectory returned None')
            else:
                # parse return value from local_trajectory
                #(local_ctrl_pnt, offset, orientation, curvature, v_target,
                # u0) = retval
                offset = retval[1]
                err = np.abs(offset)
                # print_info("[CrosstrackErrorTracker]: new error %.4f"%err)
                self.crosstrack_error_vec.append(err)
            if (self.car.laptimer.new_lap.is_set()):
                mean_err = np.mean(self.crosstrack_error_vec)
                self.print_info('current mean error = %.4f' % mean_err)

    def final(self):
        if (len(self.crosstrack_error_vec) > 100):
            mean_err = np.mean(self.crosstrack_error_vec)
            self.print_ok('mean error = %.4f' % mean_err)
        else:
            self.print_warning('insufficient data')
