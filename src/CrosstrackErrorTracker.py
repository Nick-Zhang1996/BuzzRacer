import numpy as np
from common import *
from Extension import Extension

class CrosstrackErrorTracker(Extension):
    def __init__(self,main):
        Extension.__init__(self,main)
        print_ok("[CrosstrackErrorTracker]: in use")
        self.crosstrack_error_vec = []
        self.car = self.main.cars[0]

    def update(self):
        if (self.car.critical_lap.is_set()):
            states = self.car.state
            retval = self.main.track.localTrajectory(states,wheelbase=self.car.lr,return_u=True)
            if retval is None:
                print_warning("[CrosstrackErrorTracker]: localTrajectory returned None")
            else:
                # parse return value from localTrajectory
                (local_ctrl_pnt,offset,orientation,curvature,v_target,u0) = retval
                err = np.abs(offset)
                #print_info("[CrosstrackErrorTracker]: new error %.4f"%err)
                self.crosstrack_error_vec.append(err)
            if (self.car.laptimer.new_lap.is_set()):
                mean_err = np.mean(self.crosstrack_error_vec)
                print_info("[CrosstrackErrorTracker]: current mean error = %.4f"%mean_err)


    def final(self):
        if (len(self.crosstrack_error_vec) > 100):
            mean_err = np.mean(self.crosstrack_error_vec)
            print_ok("[CrosstrackErrorTracker]: mean error = %.4f"%mean_err)
        else:
            print_warning("[CrosstrackErrorTracker]: insufficient data")

