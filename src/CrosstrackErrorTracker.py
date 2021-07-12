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
                #print_info("[CrosstrackErrorTracker]: new error %.3f"%err)
                self.crosstrack_error_vec.append(err)

    def final(self):
        mean_err = np.mean(self.crosstrack_error_vec)
        print_ok("[CrosstrackErrorTracker] mean error = %.3f"%mean_err)

