# record speed profile in debug_dict
from extension.Extension import *
from common import *
class SpeedTracker(Extension):
    def __init__(self, main):
        Extension.__init__(self,main)
    def init(self):
        for car in self.main.cars:
            car.debug_dict.update({'target_v':[]})
    def update(self):
        for car in self.main.cars:
            state = car.states
            retval = self.main.track.localTrajectory(state)
            (local_ctrl_pnt,offset,orientation,curvature,v_target) = retval
            car.debug_dict['target_v'].append(v_target)
