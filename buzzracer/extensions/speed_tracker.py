''' Record speed profile in debug_dict '''

from buzzracer.extensions.extension import Extension

class SpeedTracker(Extension):
    ''' Record speed profile in debug_dict '''
    def __init__(self):
        Extension.__init__(self, 'speed_tracker')

    def init(self):
        for car in self.main.cars:
            car.debug_dict.update({'target_v': []})

    def update(self):
        for car in self.main.cars:
            state = car.states
            retval = self.main.track.local_trajectory(state)
            #(local_ctrl_pnt, offset, orientation, curvature, v_target) = retval
            (_, _, _, _, v_target) = retval
            car.debug_dict['target_v'].append(v_target)
