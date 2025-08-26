''' Track position tracking performance and control effort '''
import numpy as np
from extension.Extension import Extension


class PerformanceTracker(Extension):
    ''' Extension to track control effort and other performance metrics. '''
    def __init__(self):
        Extension.__init__(self,'performance_tracker')
        self.car = self.main.cars[0]
        self.control_effort_vec = []
        self.terminal_cov = 0
        self.mean_control_effort = None

    def post_update(self):
        self.control_effort_vec.append(self.main.cars[0].controller.utru)
        return

    def final(self):
        '''
        self.pos_2_norm = pos_2_norm = np.mean(self.car.controller.pos_2_norm_vec)
        self.print_ok(self.prefix()+"Position 2 norm = %.6f"%(pos_2_norm))

        self.state_2_norm = state_2_norm = np.mean(self.car.controller.state_2_norm_vec)
        self.print_ok(self.prefix()+"state 2 norm = %.6f"%(state_2_norm))

        self.pos_area = pos_area = np.mean(self.car.controller.pos_area_vec)
        self.print_ok(self.prefix()+"1 sigma pos area= %.6f"%(pos_area))
        self.terminal_cov = pos_2_norm
        '''

        mean_control_effort = np.mean(self.control_effort_vec)
        self.print_ok(self.prefix() + "mean control effort u'Ru = %.5f" %
                 (mean_control_effort))
        self.mean_control_effort = mean_control_effort

        self.car.controller.p.summary()
