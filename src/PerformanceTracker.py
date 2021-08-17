import numpy as np
from common import *
from Extension import Extension

class PerformanceTracker(Extension):
    def __init__(self,main):
        Extension.__init__(self,main)
        print_ok("[PerformanceTracker]: in use")
        self.car = self.main.cars[0]

        self.control_effort_vec = []

    def postUpdate(self):
        self.control_effort_vec.append(self.main.cars[0].controller.utru)
        return

    def final(self):
        terminal_cov = np.mean(self.car.controller.terminal_cov_vec)
        print_ok(self.prefix()+"terminal covariance = %.5f"%(terminal_cov))
        mean_control_effort = np.mean(self.control_effort_vec)
        print_ok(self.prefix() + "mean control effort u'Ru = %.5f"%(mean_control_effort))

