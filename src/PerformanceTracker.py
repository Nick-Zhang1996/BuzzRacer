import numpy as np
from common import *
from Extension import Extension

class PerformanceTracker(Extension):
    def __init__(self,main):
        Extension.__init__(self,main)
        print_ok("[PerformanceTracker]: in use")
        self.car = self.main.cars[0]

    def update(self):
        return

    def final(self):
        terminal_cov = np.mean(self.car.controller.terminal_cov_vec)
        print_ok("[PerformanceTracker]: terminal covariance = %.5f"%(terminal_cov))

