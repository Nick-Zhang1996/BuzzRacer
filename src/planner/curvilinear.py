# test curvature calculation methods in curvilinear frame
import numpy as np

class Curvilinear:
    def __init__(self):
        return

    def prep(self):
        t = np.linspace(0,np.pi)
        x = 3*t**3+4*t
        y = 6*t**2 + 10
        dx = 9*t**2+4
        dy = 12*t
        ddx = 18*t
        ddy = 12*np.ones_like(t)

    def run(self):
