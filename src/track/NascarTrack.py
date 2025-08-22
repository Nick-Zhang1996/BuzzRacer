from common import *
import numpy as np
from track.CurvilinearTrack import CurvilinearTrack


class NascarTrack(CurvilinearTrack):
    def __init__(self, main, config):
        CurvilinearTrack.__init__(self, main, config)
        # NOTE nascar track params
        self.length = 2.0
        self.radius = 1.0

        # NOTE track specific
        self.start_pos = (0, 0)
        self.start_dir = 0

        ConfigObject.__init__(self, config)

        # NOTE build nascar track
        ds = 0.01
        dphi = ds/self.radius
        n_straight = int(self.length/ds)
        n_curve = int(np.pi/dphi)

        xx = []
        yy = []
        xx.append(np.linspace(0, self.length, n_straight))
        yy.append(0*np.linspace(0, self.length, n_straight))

        theta_vec = np.linspace(-np.pi/2, np.pi/2, n_curve)[1:-1]
        xx.append(self.length + np.cos(theta_vec)*self.radius)
        yy.append(self.radius + np.sin(theta_vec)*self.radius)

        xx.append(np.linspace(self.length, 0, n_straight))
        yy.append(2*self.radius + 0*np.linspace(self.length, 0, n_straight))

        theta_vec = np.linspace(np.pi/2, 1.5*np.pi, n_curve)[1:-1]
        xx.append(np.cos(theta_vec)*self.radius)
        yy.append(self.radius + np.sin(theta_vec)*self.radius)

        xx = np.hstack(xx)
        yy = np.hstack(yy)
        r = np.vstack([xx, yy]).T
        self.buildContinuousTrack(r)
