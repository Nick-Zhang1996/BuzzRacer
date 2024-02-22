# a curvilinear track that lookslike a cat's face

from common import *
import numpy as np
from track.CurvilinearTrack import CurvilinearTrack

class SineTrack(CurvilinearTrack):
    def __init__(self,main,config):
        CurvilinearTrack.__init__(self,main,config)
        self.resolution = 50

        # NOTE track params
        self.scale = 1

        # NOTE track specific
        self.start_pos = (0, 0)
        self.start_dir = np.pi/4

        ConfigObject.__init__(self,config)

        # NOTE build nascar track
        ds = 0.01
        # 7.6 is curve length for sine [0,2pi]
        n_sine = int(7.6*1.5*self.scale/ds)
        n_circle = int(np.pi*3.5*np.pi*self.scale/ds)

        xx = []
        yy = []
        t = np.linspace(0.01,3*np.pi-0.01,n_sine)
        xx.append(t*self.scale)
        yy.append(np.sin(t)*self.scale)

        t = np.linspace(0.25*np.pi,-0.25*np.pi,n_circle)
        xx.append(1.5*np.pi+np.cos(t)*3/np.sqrt(2)*np.pi*self.scale)
        yy.append(-1.5*np.pi + np.sin(t)*3/np.sqrt(2)*np.pi*self.scale)

        t = np.linspace(3*np.pi-0.01,0.01,n_sine)
        xx.append(t*self.scale)
        yy.append(-3*np.pi-np.sin(t)*self.scale)

        t = np.linspace(1.25*np.pi,0.75*np.pi,n_circle)
        xx.append(1.5*np.pi+np.cos(t)*3/np.sqrt(2)*np.pi*self.scale)
        yy.append(-1.5*np.pi + np.sin(t)*3/np.sqrt(2)*np.pi*self.scale)

        xx = np.hstack(xx)
        yy = np.hstack(yy)
        r = np.vstack([xx,yy]).T
        self.buildContinuousTrack(r)
