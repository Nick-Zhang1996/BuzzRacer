from common import *
import numpy as np
from track.CurvilinearTrack import CurvilinearTrack
#import matplotlib.pyplot as plt

class TriangleTrack(CurvilinearTrack):
    def __init__(self,main,config):
        CurvilinearTrack.__init__(self,main,config)
        # NOTE track params
        self.scale = 2.0
        self.radius = 1.0
        self.resolution = 100

        # NOTE track specific
        self.start_pos = (self.scale*self.radius, 0)
        self.start_dir = 0

        ConfigObject.__init__(self,config)

        ds = 0.01
        n_straight = int(self.scale/ds)
        n_curve = int(0.5*np.pi*2*self.radius/ds)

        xx = []
        yy = []

        theta_vec = np.linspace(np.pi,1.5*np.pi,n_curve)[1:-1]
        xx.append(self.radius + np.cos(theta_vec)*self.radius)
        yy.append(self.radius + np.sin(theta_vec)*self.radius)

        t = np.linspace(self.radius,self.radius+1.0,n_straight)
        xx.append(t)
        yy.append(0*t)

        theta_vec = np.linspace(-0.5*np.pi,0.25*np.pi,n_curve)[1:-1]
        xx.append(self.radius + 1.0 + np.cos(theta_vec)*self.radius)
        yy.append(self.radius + np.sin(theta_vec)*self.radius)


        x0 = self.radius + 1.0 + np.cos(0.25*np.pi)*self.radius
        y0 = self.radius + np.sin(0.25*np.pi)*self.radius
        x1 = self.radius + np.cos(0.25*np.pi)*self.radius
        y1 = self.radius + 1.0 + np.sin(0.25*np.pi)*self.radius
        xx.append(np.linspace(x0,x1,n_straight))
        yy.append(np.linspace(y0,y1,n_straight))

        theta_vec = np.linspace(0.25*np.pi,np.pi,n_curve)[1:-1]
        xx.append(self.radius + np.cos(theta_vec)*self.radius)
        yy.append(self.radius + 1.0 + np.sin(theta_vec)*self.radius)

        t = np.linspace(self.radius+1.0,self.radius,n_straight)
        xx.append(t*0)
        yy.append(t)

        xx = np.hstack(xx)
        yy = np.hstack(yy)
        r = np.vstack([xx,yy]).T*self.scale
        #plt.plot(r[:,0],r[:,1])
        #plt.show()
        self.buildContinuousTrack(r)
