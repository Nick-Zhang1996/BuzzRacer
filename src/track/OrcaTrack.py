# Orca track from ETH Zurich
# TODO drawTrack
# TODO drawCar at the right size
# TODO sliding window for visualization
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0,'..')

from common import *
from track.Track import Track
import json
from scipy.interpolate import splprep, splev,CubicSpline,interp1d
import matplotlib.pyplot as plt

class OrcaTrack(Track):
    def __init__(self,main,config):
        super().__init__(main,config)
        self.setResolution(300)
        self.loadTrack()
        self.buildContinuousTrack()

    def loadTrack(self,):
        config = json.load(open('../copg/car_racing/config.json'))
        #self.path = config['data_dir']
        self.path = '../copg/car_racing_simulator/ORCA_OPT/'
        self.N = config['n_track']
        # ref point location
        self.X = np.loadtxt(self.path + "x_center.txt")[:, 0]
        self.Y = np.loadtxt(self.path + "x_center.txt")[:, 1]
        # ??
        self.s = np.loadtxt(self.path + "s_center.txt")
        self.phi = np.loadtxt(self.path + "phi_center.txt")
        self.kappa = np.loadtxt(self.path + "kappa_center.txt")
        self.diff_s = np.mean(np.diff(self.s))

        self.d_upper = np.loadtxt(self.path + "con_inner.txt")
        self.d_lower = np.loadtxt(self.path + "con_outer.txt")
        self.d_upper[520:565] = np.clip(self.d_upper[520:565], 0.2, 1)
        # self.d_lower[520:565] = np.clip(self.d_lower[520:565], -1, -0.2)
        self.border_angle_upper = np.loadtxt(self.path + "con_angle_inner.txt")
        self.border_angle_lower = np.loadtxt(self.path + "con_angle_outer.txt")

    def buildSpline(self, coord_vec):
        s_vec = self.s
        m = len(s_vec)+1
        smoothing_factor = 0.01*(m)
        spline, u = splprep(coord_vec.T, u=s_vec, s=smoothing_factor, per=1) 
        return spline

    def buildContinuousTrack(self):
        s_vec = self.s
        # n*2
        r_vec = np.vstack([self.X,self.Y]).T

        lateral = np.vstack([np.cos(self.phi+np.pi/2), np.sin(self.phi+np.pi/2)]).T
        
        # boundary
        upper = r_vec + lateral * self.d_upper[:,np.newaxis]
        lower = r_vec + lateral * self.d_lower[:,np.newaxis]


        x_min = np.min( np.hstack([upper[:,0],lower[:,0]]) ) - 0.1
        x_max = np.max( np.hstack([upper[:,0],lower[:,0]]) ) + 0.1
        y_min = np.min( np.hstack([upper[:,1],lower[:,1]]) ) - 0.1
        y_max = np.max( np.hstack([upper[:,1],lower[:,1]]) ) + 0.1

        # shift track to first quadrant, x,y>0
        self.x_limit = x_max - x_min
        self.y_limit = y_max - y_min
        upper[:,0] -= x_min
        upper[:,1] -= y_min
        lower[:,0] -= x_min
        lower[:,1] -= y_min
        r_vec[:,0] -= x_min
        r_vec[:,1] -= y_min

        self.r_vec = r_vec
        self.upper = upper
        self.lower = lower

        self.raceline_len_m = s_vec[-1]
        self.raceline_m = self.buildSpline(r_vec)
        self.upper_fun = self.buildSpline(upper)
        self.lower_fun = self.buildSpline(lower)


        '''
        plt.plot(upper[:,0],upper[:,1])
        plt.plot(lower[:,0],lower[:,1])
        plt.plot(r_vec[:,0],r_vec[:,1],'--')
        plt.show()
        '''
        

    # draw a picture of the track
    def drawTrack(self):
        x_pix = int(self.x_limit*self.resolution)
        y_pix = int(self.y_limit*self.resolution)
        # height, width
        img = 255*np.ones([y_pix,x_pix,3],dtype=np.uint8)
        print(f'{np.min(self.lower)}')
        img = self.drawPolyline(self.upper,img,lineColor=(0,0,0),thickness=2)
        img = self.drawPolyline(self.lower,img,lineColor=(0,0,0),thickness=2)
        img = self.drawPolyline(self.r_vec,img,lineColor=(0,0,255),thickness=1)
        return img


    # draw a raceline
    def drawRaceline(self,img=None):
        raise NotImplementedError

if __name__=='__main__':
    track = OrcaTrack(None,None)
    track.loadTrack()
    track.buildContinuousTrack()
    img = track.drawTrack()
    plt.imshow(img)
    plt.show()

