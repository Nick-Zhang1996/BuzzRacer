# a track defined by a spline
from common import *
import cv2
import numpy as np
from math import cos,sin,pi,atan2,radians,degrees,tan
from scipy.interpolate import splprep, splev,CubicSpline,interp1d
import matplotlib.pyplot as plt

from track.Track import Track

class CurvilinearTrack(Track):
    def __init__(self,main,config):
        Track.__init__(self,main,config)
        # default parameters, to be override
        self.width = 0.5
        self.resolution = 200

        # NOTE nascar track params
        self.length = 2.0
        self.radius = 1.0

        ConfigObject.__init__(self,config)

        # NOTE build nascar track
        xx = []
        yy = []
        xx.append(np.linspace(0,self.length))
        yy.append(0*np.linspace(0,self.length))

        theta_vec = np.linspace(-np.pi/2,np.pi/2)[1:-1]
        xx.append(self.length + np.cos(theta_vec)*self.radius)
        yy.append(self.radius + np.sin(theta_vec)*self.radius)

        xx.append(np.linspace(self.length,0))
        yy.append(2*self.radius + 0*np.linspace(self.length,0))

        theta_vec = np.linspace(np.pi/2,1.5*np.pi)[1:-1]
        xx.append(np.cos(theta_vec)*self.radius)
        yy.append(self.radius + np.sin(theta_vec)*self.radius)

        xx = np.hstack(xx)
        yy = np.hstack(yy)

        n = len(xx)
        s = 0
        ss = [s]
        for i in range(n-1):
            s += ((xx[(i+1)%n]-xx[i])**2 +(yy[(i+1)%n]-yy[i])**2 )**0.5
            ss.append(s)
        self.ss = ss
        self.raceline_len_m = s
        self.r = np.vstack([xx,yy]).T
        assert (len(self.r.shape) == 2)
        assert (self.r.shape[1] == 2)

        tck, u = splprep([xx,yy], u=ss,s=0,per=1)
        self.raceline_s = tck

        # let raceline curve be r(u)
        # dr = r'(u), parameterized with xx/u
        dr = np.array(splev(ss,self.raceline_s,der=1))
        # ddr = r''(u)
        ddr = np.array(splev(ss,self.raceline_s,der=2))
        _norm = lambda x:np.linalg.norm(x,axis=0)
        # radius of curvature can be calculated as R = |y'|^3/sqrt(|y'|^2*|y''|^2-(y'*y'')^2)
        curvature = 1.0/(_norm(dr)**3/(_norm(dr)**2*_norm(ddr)**2 - np.sum(dr*ddr,axis=0)**2)**0.5)
        self.curvature, u = splprep(curvature.reshape(1,-1), u=ss,s=0,per=1)

        self.buildContinuousTrack()

        # NOTE track specific
        self.start_pos = (0, 0)
        self.start_dir = 0
        return

    def drawRaceline(self,img):
        return img

    #state: x,y,theta,vf,vs,omega
    # x,y referenced from skidpad frame
    def localTrajectory(self,state,ccw=True):
        x = state[0]
        y = state[1]
        heading = state[2]
        vf = state[3]
        vs = state[4]
        omega = state[5]

        # find the coordinate of center of front axle
        #wheelbase = 98e-3
        wheelbase = 108e-3
        x += wheelbase*cos(heading)
        y += wheelbase*sin(heading)

        # TODO optimize this
        dxx = self.r[:,0]-x
        dyy = self.r[:,1]-y
        index = np.argmin(dxx**2+dyy**2)
        raceline_point = (self.r[index])

        # find offset
        # positive offset means car is to the left of the trajectory(need to turn right)
        dr = self.r[index+1] - self.r[index]
        track_to_car = (x-self.r[index,0], y-self.r[index,1])
        offset = np.cross(dr/np.linalg.norm(dr),track_to_car).item()

        raceline_orientation = atan2(dr[1],dr[0])

        signed_curvature = splev(self.ss[index],self.curvature)[0].item()

        # reference point on raceline,lateral offset, tangent line orientation, curvature(signed, ccw+)
        return (raceline_point,offset,raceline_orientation,signed_curvature,2.0)

    def buildContinuousTrack(self):
        s_vec = self.ss
        # n*2
        ss = np.linspace(0,self.raceline_len_m,3000)
        r_vec = np.array(splev(ss,self.raceline_s,der=0))
        dr_vec = np.array(splev(ss,self.raceline_s,der=1))
        self.phi = np.arctan2(dr_vec[1,:], dr_vec[0,:])
        lateral = np.vstack([np.cos(self.phi+np.pi/2), np.sin(self.phi+np.pi/2)]).T
        # boundary
        upper = r_vec.T + lateral * self.width/2
        lower = r_vec.T - lateral * self.width/2

        self.x_min = np.min( np.hstack([upper[:,0],lower[:,0]]) ) - 0.1
        self.x_max = np.max( np.hstack([upper[:,0],lower[:,0]]) ) + 0.1
        self.y_min = np.min( np.hstack([upper[:,1],lower[:,1]]) ) - 0.1
        self.y_max = np.max( np.hstack([upper[:,1],lower[:,1]]) ) + 0.1

        # shift track to first quadrant, x,y>0
        '''
        self.x_limit = x_max - x_min
        self.y_limit = y_max - y_min
        upper[:,0] -= x_min
        upper[:,1] -= y_min
        lower[:,0] -= x_min
        lower[:,1] -= y_min
        r_vec[:,0] -= x_min
        r_vec[:,1] -= y_min
        '''

        self.r_vec = r_vec
        self.upper = upper
        self.lower = lower

        #self.raceline_len_m = s_vec[-1]
        #self.raceline_s = self.buildSpline(r_vec)
        #self.upper_fun = self.buildSpline(upper)
        #self.lower_fun = self.buildSpline(lower)

        '''
        plt.plot(upper[:,0],upper[:,1])
        plt.plot(lower[:,0],lower[:,1])
        plt.plot(r_vec[:,0],r_vec[:,1],'--')
        plt.show()
        '''
    def m2canvas(self,coord):
        x_new = int((np.clip(coord[0],self.x_min,self.x_max)-self.x_min) * self.resolution)
        y_new = int((self.y_max - np.clip(coord[1],self.y_min,self.y_max)) * self.resolution)
        return (x_new,y_new)

    # draw a picture of the track
    def drawTrack(self):
        x_pix = int((self.x_max - self.x_min)*self.resolution)
        y_pix = int((self.y_max - self.y_min)*self.resolution)
        # height, width
        img = 255*np.ones([y_pix,x_pix,3],dtype=np.uint8)
        img = self.drawPolyline(self.upper,img,lineColor=(0,0,0),thickness=2)
        img = self.drawPolyline(self.lower,img,lineColor=(0,0,0),thickness=2)
        return img

    # TODO
    def preciseTrackBoundary(self,coord,heading):
        r = (coord[0]**2 + coord[1]**2)**0.5
        phase = np.arctan2(coord[1],coord[0])
        rel_heading = heading - phase
        if (rel_heading > 0 and rel_heading < np.pi):
            # ccw
            left = r - (self.radius-self.width/2)
            right = (self.radius+self.width/2) - r
        else:
            # cw
            right = r - (self.radius-self.width/2)
            left = (self.radius+self.width/2) - r
        return (left,right)


if __name__ == "__main__":
    pass
