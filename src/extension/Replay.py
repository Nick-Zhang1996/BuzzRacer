# replay state history
import os
from common import *
import pickle
from extension.Extension import Extension
from extension import Simulator
from math import atan2,radians,degrees,sin,cos,pi,tan,copysign,asin,acos,isnan
from scipy.interpolate import splprep, splev,CubicSpline,interp1d
import matplotlib.pyplot as plt

class Replay(Simulator):
    def __init__(self,main):
        super().__init__(main)
        self.car_count = 0
        self.timestep = 0
        self.curvilinear = None
        self.log_name = None
        # rcvip
        self.basedir = self.main.basedir
        self.track = self.main.track

    def init(self):
        super().init()
        if (self.curvilinear):
            self.loadCurvilinearLog(self.log_name)
        else:
            self.loadCartesianLog(self.log_name)
        self.main.new_state_update.set()
        ## DEBUG
        #lateral_err = self.data[:,0,1]
        #plt.plot(lateral_err)
        #plt.show()
        #breakpoint()

    def loadCurvilinearLog(self,log_name):
        # time_steps * cars * (states + action)
        full_path = os.path.join(self.basedir,log_name) 
        self.print_ok(f'opening file at {full_path}')
        with open(full_path,'rb') as f:
            self.data = pickle.load(f)
        # create cars
        self.car_count = self.data.shape[1]
        assert (len(self.main.cars) == self.car_count)

    def loadCartesianLog(self,log_name):
        print_error('not implemented')

    def loadRcpTrack(self):
        N,X,Y,s,phi,kappa,diff_s,d_upper,d_lower,border_angle_upper,border_angle_lower = self.track.getOrcaStyleTrack()

        self.N = N
        self.X = X
        self.Y = Y
        self.s = s
        self.phi = phi
        self.kappa = kappa
        self.diff_s = diff_s

        self.d_upper = d_upper
        self.d_lower = d_lower
        # not really used
        self.border_angle_upper = border_angle_upper
        self.border_angle_lower = border_angle_lower
        return

    def CurvilinearToCartesian(self,state):
        progress, lateral_err, rel_heading, v_forward, v_sideways, omega,throttle,steering = state
        # TODO verify this is right, dimension
        pos = np.array(splev(progress%self.track.raceline_len_m,self.track.raceline_s,der=0))
        A = np.array([[0,-1],[1,0]])
        tangent = np.array(splev(progress%self.track.raceline_len_m,self.track.raceline_s,der=1))
        track_heading = np.arctan2(tangent[1],tangent[0])
        lateral = A @ (tangent/np.linalg.norm(tangent))
        car_pos = pos + lateral_err * lateral
        x,y = car_pos
        heading = rel_heading + track_heading
        #print(f'tangent = {tangent}')
        #print(f'lateral = {lateral}')
        #print(f'track_heading = {track_heading}')

        return (x,y,heading,v_forward,v_sideways,omega)

    def update(self):
        if (self.curvilinear):
            for (i,car) in enumerate(self.main.cars):
                car.states = self.CurvilinearToCartesian(self.data[self.timestep,i])
        else:
            for (i,car) in enumerate(self.main.cars):
                car.states = tuple(self.data[self.timestep,i].flatten())

        self.main.new_state_update.set()
        self.main.sim_t += self.main.dt
        self.matchRealTime()
        self.timestep += 1

