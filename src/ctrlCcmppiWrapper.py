# CCMPPI controller wrapper with kinematic bicycle model
import os
import sys
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), './ccmppi/')
sys.path.append(base_dir)

import numpy as np
from time import time,sleep
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev,CubicSpline,interp1d
from math import atan2,radians,degrees,sin,cos,pi,tan,copysign,asin,acos,isnan,exp,pi,atan
import random

from common import *
from timeUtil import execution_timer
from car import Car
from ccmppi import CCMPPI
from kinematicSimulator import kinematicSimulator

class ctrlCcmppiWrapper(Car):
    def __init__(self,car_setting,dt):
        np.set_printoptions(formatter={'float': lambda x: "{0:7.4f}".format(x)})

        super().__init__(car_setting,dt)
        # no need simulator to track states
        self.sim = kinematicSimulator(0,0,0,0)

        # given parameterized raceline x,y = R(s), this corresponds to raceline_s
        # last_s is the last s such that R(last_s) is closest to vehicle
        # used as a starting point for root finding
        self.last_s = None
        self.p = execution_timer(True)
        self.wheelbase = car_setting['wheelbase']
        self.ccmppi_dt = dt
        return

    # if running on real platform, set sim to None so that default values for car dimension/properties will be used
    def init(self,track,sim=None):
        self.track = track

        self.samples_count = 1024*4
        self.discretized_raceline_len = 1024
        self.horizon_steps = 15
        self.control_dim = 2
        self.state_dim = 4
        self.temperature = 0.2
        # control noise for MPPI exploration
        # NOTE tune me
        self.noise_cov = np.diag([(self.max_throttle)**2,radians(40.0/2)**2])
        #self.noise_cov = np.diag([(self.max_throttle/2)**2,radians(60.0)**2])
        self.control_limit = np.array([[-self.max_throttle,self.max_throttle],[-radians(27.1),radians(27.1)]])

        # discretize raceline for use in MPPI
        self.prepareDiscretizedRaceline()

        self.ccmppi = CCMPPI(self.samples_count,self.horizon_steps,self.state_dim,self.control_dim,self.temperature,self.ccmppi_dt,self.noise_cov,self.discretized_raceline,cuda=True,cuda_filename="ccmppi/ccmppi.cu")

        self.ccmppi.applyDiscreteDynamics = self.applyDiscreteDynamics

        if (sim is None):
            self.lr = sim.lr
        else:
            self.lr = 0.05*self.wheelbase

        return

    def prepareDiscretizedRaceline(self):
        ss = np.linspace(0,self.track.raceline_len_m,self.discretized_raceline_len)
        rr = splev(ss%self.track.raceline_len_m,self.track.raceline_s,der=0)
        drr = splev(ss%self.track.raceline_len_m,self.track.raceline_s,der=1)
        heading_vec = np.arctan2(drr[1],drr[0])
        vv = self.track.sToV(ss) 

        # parameter, distance along track
        self.ss = ss
        self.raceline_points = np.array(rr)
        self.raceline_headings = heading_vec
        self.raceline_velocity = vv

        # describe track boundary as offset from raceline
        self.createBoundary()
        self.discretized_raceline = np.vstack([self.raceline_points,self.raceline_headings,vv, self.raceline_left_boundary, self.raceline_right_boundary]).T
        return

    def createBoundary(self,show=False):
        # construct a (self.discretized_raceline_len * 2) vector
        # to record the left and right track boundary as an offset to the discretized raceline
        left_boundary = []
        right_boundary = []

        left_boundary_points = []
        right_boundary_points = []

        for i in range(self.discretized_raceline_len):
            # find normal direction
            coord = self.raceline_points[:,i]
            heading = self.raceline_headings[i]

            left, right = self.track.preciseTrackBoundary(coord,heading)
            left_boundary.append(left)
            right_boundary.append(right)

            # debug boundary points
            left_point = (coord[0] + left * cos(heading+np.pi/2),coord[1] + left * sin(heading+np.pi/2))
            right_point = (coord[0] + right * cos(heading-np.pi/2),coord[1] + right * sin(heading-np.pi/2))

            left_boundary_points.append(left_point)
            right_boundary_points.append(right_point)


            # DEBUG
            # plot left/right boundary
            '''
            left_point = (coord[0] + left * cos(heading+np.pi/2),coord[1] + left * sin(heading+np.pi/2))
            right_point = (coord[0] + right * cos(heading-np.pi/2),coord[1] + right * sin(heading-np.pi/2))
            img = self.track.drawTrack()
            img = self.track.drawRaceline(img = img)
            img = self.track.drawPoint(img,coord,color=(0,0,0))
            img = self.track.drawPoint(img,left_point,color=(0,0,0))
            img = self.track.drawPoint(img,right_point,color=(0,0,0))
            plt.imshow(img)
            plt.show()
            '''


        self.raceline_left_boundary = left_boundary
        self.raceline_right_boundary = right_boundary

        if (show):
            img = self.track.drawTrack()
            img = self.track.drawRaceline(img = img)
            img = self.track.drawPolyline(left_boundary_points,lineColor=(0,255,0),img=img)
            img = self.track.drawPolyline(right_boundary_points,lineColor=(0,0,255),img=img)
            plt.imshow(img)
            plt.show()
            return img
        return


# given state of the vehicle and an instance of track, provide throttle and steering output
# input:
#   state: (x,y,heading,v_forward,v_sideway,omega)
#   track: track object, can be RCPtrack or skidpad
#   v_override: If specified, use this as target velocity instead of the optimal value provided by track object
#   reverse: true if running in opposite direction of raceline init direction

# output:
#   (throttle,steering,valid,debug) 
# ranges for output:
#   throttle -1.0,self.max_throttle
#   steering as an angle in radians, TRIMMED to self.max_steering, left(+), right(-)
#   valid: bool, if the car can be controlled here, if this is false, then throttle will also be set to 0
#           This typically happens when vehicle is off track, and track object cannot find a reasonable local raceline
# debug: a dictionary of objects to be debugged, e.g. {offset, error in v}
    def ctrlCar(self,states,track,v_override=None,reverse=False):
        debug_dict = {'ideal_traj':[], 'rollout_traj_vec':[]}
        # profiling
        p = self.p
        p.s()
        try:
            self.predictOpponent()
            debug_dict['opponent_prediction'] = self.opponent_prediction
        except AttributeError:
            print_error("predictOpponent() AttributeError")
            pass

        p.s("local traj")
        if self.last_s is None:
            # use self.lr as wheelbase to use center of gravity in evaluation
            retval = track.localTrajectory(states,wheelbase=self.lr,return_u=True)
            if retval is None:
                print_warning("[ctrlCcmppiWrapper:ctrlCar] localTrajectory returned None")
                ret =  (0,0,False,debug_dict)
                return ret
            else:
                # parse return value from localTrajectory
                (local_ctrl_pnt,offset,orientation,curvature,v_target,u0) = retval
                # save for estimate at next step
                self.last_s = track.uToS(u0).item()
        p.e("local traj")

        p.s("prep")
        s0 = self.last_s
        # vehicle state
        # vf: forward positive
        # vs: left positive
        # convert state used in run.py : x,y,heading,vf,vs,omega 
        #    to state in ccmppi : x,y,v,heading
        x,y,heading,vf,vs,omega = states

        self.states = states = np.array([x,y,vf,heading])


        # NOTE may need revision to use previous results
        ref_control = np.zeros([self.horizon_steps,self.control_dim])
        p.e("prep")

        p.s("ccmppi")
        uu = self.ccmppi.control(states.copy(),self.opponent_prediction,self.control_limit)
        control = uu[0]
        throttle = control[0]
        steering = control[1]
        #print_info("[wrapper:ccmppi.control] T= %.2f, S = %.2f"%(throttle,degrees(steering)) )
        p.e("ccmppi")

        # DEBUG
        # simulate where mppi think where the car will end up with
        p.s("debug")

        # simulate vehicle trajectory with selected rollouts
        sampled_control = self.ccmppi.debug_dict['sampled_control']
        # use only first 100
        samples = 100
        # randomly select 100
        index = random.sample(range(sampled_control.shape[0]), samples)
        sampled_control = sampled_control[index,:,:]
        rollout_traj_vec = []

        # DEBUG
        # plot sampled trajectories
        for k in range(samples):
            this_rollout_traj = []
            sim_states = states.copy()
            for i in range(self.horizon_steps):
                sim_states = self.applyDiscreteDynamics(sim_states,sampled_control[k,i],self.ccmppi_dt)
                x,y,vf,heading = sim_states
                coord = (x,y)
                this_rollout_traj.append(coord)
            rollout_traj_vec.append(this_rollout_traj)

        # DEBUG
        # state + control
        '''
        full_state_vec = []
        sim_states = states.copy()
        k = 0
        for i in range(self.horizon_steps):
            sim_states = self.applyDiscreteDynamics(sim_states,sampled_control[k,i],self.ccmppi_dt)
            _throttle, _steering = sampled_control[k,i]
            x,y,vf,heading = sim_states
            entry = (x,y,vf,heading,_throttle,_steering)
            full_state_vec.append(entry)
        '''

        debug_dict['rollout_traj_vec'] = rollout_traj_vec

        # with synthesized control sequence

        sim_states = states.copy()
        for i in range(self.horizon_steps):
            sim_states = self.applyDiscreteDynamics(sim_states,uu[i],self.ccmppi_dt)
            x,y,vf,heading = sim_states
            coord = (x,y)
            debug_dict['ideal_traj'].append(coord)

        p.e("debug")
        p.e()

        ret =  (throttle,steering,True,debug_dict)
        return ret


    # advance car dynamics
    # for use in visualization
    def applyDiscreteDynamics(self,state,control,dt):
        return self.sim.updateCar(dt,control[0], control[1],external_states=state)
    # we assume opponent will follow reference trajectory at current speed
    def initTrackOpponents(self):
        return

    def predictOpponent(self):
        self.opponent_prediction = []
        for opponent in self.opponents:
            traj = self.track.predictOpponent(opponent.state, self.horizon_steps, self.ccmppi_dt)
            self.opponent_prediction.append(traj)


if __name__=="__main__":
    pass
