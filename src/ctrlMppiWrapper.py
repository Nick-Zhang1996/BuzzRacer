from car import Car
from math import atan2,radians,degrees,sin,cos,pi,tan,copysign,asin,acos,isnan,exp,pi
import numpy as np
from time import time
from timeUtil import execution_timer
from mppi import MPPI
from scipy.interpolate import splprep, splev,CubicSpline,interp1d

from common import *

import os
import sys
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), './mppi/')
sys.path.append(base_dir)

class ctrlMppiWrapper(Car):
    def __init__(self,car_setting,dt):
        super().__init__(car_setting,dt)

        # given parameterized raceline x,y = R(s), this corresponds to raceline_s
        # last_s is the last s such that R(last_s) is closest to vehicle
        # used as a starting point for root finding
        self.last_s = None
        self.p = execution_timer(True)
        return

    def init(self,track,sim):
        self.track = track

        # NOTE NOTE NOTE
        # update mppi_racecar.cu whenever you change parameter here
        self.mppi_dt = 0.03
        self.samples_count = 8192 #8192
        self.discretized_raceline_len = 1024
        self.horizon_steps = 30
        self.control_dim = 2
        self.state_dim = 6
        self.temperature = 1.0
        self.noise_cov = np.diag([(1.0/2)**2,radians(40.0/2)**2])
        self.control_limit = np.array([[-1.0,1.0],[-radians(27.1),radians(27.1)]])

        self.prepareDiscretizedRaceline()

        self.mppi = MPPI(self.samples_count,self.horizon_steps,self.state_dim,self.control_dim,self.temperature,self.mppi_dt,self.noise_cov,self.discretized_raceline,cuda=True,cuda_filename="mppi/mppi_racecar.cu")

        self.mppi.applyDiscreteDynamics = self.applyDiscreteDynamics
        self.mppi.evaluateStepCost = self.evaluateStepCost
        self.mppi.evaluateTerminalCost = self.evaluateTerminalCost

        self.Caf = sim.Caf
        self.Car = sim.Car
        self.lf = sim.lf
        self.lr = sim.lr
        self.Iz = sim.Iz
        self.m = sim.m
        return

    def prepareDiscretizedRaceline(self):
        ss = np.linspace(0,self.track.raceline_len_m,self.discretized_raceline_len)
        rr = splev(ss%self.track.raceline_len_m,self.track.raceline_s,der=0)
        drr = splev(ss%self.track.raceline_len_m,self.track.raceline_s,der=1)
        heading_vec = np.arctan2(drr[1],drr[0])
        # parameter, distance along track
        self.ss = ss
        self.raceline_points = np.array(rr)
        self.raceline_headings = heading_vec
        self.discretized_raceline = np.vstack([self.raceline_points,self.raceline_headings]).T
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
    def ctrlCar(self,state,track,v_override=None,reverse=False):
        p = self.p
        p.s()
        # get an estimate for current distance along raceline
        debug_dict = {'x_ref_r':[],'x_ref_l':[],'x_ref':[],'crosstrack_error':[],'heading_error':[]}
        e_cross, e_heading, v_ref, k_ref, coord_ref, valid = track.getRefPoint(state, 3, 0.01, reverse=reverse)
        debug_dict['crosstrack_error'] = e_cross
        debug_dict['heading_error'] = e_heading
        p.s("local traj")
        if self.last_s is None:
            retval = track.localTrajectory(state,wheelbase=0.102/2.0,return_u=True)
            if retval is None:
                print_warning("localTrajectory returned None")
                ret =  (0,0,False,debug_dict)
                return ret
            else:
                # parse return value from localTrajectory
                (local_ctrl_pnt,offset,orientation,curvature,v_target,u0) = retval
                self.last_s = track.uToS(u0).item()
        p.e("local traj")

        p.s("prep")
        s0 = self.last_s
        # vehicle state
        # vf: forward positive
        # vs: left positive
        x,y,heading,vf,vs,omega = state
        dx = vf*cos(heading) - vs*sin(heading)
        dy = vf*sin(heading) + vs*cos(heading)

        self.states = np.array([x,dx,y,dy,heading,omega])
        state = np.array([x,dx,y,dy,heading,omega])

        ref_control = np.zeros([self.horizon_steps,self.control_dim])
        p.e("prep")

        p.s("mppi")
        uu = self.mppi.control(state.copy(),ref_control.copy(),self.control_limit)
        control = uu[0]
        throttle = control[0]
        steering = control[1]
        p.e("mppi")
        # DEBUG
        # simulate where mppi think where the car will end up with
        # with synthesized control sequence
        p.s("debug")
        sim_state = state.copy()
        for i in range(self.horizon_steps):
            sim_state = self.applyDiscreteDynamics(sim_state,uu[i],self.mppi_dt)
            coord = (sim_state[0],sim_state[2])
            debug_dict['x_ref'].append(coord)

        ret =  (throttle,steering,True,debug_dict)
        p.e("debug")
        p.e()
        return ret


    def evaluateStepCost(self,state,control):
        heading = state[4]
        # calculate cost
        # cost = -reward + penalty
        #ids0 = self.findClosestIds(x0)
        ids = self.findClosestIds(state)

        # reward is progress along centerline
        #cost = - ( self.ss[ids] - self.ss[ids0] )

        # determine lateral offset
        cost = np.sqrt((state[0]-self.raceline_points[0,ids])**2+(state[2]-self.raceline_points[1,ids])**2) * 0.5

        # heading error cost
        # cost += abs((self.raceline_headings[ids] - heading + np.pi) % (2*np.pi) - np.pi)
        return cost*10

    def findClosestIds(self,state):
        x = state[0]
        y = state[2]
        dx = x - self.raceline_points[0]
        dy = y - self.raceline_points[1]

        dist2 = dx*dx + dy*dy
        idx = np.argmin(dist2)
        return idx

    def evaluateTerminalCost(self,state,x0):
        heading = state[4]
        # calculate cost
        # cost = -reward + penalty
        ids0 = self.findClosestIds(x0)
        ids = self.findClosestIds(state)

        # reward is progress along centerline
        cost = - ( self.ss[ids] - self.ss[ids0] )
        # instead of actual progress length, can we approximate progress with index to self.ss ?
        #cost = -(ids - ids0 + self.discretized_raceline_len)%self.discretized_raceline_len

        # sanity check, 0.5*0.1m offset equivalent to 0.1 rad(5deg) heading error
        # 10cm progress equivalent to 0.1 rad error
        # sounds bout right

        #return cost*10
        # NOTE ignoring terminal cost
        return 0.0

    # advance car dynamics
    def applyDiscreteDynamics(self,state,control,dt):
        x = state[0]
        dx = state[1]
        # left pos(+)
        y = state[2]
        dy = state[3]
        psi = state[4]
        dpsi = state[5]

        throttle = control[0]
        # left pos(+)
        steering = control[1]

        Caf = self.Caf
        Car = self.Car
        lf = self.lf
        lr = self.lr
        Iz = self.Iz
        m = self.m

        x += dx * dt
        y += dy * dt
        psi += dpsi * dt

        # dx,dy in state are in global frame, yet the dynamics equations are in car frame
        # convert here
        local_dx = dx*cos(-psi) - dy*sin(-psi)
        local_dy = dx*sin(-psi) + dy*cos(-psi)


        d_local_dx = throttle*dt
        d_local_dy = (-(2*Caf+2*Car)/(m*local_dx)*local_dy + (-local_dx - (2*Caf*lf-2*Car*lr)/(m*local_dx)) * dpsi + 2*Caf/m*steering)*dt
        d_dpsi = (-(2*lf*Caf - 2*lr*Car)/(Iz*local_dx)*local_dy - (2*lf*lf*Caf + 2*lr*lr*Car)/(Iz*local_dx)*dpsi + 2*lf*Caf/Iz*steering)*dt
        debug = steering


        local_dx += d_local_dx
        local_dy += d_local_dy
        dpsi += d_dpsi

        # convert back to global frame
        dx = local_dx*cos(psi) - local_dy*sin(psi)
        dy = local_dx*sin(psi) + local_dy*cos(psi)


        return np.array([x,dx,y,dy,psi,dpsi])
        
