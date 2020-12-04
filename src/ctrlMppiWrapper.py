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
        return

    def init(self,track,sim):
        self.track = track
        self.prepareDiscretizedRaceline()

        self.mppi_dt = 0.03
        self.samples_count = 1024
        self.horizon_steps = 20
        self.control_dim = 2
        self.state_dim = 6
        self.temperature = 1.0
        self.noise_cov = np.diag([(0.1/2)**2,radians(20.0/2)**2])

        self.mppi = MPPI(self.samples_count,self.horizon_steps,self.control_dim,self.temperature,self.mppi_dt,self.noise_cov,self.discretized_raceline,cuda=False,cuda_filename="mppi/mppi.cu")

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
        n_steps = 1000
        ss = np.linspace(0,self.track.raceline_len_m,n_steps)
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
    # NOTE this is the Stanley method, now that we have multiple control methods we may want to change its name later
    def ctrlCar(self,state,track,v_override=None,reverse=False):
        # get an estimate for current distance along raceline
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

        s0 = self.last_s
        # vehicle state
        # vf: forward positive
        # vs: left positive
        x,y,heading,vf,vs,omega = state
        dx = vf*cos(heading) - vs*sin(heading)
        dy = vf*sin(heading) + vs*cos(heading)

        self.states = np.array([x,dx,y,dy,heading,omega])
        state = np.array([x,dx,y,dy,heading,omega])

        '''
        # some testing to verify mppi model against simulation model
        #throttle = np.linspace(-0.1,1.0,300)
        #steering = np.linspace(radians(-10),radians(10),300)

        #throttle = np.random.uniform(-0.1,1.0,300)
        #steering = np.random.uniform(radians(-10),radians(10),300)

        #throttle = np.random.normal(size=(300,))*0.5
        #steering = np.random.normal(size=(300,))/2.0*radians(20)
        # simulate car 
        for i in range(30):
            self.advSim(self.mppi_dt,None,throttle[i],steering[i])
            #print(i)
            #print(self.states)

        print("adv simulation states")
        print(self.states)

        # simulate with mppi
        for i in range(50):
            state = self.applyDiscreteDynamics(state,[throttle[i],steering[i]],self.mppi_dt)
            #print(i)
            #print(x)

        print("mppi prediction")
        print(state)
        print("error")
        print(self.states-state)
        '''
        ref_control = np.zeros([self.horizon_steps,self.control_dim])
        control_limit = np.array([[-1.0,1.0],[-radians(27.1),radians(27.1)]])
        print("state")
        print(state)
        uu = self.mppi.control(state.copy(),ref_control.copy(),control_limit)
        control = uu[0]
        print(control)
        throttle = control[0]
        steering = control[1]
        # DEBUG
        # simulate where mppi think where the car will end up with
        # with synthesized control sequence
        debug_dict = {'x_ref_r':[],'x_ref_l':[],'x_ref':[]}
        sim_state = state.copy()
        for i in range(self.horizon_steps):
            sim_state = self.applyDiscreteDynamics(sim_state,uu[i],self.mppi_dt)
            coord = (sim_state[0],sim_state[2])
            debug_dict['x_ref'].append(coord)

        '''
        # simulate control = -10deg
        sim_state = state.copy()
        for i in range(self.horizon_steps):
            sim_state = self.applyDiscreteDynamics(sim_state,[0.0,-radians(5)],self.mppi_dt)
            coord = (sim_state[0],sim_state[2])
            debug_dict['x_ref_r'].append(coord)

        # simulate control = 10deg
        sim_state = state.copy()
        for i in range(self.horizon_steps):
            sim_state = self.applyDiscreteDynamics(sim_state,[0.0,radians(5)],self.mppi_dt)
            coord = (sim_state[0],sim_state[2])
            debug_dict['x_ref_l'].append(coord)
        '''

        ret =  (throttle,steering,True,debug_dict)
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
        # determine heading offset
        cost += abs((self.raceline_headings[ids] - heading + np.pi) % (2*np.pi) - np.pi)
        # sanity check, 0.5*0.1m offset equivalent to 0.1 rad(5deg) heading error
        # 10cm progress equivalent to 0.1 rad error
        # sounds bout right
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

        # determine lateral offset
        cost += np.sqrt((state[0]-self.raceline_points[0,ids])**2+(state[2]-self.raceline_points[1,ids])**2) * 0.5
        # determine heading offset
        cost += np.abs((self.raceline_headings[ids] - heading + np.pi) % (2*np.pi) - np.pi)
        # sanity check, 0.5*0.1m offset equivalent to 0.1 rad(5deg) heading error
        # 10cm progress equivalent to 0.1 rad error
        # sounds bout right

        return cost*10

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
        #x = x*cos(psi) - y*sin(psi)
        local_dx = dx*cos(-psi) - dy*sin(-psi)
        #y = x*sin(-psi) + y*cos(-psi)
        local_dy = dx*sin(-psi) + dy*cos(-psi)


        d_local_dx = throttle*dt
        d_local_dy = (-(2*Caf+2*Car)/(m*local_dx)*local_dy + (-local_dx - (2*Caf*lf-2*Car*lr)/(m*local_dx)) * dpsi + 2*Caf/m*steering)*dt
        d_dpsi = (-(2*lf*Caf - 2*lr*Car)/(Iz*local_dx)*local_dy - (2*lf*lf*Caf + 2*lr*lr*Car)/(Iz*local_dx)*dpsi + 2*lf*Caf/Iz*steering)*dt

        local_dx += d_local_dx
        local_dy += d_local_dy
        dpsi += d_dpsi

        # convert back to global frame
        dx = local_dx*cos(psi) - local_dy*sin(psi)
        dy = local_dx*sin(psi) + local_dy*cos(psi)


        return np.array([x,dx,y,dy,psi,dpsi])
        
    # DEBUG
    def advSim(self,dt,sim_states,throttle,steering):
        # simulator carries internal state and doesn't really need these
        '''
        x = sim_states['coord'][0]
        y = sim_states['coord'][1]
        psi = sim_states['heading']
        d_psi = sim_states['omega']
        Vx = sim_states['vf']
        '''

        #self.t += dt
        # NOTE page 30 of book vehicle dynamics and control
        # ref frame vehicle CG, x forward y leftward
        # this is in car frame, rotate to world frame
        psi = self.states[4]
        # change ref frame to car frame
        # vehicle longitudinal velocity
        self.Vx = self.states[1]*cos(psi) + self.states[3]*sin(psi)

        A = np.array([[0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, -(2*self.Caf+2*self.Car)/(self.m*self.Vx), 0, -self.Vx-(2*self.Caf*self.lf-2*self.Car*self.lr)/(self.m*self.Vx)],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, -(2*self.lf*self.Caf-2*self.lr*self.Car)/(self.Iz*self.Vx), 0, -(2*self.lf**2*self.Caf+2*self.lr**2*self.Car)/(self.Iz*self.Vx)]])
        B = np.array([[0,1,0,0,0,0],[0,0,0,2*self.Caf/self.m,0,2*self.lf*self.Caf/self.Iz]]).T

        u = np.array([throttle,steering])
        # active roattion matrix of angle(rad)
        R = lambda angle: np.array([[cos(angle), 0,-sin(angle),0,0,0],
                        [0, cos(angle), 0,-sin(angle),0,0],
                        [sin(angle),0,cos(angle),0,0,0],
                        [0,sin(angle),0,cos(angle),0,0],
                        [0,0,0,0,1,0],
                        [0,0,0,0,0,1]])
        self.old_states = self.states.copy()
        # A and B work in vehicle frame
        # we use R() to convert state to vehicle frame
        # before we apply A,B
        # then we convert state back to track/world frame
        self.states = self.states + R(psi) @ (A @ R(-psi) @ self.states + B @ u)*dt
        #self.states_hist.append(self.states)
        #self.local_states_hist.append(R(-psi)@self.states)

        self.throttle = throttle
        self.steering = steering

        coord = (self.states[0],self.states[2])
        heading = self.states[4]
        # longitidunal,velocity forward positive
        Vx = self.states[1] *cos(heading) + self.states[3] *sin(heading)
        # lateral, sideway velocity, left positive
        Vy = -self.states[1] *sin(heading) + self.states[3] *cos(heading)
        omega = self.states[5]
        sim_states = {'coord':coord,'heading':heading,'vf':Vx,'vs':Vy,'omega':omega}
        return sim_states
