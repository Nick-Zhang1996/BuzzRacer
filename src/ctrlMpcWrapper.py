# MPC controller for dynamic bicycle model
# used by car.py
import numpy as np
from mpc import MPC
from time import time
from timeUtil import execution_timer
from car import Car
from math import atan2,radians,degrees,sin,cos,pi,tan,copysign,asin,acos,isnan,exp,pi
from common import *

import os
import sys
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), './mppi/')
sys.path.append(base_dir)

class ctrlMpcWrapper(Car):
    def __init__(self,car_setting,dt):
        super().__init__(car_setting,dt)
        self.t = execution_timer(True)
        self.freq = []
        self.min_freq = 999
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
        # state dimension
        n = 5
        # action dimension
        m = 1
        # output dimension(y=Cx)
        l = 1

        tic = time()
        t = self.t
        t.s()

        p = self.prediction_steps
        dt = self.dt

        debug_dict = {}
        t.s("get ref point")
        e_cross, e_heading, v_ref, k_ref, coord_ref, valid = track.getRefPoint(state, p, dt, reverse=reverse)
        t.e("get ref point")

        t.s("assemble ref")
        if not valid:
            ret =  (0,0,False,debug_dict)
            return ret

        # first element of _ref is current state, we don't need that
        # FIXME
        v_target = v_ref[0]
        #v_target *= 0.5
        v_ref = v_ref[1:]
        k_ref = k_ref[1:]
        # only reference we need is dpsi_dt = Vx * K
        dpsi_dt_ref = v_ref * k_ref

        #x,y,heading,vf,vs,omega = state
        #vx = vf * cos(heading) - vs*sin(heading)
        #vy = vf * sin(heading) + vs*cos(heading)

        # assemble x0
        # state var for dynamic bicycle model w.r.t. lateral and heading error
        # e_lateral, dedt_lateral, e_heading,dedt_heading, 1(unity)
        x0 = np.array([e_cross, 0, e_heading, 0, 1])

        x,y,heading,vf,vs,omega = state
        t.e("assemble ref")

        t.s("assemble matrix")
        # assemble Ak matrices
        Car = self.Car
        Caf = self.Caf
        mass = self.m
        lf = self.lf
        lr = self.lr
        Iz = self.Iz
        Vx = vf
        getA_raw = lambda Vx,dpsi_r: \
             np.array([[0, 1, 0, 0, 0],
                        [0, -(2*Caf+2*Car)/mass/Vx, (2*Caf+2*Car)/mass, (-2*Caf*lf+2*Car*lr)/mass/Vx, (-(2*Caf*lf-2*Car*lr)/mass/Vx - Vx)*dpsi_r],
                        [0, 0, 0, 1, 0],
                        [0, -(2*Caf*lf-2*Car*lr)/Iz/Vx, (2*Caf*lf-2*Car*lr)/Iz, -(2*Caf*lf*lf+2*Car*lr*lr)/Iz/Vx, -(2*Caf*lf*lf+2*Car*lr*lr)/Iz/Vx*dpsi_r],
                        [0, 0, 0, 0, 0]])


        # assemble Bk matrices
        B = np.array([[0, 2*Caf/mass, 0, 2*Caf*lf/Iz,0]]).T

        # since
        #self.states = self.states + (Ak @ self.states + Bk @ u)*dt
        # we now compute augmented A and B
        In = np.eye(n)
        getA = lambda Vx,dpsi_r: In + getA_raw(Vx,dpsi_r) * dt

        A_vec = [getA(Vx,dpsi_r) for Vx,dpsi_r in zip(v_ref,dpsi_dt_ref)]
        # LTI model
        #v0 = v_ref[0]
        #A_vec = [getA(Vx,dpsi_r) for Vx,dpsi_r in zip([v0]*len(v_ref),[0]*len(v_ref))]

        B_vec = [B*dt] * p

        # define output matrix C, for a single state vector
        # y = Cx 
        C = np.zeros([l,n])
        C[0,0] = 1

        # J = y.T P y + u.T Q u
        P = np.zeros([l,l])
        # e_cross
        P[0,0] = 1

        Q = np.zeros([m,m])
        Q[0,0] = 1e-3
        y_ref = np.zeros([p,l])
        x0 = x0
        p = p


        # 5 deg/s
        # typical servo speed 60deg/0.1s
        # u: steering
        du_max = np.array([radians(60)/0.1*dt])*0.5
        u_max = np.array([radians(25)])
        t.e("assemble matrix")


        t.s("convert problem")
        self.mpc.convertLtv(A_vec,B_vec,C,P,Q,y_ref,x0,du_max,u_max)
        t.e("convert problem")

        t.s("solve")
        u_optimal = self.mpc.solve()
        t.e("solve")

        t.s("actuate")
        # u is stacked, so [throttle_0,steering_0, throttle_1, steering_1]
        #plt.plot(u_optimal[1::2,0])
        #plt.show()
        steering = u_optimal[0]
        #print(degrees(steering))

        # throttle is controller by other controller
        #throttle = u_optimal[0,1]
        throttle = self.calcThrottle(state,v_target)

        debug_dict['x_ref'] = coord_ref
        #debug_dict['x_ref'] = []
        #debug_dict['x_project'] = self.mpc.debug()
        ret =  (throttle,steering,True,debug_dict)
        t.e("actuate")
        tac = time()
        self.freq.append(tac-tic)
        if len(self.freq)>300:
            self.freq.pop(0)
        #print("freq = %.2f"%(1.0/(tac-tic)))
        #print("mean freq = %.2f"%(1.0/(np.mean(self.freq))))
        t.e()
        if ( 1.0/(tac-tic) < self.min_freq):
            self.min_freq = 1.0/(tac-tic)
        print("min freq = %.2f"%(self.min_freq))


        return ret

    # initialize mpc
    # sim: an instance of advCarSim so we have access to parameters
    def initMpcSim(self,sim):
        # prediction step
        self.prediction_steps = 15
        # prediction discretization dt
        # NOTE we may be able to use a finer time step in x ref calculation, this can potentially increase accuracy
        self.dt = 0.03
        # together p*mpc_dt gives prediction horizon

        self.Caf = sim.Caf
        self.Car = sim.Car
        self.lf = sim.lf
        self.lr = sim.lr
        self.Iz = sim.Iz
        self.m = sim.m
        self.mpc = MPC()
        # state dimension
        n = 5
        # action dimension
        m = 1
        # output dimension(y=Cx)
        l = 1
        p = self.prediction_steps
        self.mpc.setup(n,m,l,p)
        return

    # initialize mpc
    def initMpcReal(self):
        # prediction step
        self.prediction_steps = 5
        # prediction discretization dt
        # NOTE we may be able to use a finer time step in x ref calculation, this can potentially increase accuracy
        self.dt = 0.03
        # together p*mpc_dt gives prediction horizon

        g = 9.81
        self.m = 0.1667
        self.Caf = 5*0.25*self.m*g
        #self.Car = 5*0.25*self.m*g
        self.Car = self.Caf
        # CG to front axle
        self.lf = 0.09-0.036
        self.lr = 0.036
        # approximate as a solid box
        self.Iz = self.m/12.0*(0.1**2+0.1**2)

        self.mpc = MPC()
        # state dimension
        n = 5
        # action dimension
        m = 1
        # output dimension(y=Cx)
        l = 1
        p = self.prediction_steps
        self.mpc.setup(n,m,l,p)
        return
