# TODO
# construct cost function on y=Cx+Du instead of full state x
# apply cost to uk-uk-1

# This document defines methods related to the Car class,
# which contains the physical dimension, performance, simulation model, and control algorithm for a car
import numpy as np
import serial
from scipy import signal
from numpy import isclose 
from math import atan2,radians,degrees,sin,cos,pi,tan,copysign,asin,acos,isnan,exp,pi
import matplotlib.pyplot as plt
from PidController import PidController
from common import *
from mpc import MPC
from time import time
from timeUtil import execution_timer

class Car:
    def __init__(self,car_setting,dt):
        # debug
        self.freq = []
        self.t = execution_timer(True)
        # max allowable crosstrack error in control algorithm, if vehicle cross track error is larger than this value,
        # controller would cease attempt to correct for it, and will brake vehicle to a stop
        # unit: m
        self.max_offset = 1.0
        # controller tuning, steering->lateral offset
        # P is applied on offset
        # unit: radiant of steering per meter offset
        # the way it is set up now the first number is degree of steering per cm offset
        # for actual car 1 seems to work
        #self.P = 2.0/180*pi/0.01
        # variable P
        # define two velocity->gain map, a linear interpolation would be used to determine gain
        p1 = (1.0,2.0)
        p2 = (4.0,0.5)
        self.Pfun_slope = (p2[1]-p1[1])/(p2[0]-p1[0])
        self.Pfun_offset = p1[1] - p1[0]*self.Pfun_slope
        self.Pfun = lambda v: max(min((self.Pfun_slope*v+self.Pfun_offset),4.0),0.5)/280*pi/0.01
        # define maximum allowable throttle and steering
        # max steering is in radians, for vehicle with ackerman steering (inner wheel steer more than outer)
        # steering angle shoud be calculated by arcsin(wheelbase/turning radius), easily derived from non-slipping bicycle model
        # default values are for the MR03 chassis with Porsche 911 GT3 RS body
        self.wheelbase = car_setting['wheelbase']
        self.max_throttle = car_setting['max_throttle']
        self.max_steering_left = car_setting['max_steer_angle_left']
        self.min_pwm_left = car_setting['max_steer_pwm_left']
        self.max_steering_right = car_setting['max_steer_angle_right']
        self.max_pwm_right = car_setting['max_steer_pwm_right']

        # NOTE, parameters below may not be actively used in current version of code
        # D is applied on delta_omega, a damping on angular speed error
        self.D = radians(4)/3
        #speed controller
        P = 5 # to be more aggressive use 15
        I = 0.0 #0.1
        D = 0.4
        self.throttle_pid = PidController(P,I,D,dt,1,2)
        # low pass filter for throttle controller

        self.b, self.a = signal.butter(1,0.2,'low',analog=False,fs=50)
        self.z_throttle = [0]

        self.verr_integral = 0
        # time constant in sec 
        tc = 5
        #NOTE if using a different vicon frequency, it needs to be reflected here
        self.decay_factor = exp(-1.0/50/tc)
        self.serial_port = car_setting['serial_port']
        self.car_interface = None
        if not (self.serial_port is None):
            try:
                self.car_interface = serial.Serial(self.serial_port,115200, timeout=0.001,writeTimeout=0)
            except (FileNotFoundError,serial.serialutil.SerialException):
                print_error(" interface %s not found"%self.serial_port)
                exit(1)

    def __del__(self):
        if ((not self.serial_port is None) and (not self.car_interface is None) and self.car_interface.is_open):
            self.car_interface.close()
    def mapdata(self,x,a,b,c,d):
        y=(x-a)/(b-a)*(d-c)+c
        return int(y)

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
        coord = (state[0],state[1])

        heading = state[2]
        omega = state[5]
        vf = state[3]
        vs = state[4]

        ret = (0,0,False,{'offset':0})

        # inquire information about desired trajectory close to the vehicle
        retval = track.localTrajectory(state)
        if retval is None:
            return (0,0,False,{'offset':0})
            #return ret

        # parse return value from localTrajectory
        (local_ctrl_pnt,offset,orientation,curvature,v_target) = retval
        # FIXME
        v_target *= 0.8

        if isnan(orientation):
            return (0,0,False,{'offset':0})
            
        if reverse:
            offset = -offset
            orientation += pi

        # if vehicle cross error exceeds maximum allowable error, stop the car
        if (abs(offset) > self.max_offset):
            return (0,0,False,{'offset':offset})
        else:
            # sign convention for offset: negative offset(-) requires left steering(+)
            # this is the convention used in track class, determined arbituarily
            # control logic
            #steering = (orientation-heading) - (offset * self.P) - (omega-curvature*vf)*self.D
            steering = (orientation-heading) - (offset * self.Pfun(abs(vf)))
            # print("D/P = "+str(abs((omega-curvature*vf)*D/(offset*P))))
            # handle edge case, unwrap ( -355 deg turn -> +5 turn)
            steering = (steering+pi)%(2*pi) -pi
            if (steering>self.max_steering_left):
                steering = self.max_steering_left
            elif (steering<-self.max_steering_right):
                steering = -self.max_steering_right
            if (v_override is None):
                throttle = self.calcThrottle(state,v_target)
            else:
                throttle = self.calcThrottle(state,v_override)

            ret =  (throttle,steering,True,{'offset':offset,'dw':omega-curvature*vf,'vf':vf,'v_target':v_target,'local_ctrl_point':local_ctrl_pnt})

        return ret

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
    def ctrlCarDynamicMpc(self,state,track,v_override=None,reverse=False):
        # state dimension
        n = 5
        # action dimension
        m = 1
        # output dimension(y=Cx)
        l = 1

        tic = time()
        t = self.t
        t.s()

        p = self.mpc_prediction_steps
        dt = self.mpc_dt

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

        #debug_dict['x_ref'] = coord_ref
        debug_dict['x_ref'] = []
        #debug_dict['x_project'] = self.mpc.debug()
        ret =  (throttle,steering,True,debug_dict)
        t.e("actuate")
        tac = time()
        self.freq.append(tac-tic)
        if len(self.freq)>300:
            self.freq.pop(0)
        #print("freq = %.2f"%(1.0/(tac-tic)))
        print("mean freq = %.2f"%(1.0/(np.mean(self.freq))))
        t.e()
        return ret

    # initialize mpc
    # sim: an instance of advCarSim so we have access to parameters
    def initMpcSim(self,sim):
        # prediction step
        self.mpc_prediction_steps = 15
        # prediction discretization dt
        # NOTE we may be able to use a finer time step in x ref calculation, this can potentially increase accuracy
        self.mpc_dt = 0.03
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
        p = self.mpc_prediction_steps
        self.mpc.setup(n,m,l,p)
        return

    # initialize mpc
    def initMpcReal(self):
        # prediction step
        self.mpc_prediction_steps = 5
        # prediction discretization dt
        # NOTE we may be able to use a finer time step in x ref calculation, this can potentially increase accuracy
        self.mpc_dt = 0.03
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
        p = self.mpc_prediction_steps
        self.mpc.setup(n,m,l,p)
        return

    def actuate(self,steering,throttle):
        if not (self.car_interface is None):
            self.car_interface.write((str(self.mapdata(steering, self.max_steering_left,-self.max_steering_right,self.min_pwm_left,self.max_pwm_right))+","+str(self.mapdata(throttle,-1.0,1.0,1900,1100))+'\n').encode('ascii'))
            return True
        else:
            return False

    # provide direct pwm
    def actuatePWM(self,steeringPWM,throttlePWM):
        if not (self.car_interface is None):
            self.car_interface.write((str(int(steeringPWM))+","+str(int(throttlePWM))+'\n').encode('ascii'))
            return True
        else:
            return False

    # longitudinal acceleration has proven to be difficult to predict
    # therefore we ignore this in the EKF
    def getLongitudinalAcc(self,state,throttle):
        vf = state[3]
        acc = throttle * 4.95445214  - 1.01294228
        if (vf<0.01 and throttle<0.245):
            acc = 0
        return acc

    # PID controller for forward velocity
    def calcThrottle(self,state,v_target):
        vf = state[3]
        # PI control for throttle
        acc_target = self.throttle_pid.control(v_target,vf)
        throttle = (acc_target + 1.01294228)/4.95445214 

        return max(min(throttle,self.max_throttle),-1)

    # for simulation only
    # update car state with bicycle model, no slip
    # dt: time, in sec
    # v: velocity of rear wheel, in m/s
    # state: x,y,heading,vf(forward speed),vs(sideway speed),omega
    def updateCar(self,state,throttle,steering,dt):
        # wheelbase, in meter
        # heading of pi/2, i.e. vehile central axis aligned with y axis,
        # means theta = 0 (the x axis of car and world frame is aligned)
        # experimental acceleration model
        v = max(state[3]+(throttle-0.2)*4*dt,0)
        theta = state[2] - pi/2
        L = self.wheelbase
        dr = v*dt
        dtheta = dr*tan(steering)/L
        # specific to vehicle frame (x to right of rear axle, y to forward)
        if (steering==0):
            dx = 0
            dy = dr
        else:
            dx = - L/tan(steering)*(1-cos(dtheta))
            dy =  abs(L/tan(steering)*sin(dtheta))
        #print(dx,dy)
        # specific to world frame
        dX = dx*cos(theta)-dy*sin(theta)
        dY = dx*sin(theta)+dy*cos(theta)
        # x,y,heading,vf(forward speed),vs(sideway speed),omega
        return np.array([state[0]+dX,state[1]+dY,state[2]+dtheta,v,0,dtheta/dt])


# debugging/tuning code
if __name__ == '__main__':
    # tune PI controller for speed control
    v_log = []
    control_log = []
    integral_log = []
    t = np.linspace(0,1000,1001)
    v_targets = np.array([1+0.5*sin(2*pi/5/100*tt) for tt in t])
    integral = 0
    throttle = 0
    I = 0.1
    P = 1
    v = 0
    tc = 2
    # applied on I
    decay_factor = exp(-1.0/100/tc)
    last_v_err = 0
    for v_target in v_targets:
        v = max(v+(throttle-0.2)*0.04,0)
        v_err = v_target - v
        integral = integral*decay_factor + v_err
        throttle = min(I*integral + P*v_err,1)
        last_v_err = v_err
        control_log.append(throttle)
        v_log.append(v)

    p0, = plt.plot(v_log,label='velocity')
    p1, = plt.plot(control_log,label='output')
    p2, = plt.plot(v_targets,label='target')
    plt.legend(handles=[p0,p1,p2])
    plt.show()
