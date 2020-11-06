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
import matplotlib.pyplot as plt
from time import time

class Car:
    def __init__(self,car_setting):
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
        self.throttle_pid = PidController(P,I,D,0.01,1,2)
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
        #v_target *= 0.8

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
        tic = time()
        p = self.mpc_prediction_steps
        dt = self.mpc_dt
        # state dimension
        n = 6
        m = 2

        debug_dict = {}
        x_ref, psi_ref, v_ref, valid = track.getRefPoint(state, p, dt, reverse=False)
        # first element of _ref is current state, we don't need that
        x_ref_raw = x_ref.copy()
        x_ref = x_ref[1:,:]
        # x_ref here is a list of (x,y) convert it to full state
        o = np.zeros([p,1])
        x_ref = np.hstack([x_ref[:,0].reshape(-1,1),o,x_ref[:,1].reshape(-1,1),o,o,o])
        psi_ref = psi_ref[1:]
        v_target = v_ref[0]
        v_ref = v_ref[1:]
        if not valid:
            ret =  (0,0,False,debug_dict)
            return ret

        x,y,heading,vf,vs,omega = state
        vx = vf * cos(heading) - vs*sin(heading)
        vy = vf * sin(heading) + vs*cos(heading)

        # assemble x0
        # x format for dynamic bicycle model
        # x = x,dxdt,y,dydt,psi(heading),dpsi/dt
        # all in track frame
        x0 = np.array([x,vx,y,vy,heading,omega])

        # assemble Ak matrices
        getA_raw = lambda Vx: \
             np.array([[0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, -(2*self.Caf+2*self.Car)/(self.m*Vx), 0, -Vx-(2*self.Caf*self.lf-2*self.Car*self.lr)/(self.m*Vx)],
                        [0, 0, 0, 0, 0, 1],
                        [0, 0, 0, -(2*self.lf*self.Caf-2*self.lr*self.Car)/(self.Iz*Vx), 0, -(2*self.lf**2*self.Caf+2*self.lr**2*self.Car)/(self.Iz*Vx)]])

        # assemble Bk matrices
        B_raw = np.array([[0,1,0,0,0,0],[0,0,0,2*self.Caf/self.m,0,2*self.lf*self.Caf/self.Iz]]).T

        # active roattion matrix of angle(rad)
        R = lambda angle: np.array([[cos(angle), 0,-sin(angle),0,0,0],
                        [0, cos(angle), 0,-sin(angle),0,0],
                        [sin(angle),0,cos(angle),0,0,0],
                        [0,sin(angle),0,cos(angle),0,0],
                        [0,0,0,0,1,0],
                        [0,0,0,0,0,1]])
        
        # since
        #self.states = self.states + R(psi) @ (A @ R(-psi) @ self.states + B @ u)*dt
        # we now compute augmented A and B
        In = np.eye(n)
        getA = lambda Vx,psi: In + R(psi) @ getA_raw(Vx) @ R(-psi) * dt
        getB = lambda psi: R(psi) @ B_raw * dt

        # TODO maybe change x_ref and psi_ref to a list
        # FIXME using current vehicle heading
        #A_vec = [getA(Vx,psi) for Vx,psi in zip(v_ref,psi_ref)]
        #B_vec = [getB(psi) for psi in psi_ref]
        A_vec = [getA(Vx,heading) for Vx,psi in zip(v_ref,psi_ref)]
        B_vec = [getB(heading) for psi in psi_ref]

        # TODO add a config function to mpc

        P = np.zeros([n,n])
        # x
        P[0,0] = 1
        # y
        P[2,2] = 1
        # psi
        P[4,4] = 1

        Q = np.zeros([m,m])
        # only apply cost to steering
        Q[1,1] = 0.0
        x_ref = x_ref
        x0 = x0
        p = p
        # 5 deg/s
        # typical servo speed 60deg/0.1s
        # u: throttle,steering
        du_max = np.array([0,radians(60)/0.1*dt])
        u_max = np.array([0,radians(25)])

        self.mpc.setup(n,m,p)
        self.mpc.convertLtv(A_vec,B_vec,P,Q,x_ref,x0,du_max,u_max)
        u_optimal = self.mpc.solve()
        # u is stacked, so [throttle_0,steering_0, throttle_1, steering_1]
        plt.plot(u_optimal[1::2,0])
        plt.show()
        # TODO
        steering = u_optimal[1,0]
        #print(u_optimal)

        # throttle is controller by other controller
        #throttle = u_optimal[0,1]
        throttle = self.calcThrottle(state,v_target)

        debug_dict['x_ref'] = x_ref_raw
        debug_dict['x_project'] = self.mpc.debug()
        ret =  (throttle,steering,True,debug_dict)
        tac = time()
        #print("freq = %.2f"%(1.0/(tac-tic)))
        return ret

    # initialize mpc
    # sim: an instance of advCarSim so we have access to parameters
    def initMpc(self,sim):
        # prediction step
        self.mpc_prediction_steps = 10
        # prediction discretization dt
        # NOTE we may be able to use a finer time step in x ref calculation, this can potentially increase accuracy
        self.mpc_dt = 0.01
        # together p*mpc_dt gives prediction horizon

        self.Caf = sim.Caf
        self.Car = sim.Car
        self.lf = sim.lf
        self.lr = sim.lr
        self.Iz = sim.Iz
        self.m = sim.m
        self.mpc = MPC()
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
