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
from time import time
from timeUtil import execution_timer

class Car:
    def __init__(self,car_setting,dt):
        # max allowable crosstrack error in control algorithm, if vehicle cross track error is larger than this value,
        # controller would cease attempt to correct for it, and will brake vehicle to a stop
        # unit: m
        self.max_offset = 1.0
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
        self.optitrack_id = car_setting['optitrack_streaming_id']
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
#   track: track object, can be RCPTrack or skidpad
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
# NOTE this is a function template, inherit Car class and overload this function
    def ctrlCar(self,state,track,v_override=None,reverse=False):
        print_error("No implementation for ctrlCar() is defined, ctrlCar() in Car is a function template, inherit Car class and overload this function")
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
