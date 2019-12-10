import numpy as np
from scipy import signal
from numpy import isclose 
from math import atan2,radians,degrees,sin,cos,pi,tan,copysign,asin,acos,isnan
import matplotlib.pyplot as plt

class Car:
    def __init__(self):
        # max allowable crosstrack error, if vehicle cross track error is larger than this value, vehicle will be braked
        self.max_offset = 0.3
        # controller tuning, steering->lateral offset
        # P is applied on offset
        self.P = 0.8/180*pi/0.01
        # 5 deg of correction for every 3 rad/s overshoot
        # D is applied on delta_omega
        self.D = radians(4)/3
        self.max_throttle = 0.3
        self.max_steering = radians(24.5)
        self.throttle_P = 1
        self.throttle_I = 0.1
        self.throttle_z = [0.0]
        # denominator of throttle integral
        self.a = np.array([1.0])
        # numerator of throttle integral, second term is time constant
        tc = 3
        self.b = np.array([2,1.0])
        pass

# given state of the vehicle and an instance of track, provide throttle and steering output
# input:
#   state: (x,y,heading,v_forward,v_sideway,omega)
#   track: track object, can be RCPtrack or skidpad
#   reverse: true if running in opposite direction of raceline init direction

# output:
#   (throttle,steering,valid,debug) 
# ranges for output:
#   throttle -1.0,1.0
#   steering as an angle in radians, TRIMMED to MAX_STEERING, left positive
#   valid: bool, if the car can be controlled here, if this is false, then throttle will also be set to 0
# debug: a list of objects to be debugged
    def ctrlCar(self,state,track,reverse=False):
        coord = (state[0],state[1])
        heading = state[2]
        omega = state[5]
        vf = state[3]
        vs = state[4]
        ret = (0,0,False,0,0)

        retval = track.localTrajectory(state)
        if retval is None:
            return ret

        (local_ctrl_pnt,offset,orientation,curvature) = retval

        if isnan(orientation):
            return ret
            
        if reverse:
            offset = -offset
            orientation += pi

        # how much to compensate for per meter offset from track
        if (abs(offset) > self.max_offset):
            return (0,0,False,offset,0)
        else:
            # sign convention for offset: - requires left steering(+)
            steering = orientation-heading - offset * self.P - (omega-curvature*vf)*self.D
            #print("D/P = "+str(abs((omega-curvature*vf)*D/(offset*P))))
            # handle edge case, unwrap ( -355 deg turn -> +5 turn)
            steering = (steering+pi)%(2*pi) -pi
            if (steering>self.max_steering):
                steering = self.max_steering
            elif (steering<-self.max_steering):
                steering = -self.max_steering
            throttle = self.calcThrottle(vf,v_target)

            ret =  (throttle,steering,True,offset,(omega-curvature*vf))

        return ret

    def calcThrottle(self,v,v_target):
        # PI control for throttle
        v_err = v_target - v
        throttle_integral, self.throttle_z = signal.lfilter(self.b,self.a,[v_err],zi=self.throttle_z)
        throttle = self.throttle_P * v_err + throttle_integral * self.throttle_I
        return throttle

    # update car state with bicycle model, no slip
    # dt: time, in sec
    # v: velocity of rear wheel, in m/s
    # state: (x,y,theta), np array
    # return new state (x,y,theta)
# XXX directly copied from track.py
    def updateCar(self,state,throttle,steering,dt):
        # wheelbase, in meter
        # heading of pi/2, i.e. vehile central axis aligned with y axis,
        # means theta = 0 (the x axis of car and world frame is aligned)
        v = throttle
        theta = state[2] - pi/2
        L = 98e-3
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
# should be x,y,heading,vf,vs,omega
        return np.array([state[0]+dX,state[1]+dY,state[2]+dtheta,v,0,dtheta/dt])


if __name__ == '__main__':
    # tune PI controller 
    v_log = []
    control_log = []
    v_targets = np.ones(1000)
    integral = 0
    throttle = 0
    I = 1
    v = 0
    for v_target in v_targets:
        v += max(throttle*0.01,0)
        integral = integral*0.5 + (v_target-v)
        throttle = I*integral
        control_log.append(throttle)
        v_log.append(v)
    p0, = plt.plot(v_log,label='velocity')
    p1, = plt.plot(control_log,label='output')
    plt.legend(handles=[p0,p1])
    plt.show()
    exit(0)


    car = Car()
    sample_verr = np.ones(10)
    sample_throttle = []
    for i in range(10):
        sample_throttle.append(car.calcThrottle(0,1))
    print(np.array(sample_throttle))
    plt.plot(sample_throttle)
    plt.show()

