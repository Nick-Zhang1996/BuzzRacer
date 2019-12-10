import numpy as np
from numpy import isclose from math import atan2,radians,degrees,sin,cos,pi,tan,copysign,asin,acos,isnan

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
            # TODO PI control for throttle
            throttle = self.max_throttle
            ret =  (throttle,steering,True,offset,(omega-curvature*vf))

        return ret

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
