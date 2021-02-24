# stanley controller
from car import Car
from math import atan2,radians,degrees,sin,cos,pi,tan,copysign,asin,acos,isnan,exp,pi

class ctrlStanleyWrapper(Car):
    def __init__(self,car_setting,dt):
        super().__init__(car_setting,dt)
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
        v_target = min(v_target * 0.8, 1.2)

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
