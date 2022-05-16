from common import *
from math import isnan,pi,degrees,radians
from controller.CarController import CarController
from controller.PidController import PidController
from planner import Planner

class StanleyCarController(CarController):
    def __init__(self, car,config):
        super().__init__(car,config)
        self.debug_dict = {}
        self.max_offset = 0.4
        p1 = (1.0,2.0)
        p2 = (4.0,0.5)
        self.Pfun_slope = (p2[1]-p1[1])/(p2[0]-p1[0])
        self.Pfun_offset = p1[1] - p1[0]*self.Pfun_slope
        self.Pfun = lambda v: max(min((self.Pfun_slope*v+self.Pfun_offset),4.0),0.5)/280*pi/0.01

        #speed controller
        P = 5 # to be more aggressive use 15
        I = 0.0 #0.1
        D = 0.4
        dt = car.main.dt
        self.throttle_pid = PidController(P,I,D,dt,1,2)

        # to be overridden in config, if defined
        self.max_speed = 2.2
        self.print_ok("setting controller attributes")
        for key,value_text in config.attributes.items():
            setattr(self,key,eval(value_text))
            self.print_info(" controller.",key,'=',value_text)

        # if there's planner set it up
        # TODO put this in a parent class constructor
        try:
            config_planner = config.getElementsByTagName('planner')[0]
            planner_class = eval(config_planner.firstChild.nodeValue)
            self.planner = planner_class(config_planner)
            self.planner.main = self.main
            self.planner.car = self.car
            self.print_ok("setting planner attributes")
            for key,value_text in config_planner.attributes.items():
                setattr(self.planner,key,eval(value_text))
                self.print_info(" main.",key,'=',value_text)
            self.planner.init()
        except IndexError as e:
            self.print_info("planner not available")
            self.planner = None


    def control(self):
        # TODO do this more carefully
        if (self.planner is not None):
            self.planner.plan()
        throttle,steering,valid,debug_dict = self.ctrlCar(self.car.states,self.track)
        self.debug_dict = debug_dict
        self.car.debug_dict.update(debug_dict)
        #print("[StanleyCarController]: T= %4.1f, S= %4.1f (deg)"%( throttle,degrees(steering)))
        if valid:
            self.car.throttle = throttle
            self.car.steering = steering
        else:
            print_warning("[StanleyCarController]: car %d invalid results from ctrlCar", self.car.id)
            self.car.throttle = 0.0
            self.car.steering = 0.0
        #self.predict()
        return valid

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
    # NOTE this is the Stanley method, now that we have multiple control methods we may want to change its name later
    def ctrlCar(self,state,track,v_override=None,reverse=False):
        coord = (state[0],state[1])

        heading = state[2]
        omega = state[5]
        vf = state[3]
        vs = state[4]

        ret = (0,0,False,{'offset':0})

        # inquire information about desired trajectory close to the vehicle
        if self.planner is None:
            retval = track.localTrajectory(state)
        else:
            retval = self.planner.localTrajectory(state)
        if retval is None:
            return (0,0,False,{'offset':0})
            #return ret

        # parse return value from localTrajectory
        (local_ctrl_pnt,offset,orientation,curvature,v_target) = retval
        # for experiments
        #v_target = min(v_target*0.8, 2.2)
        v_target = min(v_target, self.max_speed)

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
            #steering = (orientation-heading) - (offset * self.car.P) - (omega-curvature*vf)*self.car.D
            steering = (orientation-heading) - (offset * self.Pfun(abs(vf)))
            # print("D/P = "+str(abs((omega-curvature*vf)*D/(offset*P))))
            # handle edge case, unwrap ( -355 deg turn -> +5 turn)
            steering = (steering+pi)%(2*pi) -pi
            if (steering>self.car.max_steering_left):
                steering = self.car.max_steering_left
            elif (steering<-self.car.max_steering_right):
                steering = -self.car.max_steering_right
            if (v_override is None):
                throttle = self.calcThrottle(state,v_target)
            else:
                throttle = self.calcThrottle(state,v_override)

            #ret =  (throttle,steering,True,{'offset':offset,'dw':omega-curvature*vf,'vf':vf,'v_target':v_target,'local_ctrl_point':local_ctrl_pnt})
            ret =  (throttle,steering,True,{})

        return ret

    # PID controller for forward velocity
    def calcThrottle(self,state,v_target):
        vf = state[3]
        # PI control for throttle
        acc_target = self.throttle_pid.control(v_target,vf)
        throttle = (acc_target + 1.01294228)/4.95445214 

        return max(min(throttle,self.car.max_throttle),-1)

