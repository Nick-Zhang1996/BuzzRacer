from common import *
from math import isnan,pi,degrees,radians,sin,cos
from controller.CarController import CarController
from controller.PidController import PidController
from planner import Planner
import matplotlib.pyplot as plt

class PurePursuitCarController(CarController):
    def __init__(self, car,config):
        super().__init__(car,config)

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
            '''
            self.print_ok("setting planner attributes")
            for key,value_text in config_planner.attributes.items():
                setattr(self.planner,key,eval(value_text))
                self.print_info(" main.",key,'=',value_text)
            '''
            self.planner.init()
        except IndexError as e:
            self.print_info("planner not available")
            self.planner = None

    def init(self):
        CarController.init(self)
        self.debug_dict = {}
        self.max_offset = 0.4

        #speed controller
        P = 5 # to be more aggressive use 15
        I = 0.0 #0.1
        D = 0.4
        dt = self.car.main.dt
        self.throttle_pid = PidController(P,I,D,dt,1,2)

        self.track.prepareDiscretizedRaceline()
        self.track.createBoundary()
        self.discretized_raceline = self.track.discretized_raceline
        self.raceline_left_boundary = self.track.raceline_left_boundary
        self.raceline_right_boundary = self.track.raceline_right_boundary

    def control(self):
        if self.planner is None:
            raceline_pnts = self.track.raceline_points.T
            raceline_headings = self.track.raceline_headings
            raceline_speed = self.track.raceline_velocity

        x,y,heading,vf,vs,omega = self.car.states
        # find control point of distance lookahead
        dist = ((raceline_pnts[:,0] - x)**2 + (raceline_pnts[:,1] - y)**2)**0.5
        idx_car = np.argmin(dist)
        idx_lookahead = np.argmin( np.abs(dist[idx_car:] - self.lookahead) ) + idx_car

        # change to local reference frame
        dx = raceline_pnts[idx_lookahead,0] - x
        dy = raceline_pnts[idx_lookahead,1] - y
        dx_body = dx*cos(heading) + dy*sin(heading)
        dy_body = -dx*sin(heading) + dy*cos(heading)

        # pure pursuit
        theta = np.arctan2(dx_body,dy_body)
        R = dist[idx_lookahead] / 2 / cos(theta)
        steering = np.arctan2(self.car.wheelbase,R)
        steering = np.copysign(steering,dy_body)

        '''
        # plot for sanity
        fig,ax = plt.subplots()
        plt.plot(x,y,'o')
        plt.plot(raceline_pnts[idx_car:idx_lookahead,0],raceline_pnts[idx_car:idx_lookahead,1])
        #plt.plot(raceline_pnts[:,0],raceline_pnts[:,1])
        plt.plot(raceline_pnts[idx_car,0],raceline_pnts[idx_car,1],'*')
        plt.plot(raceline_pnts[idx_lookahead,0],raceline_pnts[idx_lookahead,1],'o')

        # car centered
        center = (x,y)
        circle = plt.Circle(center,lookahead,fill=False)
        ax.add_patch(circle)
        ax.set_xlim((x-0.5,x+0.5))
        ax.set_ylim((y-0.5,y+0.5))
        center = (x + R*cos(heading - np.pi/2), y + R*sin(heading - np.pi/2))
        circle = plt.Circle(center,R,fill=False)
        ax.add_patch(circle)
        plt.show()
        '''

        # calculate steering
        # find reference speed
        # calculate throttle
        v_target = raceline_speed[idx_car]

        '''
        if retval is None:
            return (0,0,False,{'offset':0})
        '''

        v_target = min(v_target, self.max_speed)

        # TODO more error handling

        if (steering>self.car.max_steering_left):
            steering = self.car.max_steering_left
        elif (steering<-self.car.max_steering_right):
            steering = -self.car.max_steering_right

        throttle = self.calcThrottle(self.car.states,v_target)
        self.car.throttle = throttle
        self.car.steering = steering

        return None


    # PID controller for forward velocity
    def calcThrottle(self,state,v_target):
        vf = state[3]
        # PI control for throttle
        acc_target = self.throttle_pid.control(v_target,vf)
        throttle = (acc_target + 1.01294228)/4.95445214 

        return max(min(throttle,self.car.max_throttle),-1)

