''' Stanley controller, see https://ai.stanford.edu/~gabeh/papers/hoffmann_stanley_control07.pdf'''
from __future__ import annotations
from typing import TYPE_CHECKING
from math import isnan, pi, sin, cos

from buzzracer.types import CartesianState, Control
from buzzracer.controllers.car_controller import CarController
from buzzracer.controllers.pid_controller import PidController
if TYPE_CHECKING:
    from buzzracer.cars.car import CarParam


class StanleyCarControllerState:
    """ State class, pickleable, contains states that need to be preserved between iterations"""

    def __init__(self, config):
        # speed controller
        # P = 5 # to be more aggressive use 15
        # I = 0.0 #0.1
        # D = 0.4
        P = 1.5  # to be more aggressive use 15
        I = 0.0  # 0.1
        D = 0.005
        self.throttle_pid = PidController(P, I, D, config.dt, 1, 1000)
        self.v_override = None
        self.debug_dict = {}
        self.predicted_traj = []


class StanleyCarControllerConfig:
    """ Config class, read-only"""
    max_offset = 0.4
    max_speed = 4.0

    def __init__(self, main_config):
        p1 = (1.0, 2.0)
        p2 = (4.0, 0.5)
        self.Pfun_slope = (p2[1]-p1[1])/(p2[0]-p1[0])
        self.Pfun_offset = p1[1] - p1[0]*self.Pfun_slope
        self.dt = main_config.dt

    def Pfun(self, v):
        return max(min((self.Pfun_slope*v+self.Pfun_offset), 4.0), 0.5)/280*pi/0.01


@CarController.register(StanleyCarControllerConfig, StanleyCarControllerState)
class StanleyCarController(CarController):
    def __init__(self):
        # load config etc
        CarController.__init__(self)
        # integral limit, lpf curoff freq
        # self.throttle_pid = PidController(P,I,D,dt,1,2)
        # try:
        #     config_planner = config_minidom.getElementsByTagName('planner')[0]
        #     planner_class = eval(config_planner.firstChild.nodeValue)
        #     self.planner = planner_class(config_planner)
        #     assert self.planner is Planner
        #     self.planner.main = self.main
        #     self.planner.car = self.car
        #     self.planner.init()
        # except IndexError:
        #     self.print_info('planner not available')
        #     self.planner = None

    @staticmethod
    def control(car_state: CartesianState,
                car_params,
                track,
                config: StanleyCarControllerConfig,
                state: StanleyCarControllerState,
                reverse=False):
        ''' Given state of the vehicle and an instance of track,
        provide throttle and steering output
        Args:
          state: CartesianState (x,y,heading,v_forward,v_sideway,omega)
          car_params: CarParams, parameters of the car
          track: track object, can be RCPTrack or skidpad
          reverse: true if running in opposite direction of raceline init direction

        Outputs:
          control: Control(steering, throttle)
          valid:    If the car can be controlled here, false if too far off reference.
                    If this is false, then throttle will also be set to 0
          state: updated controller state
        '''
        coord = (car_state.x, car_state.y)

        heading = car_state.heading
        vf = car_state.v_forward

        # add in a slight lookahead distance
        lookahead = 3e-2  # TODO this should vary by car
        x_lookahead = coord[0] + cos(heading) * lookahead
        y_lookahead = coord[1] + sin(heading) * lookahead
        coord = (x_lookahead, y_lookahead)

        ctrl = Control(steering=0, throttle=0)
        fail_retval = (ctrl, False, state)

        # inquire information about desired trajectory close to the vehicle
        retval = track.local_trajectory(car_state)
        if retval is None:
            return fail_retval
            # return ret

        # parse return value from local_trajectory
        # (local_ctrl_pnt, offset, orientation, curvature, v_target) = retval
        (_, offset, orientation, _, v_target) = retval
        # for experiments
        # v_target = min(v_target*0.8, 2.2)
        v_target = min(v_target, config.max_speed)

        if isnan(orientation):
            return fail_retval

        if reverse:
            offset = -offset
            orientation += pi

        # if vehicle cross error exceeds maximum allowable error, stop the car
        if abs(offset) > config.max_offset:
            return fail_retval

        # sign convention for offset: negative offset(-) requires left steering(+)
        # this is the convention used in track class
        # control logic
        # steering = (orientation-heading) - (offset * self.car.P)
        # - (omega-curvature*vf)*self.car.D
        steering = (orientation-heading) - (offset * config.Pfun(abs(vf)))
        # print("D/P = "+str(abs((omega-curvature*vf)*D/(offset*P))))
        # handle edge case, unwrap ( -355 deg turn -> +5 turn)
        steering = (steering+pi) % (2*pi) - pi
        v_target = v_target if state.v_override is None else state.v_override
        throttle = StanleyCarController.calc_throttle(
            car_state, v_target, car_params, state.throttle_pid)

        ctrl = Control(steering=steering, throttle=throttle)
        return (ctrl, True, state)

    # PID controller for forward velocity
    @staticmethod
    def calc_throttle(state: CartesianState, v_target, car_params: CarParam, throttle_pid):
        # forgot how we got this
        # throttle = (acc_target + 1.01294228)/4.95445214

        # get ss throttle, given ss velocity, linearfit
        # TODO this should depend on car params
        def steady_state_throttle(velocity_ss):
            p = (0.06246385, 0.19171776)
            if velocity_ss > 0:
                return velocity_ss * p[0] + p[1]
            else:
                return 0

        # PID control for throttle
        throttle = throttle_pid.control(
            v_target, state.v_forward) + steady_state_throttle(v_target)

        return max(min(throttle, car_params.max_throttle), -1)
