''' Stanley controller, see https://ai.stanford.edu/~gabeh/papers/hoffmann_stanley_control07.pdf'''
from __future__ import annotations
from typing import TYPE_CHECKING
from math import isnan, pi, sin, cos

from buzzracer.types import CartesianState, Control
from buzzracer.controllers.car_controller import CarController, CarControllerConfig, CarControllerState
from buzzracer.controllers.pid_controller import PidController
if TYPE_CHECKING:
    from buzzracer.cars.car import CarParam
    from buzzracer.main import MainState, MainConfig
    from buzzracer.tracks.track import Track


class StanleyCarControllerState(CarControllerState):
    """ State class, pickleable, contains states that need to be preserved between iterations"""

    def __init__(self, config):
        # speed controller
        # P = 1.5  # to be more aggressive use 15
        # I = 0.0  # 0.1
        # D = 0.005

        P = 1.0
        I = 0.1
        D = 0.01

        self.throttle_pid = PidController(P, I, D, config.dt, 2, 10)
        self.v_override = None
        self.debug_dict = {}
        self.predicted_traj = []


class StanleyCarControllerConfig(CarControllerConfig):
    """ Config class, read-only"""

    def __init__(self, main_config: MainConfig, car_param: CarParam):
        self.max_offset = 0.4
        self.max_speed = 4.0
        self.rear_end_gap = 0.2

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
                car_params: CarParam,
                track: Track,
                controller_config: StanleyCarControllerConfig,
                controller_state: StanleyCarControllerState,
                main_state: MainState,
                car_index,
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
        heading = car_state.heading

        # add in a slight lookahead distance
        lookahead = 3e-2 + car_params.wheelbase

        ctrl = Control(steering=0, throttle=0)
        fail_retval = (ctrl, False, controller_state)

        # inquire information about desired trajectory close to the vehicle
        lookahead_point = CartesianState(x=car_state.x + lookahead*cos(heading),
                                         y=car_state.y + lookahead*sin(heading),
                                         heading=heading,
                                         v_forward=car_state.v_forward,
                                         v_sideway=car_state.v_sideway,
                                         omega=car_state.omega)

        retval = track.local_trajectory(lookahead_point)
        if retval is None:
            return fail_retval
            # return ret

        v_target = min(retval.v_target, controller_config.max_speed)

        offset = retval.lateral_err
        orientation = retval.raceline_dir

        if isnan(orientation):
            return fail_retval

        if reverse:
            offset = -offset
            orientation += pi

        # if vehicle cross error exceeds maximum allowable error, stop the car
        if abs(offset) > controller_config.max_offset:
            return fail_retval

        # sign convention for offset: negative offset(-) requires left steering(+)
        # this is the convention used in track class
        # control logic
        # steering = (orientation-heading) - (offset * self.car.P)
        # - (omega-curvature*vf)*self.car.D
        steering = (orientation-heading) - (offset *
                                            controller_config.Pfun(abs(car_state.v_forward)))
        # print("D/P = "+str(abs((omega-curvature*vf)*D/(offset*P))))
        # handle edge case, unwrap ( -355 deg turn -> +5 turn)
        steering = (steering+pi) % (2*pi) - pi
        v_target = v_target if controller_state.v_override is None else controller_state.v_override
        throttle = StanleyCarController.calc_throttle(
            car_state, v_target, car_params, controller_state.throttle_pid)
        main_state.car_target_v[car_index] = v_target

        ctrl = Control(steering=steering, throttle=throttle)
        return (ctrl, True, controller_state)

    # PID controller for forward velocity
    @staticmethod
    def calc_throttle(state: CartesianState, v_target, car_params: CarParam, throttle_pid):
        # forgot how we got this
        # throttle = (acc_target + 1.01294228)/4.95445214

        ss_throttle = car_params.ss_throttle_p0 * v_target + car_params.ss_throttle_p1
        ss_throttle = ss_throttle if v_target > 0 else 0
        # PID control for throttle
        throttle = throttle_pid.control(v_target, state.v_forward) + ss_throttle

        return max(min(throttle, car_params.max_throttle), -1)
