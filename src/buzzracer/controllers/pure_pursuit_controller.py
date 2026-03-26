from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from math import sin, cos
from buzzracer.controllers.controller import Controller, ControllerConfig, ControllerState
from buzzracer.controllers.pid_controller import PidController

if TYPE_CHECKING:
    from buzzracer.cars.car_param import CarParam
    from buzzracer.main import MainState, MainConfig
    from buzzracer.tracks.track import Track


class PurePursuitControllerConfig(ControllerConfig):
    """ Config class, read-only"""

    def __init__(self, main_config: MainConfig, car_param: CarParam):
        self.max_speed = 4.0


class PurePursuitControllerState(ControllerState):
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


@Controller.register(PurePursuitControllerConfig, PurePursuitControllerState)
class PurePursuitController(Controller):

    def __init__(self):
        super().__init__()
        self.v_override = None
        self.max_speed = 5
        self.max_offset = 0.4

        P = 1.5  # to be more aggressive use 15
        I = 0.1
        D = 0.005
        dt = car.main.dt
        # integral limit, lpf curoff freq
        # self.throttle_pid = PidController(P,I,D,dt,1,2)
        self.throttle_pid = PidController(P, I, D, dt, 1, 1000)

        self.debug_dict = {}

        # if there's planner set it up
        # TODO put this in a parent class constructor
        try:
            config_planner = config.getElementsByTagName('planner')[0]
            planner_class = eval(config_planner.firstChild.nodeValue)
            self.planner = planner_class(config_planner)
            self.planner.main = self.main
            self.planner.car = self.car
            self.planner.init()
        except IndexError:
            self.print_info('planner not available')
            self.planner = None

    def init(self):
        Controller.init(self)
        self.track.prepare_discretized_raceline()
        self.track.create_boundary()
        self.discretized_raceline = self.track.discretized_raceline
        self.raceline_left_boundary = self.track.raceline_left_boundary
        self.raceline_right_boundary = self.track.raceline_right_boundary

    def control(self):
        if self.planner is None:
            raceline_pnts = self.track.raceline_points.T
            # raceline_headings = self.track.raceline_headings
            raceline_speed = self.track.raceline_velocity
        else:
            self.planner.plan()
            raceline_pnts = self.planner.best_plan_traj_points
            # TODO
            raceline_speed = np.ones_like(raceline_pnts[:, 0]) * 2.0
            self.planner.plot_all_solutions()

        x, y, heading, vf, vs, omega = self.car.state
        # find control point of distance lookahead
        dist = ((raceline_pnts[:, 0] - x)**2 +
                (raceline_pnts[:, 1] - y)**2)**0.5
        idx_car = np.argmin(dist)
        idx_lookahead = np.argmin(
            np.abs(dist[idx_car:] - self.lookahead)) + idx_car

        # change to local reference frame
        dx = raceline_pnts[idx_lookahead, 0] - x
        dy = raceline_pnts[idx_lookahead, 1] - y
        dx_body = dx * cos(heading) + dy * sin(heading)
        dy_body = -dx * sin(heading) + dy * cos(heading)

        # pure pursuit
        theta = np.arctan2(dx_body, dy_body)
        R = dist[idx_lookahead] / 2 / cos(theta)
        steering = np.arctan2(self.car.param.wheelbase, R)
        steering = np.copysign(steering, dy_body)

        # calculate steering
        # find reference speed
        # calculate throttle
        if self.v_override is None:
            v_target = raceline_speed[idx_car]
            v_target = min(v_target, self.max_speed)
        else:
            v_target = self.v_override

        throttle = self.calc_throttle(self.car.state, v_target)
        self.car.throttle = throttle
        self.car.steering = steering

        return None

    # get ss throttle, given ss velocity, linearfit
    def steady_state_throttle(self, velocity_ss):
        # 0.25 -> 0.94
        # 0.28 -> 1.4
        # 0.31 -> 1.9
        p = (0.06246385, 0.19171776)
        if velocity_ss > 0:
            return velocity_ss * p[0] + p[1]
        else:
            return 0

    # PID controller for forward velocity
    def calc_throttle(self, state, v_target):
        vf = state[3]
        # forgot how we got this
        # throttle = (acc_target + 1.01294228)/4.95445214

        # PID control for throttle
        throttle = self.throttle_pid.control(
            v_target, vf) + self.steady_state_throttle(v_target)

        return max(min(throttle, self.car.max_throttle), -1)
