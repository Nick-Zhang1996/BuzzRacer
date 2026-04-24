''' Stanley controller, see https://ai.stanford.edu/~gabeh/papers/hoffmann_stanley_control07.pdf'''
from __future__ import annotations
from typing import TYPE_CHECKING
import logging

import numpy as np
from math import pi, sin, cos

from buzzracer.common import LoggingFilter
from buzzracer.types import CartesianState, Control
from buzzracer.controllers.controller import Controller, ControllerConfig, ControllerState
from buzzracer.controllers.pid_controller import PidController
from buzzracer.tracks.track import LocalTrajOutput
if TYPE_CHECKING:
    from buzzracer.cars.car_param import CarParam
    from buzzracer.main import MainState, MainConfig
    from buzzracer.tracks.track import Track

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addFilter(LoggingFilter(interval=1.0))


class StanleyControllerConfig(ControllerConfig):
    """ Config class, read-only"""

    def __init__(self, main_config: MainConfig, car_param: CarParam):
        super().__init__(main_config, car_param)
        self.max_offset = 0.4
        self.max_speed = 4.0
        self.rear_end_gap = 0.2
        self.use_trajectory_longitudinal_control = False
        self.trajectory_progress_gain = 2.0
        self.trajectory_progress_max_correction = 2.0
        self.trajectory_index_search_window = 8
        self.trajectory_index_reset_dist = 0.3
        self.trajectory_reference_lookahead_steps = 1

        p1 = (1.0, 2.0)
        p2 = (4.0, 0.5)
        self.Pfun_slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
        self.Pfun_offset = p1[1] - p1[0] * self.Pfun_slope
        # p1 = (1.0, 4.0)
        # p2 = (4.0, 1.0)
        # self.Pfun_slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
        # self.Pfun_offset = p1[1] - p1[0] * self.Pfun_slope

        self.dt = main_config.dt

    def Pfun(self, v):
        return max(min((self.Pfun_slope * v + self.Pfun_offset), 4.0), 0.5) / 280 * pi / 0.01


class StanleyControllerState(ControllerState):
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
        self.trajectory_target_index = None


@Controller.register(StanleyControllerConfig, StanleyControllerState)
class StanleyController(Controller):

    def __init__(self):
        # load config etc
        Controller.__init__(self)
        # integral limit, lpf curoff freq
        # self.throttle_pid = PidController(P,I,D,dt,1,2)

    @staticmethod
    def control(car_state: CartesianState,
                car_params: CarParam,
                track: Track,
                controller_config: StanleyControllerConfig,
                controller_state: StanleyControllerState,
                main_state: MainState,
                car_index,
                planner_state=None):
        ''' Given state of the vehicle and an instance of track,
        provide throttle and steering output
        Args:
          state: CartesianState (x,y,heading,v_forward,v_sideway,omega)
          car_params: CarParams, parameters of the car
          track: track object, can be RCPTrack or skidpad
          planner_state: if planner is available, planner.state object

        Outputs:
          control: Control(steering, throttle)
          valid:    If the car can be controlled here, false if too far off reference. If this is false, then throttle will also be set to 0
          state: updated controller state
          msg: Message explaining cause for invalid control
        '''
        heading = car_state.heading

        # add in a slight lookahead distance
        lookahead = 3e-2 + car_params.lf

        ctrl = Control(steering=0, throttle=0)

        # inquire information about desired trajectory close to the vehicle
        lookahead_point = CartesianState(
            x=car_state.x + lookahead * cos(heading),
            y=car_state.y + lookahead * sin(heading),
            heading=heading,
            v_forward=car_state.v_forward,
            v_sideway=car_state.v_sideway,
            omega=car_state.omega)

        if controller_config.planner:
            if not planner_state.planner_ready.is_set():
                return (ctrl, False, controller_state, 'Planner not ready')
            with planner_state.traj_sync_lock:
                shape = (6, planner_state.car_count, planner_state.cart_traj_len.value)
                cart_traj = np.frombuffer(planner_state.cart_traj_sync,
                                          dtype=np.float64,
                                          count=shape[0]*shape[1]*shape[2]
                                          ).reshape(shape, order='F').copy()
            retval = StanleyController.local_trajectory_from_traj(
                cart_traj[:, car_index, :], lookahead_point)
            trajectory_ref = StanleyController.trajectory_reference_from_traj(
                cart_traj[:, car_index, :], car_state, controller_config, controller_state)

            track_retval = track.local_trajectory(lookahead_point)
            if retval is None:
                # Fallback to stanley
                retval = track_retval
                name = car_params.name
                logger.warning("%s fallback to stanley", name)
        else:
            retval = track.local_trajectory(lookahead_point)
            trajectory_ref = None
        if retval is None:
            return (ctrl, False, controller_state, 'local_traj returned None')

        v_target = min(retval.v_target, controller_config.max_speed)

        offset = retval.lateral_err
        orientation = retval.raceline_dir

        # if vehicle cross error exceeds maximum allowable error, stop the car
        if abs(offset) > controller_config.max_offset:
            msg = f'{abs(offset)=} > {controller_config.max_offset=}:'
            return (ctrl, False, controller_state, msg)

        # sign convention for offset: negative offset(-) requires left steering(+)
        # this is the convention used in track class
        # control logic
        # steering = (orientation-heading) - (offset * self.car.P)
        # - (omega-curvature*vf)*self.car.D
        steering = (orientation - heading) - (
            offset * controller_config.Pfun(abs(car_state.v_forward)))
        # print("D/P = "+str(abs((omega-curvature*vf)*D/(offset*P))))
        # handle edge case, unwrap ( -355 deg turn -> +5 turn)
        steering = (steering + pi) % (2 * pi) - pi
        throttle_v_target = v_target
        if (trajectory_ref is not None and controller_config.use_trajectory_longitudinal_control
                and controller_state.v_override is None):
            throttle_v_target = StanleyController.adjust_speed_target_with_progress(
                trajectory_ref['v_ref'],
                trajectory_ref['progress_error'],
                controller_config)
            throttle = StanleyController.calc_throttle(
                car_state,
                throttle_v_target,
                car_params,
                controller_state.throttle_pid,
                ss_v_target=trajectory_ref['v_ref'])
        else:
            throttle_v_target = (throttle_v_target if controller_state.v_override is None
                                 else controller_state.v_override)
            throttle = StanleyController.calc_throttle(
                car_state, throttle_v_target, car_params, controller_state.throttle_pid)
        main_state.car_target_v[car_index] = throttle_v_target
        controller_state.debug_dict = {
            'v_target': float(v_target),
            'throttle_v_target': float(throttle_v_target),
            'trajectory_progress_error': None if trajectory_ref is None else float(
                trajectory_ref['progress_error']),
            'trajectory_target_index': None if trajectory_ref is None else int(
                trajectory_ref['index']),
        }

        ctrl = Control(steering=steering, throttle=throttle)
        return (ctrl, True, controller_state, 'Controller OK')

    # PID controller for forward velocity
    @staticmethod
    def calc_throttle(state: CartesianState, v_target, car_params: CarParam,
                      throttle_pid, ss_v_target=None):
        # forgot how we got this
        # throttle = (acc_target + 1.01294228)/4.95445214

        if ss_v_target is None:
            ss_v_target = v_target
        ss_throttle = car_params.ss_throttle_p0 * ss_v_target + car_params.ss_throttle_p1
        ss_throttle = ss_throttle if ss_v_target > 0 else 0
        # PID control for throttle
        throttle = throttle_pid.control(v_target, state.v_forward) + ss_throttle

        return max(min(throttle, car_params.max_throttle), -1)

    @staticmethod
    def adjust_speed_target_with_progress(v_ref: float,
                                          progress_error: float,
                                          controller_config: StanleyControllerConfig):
        correction = np.clip(
            controller_config.trajectory_progress_gain * progress_error,
            -controller_config.trajectory_progress_max_correction,
            controller_config.trajectory_progress_max_correction)
        return np.clip(v_ref + correction, a_min=0.0, a_max=controller_config.max_speed)

    @staticmethod
    def trajectory_reference_from_traj(cart_traj: np.ndarray,
                                       state: CartesianState,
                                       controller_config: StanleyControllerConfig,
                                       controller_state: StanleyControllerState):
        if cart_traj.shape[1] < 2:
            controller_state.trajectory_target_index = None
            return None

        point_count = cart_traj.shape[1]
        max_index = point_count - 2
        prev_index = controller_state.trajectory_target_index
        if prev_index is None:
            base_index = 0
        else:
            base_index = min(max(prev_index + 1, 0), max_index)

        window = max(int(controller_config.trajectory_index_search_window), 0)
        start = max(base_index - window, 0)
        end = min(base_index + window + 1, max_index + 1)
        local_idx = StanleyController.closest_traj_index(cart_traj, state, start, end)
        index = local_idx

        dx = state.x - cart_traj[0, index]
        dy = state.y - cart_traj[1, index]
        dist_sq = dx * dx + dy * dy
        reset_dist_sq = controller_config.trajectory_index_reset_dist ** 2
        if dist_sq > reset_dist_sq:
            global_idx = StanleyController.closest_traj_index(cart_traj, state, 0, max_index + 1)
            if global_idx is not None:
                index = global_idx

        lookahead_steps = max(int(controller_config.trajectory_reference_lookahead_steps), 0)
        index = min(index + lookahead_steps, max_index)
        controller_state.trajectory_target_index = index

        ref_xy = cart_traj[:2, index]
        next_xy = cart_traj[:2, index + 1]
        tangent = next_xy - ref_xy
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-9:
            tangent_hat = np.array([cos(state.heading), sin(state.heading)])
        else:
            tangent_hat = tangent / tangent_norm
        progress_error = np.dot(ref_xy - np.array([state.x, state.y]), tangent_hat)
        return {
            'index': index,
            'progress_error': progress_error,
            'v_ref': cart_traj[3, index],
        }

    @staticmethod
    def closest_traj_index(cart_traj: np.ndarray,
                           state: CartesianState,
                           start: int,
                           end: int):
        if start >= end:
            return None
        dxx = cart_traj[0, start:end] - state.x
        dyy = cart_traj[1, start:end] - state.y
        return start + int(np.argmin(dxx**2 + dyy**2))

    @staticmethod
    def local_trajectory_from_traj(cart_traj: np.ndarray, state: CartesianState):
        """ Get local_trajectory from a cartesian trajectory 
        Args:
            cart_traj: (6, horizon) array of (x,y,heading, vf, vs, omega)
            state: car state
        Return:
            LocalTrajOutput
        """
        if cart_traj.shape[1] == 0:
            return None
        dxx = state.x - cart_traj[0, :]
        dyy = state.y - cart_traj[1, :]
        dist = dxx**2 + dyy**2
        i = np.argmin(dist[:-1])  # save the very last for rdx,rdy calculation
        dx = dxx[i]  # ref point to car
        dy = dyy[i]
        rdx = cart_traj[0, i+1] - cart_traj[0, i]  # ref point to next ref point
        rdy = cart_traj[1, i+1] - cart_traj[1, i]
        # print(f'{dx=},{dy=},{rdx=},{rdy=}')
        offset = (rdx * dy - rdy * dx) / (rdx*rdx + rdy*rdy)**0.5  # cross product
        phi = np.arctan2(rdy, rdx)
        if i == len(dist)-2:
            return None

        return LocalTrajOutput(ref_point=cart_traj[:2, i],
                               lateral_err=offset,
                               raceline_dir=phi,
                               curvature=None,
                               v_target=cart_traj[3, i],
                               progress=None,
                               left_margin=None,
                               right_margin=None)
