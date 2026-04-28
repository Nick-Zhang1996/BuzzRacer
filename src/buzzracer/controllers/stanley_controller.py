''' Stanley controller, see https://ai.stanford.edu/~gabeh/papers/hoffmann_stanley_control07.pdf'''
from __future__ import annotations
from typing import TYPE_CHECKING
import logging
from dataclasses import replace
from math import pi, sin, cos

import numpy as np

from buzzracer.common import LoggingFilter
from buzzracer.types import CartesianState, CurvilinearState, Control
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
        self.trajectory_progress_gain = 4.0
        self.trajectory_progress_max_correction = 1.0

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
        name = car_params.name if car_params.name == 'mclaren_22' else None

        if controller_config.planner:
            if not planner_state.planner_ready.is_set():
                return (ctrl, False, controller_state, 'Planner not ready')
            curv_len = track.data.raceline_len_m
            curv_state = track.cart_to_curv(lookahead_point)
            with planner_state.traj_sync_lock:
                split_point = planner_state.split_point_sync.value
                wrap_s = (curv_state.progress - split_point) % curv_len + split_point
                curv_state.progress = wrap_s

                traj_len = planner_state.cart_traj_len.value
                cart_shape = (6, planner_state.car_count, traj_len)
                cart_traj = np.frombuffer(planner_state.cart_traj_sync,
                                          dtype=np.float64,
                                          count=cart_shape[0]*cart_shape[1]*cart_shape[2]
                                          ).reshape(cart_shape, order='F').copy()
                curv_shape = (5, planner_state.car_count, traj_len)
                curv_traj = np.frombuffer(planner_state.curv_traj_sync,
                                          dtype=np.float64,
                                          count=curv_shape[0]*curv_shape[1]*curv_shape[2]
                                          ).reshape(curv_shape, order='F').copy()
                traj_ts = np.frombuffer(planner_state.traj_ts_sync,
                                        dtype=np.float64,
                                        count=traj_len,
                                        ).reshape(traj_len, order='F').copy()
            retval = StanleyController.local_trajectory_from_traj(
                cart_traj[:, car_index, :], lookahead_point)
            if retval is None:
                # Fallback to stanley
                # TODO maybe something more intelligent? stay at current offset e.g.
                retval = track.local_trajectory(lookahead_point)
                logger.warning("%s fallback to stanley", name)
            else:
                progress_err = StanleyController.get_progress_err(main_state.time,
                                                                  traj_ts,
                                                                  curv_traj[:, car_index, :],
                                                                  curv_state,
                                                                  name)
                old_target_v = retval.v_target
                target_v = StanleyController.adjust_speed_target_with_progress(retval.v_target,
                                                                               progress_err,
                                                                               controller_config)

                # dv = target_v - retval.v_target
                retval = retval._replace(v_target=target_v)
                # DEBUG
                # if name is not None:
                #     # NOTE dv > 0, yet progress_err is not reducing
                #     curv_v_actual = np.diff(curv_traj[0, car_index, :]) / 0.05
                #     cart_v_actual = np.hypot(np.diff(cart_traj[0, car_index, :]),
                #                              np.diff(cart_traj[1, car_index, :])) / 0.05
                #     v_ref = curv_traj[3, car_index, :]
                #     # logger.debug(f'{curv_v_actual=}, {cart_v_actual=}, {v_ref=}')
                #     logger.debug(f'{name} {progress_err=:.2f} {old_target_v=:.2f}, {dv=:.2f}')

        else:  # No planner available
            retval = track.local_trajectory(lookahead_point)
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
        v_target = (v_target if controller_state.v_override is None else controller_state.v_override)
        throttle = StanleyController.calc_throttle(
            car_state, v_target, car_params, controller_state.throttle_pid)
        main_state.car_target_v[car_index] = v_target
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
        return np.clip(v_ref + correction, 0.0, None)

    @staticmethod
    def get_progress_err(ts: float,
                         traj_ts: np.ndarray,
                         curv_traj: np.ndarray,
                         state: CurvilinearState,
                         name=None):
        """ Return err_s = ref progress - actual progress, using ts in traj_ts """

        idx = np.searchsorted(traj_ts[1:-1], ts)
        fraction = (ts - traj_ts[idx]) / (traj_ts[idx+1] - traj_ts[idx])
        ref_s = curv_traj[0, idx] + fraction * (curv_traj[0, idx+1] - curv_traj[0, idx])
        err_s = ref_s - state.progress
        return err_s

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
