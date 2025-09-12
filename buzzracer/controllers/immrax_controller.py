from math import isnan, pi
from buzzracer.controllers.car_controller import CarController
from buzzracer.controllers.pid_controller import PidController

from buzzracer.extensions.simulators.immrax_dynamic_bycicle import DynamicBicycle

import jax
import jax.numpy as jnp

PRNG_SEED = 0


class ImmraxController(CarController):
    def __init__(self, car, config):
        # defaults for configurable parameters
        # NOTE these will be overridden
        self.max_offset = 0.4
        self.max_speed = 4.0
        # load config etc
        super().__init__(car, config)

        self.debug_dict = {}
        p1 = (1.0, 2.0)
        p2 = (4.0, 0.5)
        self.Pfun_slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
        self.Pfun_offset = p1[1] - p1[0] * self.Pfun_slope
        self.Pfun = (
            lambda v: max(min((self.Pfun_slope * v + self.Pfun_offset), 4.0), 0.5)
            / 280
            * pi
            / 0.01
        )

        # speed controller
        P = 1.5  # to be more aggressive use 15
        I = 0.0  # 0.1
        D = 0.005
        dt = car.main.dt
        # integral limit, lpf curoff freq
        # self.throttle_pid = PidController(P,I,D,dt,1,2)
        self.throttle_pid = PidController(P, I, D, dt, 1, 1000)

        self.planning_dt = 0.02
        self.planning_horizon = 50  # time steps
        self.num_samples = 2
        # TODO: randomly sample control trajectory
        self.throttle_std = car.max_throttle / 2.0
        self.steering_std = car.max_steering_left / 2.0
        self.planned_controls: jnp.ndarray = jnp.zeros(
            (self.planning_horizon, 2)
        )  # (throttle, steering)
        self.sampled_controls: jnp.ndarray = jnp.zeros(
            (self.num_samples, self.planning_horizon, 2)
        )  # (throttle, steering)
        self.prng_key = jax.random.key(PRNG_SEED)

        self.control_action = lambda t, x: self.planned_controls[
            jnp.floor(t / self.planning_dt).astype(int) % self.planning_horizon
        ]
        self.disturbance = lambda t, x: jnp.array([0.0, 0.0])
        self.predictor = DynamicBicycle(car)

        self.rollout_sampled_trajectories = jax.vmap(
            self.rollout_sampled_trajectory, in_axes=(None, 0)
        )

    def control(self):
        throttle, steering, valid, debug_dict = self.ctrl_car(
            self.car.states, self.track
        )
        self.debug_dict = debug_dict
        self.car.debug_dict.update(debug_dict)
        # self.print_info("car %d, T= %4.1f, S= %4.1f (deg)"%(self.car.id, throttle,degrees(steering)))
        if valid:
            self.car.throttle = throttle
            self.car.steering = steering
        else:
            self.print_warning(" car %d invalid results from ctrl_car", self.car.id)
            self.car.throttle = 0.0
            self.car.steering = 0.0
        # self.predict()
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
    def ctrl_car(self, state, track, v_override=None, reverse=False):
        heading = state[2]
        vf = state[3]

        self.sample_controls()
        traj = self.rollout_sampled_trajectories(
            jnp.array(state[0:6]), self.sampled_controls
        )

        # TODO: evaluate cost of each sampled trajectory, pick the best one
        self.planned_controls = self.sampled_controls[0, :, :]
        self.plot_trajectory(traj.ys[0])

        ret = (0, 0, False, {"offset": 0})

        # inquire information about desired trajectory close to the vehicle
        retval = track.local_trajectory(state)
        if retval is None:
            return (0, 0, False, {"offset": 0})

        # parse return value from local_trajectory
        (local_ctrl_pnt, offset, orientation, curvature, v_target) = retval
        # for experiments
        # v_target = min(v_target*0.8, 2.2)
        v_target = min(v_target, self.max_speed)

        if isnan(orientation):
            return (0, 0, False, {"offset": 0})

        if reverse:
            offset = -offset
            orientation += pi

        # if vehicle cross error exceeds maximum allowable error, stop the car
        if abs(offset) > self.max_offset:
            return (0, 0, False, {"offset": offset})
        else:
            # sign convention for offset: negative offset(-) requires left steering(+)
            # this is the convention used in track class, determined arbituarily
            # control logic
            # steering = (orientation-heading) - (offset * self.car.P) - (omega-curvature*vf)*self.car.D
            steering = (orientation - heading) - (offset * self.Pfun(abs(vf)))
            # print("D/P = "+str(abs((omega-curvature*vf)*D/(offset*P))))
            # handle edge case, unwrap ( -355 deg turn -> +5 turn)
            steering = (steering + pi) % (2 * pi) - pi
            if steering > self.car.max_steering_left:
                steering = self.car.max_steering_left
            elif steering < -self.car.max_steering_right:
                steering = -self.car.max_steering_right
            if v_override is None:
                throttle = self.calc_throttle(state, v_target)
            else:
                throttle = self.calc_throttle(state, v_override)

            # ret =  (throttle,steering,True,{'offset':offset,'dw':omega-curvature*vf,'vf':vf,'v_target':v_target,'local_ctrl_point':local_ctrl_pnt})
            ret = (throttle, steering, True, {})

        return ret

    def sample_controls(self):
        self.sampled_controls.at[:, :, 0].set(
            self.planned_controls[:, 0]
            + self.throttle_std
            * jax.random.normal(
                self.prng_key,
                shape=(self.num_samples, self.planning_horizon),
            )
        )  # TODO: may want to consider steady_state_throttle explicitly
        self.prng_key = jax.random.split(self.prng_key)[1]
        self.sampled_controls.at[:, :, 1].set(
            self.planned_controls[:, 1]
            + self.steering_std
            * jax.random.normal(
                self.prng_key,
                shape=(self.num_samples, self.planning_horizon),
            )
        )
        self.prng_key = jax.random.split(self.prng_key)[1]
        # TODO: clip to max steering, throttle

    def rollout_sampled_trajectory(self, x0, control_traj):
        traj = self.predictor.compute_trajectory(
            0.0,
            self.planning_horizon * self.planning_dt,
            x0,
            (self.control_action, self.disturbance),
            dt=self.planning_dt,
        )  # NOTE: this is assuming the system is time-invariant
        return traj

    # get ss throttle, given ss velocity, linearfit
    def steady_state_throttle(self, velocity_ss):
        # 0.25 -> 0.94
        # 0.28 -> 1.4
        # 0.31 -> 1.9
        p = jnp.array([0.06246385, 0.19171776])
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
        throttle = self.throttle_pid.control(v_target, vf) + self.steady_state_throttle(
            v_target
        )

        return max(min(throttle, self.car.max_throttle), -1)
