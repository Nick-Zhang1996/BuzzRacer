from math import isnan, pi, degrees, radians
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from buzzracer.extensions.simulators.immrax_dynamic_bycicle_curvilinear import (
    DynamicBicycleCurvilinear,
)
from buzzracer.types import CartesianState, CurvilinearState, Control
from buzzracer.controllers.car_controller import CarController
from buzzracer.controllers.pid_controller import PidController
from buzzracer.controllers.stanley_car_controller import StanleyCarController

from buzzracer.sysid.kinematic_bicycle_model import (
    KinematicBicycleModelFrenet,
    KinematicBicycleModelCartesian
)

# jax.config.update("jax_debug_nans", True)
PRNG_SEED = 0


@dataclass
class SampleBounds:
    min: float
    std: float
    max: float


# track class is not jittable, so we precompute curvature along raceline
# store in buffer for jittability (pure function of s)
class CurvatureLib:
    def __init__(self, track):
        self.ds = 0.01
        self.len = track.raceline_len_m
        self.buffer = jnp.zeros(int(self.len / self.ds) + 1)

        for s in jnp.arange(0, self.len, self.ds):
            self.buffer = self.buffer.at[int(s / self.ds)].set(track.curvature_s(s))

    def __call__(self, s):
        s = s % self.len
        idx = (s / self.ds).astype(int)
        # jax.debug.print("curvature lookup s={:.2f}, idx={}", s, idx)
        return self.buffer[(idx)]


class TempCar():
    def __init__(self, lr, lf):
        self.lr = lr
        self.lf = lf


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

        self.prng_key = jax.random.key(PRNG_SEED)

        self.planning_dt = 0.02
        self.planning_horizon = 50  # time steps
        self.num_samples = 100
        self.throttle_bounds = SampleBounds(
            min=car.min_throttle, std=car.max_throttle / 4, max=car.max_throttle
        )
        self.steering_bounds = SampleBounds(
            -car.max_steering_left, car.max_steering_right / 4, car.max_steering_right
        )

        self.planned_controls: jnp.ndarray = jnp.zeros(
            (self.planning_horizon, 2)
        )  # (steering, throttle)

        self.disturbance = lambda t, x: jnp.array([0.0, 0.0])
        self.curvature_lib = CurvatureLib(car.main.track)
        self.curvature = lambda t, x: self.curvature_lib(x[0])
        self.predictor = DynamicBicycleCurvilinear(car)

        self.progress_reward_weight = 1.0
        self.lateral_err_penalty_weight = 1.0
        self.track_width = 0.4  # FIXME: this should be read from track

        # TODO: I do not expect to need these long term, should remove eventually
        self.track = car.main.track
        self.car_ref = TempCar(car.lr, car.lf)
        self.stanley_controller = StanleyCarController(car, config)

    def control(self):
        curv_state = self.track.cart_to_curv(CartesianState(*self.car.state))
        # print(f"progress: {curv_state[0]:.2f}, lateral err: {curv_state[1]:.2f}")
        throttle, steering, valid, debug_dict = self.ctrl_car(
            self.car.state, self.car.sim_state, self.track
        )
        self.debug_dict = debug_dict
        self.car.debug_dict.update(debug_dict)
        self.print_info("car %d, T= %4.1f, S= %4.1f (deg)" %
                        (self.car.id, throttle, degrees(steering)))
        if valid:
            self.car.throttle = throttle
            self.car.steering = steering
        else:
            self.print_warning(" car %d invalid results from ctrl_car", self.car.id)
            self.car.throttle = 0.0
            self.car.steering = 0.0
        # self.predict()
        return valid

    def get_reference_control_from_stanley(self, state: CartesianState):
        ''' Generate an initial guess for ref control from stanley controller'''
        control_vec: list[Control] = []
        state_vec = [state]
        for _ in range(self.planning_horizon):
            throttle, steering, _, _ = self.stanley_controller.ctrl_car(
                state_vec[-1], self.track)
            control = Control(steering=steering, throttle=throttle)
            next_state = KinematicBicycleModelCartesian.advance_dynamics(
                state_vec[-1], control, self.car, self.planning_dt)
            control_vec.append(control)
            state_vec.append(next_state)

        ref_control_np = np.array([(u.steering, u.throttle) for u in control_vec])
        ref_control: jnp.ndarray = jnp.array(
            ref_control_np
        )  # (steering, throttle)
        return ref_control

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

    def ctrl_car(self, state, sim_state, track, v_override=None, reverse=False):
        heading = state[2]
        vf = state[3]

        if sim_state[3] > 1:
            ref_control = self.get_reference_control_from_stanley(CartesianState(*state))
            self.planned_controls, traj, self.prng_key = self.update_planned_controls(
                sim_state, ref_control, self.prng_key
            )

            # return (self.planned_controls[0, 1], self.planned_controls[0, 0], True, {})

            ### PLOTTING ###
            # dsdts = jnp.diff(traj[:, 0])
            # max_jump_idx = jnp.argmax(dsdts)
            # max_jump = dsdts[max_jump_idx]
            # if max_jump > 5:
            #     print(f"ERROR: huge jump in immrax ({max_jump})")
            #
            #     x = traj[max_jump_idx]
            #     u = self.planned_controls[max_jump_idx]
            #     dx = self.predictor.f(0, x, u, jnp.array([0, 0]), self.curvature(0, x))
            #
            #     curv_state = CurvilinearState(*x)
            #     curv_state = KinematicBicycleModelFrenet.advance_dynamics(
            #         curv_state,
            #         Control(u[1], u[0]),
            #         self.car,
            #         dt=self.planning_dt,
            #         curvature=self.curvature(0, x),
            #     )
            #     xp1 = jnp.array([*curv_state])
            #     buzz_dx = (xp1 - x) / self.planning_dt

            # print(f"Computed dynamics:\nmy dx={dx},\nbuzz_dx={buzz_dx}")

            curv_state = CurvilinearState(*sim_state)
            curv_traj = [curv_state]

            for i in range(self.planning_horizon):
                curv_state = KinematicBicycleModelFrenet.advance_dynamics(
                    curv_state,
                    Control(
                        steering=self.planned_controls[i, 0],
                        throttle=self.planned_controls[i, 1],
                    ),
                    self.car_ref,
                    dt=self.planning_dt,
                    curvature=self.curvature(0, jnp.array([*curv_state])),
                    # curvature=0,
                )
                curv_traj.append(curv_state)

                jump_size = (
                    curv_state.progress - curv_traj[-2].progress
                ) / self.planning_dt
                if abs(jump_size) > 10:
                    print(f"ERROR: huge jump in buzzracer ({jump_size})")

                # # compare to my dynamics
                # progress, lateral_err, heading_err, v_forward, v_sideway, rel_omega = curv_state
                # steering, throttle = self.planned_controls[i]
                # dsdt = (
                #     v_forward * jnp.cos(heading_err) - v_sideway * jnp.sin(heading_err)
                # ) / (1 - lateral_err * self.curvature(0, sim_state))

            cart_traj = [
                self.track.curv_to_cart(curv_state) for curv_state in curv_traj
            ]
            self.plot_trajectory(cart_traj)

            # DEBUG: Simulate control sequence in cartesian-based dynamics
            init_curv_state = CurvilinearState(*sim_state)
            init_cart_state = self.track.curv_to_cart(init_curv_state)
            alt_cart_traj = [init_cart_state]
            for i in range(self.planning_horizon):
                cart_state = KinematicBicycleModelCartesian.advance_dynamics(
                    alt_cart_traj[-1],
                    Control(
                        steering=self.planned_controls[i, 0],
                        throttle=self.planned_controls[i, 1],
                    ),
                    self.car_ref,
                    dt=self.planning_dt,
                )
                alt_cart_traj.append(cart_state)
            self.plot_trajectory(alt_cart_traj, color=(50, 50, 50))

            # for i in range(self.planning_horizon):
            #     print(jnp.array([*cart_traj[i]]) - traj[i])

            # TODO: compare cart_traj[0] to sim_state to traj[0]
            # Before and after curv to cart conversion?
            # print(
            #     f"buzzracer traj:\t{jnp.array([*curv_traj[0]])},\nimmrax traj:\t{traj[0]},\nsim_state:\t{jnp.array([*sim_state])}"
            # )
            # print()

            # print(f"{self.curvature(0, sim_state)}*{state[1]}={self.curvature(0, sim_state)*state[1]}")
            # control_action = lambda t, x: self.planned_controls[
            #     jnp.floor(t / self.planning_dt).astype(int) % self.planning_horizon
            # ]
            #
            # traj = self.predictor.compute_trajectory(
            #     0.0,
            #     self.planning_horizon * self.planning_dt,
            #     jnp.array([*sim_state]),
            #     (control_action, self.disturbance, lambda t, x: 0),
            #     # (control_action, self.disturbance, self.curvature),
            #     dt=self.planning_dt,
            #     solver="euler",
            # )  # NOTE: this is assuming the system is time-invariant

            cart_traj = [
                self.track.curv_to_cart(CurvilinearState(*curv_state))
                for curv_state in traj
                # for curv_state in traj.ys
            ]
            # self.plot_trajectory(cart_traj, color=(0, 0, 255))
            ### END PLOTTING ###

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

    def sample_controls(self, planned_controls: jax.Array, prng_key):
        steering_key, throttle_key, next_key = jax.random.split(prng_key, 3)
        sampled_steering = jnp.clip(
            planned_controls[:, 0]
            + self.steering_bounds.std
            * jax.random.normal(
                steering_key,
                shape=(self.num_samples, self.planning_horizon),
            ),
            self.steering_bounds.min,
            self.steering_bounds.max,
        )
        sampled_throttle = jnp.clip(
            planned_controls[:, 1]
            + self.throttle_bounds.std
            * jax.random.normal(
                throttle_key,
                shape=(self.num_samples, self.planning_horizon),
            ),
            self.throttle_bounds.min,
            self.throttle_bounds.max,
        )  # TODO: may want to consider steady_state_throttle explicitly

        return jnp.stack([sampled_steering, sampled_throttle], axis=-1), next_key

    def rollout_sampled_trajectory(self, x0, control_traj):
        def control_action(t, x):
            return control_traj[
                jnp.floor(t / self.planning_dt).astype(int) % self.planning_horizon
            ]

        traj = self.predictor.compute_trajectory(
            0.0,
            self.planning_horizon * self.planning_dt,
            x0,
            (control_action, self.disturbance, lambda t, x: 0),
            # (control_action, self.disturbance, self.curvature),
            dt=self.planning_dt,
            solver="euler",
        )  # NOTE: this is assuming the system is time-invariant
        return traj

    def evaluate_trajectory_cost(self, state, traj):
        # NOTE: can't use traj.ys here since we are inside jitted code - should consider rework
        progress = traj._ys[self.planning_horizon - 1, 0] - state[0]
        lateral_err = traj._ys[: self.planning_horizon, 1]

        # conditional_log(
        #     jnp.any(jnp.isnan(traj._ys[0])),
        #     "NaN in planned trajectory!",
        # )
        # jax.debug.print("{}", jnp.any(jnp.isnan(traj._ys[:self.planning_horizon])))

        progress_reward = self.progress_reward_weight * progress**2
        lateral_err_penalty = jnp.sum(
            jax.vmap(lambda err: self.lateral_err_penalty_weight * err**2)(lateral_err)
        )
        collision = jnp.max(jnp.abs(lateral_err)) > self.track_width

        return jax.lax.cond(
            collision,
            lambda: jnp.inf,
            lambda: lateral_err_penalty - progress_reward,
        )

    @partial(jax.jit, static_argnums=0)
    def update_planned_controls(self, state, planned_controls, prng_key):
        # print("compiling plan_control_trajectory")
        sampled_controls, next_key = self.sample_controls(planned_controls, prng_key)
        trajs = jax.vmap(self.rollout_sampled_trajectory, in_axes=(None, 0))(
            jnp.array(state[0:6]), sampled_controls
        )
        costs = jax.vmap(lambda traj: self.evaluate_trajectory_cost(state, traj))(trajs)

        best_idx = jnp.argmin(costs)

        return (
            sampled_controls[best_idx],
            trajs._ys[best_idx, : self.planning_horizon, :],
            next_key,
        )

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
