from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from immrax import AdjointEmbedding, Interval, Polytope, icentpert, interval, natif
from immrax.inclusion.cubic_spline import (
    create_cubic_spline_coeffs,
    make_spline_eval_fn,
)
from immrax.system.trajectory import RawDiscreteTrajectory, RawTrajectory

from buzzracer.controllers.car_controller import CarController
from buzzracer.controllers.pid_controller import PidController
from buzzracer.controllers.stanley_car_controller import StanleyCarController
from buzzracer.extensions.simulators.immrax_dynamic_bycicle_curvilinear import (
    DynamicBicycleCurvilinear,
)
from buzzracer.extensions.simulators.kinematic_bicycle_curvilinear_simulator import (
    KinematicBicycleModelFrenet,
)
from buzzracer.types import Control, CurvilinearState

# jax.config.update("jax_debug_nans", True)
PRNG_SEED = 0


@dataclass
class SampleBounds:
    min: float
    std: float
    max: float


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
            * jnp.pi
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
        self.num_samples = 1000
        self.throttle_bounds = SampleBounds(
            min=car.min_throttle, std=car.max_throttle / 2, max=car.max_throttle
        )
        self.steering_bounds = SampleBounds(
            -car.max_steering_left, car.max_steering_right / 1, car.max_steering_right
        )

        self.planned_controls: jnp.ndarray = jnp.zeros(
            (self.planning_horizon, 2)
        )  # (steering, throttle) for each timestep over planning horizon

        self.disturbance = lambda t, x: jnp.array([0.0, 0.0])
        self.predictor = DynamicBicycleCurvilinear(car)

        self.progress_reward_weight = 0.1
        self.lateral_err_penalty_weight = 3.0
        self.discount_factor = 0.99
        self.track_width = 0.2  # FIXME: this should be read from track

        # Construct curvature spline
        ds = 0.1
        track_len = self.track.raceline_len_m
        x_knots = jnp.arange(0, track_len, ds)
        y_knots = jnp.array([car.main.track.curvature_s(x) for x in x_knots])
        knots = jnp.vstack((x_knots, y_knots)).T
        curvature_spline = make_spline_eval_fn(*create_cubic_spline_coeffs(knots))
        self.curvature = (
            lambda t, x: jnp.atleast_1d(curvature_spline(x[0] % track_len))
        )  # This needs to be 1d for compatibility with parametope embedding version of curvature

        def curvature_int(s: Interval):
            # print(s.shape)
            start = s.lower % track_len
            width = s.width
            s1 = interval(start, jnp.minimum(track_len, start + width))
            s2 = interval(
                jnp.array(0.0),
                jnp.minimum(
                    jnp.maximum(jnp.array(0.0), start + width - track_len), start
                ),
            )

            spline_int = natif(curvature_spline)
            ret = (
                spline_int(s1) | spline_int(s2)
            ).atleast_1d()  # Since ParametricEmbedding requires all *args to have a defined `len`, we need it to be a 1d array, not just a scalar
            return ret

        def iover_s(x):
            pt, aux = x
            alpha, _ = aux
            return (interval(alpha) @ interval(-pt.y[:6], pt.y[6:]) + pt.ox[0])[0]

        self.reach_predictor = AdjointEmbedding(
            self.predictor, jnp.eye(6), jnp.zeros((0, 6))
        )
        self.disturbance_int = lambda t, x: interval(jnp.array([0.0, 0.0]))
        self.curvature_int = lambda t, x: curvature_int(iover_s(x))

        # TODO: I do not expect to need these long term, should remove eventually
        self.track = car.main.track
        self.stanley_controller = StanleyCarController(car, config)

    def control(self):
        # curv_state = self.car.sim_state
        # print(f"progress: {curv_state[0]:.2f}, lateral err: {curv_state[1]:.2f}")
        throttle, steering, valid, debug_dict = self.ctrl_car(
            self.car.state, self.car.sim_state, self.track
        )
        self.debug_dict = debug_dict
        self.car.debug_dict.update(debug_dict)
        # self.print_info(
        #     "car %d, T= %4.1f, S= %4.1f (deg)"
        #     % (self.car.id, throttle, degrees(steering))
        # )
        if valid:
            self.car.throttle = throttle
            self.car.steering = steering
        else:
            self.print_warning(" car %d invalid results from ctrl_car", self.car.id)
            self.car.throttle = 0.0
            self.car.steering = 0.0
        # self.predict()
        return valid

    def ctrl_car(self, state, sim_state, track, v_override=None, reverse=False):
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
        heading = state[2]
        vf = state[3]

        if sim_state[3] > 1:
            # ref_control = self.get_reference_control_from_stanley(
            #     CartesianState(*state)
            # )
            self.planned_controls, traj, self.prng_key = self.update_planned_controls(
                sim_state, self.planned_controls, self.prng_key
            )

            # self.__compare_to_buzzracer_traj(
            #     sim_state, plot=True, cost_compare=True, use_stanley_control=True
            # )
            ### PLOTTING ###

            # DEBUG: plot sampled trajectory
            # ============================================================
            traj_cart = [
                self.track.curv_to_cart(CurvilinearState(*state)) for state in traj
            ]
            self.plot_trajectory(traj_cart, color=(0, 0, 200))
            # ============================================================

            # DEBUG: compute + plot reachable set overapproximation of sample trajectory
            # ============================================================
            # pt0 = Polytope.from_interval(
            #     icentpert(sim_state, jnp.array([0.1, 0.1, 0.1, 0.01, 0.01, 0.01]))
            # )

            # traj_reach = self.reach_predictor.compute_reachset(
            #     0,
            #     self.planning_horizon * self.planning_dt,
            #     pt0,
            #     (self.ff_control, self.disturbance_int, self.curvature_int),
            #     dt=self.planning_dt,
            # )
            # pt, aux = traj_reach.ys
            # alpha, _ = aux
            # idx = 1
            # final_state_iover = (
            #     interval(alpha[idx])
            #     @ interval(
            #         -pt.y[idx, :6],
            #         pt.y[idx, 6:],
            #     )
            #     + pt.ox[idx]
            # )
            # print(f"Bound size: {jnp.prod(final_state_iover.width):.4g}")

            # pt = pt[idx]
            # E = onp.hstack((onp.eye(2), onp.zeros((2, pt.H.shape[1] - 2))))
            # Hi = onp.vstack((-pt.H, pt.H))
            # bi = onp.hstack((-pt.ly, pt.uy))
            # frenet_vertices = project_polytope((E, onp.zeros(2)), (Hi, bi))
            # # TODO: I need the full state information to do this conversion, but can't project VREP down to 2D for plotting
            # cartesian_vertices = [self.track.curv_to_cart(v) for v in frenet_vertices]
            # plot_polygon(cartesian_vertices)

            # ============================================================
            ### END PLOTTING ###

            return (self.planned_controls[0, 1], self.planned_controls[0, 0], True, {})

        # inquire information about desired trajectory close to the vehicle
        retval = track.local_trajectory(state)
        if retval is None:
            return (0, 0, False, {"offset": 0})

        # parse return value from local_trajectory
        (local_ctrl_pnt, offset, orientation, curvature, v_target) = retval
        # for experiments
        # v_target = min(v_target*0.8, 2.2)
        v_target = min(v_target, self.max_speed)

        if jnp.isnan(orientation):
            return (0, 0, False, {"offset": 0})

        if reverse:
            offset = -offset
            orientation += jnp.pi

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
            steering = (steering + jnp.pi) % (2 * jnp.pi) - jnp.pi
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
        planned_controls = jnp.vstack([planned_controls[1:, :], jnp.zeros((1, planned_controls.shape[1]))])

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
        )

        return jnp.stack([sampled_steering, sampled_throttle], axis=-1), next_key

    def rollout_sampled_trajectory(self, x0, control_traj):
        def control_action(t, x):
            idx = jnp.floor((t + 0.5 * self.planning_dt) / self.planning_dt).astype(int) % self.planning_horizon
            # jax.debug.print("time: {:.4f} mapped to index {:d}", t, idx)
            return control_traj[idx]

        traj = self.predictor.compute_trajectory(
            0.0,  # NOTE: this is assuming the system is time-invariant
            self.planning_horizon * self.planning_dt,
            x0,
            # (control_action, self.disturbance, lambda t, x: 0),
            (control_action, self.disturbance, self.curvature),
            dt=self.planning_dt,
            solver="euler",
        )
        return traj

    def evaluate_trajectory_cost(self, state, traj):
        progress = traj.ys[self.planning_horizon - 1, 0] - state[0]
        lateral_err = traj.ys[: self.planning_horizon, 1]
        # jax.debug.print("Progress: {}, Lateral err: {}", progress, lateral_err)

        # conditional_log(
        #     jnp.any(jnp.isnan(traj._ys[0])),
        #     "NaN in planned trajectory!",
        # )
        # jax.debug.print("{}", jnp.any(jnp.isnan(traj._ys[:self.planning_horizon])))

        progress_reward = self.progress_reward_weight * progress**2
        lateral_err_penalty = jnp.sum(
            jax.vmap(lambda err: self.lateral_err_penalty_weight * (err**2))(
                lateral_err
            )
        )

        collision = jnp.max(jnp.abs(lateral_err)) > self.track_width

        # TODO: add cost term for heading error

        return jax.lax.cond(
            collision,
            lambda: jnp.inf,
            lambda: lateral_err_penalty - progress_reward,
        )

    @partial(jax.jit, static_argnums=0)
    def update_planned_controls(self, state, planned_controls, prng_key):
        # print("compiling update_planned_controls")
        sampled_controls, next_key = self.sample_controls(planned_controls, prng_key)
        trajs = jax.vmap(self.rollout_sampled_trajectory, in_axes=(None, 0))(
            jnp.array(state[0:6]), sampled_controls
        )
        costs = jax.vmap(lambda traj: self.evaluate_trajectory_cost(state, traj))(trajs)

        best_idx = jnp.argmin(costs)
        # jax.debug.print("Best cost: {}", costs[best_idx])

        # lateral_deviation = trajs._ys[best_idx, : self.planning_horizon, 1]
        # jax.debug.print(
        #     "Lateral deviation: {:.2f}", jnp.max(jnp.abs(lateral_deviation))
        # )

        return (
            sampled_controls[best_idx],
            trajs.ys[best_idx, : self.planning_horizon, :],
            next_key,
        )

    def __compare_to_buzzracer_traj(
        self,
        sim_state,
        cost_compare: bool = False,
        plot: bool = False,
        use_stanley_control=False,
    ):
        """"""
        curv_state = CurvilinearState(*sim_state)
        curv_states = [curv_state]

        for i in range(self.planning_horizon):
            singularity_proximity = (
                1
                - curv_state.lateral_err
                * self.curvature(0, jnp.array([*curv_state])).item()
            )

            steering = (self.planned_controls[i, 0],)
            throttle = (self.planned_controls[i, 1],)
            if use_stanley_control:
                throttle, steering, _, _ = self.stanley_controller.ctrl_car(
                    self.track.curv_to_cart(curv_state), self.track
                )

            curv_state = KinematicBicycleModelFrenet.advance_dynamics(
                curv_state,
                Control(
                    steering=steering,
                    throttle=throttle,
                ),
                SimpleNamespace(params=self.car.params),
                dt=self.planning_dt,
                curvature=self.curvature(0, jnp.array([*curv_state])).item(),
                # curvature=0,
            )
            curv_states.append(curv_state)

            jump_size = (
                curv_state.progress - curv_states[-2].progress
            ) / self.planning_dt
            if jnp.abs(jump_size) > 10:
                print(
                    f"!!!!!: huge jump in buzzracer ({jump_size:.2f}) -- {singularity_proximity=:.4g}"
                )

        if cost_compare:
            state = jnp.array(curv_states[0])
            curv_traj = RawDiscreteTrajectory(
                jnp.arange(
                    0, self.planning_horizon * self.planning_dt, self.planning_dt
                ),
                jnp.array(curv_states),
            )
            cost = self.evaluate_trajectory_cost(state, curv_traj)
            print(f"Buzzracer dynamics cost: {cost:.4g}")

        if plot:
            cart_traj = [
                self.track.curv_to_cart(curv_state) for curv_state in curv_states
            ]
            self.plot_trajectory(cart_traj)

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
