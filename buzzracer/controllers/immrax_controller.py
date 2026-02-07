import time
from dataclasses import dataclass
from functools import partial
from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from immrax import AdjointEmbedding, Polytope, icentpert, interval
from immrax.system.trajectory import RawTrajectory
from immrax.utils import timed

from buzzracer.controllers.car_controller import CarController
from buzzracer.controllers.stanley_car_controller import StanleyCarController
from buzzracer.extensions.simulators.immrax_dynamic_bicycle_cartesian import (
    DynamicBicycleCartesian,
)
from buzzracer.sysid.kinematic_bicycle_model import KinematicBicycleModelCartesian
from buzzracer.types import CartesianState, Control

# jax.config.update("jax_debug_nans", True)

PRNG_SEED = 0


def _get_control_index(t, planning_dt, planning_horizon):
    """Compute the control index for time t."""
    return (
        jnp.floor((t + 0.5 * planning_dt) / planning_dt).astype(int) % planning_horizon
    )


@dataclass
class SampleBounds:
    min: float
    std: float
    max: float


class ImmraxController(CarController):
    def __init__(self, car, config):
        # load config etc
        super().__init__(car, config)

        self.debug_dict = {}

        self.prng_key = jax.random.key(PRNG_SEED)

        self.planning_dt = 0.02
        self.planning_horizon = 30  # time steps
        self.num_samples = 1024
        self.throttle_bounds = SampleBounds(
            min=car.min_throttle, std=car.max_throttle / 2, max=car.max_throttle
        )
        self.steering_bounds = SampleBounds(
            -car.max_steering_left,
            car.max_steering_right / 1,
            car.max_steering_right,
        )

        self.predictor = DynamicBicycleCartesian(car)

        # Pre-allocate constant disturbance array to avoid repeated allocation
        self._zero_disturbance = jnp.array([0.0, 0.0])
        self.disturbance = lambda t, x: self._zero_disturbance

        # args: sys, alpha_p0, N0
        # N0 is null vectors
        self.reach_predictor = AdjointEmbedding(
            self.predictor, jnp.eye(6), jnp.zeros((0, 6))
        )
        # Pre-compute interval disturbance to avoid repeated allocation
        self._zero_disturbance_interval = interval(self._zero_disturbance)
        self.disturbance_int = lambda t, x: self._zero_disturbance_interval

        self.stanley_controller = StanleyCarController(car, config)
        initial = jnp.array(self.car.state[:6])  # [x, y, heading, vf, vs, omega]
        _, self.planned_controls = self.__plan_stanley_traj(
            initial
        )  # (steering, throttle) for each timestep over planning horizon

        self.track = car.main.track

        # Pre-compute constant for terminal cost
        self._terminal_cost_offset = 8.0 * self.planning_dt * self.planning_horizon

        # Pre-compute track arrays as JAX arrays to avoid repeated conversions in JIT
        self._jax_ss = jnp.array(self.track.ss)
        self._jax_raceline_velocity = jnp.array(self.track.raceline_velocity)
        self._jax_raceline_left_boundary = jnp.array(self.track.raceline_left_boundary)
        self._jax_raceline_right_boundary = jnp.array(
            self.track.raceline_right_boundary
        )
        # Raceline arrays for Cartesian-to-Frenet projection
        self._jax_raceline_x = jnp.array(self.track.raceline_points[0])  # (N,)
        self._jax_raceline_y = jnp.array(self.track.raceline_points[1])  # (N,)
        self._jax_raceline_headings = jnp.array(self.track.raceline_headings)  # (N,)

        # Visualization settings
        self.enable_trajectory_visualization = False  # Set True only for debugging
        self._viz_skip_count = 10  # Only visualize every N updates
        self._viz_counter = 0

        # Timing instrumentation
        self._timing_log = []
        self._last_update_time_ms = 0.0

        # JIT warmup
        self.print_info("Warming up JIT compilation...")
        _t0 = time.perf_counter()
        pt0 = Polytope.from_interval(interval(initial))
        _ = self.update_planned_controls(initial, self.planned_controls, self.prng_key)
        __ = self.rollout_reachset(pt0, self.planned_controls)
        jax.block_until_ready(_[0])
        jax.block_until_ready(__)
        _t1 = time.perf_counter()
        self.print_info(f"JIT compilation time: {(_t1 - _t0) * 1000:.1f}ms")

    def control(self):
        throttle, steering, valid, debug_dict = self.ctrl_car(
            self.car.state, self.car.sim_state, self.track
        )
        # Timing log
        self._timing_log.append(self._last_update_time_ms)
        if len(self._timing_log) % 100 == 0:
            self.print_info(
                f"Control loop: mean={np.mean(self._timing_log[-100:]):.2f}ms, std={np.std(self._timing_log[-100:]):.2f}ms"
            )
        self.debug_dict = debug_dict
        self.car.debug_dict.update(debug_dict)
        if valid:
            self.car.throttle = throttle
            self.car.steering = steering
        else:
            self.print_warning(" car %d invalid results from ctrl_car", self.car.id)
            self.car.throttle = 0.0
            self.car.steering = 0.0
        return valid

    def ctrl_car(
        self, state, sim_state: CartesianState, track, v_override=None, reverse=False
    ):
        """
        given state of the vehicle and an instance of track, provide throttle and steering output
        input:
          state: (x,y,heading,v_forward,v_sideway,omega)
          track: track object, can be RCPTrack or skidpad
          v_override: If specified, use this as target velocity instead of the optimal value provided by track object
          reverse: true if running in opposite direction of raceline init direction

        output:
          (throttle,steering,valid,debug)
        ranges for output:
          throttle -1.0,self.max_throttle
          steering as an angle in radians, TRIMMED to self.max_steering, left(+), right(-)
          valid: bool, if the car can be controlled here, if this is false, then throttle will also be set to 0
                  This typically happens when vehicle is off track, and track object cannot find a reasonable local raceline
        debug: a dictionary of objects to be debugged, e.g. {offset, error in v}
        """

        res, controller_compute_time = self.update_planned_controls(
            jnp.asarray(state), self.planned_controls, self.prng_key
        )
        self.planned_controls, traj, self.prng_key = res
        self._last_update_time_ms = controller_compute_time * 1000

        # Reachability computation
        pt0 = Polytope.from_interval(
            icentpert(
                jnp.array(state[:6]), jnp.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
            )
        )
        traj_reach, reachset_compute_time = self.rollout_reachset(
            pt0, self.planned_controls
        )

        return (self.planned_controls[0, 1], self.planned_controls[0, 0], True, {})

    def sample_controls(self, planned_controls: jax.Array, prng_key):
        steering_key, throttle_key, next_key = jax.random.split(prng_key, 3)
        # Shift controls: roll up by 1 and zero-out the last row
        shifted_controls = jnp.roll(planned_controls, -1, axis=0).at[-1, :].set(0.0)

        # Generate noise for all samples at once
        noise_shape = (self.num_samples, self.planning_horizon)
        steering_noise = self.steering_bounds.std * jax.random.normal(
            steering_key, shape=noise_shape
        )
        throttle_noise = self.throttle_bounds.std * jax.random.normal(
            throttle_key, shape=noise_shape
        )

        # Add noise to shifted controls and clip
        sampled_steering = jnp.clip(
            shifted_controls[:, 0] + steering_noise,
            self.steering_bounds.min,
            self.steering_bounds.max,
        )
        sampled_throttle = jnp.clip(
            shifted_controls[:, 1] + throttle_noise,
            self.throttle_bounds.min,
            self.throttle_bounds.max,
        )

        return jnp.stack([sampled_steering, sampled_throttle], axis=-1), next_key

    def rollout_sampled_trajectory(
        self, x0: jax.Array, control_traj: jax.Array
    ) -> RawTrajectory:
        # Local closure capturing control_traj. This works inside JIT because
        # control_traj is a traced array and the closure structure is stable.
        def control_action(t, x):
            idx = _get_control_index(t, self.planning_dt, self.planning_horizon)
            return control_traj[idx]

        traj = self.predictor.compute_trajectory(
            0.0,  # NOTE: this is assuming the system is time-invariant
            self.planning_horizon * self.planning_dt,
            x0,
            (control_action, self.disturbance),  # Cartesian model: no curvature
            dt=self.planning_dt,
            solver="euler",
        )
        return traj

    @partial(jax.jit, static_argnums=0)
    @timed
    def rollout_reachset(self, pt0, planned_controls):
        """Compute reachable set with planned_controls as a traced argument.

        Mirrors rollout_sampled_trajectory: the closure captures planned_controls
        as a traced JAX array so it stays dynamic under JIT, rather than being
        baked in as a compile-time constant. The outer JIT here makes the inner
        compute_reachset JIT a no-op, so the static_argnums on inputs is
        irrelevant and the closure correctly captures a tracer.
        """

        def ff_control(t, x):
            idx = _get_control_index(t, self.planning_dt, self.planning_horizon)
            return interval(planned_controls[idx])

        return self.reach_predictor.compute_reachset(
            0,
            self.planning_horizon * self.planning_dt,
            pt0,
            (ff_control, self.disturbance_int),
            dt=self.planning_dt,
        )

    def evaluate_trajectory_cost(self, state: jax.Array, traj: RawTrajectory):
        return self.__evaluate_trajectory_cost_cartesian(state, traj)

    def __evaluate_trajectory_cost_cartesian(
        self, state: jax.Array, traj: RawTrajectory
    ):
        """Evaluate trajectory cost in Cartesian frame."""
        traj_slice = traj.ys[: self.planning_horizon]

        # Extract Cartesian state components
        xy_vals = traj_slice[:, :2]  # (x, y) positions
        vel = traj_slice[:, 3]  # forward velocity

        # Project to Frenet for cost computation
        progress_vals, lateral_err = self._project_to_frenet_batch(xy_vals)

        # Map progress to track indices
        index = jnp.searchsorted(self._jax_ss, progress_vals % self._jax_ss[-1])
        ref_vel = self._jax_raceline_velocity[index]
        bounds_left = self._jax_raceline_left_boundary[index]
        bounds_right = self._jax_raceline_right_boundary[index]

        # Step cost: penalize lateral error and velocity deviation
        vel_diff_sq = (vel - ref_vel) ** 2
        step_cost = jnp.sum(
            5.0 * jnp.abs(lateral_err)
            + 2.5 * vel_diff_sq
            - 10.0 * jnp.minimum(vel, 0.0)
        )

        # Speed cost: penalize low speeds
        speed_cost = 10.0 * jnp.sum(vel <= 1.5)

        # Boundary cost: soft penalty near boundaries
        abs_lateral_err = jnp.abs(lateral_err)
        bound = jnp.where(lateral_err > 0, bounds_left, bounds_right)
        boundary_arg = -100.0 * (bound - (abs_lateral_err + 0.05))
        boundary_cost = 10.0 * jnp.sum(
            jnp.maximum(0.0, (2.0 / jnp.pi) * jnp.arctan(boundary_arg) + 1.0)
        )

        # Collision cost: hard penalty for boundary violation
        collision = jnp.logical_or(
            lateral_err > bounds_left, lateral_err < -bounds_right
        )
        collision_cost = jnp.where(jnp.any(collision), jnp.inf, 0.0)

        # Terminal cost: compute progress made
        initial_xy = jnp.array(state[:2])
        initial_progress, _ = self._project_to_frenet_batch(initial_xy[None, :])
        progress_made = progress_vals[-1] - initial_progress[0]
        # Handle wrap-around in both directions
        track_len = self.track.raceline_len_m
        progress_made = jnp.where(
            progress_made < -track_len / 2,
            progress_made + track_len,  # Forward lap crossing
            jnp.where(
                progress_made > track_len / 2,
                progress_made - track_len,  # Backward lap crossing
                progress_made,
            ),
        )
        terminal_cost = self._terminal_cost_offset - 2.0 * progress_made

        return step_cost + speed_cost + boundary_cost + collision_cost + terminal_cost

    @partial(jax.jit, static_argnums=0)
    @timed
    def update_planned_controls(self, state, planned_controls, prng_key):
        sampled_controls, next_key = self.sample_controls(planned_controls, prng_key)
        trajs = jax.vmap(self.rollout_sampled_trajectory, in_axes=(None, 0))(
            jnp.array(state[0:6]), sampled_controls
        )
        costs = jax.vmap(lambda traj: self.evaluate_trajectory_cost(state, traj))(trajs)

        best_idx = jnp.argmin(costs)

        return (
            sampled_controls[best_idx],
            trajs.ys[best_idx, : self.planning_horizon, :],
            next_key,
        )

    def _project_to_frenet_batch(
        self, xy_vals: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Project Cartesian (x, y) points onto raceline.

        Args:
            xy_vals: (N, 2) array of [x, y] positions

        Returns:
            progress: (N,) progress values along raceline
            lateral_err: (N,) signed lateral errors (positive = left of raceline)
        """
        # Compute squared distances to all raceline points
        # xy_vals: (N, 2), raceline: (M, 2) -> distances: (N, M)
        dx = xy_vals[:, 0:1] - self._jax_raceline_x[None, :]  # (N, M)
        dy = xy_vals[:, 1:2] - self._jax_raceline_y[None, :]  # (N, M)
        dists_sq = dx**2 + dy**2

        # Find closest raceline point index for each trajectory point
        closest_idx = jnp.argmin(dists_sq, axis=1)  # (N,)

        # Get progress at closest points
        progress = self._jax_ss[closest_idx]  # (N,)

        # Compute signed lateral error using cross product with tangent
        raceline_heading = self._jax_raceline_headings[closest_idx]  # (N,)
        dx_closest = xy_vals[:, 0] - self._jax_raceline_x[closest_idx]
        dy_closest = xy_vals[:, 1] - self._jax_raceline_y[closest_idx]

        # Dot product with left-pointing normal: D · N where N = (-sin(h), cos(h))
        # Positive when point is to the LEFT of raceline direction
        lateral_err = -dx_closest * jnp.sin(raceline_heading) + dy_closest * jnp.cos(
            raceline_heading
        )

        return progress, lateral_err

    def __plan_stanley_traj(
        self, sim_state: jax.Array
    ) -> Tuple[List[CartesianState], jax.Array]:
        """Plan a trajectory using the Stanley controller for warm-starting the sampler.

        Args:
            sim_state: Initial Cartesian state [x, y, heading, vf, vs, omega]

        Returns:
            cart_states: List of CartesianState along the trajectory
            control_traj: Array of shape (planning_horizon, 2) with [steering, throttle]
        """
        cart_state = CartesianState(*sim_state)
        cart_states = [cart_state]
        control_traj = jnp.zeros((self.planning_horizon, 2))

        for i in range(self.planning_horizon):
            throttle, steering, _, _ = self.stanley_controller.ctrl_car(
                cart_state, self.track
            )
            control_traj = control_traj.at[i, :].set(jnp.array([steering, throttle]))

            cart_state = KinematicBicycleModelCartesian.advance_dynamics(
                cart_state,
                Control(steering=steering, throttle=throttle),
                self.car,
                dt=self.planning_dt,
            )
            cart_states.append(cart_state)

        return cart_states, control_traj
