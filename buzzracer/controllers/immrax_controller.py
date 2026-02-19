import time
from dataclasses import dataclass
from functools import partial
from typing import List, Tuple

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from immrax import (
    AdjointEmbedding,
    Polytope,
    RawDiscreteTrajectory,
    interval,
)
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
        # Allow transfers during initialization (reset any previous guard state)
        jax.config.update("jax_transfer_guard", "allow")

        # load config etc
        super().__init__(car, config)
        self.print_debug_enable()

        self.debug_dict = {}

        # Sampler settings
        # ============================================================
        self.prng_key = jax.random.key(jax.device_put(PRNG_SEED))

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

        # Trajectory rollout settings
        # ============================================================
        self.predictor = DynamicBicycleCartesian(car)

        # Pre-allocate constant disturbance array to avoid repeated allocation
        self._zero_disturbance = jax.device_put(np.array([0.0, 0.0]))
        self.disturbance = lambda t, x: self._zero_disturbance

        # args: sys, alpha_p0, N0
        # N0 is null vectors
        self.reach_predictor = AdjointEmbedding(
            self.predictor, jax.device_put(np.eye(6)), jax.device_put(np.zeros((0, 6)))
        )
        # Pre-compute interval disturbance to avoid repeated allocation
        self._zero_disturbance_interval = interval(self._zero_disturbance)
        self.disturbance_int = lambda t, x: self._zero_disturbance_interval

        self.stanley_controller = StanleyCarController(car, config)
        initial = jax.device_put(
            np.array(self.car.state[:6])
        )  # [x, y, heading, vf, vs, omega]
        _, self.planned_controls = self.__plan_stanley_traj(
            initial
        )  # (steering, throttle) for each timestep over planning horizon

        # Reachable set data (populated each control step)
        self.reach_intervals = []
        self.reach_center_trajectory = jax.device_put(np.empty((0, 2)))

        self.track = car.main.track

        # Cost computation settings
        # ============================================================
        # Pre-compute constant for terminal cost
        self._terminal_cost_offset = 8.0 * self.planning_dt * self.planning_horizon

        # Pre-compute track arrays with explicit device transfer
        self._jax_ss = jax.device_put(np.asarray(self.track.ss))
        self._jax_raceline_velocity = jax.device_put(
            np.asarray(self.track.raceline_velocity)
        )
        self._jax_raceline_left_boundary = jax.device_put(
            np.asarray(self.track.raceline_left_boundary)
        )
        self._jax_raceline_right_boundary = jax.device_put(
            np.asarray(self.track.raceline_right_boundary)
        )
        # Raceline arrays for Cartesian-to-Frenet projection
        self._jax_raceline_x = jax.device_put(
            np.asarray(self.track.raceline_points[0])
        )  # (N,)
        self._jax_raceline_y = jax.device_put(
            np.asarray(self.track.raceline_points[1])
        )  # (N,)
        self._jax_raceline_headings = jax.device_put(
            np.asarray(self.track.raceline_headings)
        )  # (N,)

        # Pre-compute constants for compute_trajectory/compute_reachset to avoid runtime transfers
        self._t_start = jax.device_put(np.array(0.0))
        self._t_end = jax.device_put(np.array(self.planning_horizon * self.planning_dt))
        self._planning_dt_jax = jax.device_put(np.array(self.planning_dt))

        self._state_eye = jax.device_put(np.eye(6))
        self._state_pert = jax.device_put(
            np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        )
        self._state_pert = jnp.concatenate((self._state_pert, self._state_pert))

        # Visualization settings
        # ============================================================
        self._viz_skip_count = 1  # Only visualize every N updates
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

        # Disallow implicit host-to-device transfers after initialization
        jax.config.update("jax_transfer_guard", "disallow")

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

        sim_state_jax = jax.device_put(np.asarray(state))
        res, controller_compute_time = self.update_planned_controls(
            sim_state_jax, self.planned_controls, self.prng_key
        )
        self.planned_controls, traj, self.prng_key = res
        self._last_update_time_ms = controller_compute_time * 1000

        self.__plot_planned_trajectory(traj)
        # self.__debug_compare_costs(state, traj)
        # self.__plot_track_bounds(sim_state)

        pt0 = Polytope(sim_state_jax, self._state_eye, self._state_pert)
        traj_reach, reach_time = self.rollout_reachset(pt0, self.planned_controls)
        # (
        #     overflow_timestep,
        #     num_timesteps,
        #     valid_intervals,
        #     center_points,
        #     final_widths,
        # ) = self.__analyze_reachset_validity(traj_reach)
        # self.__update_reachset_visualization(valid_intervals, center_points)
        # self.__log_reachability_results(
        #     overflow_timestep, num_timesteps, reach_time_ms, final_widths
        # )

        # Necessary transfer to interface w/ Buzzracer sim
        controls = jax.device_get(self.planned_controls)
        result = (
            float(controls[0, 1]),
            float(controls[0, 0]),
            True,
            {},
        )

        return result

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
            self._t_start,  # NOTE: this is assuming the system is time-invariant
            self._t_end,
            x0,
            (control_action, self.disturbance),  # Cartesian model: no curvature
            dt=self._planning_dt_jax,
            solver="euler",
        )
        return traj

    @timed
    @partial(jax.jit, static_argnums=0)
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
            self._t_start,
            self._t_end,
            pt0,
            (ff_control, self.disturbance_int),
            dt=self._planning_dt_jax,
        )

    def evaluate_trajectory_cost(self, state: jax.Array, traj: RawTrajectory):
        return self.__evaluate_trajectory_cost_cartesian(state, traj)

    def __compute_cost_components(self, state: jax.Array, traj: RawTrajectory):
        """Compute individual cost components as a dict of scalars.

        Returns a dict with keys: step_cost, boundary_cost,
        collision_cost, terminal_cost.
        """
        traj_slice = traj.ys[: self.planning_horizon]

        # Extract Cartesian state components
        xy_vals = traj_slice[:, :2]  # (x, y) positions
        vel = traj_slice[:, 3]  # forward velocity

        # Project to Frenet for cost computation
        progress_vals, lateral_err = self._project_to_frenet_batch(xy_vals)
        abs_lateral_err = jnp.abs(lateral_err)

        # Map progress to track indices
        index = jnp.searchsorted(self._jax_ss, progress_vals % self._jax_ss[-1])
        ref_vel = self._jax_raceline_velocity[index]
        bounds_left = self._jax_raceline_left_boundary[index]
        bounds_right = self._jax_raceline_right_boundary[index]

        # Step cost: penalize lateral error and velocity deviation
        vel_diff_sq = (vel - ref_vel) ** 2
        step_cost = jnp.sum(
            5.0 * abs_lateral_err + 2.5 * vel_diff_sq - 10.0 * jnp.minimum(vel, 0.0)
        )

        # Speed cost: penalize low speeds
        speed_cost = 10.0 * jnp.sum(vel <= 1.5)

        # Boundary cost: soft penalty near boundaries
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

        return {
            "step_cost": step_cost,
            "speed_cost": speed_cost,
            "boundary_cost": boundary_cost,
            "collision_cost": collision_cost,
            "terminal_cost": terminal_cost,
        }

    def __evaluate_trajectory_cost_cartesian(
        self, state: jax.Array, traj: RawTrajectory
    ):
        components = self.__compute_cost_components(state, traj)
        return sum(components.values())

    @timed
    @partial(jax.jit, static_argnums=0)
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

    # ============================================================
    # Debug functions
    # ============================================================

    def __analyze_reachset_validity(self, traj_reach):
        """Analyze reachable set for overflow and track boundary violations.

        Returns:
            overflow_timestep: Index of the first overflowed timestep (or num_timesteps if none).
            num_timesteps: Total number of timesteps in the reachable set.
            valid_intervals: List of valid state intervals for visualization.
            center_points: List of (x, y) center points for valid timesteps.
            final_widths: State widths at last valid timestep, or None if none valid.
        """
        pt, aux = traj_reach.ys
        alpha, _ = aux
        num_timesteps = pt.y.shape[0]

        overflow_timestep = num_timesteps
        final_widths = None
        valid_intervals = []
        center_points = []
        for idx in range(num_timesteps):
            state_iover = (
                interval(alpha[idx]) @ interval(-pt.y[idx, :6], pt.y[idx, 6:])
                + pt.ox[idx]
            )
            widths = state_iover.width

            # Check for overflow (NaN or Inf in widths, or width > 1e6)
            if (
                jnp.any(jnp.isnan(widths))
                or jnp.any(jnp.isinf(widths))
                or jnp.any(widths > 1e6)
            ):
                overflow_timestep = idx
                break

            self.__check_reachset_track_collision(idx, pt.ox[idx, :2])

            valid_intervals.append(state_iover)
            center_points.append(pt.ox[idx, :2])
            final_widths = widths

        return (
            overflow_timestep,
            num_timesteps,
            valid_intervals,
            center_points,
            final_widths,
        )

    def __check_reachset_track_collision(self, timestep_idx, center_xy_1d):
        """Check if a reachable set center point is inside track boundaries and warn if not."""
        center_xy = center_xy_1d[None, :]  # (1, 2)
        center_progress, center_lateral_err = self._project_to_frenet_batch(center_xy)
        bound_idx = jnp.searchsorted(self._jax_ss, center_progress % self._jax_ss[-1])
        left_bound = self._jax_raceline_left_boundary[bound_idx[0]]
        right_bound = self._jax_raceline_right_boundary[bound_idx[0]]

        center_collision = (center_lateral_err[0] > left_bound) | (
            center_lateral_err[0] < -right_bound
        )
        if center_collision:
            self.print_warning(
                f"Reachable set center exits track at timestep {timestep_idx}! "
                f"lateral_err={float(center_lateral_err[0]):.4f}, "
                f"bounds=[-{float(right_bound):.4f}, +{float(left_bound):.4f}]"
            )

    def __update_reachset_visualization(self, valid_intervals, center_points):
        """Store reachable set data and draw to current visualization context."""
        self.reach_intervals = valid_intervals
        self.reach_center_trajectory = (
            jnp.stack(center_points) if center_points else jnp.empty((0, 2))
        )
        self.draw_reachset()

    def draw_reachset(self):
        """Draw the most recent reachable set projection onto (x, y) on visualization image."""
        if not self.car.main.visualization.update_visualization.is_set():
            return
        img = self.car.main.visualization.visualization_img
        track = self.car.main.track

        # Draw interval boxes as rectangles
        for iover in self.reach_intervals:
            # Extract x, y bounds from the interval
            x_lo = float(iover[0].lower)
            x_hi = float(iover[0].upper)
            y_lo = float(iover[1].lower)
            y_hi = float(iover[1].upper)

            # Convert world coordinates to canvas coordinates
            pt1 = track.m2canvas((x_lo, y_hi))  # top-left in world = top-left on canvas
            pt2 = track.m2canvas(
                (x_hi, y_lo)
            )  # bottom-right in world = bottom-right on canvas

            # Draw filled rectangle with transparency (blue)
            overlay = img.copy()
            cv2.rectangle(overlay, pt1, pt2, (255, 100, 100), -1)  # BGR blue
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

            # Draw rectangle outline
            cv2.rectangle(img, pt1, pt2, (255, 100, 100), 1)

        # Draw center trajectory as small circles
        if self.reach_center_trajectory.shape[0] > 0:
            for i in range(self.reach_center_trajectory.shape[0]):
                coord = (
                    float(self.reach_center_trajectory[i, 0]),
                    float(self.reach_center_trajectory[i, 1]),
                )
                img = track.draw_circle(img, coord, 0.015, color=(0, 0, 200))  # BGR red

        self.car.main.visualization.visualization_img = img

    def __log_reachability_results(
        self, overflow_timestep, num_timesteps, reach_time, final_widths
    ):
        """Log reachability analysis results including valid steps and bound widths."""
        state_names = ["x", "y", "heading", "v_forward", "v_sideway", "omega"]
        reach_time_ms = reach_time * 1000
        self.print_info(
            f"Reachability: {overflow_timestep}/{num_timesteps} steps valid, "
            f"compute time: {reach_time_ms:.2f}ms"
        )

        if final_widths is not None:
            # Compute relative widths (normalized by smallest non-zero width)
            min_width = jnp.min(final_widths[final_widths > 1e-10])
            relative_widths = final_widths / min_width
            width_strs = [
                f"{state_names[i]}={final_widths[i]:.4g} ({relative_widths[i]:.1f}x)"
                for i in range(6)
            ]
            self.print_info(f"Final valid bound widths: {', '.join(width_strs)}")
        elif overflow_timestep == 0:
            self.print_warning("Reachable set overflowed at first timestep!")

    def __evaluate_trajectory_cost_breakdown(
        self, state: jax.Array, traj: RawTrajectory
    ):
        """Evaluate trajectory cost with component breakdown for debugging."""
        components = self.__compute_cost_components(state, traj)
        components["total"] = sum(components.values())
        return components

    def __debug_compare_costs(self, sim_state: jax.Array, best_sampled_traj: jax.Array):
        """Compare cost breakdowns between Stanley reference and best sampled trajectory."""
        state_arr = jnp.array(sim_state[:6])

        # Get Stanley trajectory
        _, stanley_controls = self.__plan_stanley_traj(state_arr)
        stanley_traj = self.rollout_sampled_trajectory(state_arr, stanley_controls)

        # Create RawTrajectory from best sampled traj for cost evaluation
        class FakeTraj:
            def __init__(self, ys):
                self.ys = ys

        best_traj_obj = FakeTraj(best_sampled_traj)

        # Get cost breakdowns
        stanley_costs = self.__compute_cost_components(state_arr, stanley_traj)
        sampled_costs = self.__compute_cost_components(state_arr, best_traj_obj)

        stanley_total = sum(stanley_costs.values())
        sampled_total = sum(sampled_costs.values())

        # Print comparison
        self.print_info("=" * 60)
        self.print_info("COST COMPARISON: Stanley vs Best Sampled")
        self.print_info("-" * 60)
        self.print_info(
            f"{'Component':<20} {'Stanley':>15} {'Sampled':>15} {'Diff':>15}"
        )
        self.print_info("-" * 60)
        for key in stanley_costs:
            s_val = float(stanley_costs[key])
            b_val = float(sampled_costs[key])
            diff = b_val - s_val
            self.print_info(f"{key:<20} {s_val:>15.2f} {b_val:>15.2f} {diff:>+15.2f}")
        self.print_info("-" * 60)
        self.print_info(
            f"{'total':<20} {float(stanley_total):>15.2f} {float(sampled_total):>15.2f} "
            f"{float(sampled_total - stanley_total):>+15.2f}"
        )

        # Additional debug: show lateral error and boundary values
        self.print_info("Lateral error debug (sampled trajectory):")
        traj_slice = best_sampled_traj[: self.planning_horizon]
        xy_vals = traj_slice[:, :2]
        progress_vals, lateral_err = self._project_to_frenet_batch(xy_vals)
        index = jnp.searchsorted(self._jax_ss, progress_vals % self._jax_ss[-1])
        bounds_left = self._jax_raceline_left_boundary[index]
        bounds_right = self._jax_raceline_right_boundary[index]

        # Show first 3 and last 2 timesteps
        for i in [0, 1, 2, self.planning_horizon - 2, self.planning_horizon - 1]:
            collision_left = lateral_err[i] > bounds_left[i]
            collision_right = lateral_err[i] < -bounds_right[i]
            self.print_info(
                f"  t={i}: lat_err={float(lateral_err[i]):.4f}, "
                f"bounds=[-{float(bounds_right[i]):.4f}, +{float(bounds_left[i]):.4f}], "
                f"collision={'LEFT' if collision_left else ('RIGHT' if collision_right else 'OK')}"
            )
        self.print_info("=" * 60)

    def __plot_planned_trajectory(self, traj):
        self._viz_counter += 1
        if self._viz_counter >= self._viz_skip_count:
            self._viz_counter = 0
            traj_cart = [CartesianState(*state) for state in traj]
            self.plot_trajectory(traj_cart, color=(0, 0, 200))

    def __plot_track_bounds(self, sim_state: CartesianState):
        # Compute track bounds using Cartesian trajectory
        planned_traj = self.rollout_sampled_trajectory(
            jnp.array([*sim_state]), self.planned_controls
        )
        planned_ys = planned_traj.ys[: self.planning_horizon, :]
        xy_vals = planned_ys[:, :2]  # (x, y) positions

        # Project to get progress values and find boundary widths
        progress_vals, _ = self._project_to_frenet_batch(xy_vals)
        index = jnp.searchsorted(self._jax_ss, progress_vals % self._jax_ss[-1])
        bounds_left = jnp.take(self._jax_raceline_left_boundary, index)
        bounds_right = jnp.take(self._jax_raceline_right_boundary, index)

        # Get raceline headings at these progress values
        headings = self._jax_raceline_headings[index]

        # Compute boundary points in Cartesian coordinates
        # Left boundary: offset perpendicular (90 degrees left of heading)
        left_x = xy_vals[:, 0] + bounds_left * jnp.cos(headings + jnp.pi / 2)
        left_y = xy_vals[:, 1] + bounds_left * jnp.sin(headings + jnp.pi / 2)
        # Right boundary: offset perpendicular (90 degrees right of heading)
        right_x = xy_vals[:, 0] - bounds_right * jnp.cos(headings + jnp.pi / 2)
        right_y = xy_vals[:, 1] - bounds_right * jnp.sin(headings + jnp.pi / 2)

        # Convert to CartesianState for plotting (use trajectory velocities)
        bounds_left_cart = [
            CartesianState(
                left_x[i],
                left_y[i],
                headings[i],
                planned_ys[i, 3],
                planned_ys[i, 4],
                planned_ys[i, 5],
            )
            for i in range(self.planning_horizon)
        ]
        bounds_right_cart = [
            CartesianState(
                right_x[i],
                right_y[i],
                headings[i],
                planned_ys[i, 3],
                planned_ys[i, 4],
                planned_ys[i, 5],
            )
            for i in range(self.planning_horizon)
        ]

        self.plot_trajectory(bounds_left_cart, color=(0, 255, 0))
        self.plot_trajectory(bounds_right_cart, color=(0, 255, 0))

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

    def _debug_log_best_cost_components(
        self, state: jax.Array, best_traj_ys: jax.Array
    ):
        """Re-evaluate cost on the best sampled trajectory and log individual components.

        Args:
            state: Current vehicle state (Cartesian) as a JAX array.
            best_traj_ys: State array of the best trajectory, shape (planning_horizon, state_dim).
        """
        ts = jnp.arange(0, self.planning_horizon * self.planning_dt, self.planning_dt)
        traj = RawDiscreteTrajectory(ts, best_traj_ys)
        components = self.__compute_cost_components(state, traj)
        total = sum(components.values())
        parts = " | ".join(f"{k}={float(v):.4g}" for k, v in components.items())
        self.print_info(f"Best cost: {float(total):.4g} ({parts})")
