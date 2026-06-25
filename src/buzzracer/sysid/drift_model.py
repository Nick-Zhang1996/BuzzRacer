"""Simplified drift dynamics model based on Drifting_Controller (May 2026)."""
from __future__ import annotations

from math import atan, cos, hypot, sin
from typing import TYPE_CHECKING

import numpy as np

from buzzracer.sysid.vehicle_dynamics import VehicleDynamics
from buzzracer.types import CartesianState, Control

if TYPE_CHECKING:
    from buzzracer.cars.car_param import CarParam


class DriftModel(VehicleDynamics):
    """Cartesian rigid-body drift model with quasi-steady drivetrain coupling."""

    state_type = CartesianState
    _EPS = 1e-9
    _ROOT_SAMPLES = 17
    _ROOT_ITERS = 16

    @staticmethod
    def _combined_slip_force(slip_x: float,
                             slip_y: float,
                             peak_force: float,
                             shape: float) -> tuple[float, float]:
        """Resolve a combined-slip tire force from a slip-velocity vector."""
        slip_mag = hypot(slip_x, slip_y)
        if slip_mag < DriftModel._EPS:
            return (0.0, 0.0)

        force_mag = peak_force * atan(shape * slip_mag)/np.pi*2
        scale = force_mag / slip_mag
        return (slip_x * scale, slip_y * scale)

    @staticmethod
    def _rear_force_balance_residual(wheel_speed: float,
                                     throttle: float,
                                     vx: float,
                                     rear_lateral_velocity: float,
                                     car_param: CarParam) -> float:
        """Residual of the quasi-steady rear-wheel drivetrain balance."""
        wheel_radius = max(car_param.drift_wheel_radius, DriftModel._EPS)
        slip_x = -vx + wheel_speed * wheel_radius
        slip_y = -rear_lateral_velocity
        tire_fx, _ = DriftModel._combined_slip_force(
            slip_x,
            slip_y,
            car_param.drift_rear_tire_A,
            car_param.drift_rear_tire_B,
        )
        motor_fx = (
            car_param.drift_motor_torque_coeff
            * (throttle - car_param.drift_motor_back_emf * wheel_speed)
            / wheel_radius
        )
        return tire_fx - motor_fx

    @staticmethod
    def _bisect_wheel_speed(lower: float,
                            upper: float,
                            lower_residual: float,
                            throttle: float,
                            vx: float,
                            rear_lateral_velocity: float,
                            car_param: CarParam) -> float:
        """Refine a sign-changing wheel-speed bracket."""
        for _ in range(DriftModel._ROOT_ITERS):
            mid = 0.5 * (lower + upper)
            mid_residual = DriftModel._rear_force_balance_residual(
                mid, throttle, vx, rear_lateral_velocity, car_param)
            if lower_residual * mid_residual <= 0.0:
                upper = mid
            else:
                lower = mid
                lower_residual = mid_residual
        return 0.5 * (lower + upper)

    @staticmethod
    def _solve_wheel_speed(vx: float,
                           rear_lateral_velocity: float,
                           throttle: float,
                           car_param: CarParam) -> float:
        """Solve the hidden quasi-steady wheel speed from drivetrain balance."""
        wheel_radius = max(car_param.drift_wheel_radius, DriftModel._EPS)
        back_emf = max(abs(car_param.drift_motor_back_emf), DriftModel._EPS)
        free_speed = throttle / back_emf
        road_speed = vx / wheel_radius
        span = max(abs(free_speed), abs(road_speed), 1.0)
        lower = min(free_speed, road_speed) - 2.0 * span - 10.0
        upper = max(free_speed, road_speed) + 2.0 * span + 10.0

        prev_speed = lower
        prev_residual = DriftModel._rear_force_balance_residual(
            prev_speed, throttle, vx, rear_lateral_velocity, car_param)
        best_speed = prev_speed
        best_error = abs(prev_residual)
        if best_error < DriftModel._EPS:
            return best_speed

        for idx in range(1, DriftModel._ROOT_SAMPLES):
            alpha = idx / (DriftModel._ROOT_SAMPLES - 1)
            wheel_speed = lower + (upper - lower) * alpha
            residual = DriftModel._rear_force_balance_residual(
                wheel_speed, throttle, vx, rear_lateral_velocity, car_param)

            error = abs(residual)
            if error < best_error:
                best_speed = wheel_speed
                best_error = error
                if error < DriftModel._EPS:
                    return best_speed

            if prev_residual * residual < 0.0:
                return DriftModel._bisect_wheel_speed(
                    prev_speed,
                    wheel_speed,
                    prev_residual,
                    throttle,
                    vx,
                    rear_lateral_velocity,
                    car_param,
                )

            prev_speed = wheel_speed
            prev_residual = residual

        return best_speed

    @staticmethod
    def advance_dynamics(state: CartesianState,
                         control: Control,
                         car_param: CarParam,
                         dt: float,
                         curvature: float = None) -> CartesianState:
        """Advance the simplified drift model by one Euler step."""
        del curvature

        lf = car_param.lf
        lr = car_param.lr
        m = car_param.m
        Iz = car_param.Iz
        wheel_radius = max(car_param.drift_wheel_radius, DriftModel._EPS)

        throttle = float(np.clip(
            control.throttle,
            car_param.min_throttle,
            car_param.max_throttle,
        ))
        steering = control.steering

        vx = state.v_forward
        vy = state.v_sideway
        yaw_rate = state.omega

        front_lateral_velocity = vy + yaw_rate * lf
        rear_lateral_velocity = vy - yaw_rate * lr

        # The PDF's simplified model treats wheel speed as a fast hidden state.
        wheel_speed = DriftModel._solve_wheel_speed(
            vx,
            rear_lateral_velocity,
            throttle,
            car_param,
        )
        wheel_linear_speed = wheel_speed * wheel_radius

        front_slip_x = -vx + wheel_linear_speed * cos(steering)
        front_slip_y = -front_lateral_velocity + wheel_linear_speed * sin(steering)
        rear_slip_x = -vx + wheel_linear_speed
        rear_slip_y = -rear_lateral_velocity

        print(f'{rear_slip_x=}')

        Ffx, Ffy = DriftModel._combined_slip_force(
            front_slip_x,
            front_slip_y,
            car_param.drift_front_tire_A,
            car_param.drift_front_tire_B,
        )
        Frx, Fry = DriftModel._combined_slip_force(
            rear_slip_x,
            rear_slip_y,
            car_param.drift_rear_tire_A,
            car_param.drift_rear_tire_B,
        )

        d_vx = (Ffx + Frx) / m + yaw_rate * vy
        d_vy = (Ffy + Fry) / m - yaw_rate * vx
        d_yaw_rate = (lf * Ffy - lr * Fry) / Iz

        vx_next = vx + d_vx * dt
        vy_next = vy + d_vy * dt
        yaw_rate_next = yaw_rate + d_yaw_rate * dt

        vx_global = vx_next * cos(state.heading) - vy_next * sin(state.heading)
        vy_global = vx_next * sin(state.heading) + vy_next * cos(state.heading)

        return CartesianState(
            x=state.x + vx_global * dt,
            y=state.y + vy_global * dt,
            heading=state.heading + yaw_rate * dt + 0.5 * d_yaw_rate * dt * dt,
            v_forward=vx_next,
            v_sideway=vy_next,
            omega=yaw_rate_next,
        )
