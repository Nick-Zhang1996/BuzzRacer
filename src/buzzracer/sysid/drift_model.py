"""Simplified drift dynamics model based on Drifting_Controller (May 2026)."""
from __future__ import annotations

from math import atan, cos, hypot, sin

import matplotlib.pyplot as plt
import numpy as np

from buzzracer.cars.car_param import CarParam
from buzzracer.sysid.vehicle_dynamics import VehicleDynamics
from buzzracer.types import CartesianState, Control


class DriftModel(VehicleDynamics):
    """Cartesian rigid-body drift model with quasi-steady drivetrain coupling."""

    state_type = CartesianState
    _EPS = 1e-9
    _ROOT_ITERS = 10

    @staticmethod
    def _combined_slip_force(
        slip_x: float,
        slip_y: float,
        peak_force: float,
        shape: float,
    ) -> tuple[float, float]:
        """Resolve a combined-slip tire force from a slip-velocity vector."""
        slip_mag = hypot(slip_x, slip_y)
        if slip_mag < DriftModel._EPS:
            return (0.0, 0.0)

        force_mag = peak_force * atan(shape * slip_mag) / np.pi * 2
        scale = force_mag / slip_mag
        return (slip_x * scale, slip_y * scale)

    @staticmethod
    def _rear_force_balance_residual(
        wheel_speed: float,
        throttle: float,
        vx: float,
        rear_lateral_velocity: float,
        car_param: CarParam,
    ) -> float:
        """Residual of the rear tire force balance, for rear-only checks."""
        wheel_radius = max(car_param.drift_wheel_radius, DriftModel._EPS)
        slip_x = -vx + wheel_speed * wheel_radius
        slip_y = -rear_lateral_velocity
        tire_fx, _ = DriftModel._combined_slip_force(
            slip_x,
            slip_y,
            car_param.drift_rear_tire_A,
            car_param.drift_rear_tire_B,
        )
        motor_fx = DriftModel.get_motor_fx(car_param, throttle, wheel_speed)
        return tire_fx - motor_fx

    @staticmethod
    def _drivetrain_force_balance_residual(
        wheel_speed: float,
        throttle: float,
        vx: float,
        front_lateral_velocity: float,
        rear_lateral_velocity: float,
        steering: float,
        car_param: CarParam,
    ) -> float:
        """Residual of the locked, all-wheel-drive force balance."""
        wheel_radius = max(car_param.drift_wheel_radius, DriftModel._EPS)
        wheel_linear_speed = wheel_speed * wheel_radius

        front_slip_x = -vx + wheel_linear_speed * cos(steering)
        front_slip_y = -front_lateral_velocity + wheel_linear_speed * sin(steering)
        rear_slip_x = -vx + wheel_linear_speed
        rear_slip_y = -rear_lateral_velocity

        front_fx, front_fy = DriftModel._combined_slip_force(
            front_slip_x,
            front_slip_y,
            car_param.drift_front_tire_A,
            car_param.drift_front_tire_B,
        )
        rear_fx, _ = DriftModel._combined_slip_force(
            rear_slip_x,
            rear_slip_y,
            car_param.drift_rear_tire_A,
            car_param.drift_rear_tire_B,
        )

        # The front wheel is steered, so its longitudinal tire force is the
        # component along the wheel heading. This includes its lateral force.
        front_drive_force = front_fx * cos(steering) + front_fy * sin(steering)
        motor_fx = DriftModel.get_motor_fx(car_param, throttle, wheel_speed)
        return front_drive_force + rear_fx - motor_fx

    @staticmethod
    def get_motor_fx(car_param, throttle, wheel_speed):
        motor_fx = (
            car_param.drift_motor_torque_coeff
            * (throttle - car_param.drift_motor_back_emf * wheel_speed)
            / car_param.drift_wheel_radius
        )
        return motor_fx

    @staticmethod
    def _bisect_wheel_speed(
        lower: float,
        upper: float,
        lower_residual: float,
        throttle: float,
        vx: float,
        front_lateral_velocity: float,
        rear_lateral_velocity: float,
        steering: float,
        car_param: CarParam,
    ) -> float:
        """Refine a sign-changing all-wheel-drive wheel-speed bracket."""
        for _ in range(DriftModel._ROOT_ITERS):
            mid = 0.5 * (lower + upper)
            mid_residual = DriftModel._drivetrain_force_balance_residual(
                mid,
                throttle,
                vx,
                front_lateral_velocity,
                rear_lateral_velocity,
                steering,
                car_param,
            )
            if lower_residual * mid_residual <= 0.0:
                upper = mid
            else:
                lower = mid
                lower_residual = mid_residual
        return 0.5 * (lower + upper)

    @staticmethod
    def _solve_wheel_speed(
        vx: float,
        front_lateral_velocity: float,
        rear_lateral_velocity: float,
        throttle: float,
        steering: float,
        car_param: CarParam,
    ) -> float:
        """Solve the common front/rear wheel speed from drivetrain balance."""
        wheel_radius = max(car_param.drift_wheel_radius, DriftModel._EPS)
        road_speed = vx / wheel_radius
        span = max(abs(road_speed), 100.0)
        lower = road_speed - span
        upper = road_speed + span

        # The calibrated back-EMF curve is nonlinear, so the residual can have
        # the same sign at both range endpoints. Find a local sign change.
        wheel_speeds = np.linspace(lower, upper, 101)
        previous_speed = wheel_speeds[0]
        previous_residual = DriftModel._drivetrain_force_balance_residual(
            previous_speed,
            throttle,
            vx,
            front_lateral_velocity,
            rear_lateral_velocity,
            steering,
            car_param,
        )

        for speed in wheel_speeds[1:]:
            residual = DriftModel._drivetrain_force_balance_residual(
                speed,
                throttle,
                vx,
                front_lateral_velocity,
                rear_lateral_velocity,
                steering,
                car_param,
            )
            if previous_residual * residual <= 0.0:
                return DriftModel._bisect_wheel_speed(
                    previous_speed,
                    speed,
                    previous_residual,
                    throttle,
                    vx,
                    front_lateral_velocity,
                    rear_lateral_velocity,
                    steering,
                    car_param,
                )
            previous_speed = speed
            previous_residual = residual
        raise ValueError('wheel-speed root is not bracketed')

    @staticmethod
    def advance_dynamics(
        state: CartesianState,
        control: Control,
        car_param: CarParam,
        dt: float,
        curvature: float = None,
    ) -> CartesianState:
        """Advance the simplified drift model by one Euler step."""
        del curvature

        lf = car_param.lf
        lr = car_param.lr
        m = car_param.m
        Iz = car_param.Iz
        wheel_radius = max(car_param.drift_wheel_radius, DriftModel._EPS)

        throttle = float(
            np.clip(
                control.throttle,
                car_param.min_throttle,
                car_param.max_throttle,
            )
        )
        steering = control.steering

        vx = state.v_forward
        vy = state.v_sideway
        yaw_rate = state.omega

        front_lateral_velocity = vy + yaw_rate * lf
        rear_lateral_velocity = vy - yaw_rate * lr

        # The PDF's simplified model treats wheel speed as a fast hidden state.
        wheel_speed = DriftModel._solve_wheel_speed(
            vx,
            front_lateral_velocity,
            rear_lateral_velocity,
            throttle,
            steering,
            car_param,
        )
        wheel_linear_speed = wheel_speed * wheel_radius

        front_slip_x = -vx + wheel_linear_speed * cos(steering)
        front_slip_y = -front_lateral_velocity + wheel_linear_speed * sin(steering)
        rear_slip_x = -vx + wheel_linear_speed
        rear_slip_y = -rear_lateral_velocity

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
        print(d_vx)

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
