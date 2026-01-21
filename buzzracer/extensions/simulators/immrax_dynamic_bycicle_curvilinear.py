"""Simulator for an Ackermann steering vehicle with dynamic bicycle model"""
# page 30 of book Vehicle Dynamics and Control

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from immrax import System

if TYPE_CHECKING:
    from buzzracer.cars.car import Car

MAX_CLIP_FLOAT = 1e3
FLOAT_EPS = 1.0 / MAX_CLIP_FLOAT


def my_sign(input):
    return jnp.where(input > 0, 1, -1)


# NOTE: duplicated from `tire.py`, modified to use jax.numpy
def tire_curve(slip):
    C = 1.6
    B = 2.3
    D = 1.1
    # C: tail shape
    retval = D * jnp.sin(C * jnp.arctan(B * slip))
    return retval


class DynamicBicycleCurvilinear(System):
    max_v: float = 3.0

    def __init__(self, car: Car) -> None:
        self.evolution = "continuous"
        self.xlen = 6  # car_states = progress, lateral_err, heading_err, v_forward, v_sideway, rel_omega
        self.name = "Dynamic Bicycle Model"

        car.init_param()  # FIXME: This is a horrible hack
        self.lf = car.params.lf
        self.lr = car.params.lr
        self.Iz = car.params.Iz
        self.m = car.params.m

    def f(
        self, t, x: jax.Array, u: jax.Array, w: jax.Array, curvature: jax.Array
    ) -> jax.Array:
        progress, lateral_err, heading_err, v_forward, v_sideway, rel_omega = x
        steering, throttle = u

        # Pre-compute values shared between kinematic and dynamic models
        # FIXME: I have no idea why curvature picks up an extra dimension, but we need to remove it
        kappa = curvature.squeeze()
        cos_psi = jnp.cos(heading_err)
        sin_psi = jnp.sin(heading_err)
        denom = 1 - lateral_err * kappa
        safe_denom = denom + my_sign(denom) * FLOAT_EPS
        dsdt = (v_forward * cos_psi - v_sideway * sin_psi) / safe_denom
        dndt = v_forward * sin_psi + v_sideway * cos_psi

        # Pre-compute acceleration term (used in both models)
        # FIXME: this branch prevents parametope reachset calculation, needs custom_if logic
        acc_rw = jax.lax.select(
            v_forward > 0,
            6.17 * (throttle - v_forward / 15.2 - 0.333),
            0.0,
        )

        def kinematic_model():
            # Origin at CG, beta is the angle between CG velocity and car orientation
            beta = jnp.arctan(jnp.tan(steering) * self.lr / (self.lf + self.lr))
            cos_beta = jnp.cos(beta)
            sin_beta = jnp.sin(beta)

            acc_cg = acc_rw / (cos_beta + my_sign(cos_beta) * FLOAT_EPS)
            d_v_forward_dt = jnp.clip(
                acc_cg * cos_beta, -MAX_CLIP_FLOAT, MAX_CLIP_FLOAT
            )
            d_v_sideway_dt = jnp.clip(
                acc_cg * sin_beta, -MAX_CLIP_FLOAT, MAX_CLIP_FLOAT
            )

            total_v = jnp.sqrt(v_forward**2 + v_sideway**2)
            d_heading_dt = total_v / self.lr * sin_beta
            d_rel_heading_dt = d_heading_dt - kappa * dsdt

            return d_rel_heading_dt, d_v_forward_dt, d_v_sideway_dt, 0.0

        def dynamic_model():
            # Reference angular velocity
            omega_ref = jnp.clip(dsdt * kappa, -MAX_CLIP_FLOAT, MAX_CLIP_FLOAT)
            # Total angular velocity in inertial frame
            omega = omega_ref + rel_omega

            # Slip angle of front/rear tires
            v_forward_safe = jnp.maximum(v_forward, FLOAT_EPS)
            slip_f = steering - jnp.arctan(
                (omega * self.lf + v_sideway) / v_forward_safe
            )
            slip_r = jnp.arctan((omega * self.lr - v_sideway) / v_forward_safe)

            # Lateral forces from front and rear tires (pre-compute shared factor)
            weight_factor = self.m * 9.8 / (self.lr + self.lf)
            Ffy = tire_curve(slip_f) * weight_factor * self.lr
            Fry = 1.15 * tire_curve(slip_r) * weight_factor * self.lf

            # in body frame
            d_vy_body = jnp.clip(
                (Fry + Ffy) / self.m - v_forward * omega,
                -MAX_CLIP_FLOAT,
                MAX_CLIP_FLOAT,
            )
            d_vx_body = jnp.clip(
                acc_rw + omega * v_sideway, -MAX_CLIP_FLOAT, MAX_CLIP_FLOAT
            )
            d_rel_heading_dt = rel_omega
            # NOTE ignoring d_omega_ref_dt, i.e. curvature time rate
            d_rel_omega = (Ffy * self.lf - Fry * self.lr) / self.Iz

            return d_rel_heading_dt, d_vx_body, d_vy_body, d_rel_omega

        # FIXME: cond is not in the inclusion registry

        # d_rel_heading_dt, d_vx_body, d_vy_body, d_rel_omega = (
        #     kinematic_model()
        #     # dynamic_model()
        # )
        d_rel_heading_dt, d_vx_body, d_vy_body, d_rel_omega = jax.lax.cond(
            v_forward < 0.1, kinematic_model, dynamic_model
        )

        return jnp.array(
            [dsdt, dndt, d_rel_heading_dt, d_vx_body, d_vy_body, d_rel_omega]
        )
