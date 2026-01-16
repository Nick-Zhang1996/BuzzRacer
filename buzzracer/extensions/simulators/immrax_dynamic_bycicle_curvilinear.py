"""Simulator for an Ackermann steering vehicle with dynamic bicycle model"""
# page 30 of book Vehicle Dynamics and Control

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from immrax import System

if TYPE_CHECKING:
    from buzzracer.cars.car import Car


def conditional_log(condition: bool, format_str: str, *args):
    if condition:
        print(format_str.format(*args))


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
        # print(f"{t.shape=}, {x.shape=}, {u.shape=}, {w.shape=}, {curvature.shape=}")
        progress, lateral_err, heading_err, v_forward, v_sideway, rel_omega = x
        steering, throttle = u

        def kinematic_model():
            # Origin at CG, beta is the angle between CG velocity and car orientation
            beta = jnp.arctan(jnp.tan(steering) * self.lr / (self.lf + self.lr))

            # jax.debug.print("curvature.shape={}", curvature.shape)

            dsdt = (
                (v_forward * jnp.cos(heading_err) - v_sideway * jnp.sin(heading_err))
                / (1 - lateral_err * curvature.squeeze())
            )  # FIXME: I have no idea why curvature picks up an extra dimension, but we need to remove it
            # print(
            #     f"{curvature.shape=}\n{(1 - lateral_err * curvature).shape=}\n{dsdt.shape=}"
            # )
            dndt = v_forward * jnp.sin(heading_err) + v_sideway * jnp.cos(heading_err)
            # acceleration at rear wheel
            acc_rw = jax.lax.select(
                0 < v_forward,
                6.17 * (throttle - v_forward / 15.2 - 0.333),
                0.0,
            )  # FIXME: this branch prevents parametope reachset calculation, needs custom_if logic
            acc_cg = acc_rw / jnp.cos(beta)
            d_v_forward_dt = acc_cg * jnp.cos(beta)
            d_v_sideway_dt = acc_cg * jnp.sin(beta)

            total_v = jnp.sqrt(v_forward**2 + v_sideway**2)
            d_heading_dt = total_v / self.lr * jnp.sin(beta)
            d_rel_heading_dt = d_heading_dt - curvature.squeeze() * dsdt

            return dsdt, dndt, d_rel_heading_dt, d_v_forward_dt, d_v_sideway_dt, 0.0

        def dynamic_model():
            dsdt = (
                v_forward * jnp.cos(heading_err) - v_sideway * jnp.sin(heading_err)
            ) / (1 - lateral_err * curvature.squeeze())
            dndt = v_forward * jnp.sin(heading_err) + v_sideway * jnp.cos(heading_err)

            # Reference angular velocity
            omega_ref = dsdt * curvature.squeeze()
            omega_ref = jnp.clip(omega_ref, -1e3, 1e3)
            # Total angular velocity in inertial frame
            omega = omega_ref + rel_omega
            # jax.debug.print("omega isinf? {0}", jnp.isinf(omega))

            # Slip angle of front/rear tires
            slip_f = (
                -jnp.arctan(
                    (omega * self.lf + v_sideway) / jnp.maximum(v_forward, 1e-3)
                )
                + steering
            )
            slip_r = jnp.arctan(
                (omega * self.lr - v_sideway) / jnp.maximum(v_forward, 1e-3)
            )

            # Lateral forces from front and rear tires
            Ffy = tire_curve(slip_f) * self.m * 9.8 * self.lr / (self.lr + self.lf)
            Fry = (
                1.15 * tire_curve(slip_r) * self.m * 9.8 * self.lf / (self.lr + self.lf)
            )

            # in body frame
            d_vy_body = 1.0 / self.m * (Fry + Ffy - self.m * v_forward * omega)
            d_vx_body = (
                jax.lax.select(
                    v_forward > 0, 6.17 * (throttle - v_forward / 15.2 - 0.333), 0.0
                )
                + omega * v_sideway
            )
            d_rel_heading_dt = rel_omega
            # NOTE ignoring d_omega_ref_dt, i.e. curvature time rate
            d_rel_omega = 1.0 / self.Iz * (Ffy * self.lf - Fry * self.lr)

            # jax.debug.callback(
            #     conditional_log,
            #     jnp.isinf(
            #         jnp.array(
            #             [
            #                 dsdt,
            #                 dndt,
            #                 d_rel_heading_dt,
            #                 d_vx_body,
            #                 d_vy_body,
            #                 d_rel_omega,
            #             ]
            #         )
            #     ).any(),
            #     "inf in dynamic_bycicle: {0}, {1}, {2}, {3}, {4}, {5}",
            #     jnp.isinf(dsdt),
            #     jnp.isinf(dndt),
            #     jnp.isinf(d_rel_heading_dt),
            #     jnp.isinf(d_vx_body),
            #     jnp.isinf(d_vy_body),
            #     jnp.isinf(d_rel_omega),
            # )

            return dsdt, dndt, d_rel_heading_dt, d_vx_body, d_vy_body, d_rel_omega

        # FIXME: cond is not in the inclusion registry
        # However, we are already restricting the use of the sampling based controller to the case where v_forward > 1.0
        # Therefore, should always use dynamic_model
        # dsdt, dndt, d_rel_heading_dt, d_vx_body, d_vy_body, d_rel_omega = (
        #     kinematic_model()
        #     # dynamic_model()
        # )
        dsdt, dndt, d_rel_heading_dt, d_vx_body, d_vy_body, d_rel_omega = jax.lax.cond(
            v_forward < 0.1, kinematic_model, dynamic_model
        )

        return jnp.array(
            [dsdt, dndt, d_rel_heading_dt, d_vx_body, d_vy_body, d_rel_omega]
        )
