"""Simulator for an Ackermann steering vehicle with dynamic bicycle model"""
# page 30 of book Vehicle Dynamics and Control
# "forward" states are longitudinal, "sideway" states are lateral

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from buzzracer.cars.car import Car

import jax
import jax.numpy as jnp
from immrax import System, lt
from immrax.comparison import IntervalRelation


# NOTE: duplicated from `tire.py`, modified to use jax.numpy
def tire_curve(slip):
    C = 1.6
    B = 2.3
    D = 1.1
    # C: tail shape
    retval = D * jnp.sin(C * jnp.arctan(B * slip))
    return retval


class DynamicBicycleCartesian(System):
    def __init__(self, car: Car) -> None:
        self.evolution = "continuous"
        self.xlen = 6  # car_states = x, y, heading, x_vel, y_vel, heading_vel
        self.name = "Dynamic Bicycle Model"

        car.init_param()  # FIXME: This is a horrible hack
        self.lf = car.params.lf
        self.lr = car.params.lr
        self.L = car.params.lf + car.params.lr
        self.Iz = car.params.Iz
        self.m = car.params.m

    def f(self, t, x: jnp.ndarray, u: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
        x, y, heading, v_forward, v_sideway, omega = x
        a_forward, a_sideway, a_heading = 0.0, 0.0, 0.0
        steering, throttle = u
        w_v_sideway, w_v_heading = w

        def kinematic_model():
            beta = jnp.atan(self.lr / self.L * jnp.tan(steering))

            # motor model
            a_forward = 6.17 * (throttle - v_forward / 15.2 - 0.333)
            v_sideway_out = jnp.sqrt(v_forward**2 + v_sideway**2) * jnp.sin(beta)
            heading_vel = v_forward / self.L * jnp.tan(steering)

            return (
                v_forward,
                v_sideway_out,
                heading_vel,
                a_forward,
                a_sideway,
                a_heading,
            )

        def dynamic_model():
            slip_f = -jnp.atan((omega * self.lf + v_sideway) / v_forward) + steering
            slip_r = jnp.atan((omega * self.lr - v_sideway) / v_forward)

            Ffy = tire_curve(slip_f) * self.m * 9.8 * self.lr / (self.lr + self.lf)
            Fry = (
                # 1.15 * tire_curve(slip_r) * self.m * 9.8 * self.lf / (self.lr + self.lf)
                tire_curve(slip_r) * self.m * 9.8 * self.lf / (self.lr + self.lf)
            )

            # Dynamics
            a_forward = 6.17 * (throttle - v_forward / 15.2 - 0.333)
            a_sideway = (
                1.0
                / self.m
                * (
                    # Fry + Ffy - self.m * v_forward * omega
                    Fry + Ffy * jnp.cos(steering) - self.m * v_forward * omega
                )  # FIXME: model mismatch. Original doesn't have cos here or in a_heading
            )
            a_heading = (
                # 1.0 / self.Iz * (Ffy * self.lf - Fry * self.lr)
                1.0 / self.Iz * (Ffy * self.lf * jnp.cos(steering) - Fry * self.lr)
            )
            return v_forward, v_sideway, omega, a_forward, a_sideway, a_heading

        # for small longitudinal velocity use kinematic model
        # for tire slip, ratio between lateral and longitudinal speed matters, avoid singularities
        # v_forward, v_sideway, omega, a_forward, a_sideway, a_heading = kinematic_model()
        v_forward, v_sideway, omega, a_forward, a_sideway, a_heading = jax.lax.cond(
            lt(
                v_forward,
                0.1
                * jnp.ones_like(
                    v_forward
                ),  # Adjoint reachability very sensitive to small v_forward
                # IntervalRelation.ALL,
                IntervalRelation.PRECEDES
                | IntervalRelation.FINISHED_BY
                | IntervalRelation.CONTAINS
                | IntervalRelation.STARTED_BY,
            ),
            # kinematic_model,
            kinematic_model,
            dynamic_model,
            # dynamic_model,
        )

        v_sideway += w_v_sideway
        omega += w_v_heading

        return jnp.array(
            [
                v_forward * jnp.cos(heading)
                - v_sideway * jnp.sin(heading),  # conversion from body to global frame
                v_forward * jnp.sin(heading) + v_sideway * jnp.cos(heading),
                omega,
                a_forward,
                a_sideway,
                a_heading,
            ]
        )
