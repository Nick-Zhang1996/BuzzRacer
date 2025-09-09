"""Simulator for an Ackermann steering vehicle with dynamic bicycle model"""
# page 30 of book Vehicle Dynamics and Control

from buzzracer.cars.car import Car
import jax
import jax.numpy as jnp
from immrax import System


# NOTE: duplicated from `tire.py`, modified to use jax.numpy
def tire_curve(slip):
    C = 1.6
    B = 2.3
    D = 1.1
    # C: tail shape
    retval = D * jnp.sin(C * jnp.arctan(B * slip))
    return retval


class DynamicBicycle(System):
    max_v: float = 3.0

    def __init__(self, car: Car) -> None:
        self.evolution = "continuous"
        self.xlen = 6  # car_states = x, y, heading, x_vel, y_vel, heading_vel
        self.name = "Dynamic Bicycle Model"

        car.init_param()  # FIXME: This is a horrible hack
        self.lf = car.lf
        self.lr = car.lr
        self.L = car.L
        self.Iz = car.Iz
        self.m = car.m

    def f(self, t, x: jnp.ndarray, u: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
        x, y, heading, x_vel, y_vel, heading_vel = x
        x_acc, y_acc, heading_acc = 0.0, 0.0, 0.0
        steering, throttle = u

        def kinematic_model():
            beta = jnp.atan(self.lr / self.L * jnp.tan(steering))

            # motor model
            x_acc = 6.17 * (throttle - x_vel / 15.2 - 0.333)
            y_velocity = jnp.linalg.norm(jnp.array([x_vel, y_vel])) * jnp.sin(beta)
            heading_vel = x_vel / self.L * jnp.tan(steering)

            return x_vel, y_velocity, heading_vel, x_acc, y_acc, heading_acc

        def dynamic_model():
            slip_f = -jnp.arctan((heading_vel * self.lf + y_vel) / x_vel) + steering
            slip_r = jnp.arctan((heading_vel * self.lr - y_vel) / x_vel)

            Ffy = tire_curve(slip_f) * self.m * 9.8 * self.lr / (self.lr + self.lf)
            Fry = (
                1.15 * tire_curve(slip_r) * self.m * 9.8 * self.lf / (self.lr + self.lf)
            )

            # Dynamics
            x_acc = 6.17 * (throttle - x_vel / 15.2 - 0.333)
            y_acc = (
                1.0
                / self.m
                * (Fry + Ffy * jnp.cos(steering) - self.m * x_vel * heading_vel)
            )
            heading_acc = (
                1.0 / self.Iz * (Ffy * self.lf * jnp.cos(steering) - Fry * self.lr)
            )
            return x_vel, y_vel, heading_vel, x_acc, y_acc, heading_acc

        # for small longitudinal velocity use kinematic model
        # for tire slip, ratio between lateral and longitudinal speed matters, avoid singularities
        x_vel, y_vel, heading_vel, x_acc, y_acc, heading_acc = jax.lax.cond(
            x_vel < 0.05, kinematic_model, dynamic_model
        )

        return jnp.array(
            [
                x_vel * jnp.cos(heading)
                - y_vel * jnp.sin(heading),  # conversion from body to global frame
                x_vel * jnp.sin(heading) + y_vel * jnp.cos(heading),
                heading_vel,
                x_acc,
                y_acc,
                heading_acc,
            ]
        )
