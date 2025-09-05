''' Dynamic bicycle model with pacjka tire model'''
from math import sin, cos

import numpy as np

from buzzracer.cars.car import Car
from buzzracer.types import CartesianState, CurvilinearState, Control
from buzzracer.sysid.vehicle_dynamics import VehicleDynamics
from buzzracer.sysid.kinematic_bicycle_model import KinematicBicycleModel
from buzzracer.sysid.tire import tire_curve

class DynamicBicycleModel(VehicleDynamics):
    ''' Dynamic bicycle model with pacjka tire model
    Follows Vehicle Dynamics and Control, 2nd Edition, Sec 2.3 with nonlinear tire function'''

    @staticmethod
    def advance_dynamics(state: CartesianState,
                         control: Control, car: Car,
                         dt: float) -> CartesianState:
        ''' Step dynamics forward by dt, x+ = x + f(x,u)*dt

        Args:
            state: Current state of the vehicle
            control: Control for the vehicle
            car: Car object to supply vehicle sysid parameters like mass, Iz, wheelbase
            dt: time step in seconds e.g. 0.01
        Return:
            states at next timestep
        '''
        lf = car.lf
        lr = car.lr

        Iz = car.Iz
        m = car.m

        # for small longitudinal velocity use kinematic model
        # to avoid numerical instability caused by 1/vx
        if state.v_forward < 0.05:
            return KinematicBicycleModel.advance_dynamics(state, control, car, dt)

        vx = state.v_forward
        vy = state.v_sideway
        omega = state.omega
        # Slip angle of front/rear tires
        slip_f = -np.arctan((omega * lf + vy) / vx) + control.steering
        slip_r = np.arctan((omega * lr - vy) / vx)

        # Lateral forces from front and rear tires
        Ffy = tire_curve(slip_f) * m * 9.8 * lr / (lr + lf)
        Fry = 1.15 * tire_curve(slip_r) * m * 9.8 * lf / (lr + lf)

        # Dynamics
        d_vx = 6.17 * (control.throttle - vx / 15.2 - 0.333)
        d_vy = 1.0 / m * (Fry + Ffy - m * vx * omega)
        d_omega = 1.0 / Iz * (Ffy * lf - Fry * lr)

        # Discretization
        vx = state.v_forward + d_vx * dt
        vy = state.v_sideway + d_vy * dt
        omega = state.omega + d_omega * dt

        # Back to global frame
        vxg = vx * cos(state.heading) - vy * sin(state.heading)
        vyg = vx * sin(state.heading) + vy * cos(state.heading)

        # Update x,y, heading
        x = state.x + vxg * dt
        y = state.y + vyg * dt
        heading = state.heading + omega * dt + 0.5 * d_omega * dt * dt

        return CartesianState(x=x,
                              y=y,
                              heading=heading,
                              v_forward=vx,
                              v_sideway=vy,
                              omega=omega)

