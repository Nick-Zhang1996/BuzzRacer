''' Dynamic bicycle model with pacjka tire model'''
# pylint: disable-next=line-too-long
# Ref:https://ftp.idu.ac.id/wp-content/uploads/ebook/tdg/TERRAMECHANICS%20AND%20MOBILITY/epdf.pub_vehicle-dynamics-and-control-2nd-edition.pdf
from __future__ import annotations
from typing import TYPE_CHECKING
from math import sin, cos

import numpy as np
import torch

from buzzracer.types import CartesianState, CurvilinearState, Control
from buzzracer.sysid.vehicle_dynamics import VehicleDynamics
from buzzracer.sysid.kinematic_bicycle_model import KinematicBicycleModelCartesian
from buzzracer.sysid.kinematic_bicycle_model import KinematicBicycleModelFrenet
from buzzracer.sysid.tire import tire_curve

if TYPE_CHECKING:
    from buzzracer.cars.car import CarParam


class DynamicBicycleModelCartesian(VehicleDynamics):
    ''' Dynamic bicycle model with pacjka tire model
    Follows Vehicle Dynamics and Control, 2nd Edition, Sec 2.3 with nonlinear tire function'''

    state_type = CartesianState

    @staticmethod
    def advance_dynamics(state: CartesianState,
                         control: Control,
                         car_param: CarParam,
                         dt: float,
                         curvature: float = None,
                         use_torch: bool = False) -> CartesianState:
        ''' Step dynamics forward by dt, x+ = x + f(x,u)*dt

        Args:
            state: Current state of the vehicle
            control: Control for the vehicle
            car_param: CarParam object to supply vehicle sysid parameters like mass, Iz, wheelbase
            dt: time step in seconds e.g. 0.01
            curvature: signed curvature of ref curve, ccw positive (only used for CurvilinearState)
        Return:
            states at next timestep
        '''
        lf = car_param.lf
        lr = car_param.lr

        Iz = car_param.Iz
        m = car_param.m

        # for small longitudinal velocity use kinematic model
        # to avoid numerical instability caused by 1/vx
        if state.v_forward < 0.05:
            return KinematicBicycleModelCartesian.advance_dynamics(state, control, car_param, dt, use_torch=use_torch)

        vx = state.v_forward
        vy = state.v_sideway
        omega = state.omega
        # Slip angle of front/rear tires
        if use_torch:
            slip_f = -torch.arctan((omega * lf + vy) / vx) + control.steering
            slip_r = torch.arctan((omega * lr - vy) / vx)
        else:
            slip_f = -np.arctan((omega * lf + vy) / vx) + control.steering
            slip_r = np.arctan((omega * lr - vy) / vx)

        # Lateral forces from front and rear tires
        Ffy = tire_curve(slip_f, use_torch) * m * 9.8 * lr / (lr + lf)
        Fry = 1.15 * tire_curve(slip_r, use_torch) * m * 9.8 * lf / (lr + lf)

        # Dynamics
        d_vx = 6.17 * (control.throttle - vx / 15.2 - 0.333) * (vx > 0)
        # d_vy is measured in body-attached frame
        d_vy = 1.0 / m * (Fry + Ffy - m * vx * omega)
        d_omega = 1.0 / Iz * (Ffy * lf - Fry * lr)

        # Discretization
        vx = state.v_forward + d_vx * dt
        vy = state.v_sideway + d_vy * dt
        omega = state.omega + d_omega * dt

        # Back to global frame
        if use_torch:
            vxg = vx * torch.cos(state.heading) - vy * torch.sin(state.heading)
            vyg = vx * torch.sin(state.heading) + vy * torch.cos(state.heading)
        else:
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


class DynamicBicycleModelFrenet(VehicleDynamics):
    ''' Dynamic Bicycle Model (Frenet frame)'''
    state_type = CurvilinearState

    @staticmethod
    def advance_dynamics(state: CurvilinearState,
                         control: Control,
                         car_param: CarParam,
                         dt: float,
                         curvature: float = None,
                         use_torch: bool = False) -> CurvilinearState:
        ''' Step dynamics forward by dt, x+ = x + f(x,u)*dt

        Args:
            state: Current state of the vehicle
            control: Control for the vehicle
            car_param: CarParam object to supply vehicle sysid parameters like mass, Iz, wheelbase
            dt: time step in seconds e.g. 0.01
            curvature: signed curvature of ref curve, ccw positive (only used for CurvilinearState)
        Return:
            state at next timestep
        Ref: Vehicle Dynamics and Control, 2nd Edition, Sec 2.2
        Ref: https://arxiv.org/pdf/2005.00826
        '''
        if state.v_forward < 0.1:
            return KinematicBicycleModelFrenet.advance_dynamics(state, control, car_param, dt, curvature, use_torch=use_torch)

        lf = car_param.lf
        lr = car_param.lr

        Iz = car_param.Iz
        m = car_param.m

        if use_torch:
            dsdt = (state.v_forward * torch.cos(state.heading_err)
                    - state.v_sideway * torch.sin(state.heading_err)
                    ) / (1-state.lateral_err*curvature)
            dndt = (state.v_forward * torch.sin(state.heading_err)
                    + state.v_sideway * torch.cos(state.heading_err)
                    )

            # Reference angular velocity
            omega_ref = dsdt * curvature
            # Total angular velocity in inertial frame
            omega = omega_ref + state.rel_omega

            # Slip angle of front/rear tires
            slip_f = -torch.arctan(
                (omega * lf + state.v_sideway) / torch.max([state.v_forward, 1e-3])
            ) + control.steering
            slip_r = torch.arctan((omega * lr - state.v_sideway) /
                                  torch.max([state.v_forward, 1e-3]))

            # Lateral forces from front and rear tires
            Ffy = tire_curve(slip_f, use_torch) * m * 9.8 * lr / (lr + lf)
            Fry = 1.15 * tire_curve(slip_r, use_torch) * m * 9.8 * lf / (lr + lf)

            # in body frame
            d_vy_body = 1.0 / m * (Fry + Ffy - m * state.v_forward * omega)
            d_vx_body = 6.17 * (control.throttle - state.v_forward / 15.2 - 0.333) * (state.v_forward > 0) + \
                omega * state.v_sideway
            d_rel_heading_dt = state.rel_omega

            # NOTE ignoring d_omega_ref_dt, i.e. curvature time rate
            d_rel_omega = 1.0 / Iz * (Ffy * lf - Fry * lr)
        else:
            dsdt = (state.v_forward * np.cos(state.heading_err)
                    - state.v_sideway * np.sin(state.heading_err)
                    ) / (1-state.lateral_err*curvature)
            dndt = (state.v_forward * np.sin(state.heading_err)
                    + state.v_sideway * np.cos(state.heading_err)
                    )

            # Reference angular velocity
            omega_ref = dsdt * curvature
            # Total angular velocity in inertial frame
            omega = omega_ref + state.rel_omega

            # Slip angle of front/rear tires
            slip_f = -np.arctan(
                (omega * lf + state.v_sideway) / np.max([state.v_forward, 1e-3])
            ) + control.steering
            slip_r = np.arctan((omega * lr - state.v_sideway) / np.max([state.v_forward, 1e-3]))

            # Lateral forces from front and rear tires
            Ffy = tire_curve(slip_f) * m * 9.8 * lr / (lr + lf)
            Fry = 1.15 * tire_curve(slip_r) * m * 9.8 * lf / (lr + lf)

            # in body frame
            d_vy_body = 1.0 / m * (Fry + Ffy - m * state.v_forward * omega)
            d_vx_body = 6.17 * (control.throttle - state.v_forward / 15.2 - 0.333) * (state.v_forward > 0) + \
                omega * state.v_sideway
            d_rel_heading_dt = state.rel_omega

            # NOTE ignoring d_omega_ref_dt, i.e. curvature time rate
            d_rel_omega = 1.0 / Iz * (Ffy * lf - Fry * lr)

        return CurvilinearState(
            progress=state.progress + dsdt * dt,
            lateral_err=state.lateral_err + dndt * dt,
            heading_err=state.heading_err + d_rel_heading_dt * dt,
            v_forward=state.v_forward + d_vx_body * dt,
            v_sideway=state.v_sideway + d_vy_body * dt,
            rel_omega=state.rel_omega + d_rel_omega * dt
        )
