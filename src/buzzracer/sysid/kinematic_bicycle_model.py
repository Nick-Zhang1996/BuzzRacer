''' Kinematic bicycle model with pacjka tire model'''
# pylint: disable-next=line-too-long
# Reference: https://ftp.idu.ac.id/wp-content/uploads/ebook/tdg/TERRAMECHANICS%20AND%20MOBILITY/epdf.pub_vehicle-dynamics-and-control-2nd-edition.pdf
from __future__ import annotations
from typing import TYPE_CHECKING

import casadi as ca
import numpy as np
import torch

from buzzracer.types import CartesianState, CurvilinearState, Control
from buzzracer.sysid.vehicle_dynamics import VehicleDynamics

if TYPE_CHECKING:
    from buzzracer.cars.car import CarParam


class KinematicBicycleModelCartesian(VehicleDynamics):
    ''' Kinematic Bicycle Model (Cartesian frame)
    Follows Vehicle Dynamics and Control, 2nd Edition, Sec 2.2'''
    state_type = CartesianState

    @staticmethod
    def advance_dynamics(state: CartesianState,
                         control: Control,
                         car_param: CarParam,
                         dt: float,
                         curvature: float = None,
                         use_torch: bool = False,
                         simple_throttle: bool = False) -> CartesianState:
        ''' Step dynamics forward by dt, x+ = x + f(x,u)*dt

        Args:
            state: Current state of the vehicle
            control: Control for the vehicle
            car: CarParam object to supply vehicle sysid parameters like mass, Iz, wheelbase
            dt: time step in seconds e.g. 0.01
            curvature: signed curvature of ref curve, ccw positive (only used for CurvilinearState)
            use_torch: use torch
            simple_throttle: If true, throttle = acceleration
        Return:
            state at next timestep

        '''

        del curvature
        if use_torch:
            beta = torch.arctan(np.tan(control.steering) * car_param.lr /
                                (car_param.lf + car_param.lr))
            dxdt = state.v_forward * torch.cos(state.heading + beta)
            dydt = state.v_forward * torch.sin(state.heading + beta)
            dvdt = 6.17 * (control.throttle + (- state.v_forward /
                           15.2 - 0.333) * (state.v_forward > 0))
            dheadingdt = state.v_forward * \
                torch.cos(beta) / (car_param.lf + car_param.lr) * torch.tan(control.steering)
        else:
            beta = np.arctan(np.tan(control.steering) * car_param.lr /
                             (car_param.lf + car_param.lr))
            dxdt = state.v_forward * np.cos(state.heading + beta)
            dydt = state.v_forward * np.sin(state.heading + beta)
            if simple_throttle:
                dvdt = control.throttle
            else:
                dvdt = 6.17 * (control.throttle + (- state.v_forward /
                                                   15.2 - 0.333) * (state.v_forward > 0))
            dheadingdt = state.v_forward * \
                np.cos(beta) / (car_param.lf + car_param.lr) * np.tan(control.steering)

        x = state.x + dt * dxdt
        y = state.y + dt * dydt
        v = state.v_forward + dt * dvdt
        heading = state.heading + dt * dheadingdt
        return CartesianState(x=x,
                              y=y,
                              heading=heading,
                              v_forward=v,
                              v_sideway=0,
                              omega=dheadingdt)


class KinematicBicycleModelFrenet(VehicleDynamics):
    ''' Kinematic Bicycle Model (Frenet frame)'''
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
        '''

        # Origin at CG, beta is the angle between CG velocity and car orientation
        if use_torch:
            beta = torch.arctan(torch.tan(control.steering) *
                                car_param.lr / (car_param.lf + car_param.lr))

            dsdt = (state.v_forward * torch.cos(state.heading_err) - state.v_sideway *
                    torch.sin(state.heading_err))/(1-state.lateral_err*curvature)
            dndt = state.v_forward * torch.sin(state.heading_err) + \
                state.v_sideway * torch.cos(state.heading_err)
            # acceleration at rear wheel
            acc_rw = 6.17 * (control.throttle + (- state.v_forward /
                             15.2 - 0.333) * (state.v_forward > 0))
            acc_cg = acc_rw / torch.cos(beta)
            d_v_forward_dt = acc_cg * torch.cos(beta)
            d_v_sideway_dt = acc_cg * torch.sin(beta)

            total_v = torch.sqrt(state.v_forward**2 + state.v_sideway**2)
            d_heading_dt = total_v / car_param.lr * torch.sin(beta)
            d_rel_heading_dt = d_heading_dt - curvature * dsdt

        else:
            beta = np.arctan(np.tan(control.steering) * car_param.lr /
                             (car_param.lf + car_param.lr))

            dsdt = (state.v_forward * np.cos(state.heading_err) - state.v_sideway *
                    np.sin(state.heading_err))/(1-state.lateral_err*curvature)
            dndt = state.v_forward * np.sin(state.heading_err) + \
                state.v_sideway * np.cos(state.heading_err)
            # acceleration at rear wheel
            acc_rw = 6.17 * (control.throttle + (- state.v_forward /
                             15.2 - 0.333) * (state.v_forward > 0))
            acc_cg = acc_rw / np.cos(beta)
            d_v_forward_dt = acc_cg * np.cos(beta)
            d_v_sideway_dt = acc_cg * np.sin(beta)

            total_v = np.sqrt(state.v_forward**2 + state.v_sideway**2)
            d_heading_dt = total_v / car_param.lr * np.sin(beta)
            d_rel_heading_dt = d_heading_dt - curvature * dsdt

        return CurvilinearState(
            progress=state.progress + dsdt * dt,
            lateral_err=state.lateral_err + dndt * dt,
            heading_err=state.heading_err + d_rel_heading_dt * dt,
            v_forward=state.v_forward + d_v_forward_dt * dt,
            v_sideway=state.v_sideway + d_v_sideway_dt * dt,
            rel_omega=0
        )

    @staticmethod
    def advance_dynamics_casadi(state,
                                control,
                                car_param: CarParam,
                                dt: float,
                                curvature):
        """CasADi-compatible discrete-time Frenet kinematic model."""
        beta = ca.atan(ca.tan(control[0]) * car_param.lr /
                       (car_param.lf + car_param.lr))

        dsdt = (state[3] * ca.cos(state[2]) - state[4] * ca.sin(state[2])) / (
            1 - state[1] * curvature)
        dndt = state[3] * ca.sin(state[2]) + state[4] * ca.cos(state[2])
        rolling = ca.if_else(state[3] > 0, 1.0, 0.0)
        acc_rw = 6.17 * (control[1] + (- state[3] / 15.2 - 0.333) * rolling)
        acc_cg = acc_rw / ca.cos(beta)
        d_v_forward_dt = acc_cg * ca.cos(beta)
        d_v_sideway_dt = acc_cg * ca.sin(beta)

        total_v = ca.sqrt(state[3] * state[3] + state[4] * state[4])
        d_heading_dt = total_v / car_param.lr * ca.sin(beta)
        d_rel_heading_dt = d_heading_dt - curvature * dsdt

        return ca.vertcat(
            state[0] + dsdt * dt,
            state[1] + dndt * dt,
            state[2] + d_rel_heading_dt * dt,
            state[3] + d_v_forward_dt * dt,
            state[4] + d_v_sideway_dt * dt,
            0.0,
        )
