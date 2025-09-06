''' Dynamic bicycle model with pacjka tire model'''
# pylint disable-next=line-too-long
# Ref:https://ftp.idu.ac.id/wp-content/uploads/ebook/tdg/TERRAMECHANICS%20AND%20MOBILITY/epdf.pub_vehicle-dynamics-and-control-2nd-edition.pdf
from math import sin, cos

import numpy as np

from buzzracer.cars.car import Car
from buzzracer.types import CartesianState, CurvilinearState, Control
from buzzracer.sysid.vehicle_dynamics import VehicleDynamics
from buzzracer.sysid.kinematic_bicycle_model import KinematicBicycleModelCartesian
from buzzracer.sysid.tire import tire_curve

class DynamicBicycleModelCartesian(VehicleDynamics):
    ''' Dynamic bicycle model with pacjka tire model
    Follows Vehicle Dynamics and Control, 2nd Edition, Sec 2.3 with nonlinear tire function'''

    @staticmethod
    def advance_dynamics(state: CartesianState,
                         control: Control,
                         car: Car,
                         dt: float,
                         curvature: float=None) -> CartesianState:
        ''' Step dynamics forward by dt, x+ = x + f(x,u)*dt

        Args:
            state: Current state of the vehicle
            control: Control for the vehicle
            car: Car object to supply vehicle sysid parameters like mass, Iz, wheelbase
            dt: time step in seconds e.g. 0.01
            curvature: signed curvature of ref curve, ccw positive (only used for CurvilinearState)
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
            return KinematicBicycleModelCartesian.advance_dynamics(state, control, car, dt)

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
        # d_vy is measured in body-attached frame
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


class DynamicBicycleModelFrenet(VehicleDynamics):
    ''' Dynamic Bicycle Model (Frenet frame)'''

    @staticmethod
    def advance_dynamics(state: CurvilinearState, control: Control,
                         car: Car, dt: float, curvature: float=None) -> CurvilinearState:
        ''' Step dynamics forward by dt, x+ = x + f(x,u)*dt

        Args:
            state: Current state of the vehicle
            control: Control for the vehicle
            car: Car object to supply vehicle sysid parameters like mass, Iz, wheelbase
            dt: time step in seconds e.g. 0.01
            curvature: signed curvature of ref curve, ccw positive (only used for CurvilinearState)
        Return:
            state at next timestep
        Ref: https://arxiv.org/pdf/2301.04316
        Ref: Vehicle Dynamics and Control, 2nd Edition, Sec 2.2
        '''

        lf = car.lf
        lr = car.lr

        Iz = car.Iz
        m = car.m

        dvdt = 6.17 * (control.throttle - state.v_forward / 15.2 - 0.333)

        vx = state.v_forward
        vy = state.v_sideway
        n = state.lateral_err
        omega_ref = vx * curvature
        # Slip angle of front/rear tires
        slip_f = -np.arctan(((state.rel_omega + omega_ref) * lf + vy) / vx) + control.steering
        slip_r = np.arctan(((state.rel_omega + omega_ref) * lr - vy) / vx)

        # Lateral forces from front and rear tires
        Ffy = tire_curve(slip_f) * m * 9.8 * lr / (lr + lf)
        Fry = 1.15 * tire_curve(slip_r) * m * 9.8 * lf / (lr + lf)

        # in body-aligned inertial frame
        d_vy_body = 1.0 / m * (Fry + Ffy - m * vx * (state.rel_omega + omega_ref))
        d_vx_body = 6.17 * (control.throttle - state.v_forward / 15.2 - 0.333)

        # in ref-curve-tangent-aligned inertial frame
        d_vx = d_vx_body*np.cos(state.rel_heading) - d_vy_body*np.sin(state.rel_heading)
        d_vy = d_vx_body*np.sin(state.rel_heading) + d_vy_body*np.cos(state.rel_heading)

        dsdt = d_vx /(1-state.lateral_err*curvature)
        dndt = d_vy
        dphidt = state.rel_omega


        # NOTE ignoring d_omega_ref_dt, i.e. curvature time rate
        d_omega = 1.0 / Iz * (Ffy * lf - Fry * lr)


        return CurvilinearState(
                                progress=state.progress + dsdt * dt,
                                lateral_err=state.lateral_err + dndt * dt,
                                rel_heading=state.rel_heading + dphidt * dt,
                                v_forward=state.v_forward + dvdt * dt,
                                v_sideway=state.v_sideway + d_vy * dt,
                                rel_omega=d_omega * dt
                                )
