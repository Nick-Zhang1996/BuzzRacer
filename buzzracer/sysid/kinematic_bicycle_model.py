''' Kinematic bicycle model with pacjka tire model'''
# pylint: disable-next=line-too-long
# Reference: https://ftp.idu.ac.id/wp-content/uploads/ebook/tdg/TERRAMECHANICS%20AND%20MOBILITY/epdf.pub_vehicle-dynamics-and-control-2nd-edition.pdf
import numpy as np

from buzzracer.types import CartesianState, CurvilinearState, Control
from buzzracer.cars.car import Car
from buzzracer.sysid.vehicle_dynamics import VehicleDynamics

class KinematicBicycleModelCartesian(VehicleDynamics):
    ''' Kinematic Bicycle Model (Cartesian frame)
    Follows Vehicle Dynamics and Control, 2nd Edition, Sec 2.2'''

    @staticmethod
    def advance_dynamics(state: CartesianState, control: Control,
                         car: Car, dt: float, curvature: float=None) -> CartesianState:
        ''' Step dynamics forward by dt, x+ = x + f(x,u)*dt

        Args:
            state: Current state of the vehicle
            control: Control for the vehicle
            car: Car object to supply vehicle sysid parameters like mass, Iz, wheelbase
            dt: time step in seconds e.g. 0.01
            curvature: signed curvature of ref curve, ccw positive (only used for CurvilinearState)
        Return:
            state at next timestep

        '''

        del curvature
        beta = np.arctan(np.tan(control.steering) * car.lr / (car.lf + car.lr))
        dxdt = state.v_forward * np.cos(state.heading + beta)
        dydt = state.v_forward * np.sin(state.heading + beta)
        dvdt = 6.17 * (control.throttle - state.v_forward / 15.2 - 0.333)
        dheadingdt = state.v_forward * np.cos(beta) / (car.lf + car.lr) * np.tan(control.steering)

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

        beta = np.arctan(np.tan(control.steering) * car.lr / (car.lf + car.lr))

        dsdt = state.v_forward*np.cos(state.rel_heading)/(1-state.lateral_err*curvature)
        dndt = state.v_forward*np.sin(state.rel_heading)
        # Origin at rear axle center
        # dphidt = state.v_forward * (
        #               np.tan(control.steering) / (car.lf+car.lr)
        #               - curvature*np.cos(state.rel_heading)/(1-curvature*state.lateral_err)
        #               )
        # Origin at CG
        dphidt = state.v_forward * (
            np.cos(beta) / (car.lf + car.lr) * np.tan(control.steering)
            - curvature*np.cos(state.rel_heading)/(1-curvature*state.lateral_err)
            )
        dvdt = 6.17 * (control.throttle - state.v_forward / 15.2 - 0.333)

        return CurvilinearState(
                                progress=state.progress + dsdt * dt,
                                lateral_err=state.lateral_err + dndt * dt,
                                rel_heading=state.rel_heading + dphidt * dt,
                                v_forward=state.v_forward + dvdt * dt,
                                v_sideway=0,
                                rel_omega=0
                                )
