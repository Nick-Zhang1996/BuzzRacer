''' Kinematic bicycle model with pacjka tire model'''
import numpy as np

from buzzracer.types import CartesianState, CurvilinearState, Control
from buzzracer.cars.car import Car
from buzzracer.sysid.vehicle_dynamics import VehicleDynamics

class KinematicBicycleModel(VehicleDynamics):
    ''' Kinematic Bicycle Model
    Follows Vehicle Dynamics and Control, 2nd Edition, Sec 2.2'''

    @staticmethod
    def advance_dynamics(state: CartesianState, control: Control,
                         car: Car, dt: float) -> CartesianState:
        ''' Step dynamics forward by dt, x+ = x + f(x,u)*dt

        Args:
            state: Current state of the vehicle
            control: Control for the vehicle
            car: Car object to supply vehicle sysid parameters like mass, Iz, wheelbase
            dt: time step in seconds e.g. 0.01
        Return:
            state at next timestep

        '''

        beta = np.arctan(np.tan(control.steering) * car.lr / (car.lf + car.lr))
        dxdt = state.v_forward * np.cos(state.heading + beta)
        dydt = state.v_forward * np.sin(state.heading + beta)
        dvdt = 6.17 * (control.throttle - state.v_forward / 15.2 - 0.333)
        omega = dheadingdt = state.v_forward * np.cos(beta) / (car.lf + car.lr) * np.tan(control.steering)

        x = state.x + dt * dxdt
        y = state.y + dt * dydt
        v = state.v_forward + dt * dvdt
        heading = state.heading + dt * dheadingdt
        return CartesianState(x=x,
                              y=y,
                              heading=heading,
                              v_forward=v,
                              v_sideway=0,
                              omega=omega)
