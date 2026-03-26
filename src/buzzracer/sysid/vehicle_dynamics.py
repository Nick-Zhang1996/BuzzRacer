''' Base class for vehicle dynamics'''
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from buzzracer.types import CartesianState, CurvilinearState, Control


if TYPE_CHECKING:
    from buzzracer.cars.car import CarParam


class VehicleDynamics(ABC):
    """ Base class for vehicle dynamics model."""
    state_type = CartesianState

    @staticmethod
    @abstractmethod
    def advance_dynamics(state: CartesianState | CurvilinearState,
                         control: Control,
                         car_param: CarParam,
                         dt: float,
                         curvature: float = None) -> CartesianState | CurvilinearState:
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
        raise NotImplementedError
