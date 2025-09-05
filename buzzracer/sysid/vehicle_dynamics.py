''' Base class for vehicle dynamics'''
from abc import ABC, abstractmethod
from buzzracer.types import CartesianState, CurvilinearState, Control
from buzzracer.cars.car import Car

class VehicleDynamics(ABC):
    """ Base class for vehicle dynamics model."""

    def __init__(self):
        self.curvilinear = None
        ''' If True, then use CurvilinearState, else use CartesianState'''

    @staticmethod
    @abstractmethod
    def advance_dynamics(state: CartesianState | CurvilinearState,
                         control: Control, car: Car,
                         dt: float) -> CartesianState | CurvilinearState:
        ''' Step dynamics forward by dt, x+ = x + f(x,u)*dt

        Args:
            state: Current state of the vehicle
            control: Control for the vehicle
            car: Car object to supply vehicle sysid parameters like mass, Iz, wheelbase
            dt: time step in seconds e.g. 0.01
        Return:
            states at next timestep
        '''
        raise NotImplementedError

