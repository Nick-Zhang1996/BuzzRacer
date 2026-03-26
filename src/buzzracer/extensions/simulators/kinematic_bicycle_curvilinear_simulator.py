''' Simulate vehicle dynamics in Curvilinear/Frenet reference frame with Kinematic Bicycle Model '''
from __future__ import annotations
from typing import TYPE_CHECKING

from buzzracer.extensions.simulator import Simulator, SimulatorConfig
from buzzracer.extensions.extension import Extension, ExtensionState
from buzzracer.types import CartesianState, CurvilinearState, Control
from buzzracer.sysid.kinematic_bicycle_model import KinematicBicycleModelFrenet
if TYPE_CHECKING:
    from buzzracer.cars.car import CarParam


@Extension.register('simulator', SimulatorConfig, ExtensionState)
class KinematicBicycleCurvilinearSimulator(Simulator):
    ''' Simulate vehicle dynamics in Curvilinear/Frenet reference frame 
        with Kinematic Bicycle Model
    '''
    state_type = CurvilinearState

    def __init__(self, config, state):
        super().__init__(config, state)
        self.track = self.main.track

    def init(self):
        super().init()
        for car in self.main.cars:
            self.add_car(car)
        self.main.new_state_update.set()

    def add_car(self, car):
        """ Register a car to use this simulation. 

        from car.state =  (x,y,heading,v_forward,v_sideway,omega)
        initialize car.sim_state: CurvilinearState
        """
        super().add_car(car)
        cart_state = CartesianState(*car.state)
        if self.state_type == CurvilinearState:
            curv_state = self.main.track.cart_to_curv(cart_state)
            car.sim_state = curv_state
        elif self.state_type == CartesianState:
            car.sim_state = cart_state
        else:
            raise RuntimeError(f'unknown state type {self.state_type}')

        car.state_dim = 6
        car.control_dim = 2

    @staticmethod
    def advance_dynamics(state: CurvilinearState,
                         control: Control,
                         car_param: CarParam,
                         dt: float,
                         curvature: float = None):
        """advance dynamics by self.dt.
        using car frame origined at CG with x pointing forward, y leftward
        Args:
            state: state of the car
            control: (steering,throttle) steering in rad, left positive, throttle in [-1,1], 
                    positive indicates acceleration
            car_param: CarParam object, contains information about the car's kinematics, 
                also contains car.sim_state for simulators that do not use car.state for update
            dt: Time step to advance dynamics by, unit:seconds
            curvature: signed curvature
        Return: 
            state at next time step.
        """
        return KinematicBicycleModelFrenet.advance_dynamics(state, control, car_param, dt, curvature)
