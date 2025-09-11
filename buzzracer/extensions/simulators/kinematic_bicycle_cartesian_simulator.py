''' Simulator for Ackerman steering vehicle with Kinematic Bicycle Model
Refer to paper
The Kinematic Bicycle Model: 
    a Consistent Model for Planning Feasible Trajectories for Autonomous Vehicles
'''
import numpy as np
from buzzracer.extensions.simulator import Simulator
from buzzracer.types import CartesianState, Control
from buzzracer.cars.car import Car
from buzzracer.sysid.kinematic_bicycle_model import KinematicBicycleModelCartesian


class KinematicBicycleCartesianSimulator(Simulator):
    ''' Simulator for Ackerman steering vehicle with Kinematic Bicycle Model '''

    max_v = 3.0
    simple_throttle_model = False
    state_type = CartesianState
    ''' If True, throttle is the acceleration without mapping'''

    def __init__(self):
        super().__init__()
        KinematicBicycleCartesianSimulator.dt = self.main.dt

        # for when a specific car instance is not speciied
        self.lr = 45e-3
        self.lf = 45e-3
        self.simple_throttle_model = False

    def init(self):
        super().init()
        KinematicBicycleCartesianSimulator.simple_throttle_model = self.simple_throttle_model

        for car in self.main.cars:
            self.add_car(car)

        self.main.new_state_update.set()

    @staticmethod
    def advance_dynamics(state:  CartesianState,
                         control: Control,
                         car: Car,
                         dt: float,
                         curvature: float = None) -> CartesianState:
        """advance dynamics by dt.

        Args:
            state: state of the car, may be CartesianState or CurvilinearState
            control: (steering,throttle) steering in rad, left positive, throttle in [-1,1], 
                    positive indicates acceleration
            car: Car object, contains information about the car's kinematics, 
                also contains car.sim_state for simulators that do not use car.state for update
            dt: Time step to advance dynamics by, unit:seconds
            curvature: unused, only for curvilinear
        Return: 
            state at next time step.
        """
        x, y, heading, vx, vy, omega = state
        _state = CartesianState(x=x,
                                y=y,
                                heading=heading,
                                v_forward=vx,
                                v_sideway=vy,
                                omega=omega)
        steering, throttle = control
        _control = Control(steering=steering, throttle=throttle)
        next_car_state = KinematicBicycleModelCartesian.advance_dynamics(_state, _control, car, dt)
        return np.array(next_car_state)
