''' Simulator for Ackerman steering vehicle with Kinematic Bicycle Model
Refer to paper
The Kinematic Bicycle Model: 
    a Consistent Model for Planning Feasible Trajectories for Autonomous Vehicles
'''
import numpy as np
from buzzracer.extensions.simulator import Simulator
from buzzracer.types import CartesianState, Control
from buzzracer.sysid.kinematic_bicycle_model import KinematicBicycleModelCartesian

class KinematicSimulator(Simulator):
    ''' Simulator for Ackerman steering vehicle with Kinematic Bicycle Model '''

    max_v = 3.0
    simple_throttle_model = False
    ''' If True, throttle is the acceleration without mapping'''

    def __init__(self):
        super().__init__()
        KinematicSimulator.dt = self.main.dt

        # for when a specific car instance is not speciied
        self.lr = 45e-3
        self.lf = 45e-3
        self.simple_throttle_model = False

    def init(self):
        super().init()
        KinematicSimulator.simple_throttle_model = self.simple_throttle_model

        for car in self.main.cars:
            self.add_car(car)

        self.main.new_state_update.set()

    @staticmethod
    def advance_dynamics(car_states, control, car, dt):
        """advance dynamics by dt.

        Args:
            car_states: Cartesian state of the car, (x,y,heading,v_forward,v_sideway,omega)
            control: (steering,throttle) steering in rad, left positive, throttle in [-1,1], 
                    positive indicates acceleration
            car: Car object, contains information about the car's kinematics, 
                also contains car.sim_states for simulators that do not use car.states for update
            dt: Time step to advance dynamics by, unit:seconds
        Return: 
            state at next time step.
        """
        x, y, heading, vx, vy, omega = car_states
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
