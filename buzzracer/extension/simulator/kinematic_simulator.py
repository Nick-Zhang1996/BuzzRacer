''' Simulator for Ackerman steering vehicle with Kinematic Bicycle Model
Refer to paper
The Kinematic Bicycle Model: 
    a Consistent Model for Planning Feasible Trajectories for Autonomous Vehicles
'''
import numpy as np
from buzzracer.extension.Simulator import Simulator

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
        lr = car.lr
        lf = car.lf
        dt = KinematicSimulator.dt

        x, y, heading, v_forward, v_sideway, omega = car_states
        v = v_forward
        throttle = control[1]
        steering = control[0]

        beta = np.arctan(np.tan(steering) * lr / (lf+lr))
        dXdt = v * np.cos(heading + beta)
        dYdt = v * np.sin(heading + beta)
        try:
            if KinematicSimulator.simple_throttle_model:
                if (v > KinematicSimulator.max_v):
                    dvdt = -0.01
                else:
                    dvdt = throttle
            else:
                dvdt = 6.17*(throttle - v/15.2 - 0.333)
        except AttributeError:
            dvdt = 6.17*(throttle - v/15.2 - 0.333)
        omega = dheadingdt = v/lr*np.sin(beta)

        x += dt * dXdt
        y += dt * dYdt
        v += dt * dvdt
        heading += dt * dheadingdt

        v_forward = v
        v_sideway = 0
        car_states = x, y, heading, v_forward, v_sideway, omega
        return np.array(car_states)
