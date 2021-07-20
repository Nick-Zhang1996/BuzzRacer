# refer to paper
# The Kinematic Bicycle Model: a Consistent Model for Planning Feasible Trajectories for Autonomous Vehicles?
import numpy as np
from math import radians
from common import *
from Extension import Extension
from threading import Event,Lock

class KinematicSimulator(Extension):

    def __init__(self,main):
        Extension.__init__(self,main)
        print_ok("[KinematicSimulator]: in use")

        # for when a specific car instance is not speciied
        self.lr = 45e-3
        self.lf = 45e-3

        # ensure experiment_type hasn't been initialized
        flag_is_unique = False
        try:
            main.experiment_type != ExperimentType.Simulation
        except (AttributeError):
            flag_is_unique = True
        if (not flag_is_unique):
            print_error("[KinematicSimulator]: another state update source has been initialized")

        main.experiment_type = ExperimentType.Simulation
        main.sim_t = 0
        self.cars = self.main.cars
        for car in self.cars:
            self.addCar(car)

    # add a car to be KinematicSimulator
    # car needs to have .lf, .lr, .L .states (x,y,heading,v_forward,v_sideways,omega)
    def addCar(self,car):
        x,y,heading,v_forward,v_sideways,omega = car.states
        car.sim_states = np.array([x,y,v_forward,heading])

        car.new_state_update = Event()
        car.new_state_update.set()

        car.steering = steering = 0
        car.throttle = throttle = 0

    def update(self): 
        #print_ok("[KinematicSimulator]: update")
        for car in self.cars:
            car.sim_states = self.advanceDynamics(car.sim_states, (car.throttle, car.steering), car)
            x,y,v,heading = car.sim_states
            car.states = (x,y,heading,v,0,0)
            car.new_state_update.set()
        self.main.sim_t += self.main.dt

    @staticmethod
    def advanceDynamics(sim_states,control, car=None):
        if (car is None):
            lr = self.lr
            lf = self.lf
        else:
            lr = car.lr
            lf = car.lf
        
        '''
        throttle = np.clip(throttle, -1.0, 1.0)
        steering = np.clip(throttle, -radians(27), radians(27))
        '''
        x,y,v,heading = sim_states
        throttle = control[0]
        steering = control[1]
        dt = KinematicSimulator.main.dt

        beta = np.arctan( np.tan(steering) * lr / (lf+lr))
        dXdt = v * np.cos( heading + beta )
        dYdt = v * np.sin( heading + beta )
        dvdt = throttle
        dheadingdt = v/lr*np.sin(beta)

        x += dt * dXdt
        y += dt * dYdt
        v += dt * dvdt
        heading += dt * dheadingdt
        return np.array([x,y,v,heading])

