# dynamic simulator of a passenger vehicle
# page 30 of book vehicle dynamics and control
import numpy as np
from math import sin,cos,tan,radians,degrees,pi
import matplotlib.pyplot as plt
from Simulator import Simulator
from common import *
from threading import Event,Lock

class DynamicSimulator(Simulator):
    def __init__(self,main):
        super().__init__(main)
        DynamicSimulator.max_v = 3.0

    def init(self):
        super().init()
        self.cars = self.main.cars
        DynamicSimulator.dt = self.main.dt
        for car in self.cars:
            self.addCar(car)
        self.main.new_state_update = Event()
        self.main.new_state_update.set()

    # add a car to be DynamicSimu  
    # car needs to (x,y,heading,v_forward,v_sideway,omega)
    def addCar(self,car):
        x,y,heading,v_forward,v_sideway,omega = car.states
        car.Vx = v_forward
        car.Vy = v_sideway

        car.x = x
        car.y = y
        car.psi = heading

        car.d_x = car.Vx*cos(car.psi)-car.Vy*sin(car.psi)
        car.d_y = car.Vx*sin(car.psi)+car.Vy*cos(car.psi)
        car.d_psi = 0
        car.sim_states = np.array([car.x,car.d_x,car.y,car.d_y,car.psi,car.d_psi])

        car.state_dim = 6
        car.control_dim = 2
        
        # not implemented: support for artificially added noise
        noise = False
        car.noise = noise
        if noise:
            car.noise_cov = noise_cov
            assert np.array(noise_cov).shape == (6,6)

        #car.states_hist = []
        car.local_states_hist = []
        car.norm = []


    # advance vehicle dynamics
    # NOTE using car frame origined at CG with x pointing forward, y leftward
    # this method does NOT update car.sim_states, only returns a sim_state
    # this is to make itself useful for when update is not necessary
    @staticmethod
    def advanceDynamics(car_states, control, car):
        dt = DynamicSimulator.dt
        x,y,psi,v_forward,v_sideway,d_psi = car_states

        d_x = v_forward*cos(psi)-v_sideway*sin(psi)
        d_y = v_forward*sin(psi)+v_sideway*cos(psi)
        sim_states = np.array([x,d_x,y,d_y,psi,d_psi])

        throttle, steering = control
        u = np.array([throttle,steering])
        # NOTE page 30 of book vehicle dynamics and control
        A = np.array([[0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, -(2*car.Caf+2*car.Car)/(car.m*v_forward), 0, -v_forward-(2*car.Caf*car.lf-2*car.Car*car.lr)/(car.m*v_forward)],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, -(2*car.lf*car.Caf-2*car.lr*car.Car)/(car.Iz*v_forward), 0, -(2*car.lf**2*car.Caf+2*car.lr**2*car.Car)/(car.Iz*v_forward)]])
        B = np.array([[0,1,0,0,0,0],[0,0,0,2*car.Caf/car.m,0,2*car.lf*car.Caf/car.Iz]]).T

        # vehicle frame: origin at CG, x forward y leftward
        # active roattion matrix of angle(rad)
        R = lambda angle: np.array([[cos(angle), 0,-sin(angle),0,0,0],
                        [0, cos(angle), 0,-sin(angle),0,0],
                        [sin(angle),0,cos(angle),0,0,0],
                        [0,sin(angle),0,cos(angle),0,0],
                        [0,0,0,0,1,0],
                        [0,0,0,0,0,1]])

        # dynamics (A and B) work in vehicle frame
        # we use R() to convert state from global frame to vehicle frame
        # then we apply A,B
        # finally we convert state back to track/world frame
        if (car.noise):
            plant_noise = np.random.multivariate_normal([0.0]*car.state_dim, car.noise_cov, size=1).flatten()
            sim_states = sim_states + R(psi) @ (A @ R(-psi) @ sim_states + B @ u + plant_noise)*dt
        else:
            sim_states = sim_states + R(psi) @ (A @ R(-psi) @ sim_states + B @ u)*dt
        #car.states_hist.append(sim_states)
        #car.local_states_hist.append(R(-psi)@sim_states)

        x = sim_states[0]
        y = sim_states[2]
        psi = sim_states[4] # heading
        # longitidunal,velocity forward positive
        v_forward = sim_states[1] *cos(psi) + sim_states[3] *sin(psi)
        # lateral, sideway velocity, left positive
        v_sideway = -sim_states[1] *sin(psi) + sim_states[3] *cos(psi)
        omega = sim_states[5]
        car_states = x,y,psi,v_forward,v_sideway,omega
        return np.array(car_states)

    def update(self): 
        #print_ok(self.prefix() + "update")
        for car in self.cars:
            car.states = self.advanceDynamics(car.states, (car.throttle, car.steering), car)
            #print(self.prefix()+str(car.states))
        self.main.new_state_update.set()
        self.main.sim_t += self.main.dt
        self.matchRealTime()

