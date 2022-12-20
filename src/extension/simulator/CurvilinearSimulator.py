# use curvilinear ref frame dynamics from copg

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from extension import Simulator

from sysid.tire import tireCurve
import numpy as np
from math import sin,cos,tan,radians,degrees,pi,atan
import matplotlib.pyplot as plt
from Simulator import Simulator
from common import *
from threading import Event,Lock
from simulator.KinematicSimulator import KinematicSimulator
from copg.rcvip_simulator.VehicleModel import VehicleModel

class CurvilinearSimulator(Simulator):
    def __init__(self,main):
        super().__init__(main)

    def init(self):
        super().init()

        self.cars = self.main.cars
        CurvilinearSimulator.dt = self.main.dt
        KinematicSimulator.dt = CurvilinearSimulator.dt
        KinematicSimulator.max_v = 100
        for car in self.cars:
            self.addCar(car)
        self.main.new_state_update.set()

        CurvilinearSimulator.vehicle_model = VehicleModel(1,'cpu','rcp',dt=self.main.dt)

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
    #    x,y,psi,v_forward,v_sideway,d_psi = car_states
    @staticmethod
    def advanceDynamics(car_states, control, car):
        #x,y,psi,v_forward,v_sideway,d_psi = car_states
        local_state = CurvilinearSimulator.vehicle_model.fromGlobalToLocal(car_states)
        # control = throttle, steering
        new_local_state = CurvilinearSimulator.vehicle_model.dynModelBlendBatch(local_state, control)
        global_state = CurvilinearSimulator.vehicle_model.fromLocalToGlobal(new_local_state).flatten()
        return global_state.flatten()

    def update(self): 
        #print_ok(self.prefix() + "update")
        for car in self.cars:
            car.states = self.advanceDynamics(car.states, (car.throttle, car.steering), car)
            #print(self.prefix()+str(car.states))
            #print(self.prefix()+"T: %.1f, S:%.1f"%(car.throttle, degrees(car.steering)))
        if (self.state_noise_enabled):
            self.addStateNoise()
        self.main.new_state_update.set()
        self.main.sim_t += self.main.dt
        self.matchRealTime()

