# dynamic simulator of a passenger vehicle
# page 30 of book vehicle dynamics and control
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from extension import Simulator

import numpy as np
from math import sin,cos,tan,radians,degrees,pi,atan
import matplotlib.pyplot as plt
from Simulator import Simulator
from common import *
from threading import Event,Lock
from simulator.KinematicSimulator import KinematicSimulator

class DynamicSimulator(Simulator):
    def __init__(self,main):
        super().__init__(main)
        DynamicSimulator.max_v = 3.0
        DynamicSimulator.using_kinematics = False

    def init(self):
        super().init()
        self.cars = self.main.cars
        DynamicSimulator.dt = self.main.dt
        KinematicSimulator.dt = DynamicSimulator.dt
        KinematicSimulator.max_v = 100
        for car in self.cars:
            self.addCar(car)
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
    #    x,y,psi,v_forward,v_sideway,d_psi = car_states
    @staticmethod
    def advanceDynamics(car_states, control, car):
        lf = car.lf
        lr = car.lr
        L = car.L

        Df = car.Df
        Dr = car.Dr
        B = car.B
        C = car.C
        Cm1 = car.Cm1
        Cm2 = car.Cm2
        Cr = car.Cr
        Cd = car.Cd
        Iz = car.Iz
        m = car.m
        dt = DynamicSimulator.dt

        # NOTE here vx = vf, vy = vs, different convention
        x,y,heading,vx,vy,omega = car_states
        throttle, steering = control

        # for small longitudinal velocity use kinematic model
        if (vx<0.05):
            beta = atan(lr/L*tan(steering))
            norm = lambda a,b:(a**2+b**2)**0.5
            # motor model
            d_vx = 0.425*(15.2*throttle - vx - 3.157)
            vx = vx + d_vx * dt
            vy = norm(vx,vy)*sin(beta)
            d_omega = 0.0
            omega = vx/L*tan(steering)

            slip_f = 0
            slip_r = 0
            Ffy = 0
            Fry = 0

        else:
            slip_f = -np.arctan((omega*lf + vy)/vx) + steering
            slip_r = np.arctan((omega*lr - vy)/vx)

            Ffy = Df * np.sin( C * np.arctan(B *slip_f)) * 9.8 * lr / (lr + lf) * m
            Fry = Dr * np.sin( C * np.arctan(B *slip_r)) * 9.8 * lf / (lr + lf) * m

            # Dynamics
            #d_vx = 1.0/m * (Frx - Ffy * np.sin( steering ) + m * vy * omega)
            d_vx = 0.425*(15.2*throttle - vx - 3.157)
            d_vy = 1.0/m * (Fry + Ffy * np.cos( steering ) - m * vx * omega)
            d_omega = 1.0/Iz * (Ffy * lf * np.cos( steering ) - Fry * lr)

            # discretization
            vx = vx + d_vx * dt
            vy = vy + d_vy * dt
            omega = omega + d_omega * dt 

        # back to global frame
        vxg = vx*cos(heading)-vy*sin(heading)
        vyg = vx*sin(heading)+vy*cos(heading)

        # update x,y, heading
        x += vxg*dt
        y += vyg*dt
        heading += omega*dt + 0.5* d_omega * dt * dt

        car_states = x,y,heading,vx,vy,omega
        return np.array(car_states)


    def update(self): 
        #print_ok(self.prefix() + "update")
        for car in self.cars:
            car.states = self.advanceDynamics(car.states, (car.throttle, car.steering), car)
            #print(self.prefix()+str(car.states))
            #print(self.prefix()+"T: %.1f, S:%.1f"%(car.throttle, degrees(car.steering)))
        self.main.new_state_update.set()
        self.main.sim_t += self.main.dt
        self.matchRealTime()

