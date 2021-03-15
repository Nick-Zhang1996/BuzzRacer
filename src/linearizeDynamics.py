# functions to generate a linearized dynamics model from reference trajectory and control
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from common import *
from laptimer import Laptimer
from RCPTrack import RCPtrack
from math import pi,radians,degrees,asin,acos,isnan
from ethCarSim import ethCarSim

class LinearizeDynamics():
    def __init__(self):
        self.dt = 0.01
        self.sim = ethCarSim(0,0,0)
        self.n = 6
        self.m = 2
        return 

    # read a log
    # use trajectory of second lap as reference trajectory
    # this sets and saves reference state and trajectory
    def getRefTraj(self, logname, lap_no = 2):

        # read log
        with open(logname, 'rb') as f:
            data = pickle.load(f)
        data = np.array(data)

        # data.shape = time_seq,car_no,data_entry_id
        data = data.squeeze(1)

        t = data[:,0]
        t = t-t[0]
        x = data[:,1]
        y = data[:,2]
        heading = data[:,3]
        # calculate speed from pos, to use as no-bias but noise reference
        dt = 0.01
        dx = np.diff(x)
        dy = np.diff(y)
        dheading = np.diff(heading)
        steering = data[:,4]
        throttle = data[:,5]

        '''
        exp_kf_x = data[:,6]
        exp_kf_y = data[:,7]
        exp_kf_v = data[:,8]
        exp_kf_theta = data[:,9]
        exp_kf_omega = data[:,10]
        '''

        # search for log_no lap
        self.track = RCPtrack()
        self.track.startPos = (0.6*3.5,0.6*1.75)
        self.track.startDir = radians(90)
        self.track.load()
        laptimer = Laptimer(self.track.startPos, self.track.startDir)
        current_lap = 0
        index = 0
        start_index = -1
        end_index = -1
        
        try:
            while (current_lap < lap_no + 1):
                retval = laptimer.update((x[index],y[index]),t[index])
                if retval:
                    current_lap += 1
                    if (current_lap == lap_no):
                        start_index = index
                    if (current_lap == lap_no+1):
                        end_index = index
                        self.ref_laptime = t[end_index] - t[start_index]
                index += 1
        except IndexError:
            print_error("specified lap not found in log, maybe not long enough")
        print(" reference laptime %.2fs "%(self.ref_laptime))

        # assemble ref traj
        # state: [X,dX,Y,dY,phi,omega]
        # control: [throttle, steering]

        i = start_index
        f = end_index
        self.ref_traj = np.vstack([x[i:f],dx[i:f],y[i:f],dy[i:f],heading[i:f],dheading[i:f]])
        self.ref_ctrl = np.vstack([throttle[i:f],steering[i:f]])

        # plot the lap
        plt.plot(x[i:f],y[i:f])
        plt.show()

        return

    def testGetRefTraj(self):
        self.getRefTraj("../log/ethsim/full_state1.p")
        return

    # differentiate dynamics around nominal state and control
    # return: A, B, d, s.t. x_k+1 = Ax + Bu + d
    def diff(self, nominal_state, nominal_ctrl):
        nominal_state = np.array(nominal_state)
        nominal_ctrl = np.array(nominal_ctrl)
        epsilon = 1e-3

        # A = df/dx * dt
        A = np.zeros((self.n,self.n),dtype=np.float)
        # find A
        for i in range(self.n):
            # d x / d x_i, ith row in A
            x_l = nominal_state.copy()
            x_l[i] -= epsilon
            self.sim.states = np.array(x_l)
            self.sim.updateCar(self.dt,None,nominal_ctrl[0],nominal_ctrl[1])
            x_post_l = np.array(self.sim.states)


            x_r = nominal_state.copy()
            x_r[i] += epsilon
            self.sim.states = np.array(x_r)
            self.sim.updateCar(self.dt,None,nominal_ctrl[0],nominal_ctrl[1])
            x_post_r = np.array(self.sim.states)

            A[:,i] += (x_post_r - x_post_l) / (2*epsilon) * self.dt

        # B = df/du * dt
        B = np.zeros((self.n,self.m),dtype=np.float)
        # find A
        for i in range(self.m):
            # d x / d u_i, ith row in B
            x0 = nominal_state.copy()

            u_l = nominal_ctrl.copy()
            u_l[i] -= epsilon
            self.sim.updateCar(self.dt,None,*u_l)
            x_post_l = np.array(self.sim.states)

            x0 = nominal_state.copy()
            u_r = nominal_ctrl.copy()
            u_r[i] += epsilon
            self.sim.updateCar(self.dt,None,*u_r)
            x_post_r = np.array(self.sim.states)

            B[:,i] += (x_post_r - x_post_l) / (2*epsilon) * self.dt

        x0 = nominal_state.copy()
        u0 = nominal_ctrl.copy()
        self.sim.states = np.array(x0.copy())
        self.sim.updateCar(self.dt,None,nominal_ctrl[0],nominal_ctrl[1])
        x_post = np.array(self.sim.states)
        # d = x_k+1 - Ak*x_k - Bk*u_k
        d = x_post - A @ x0 - B @ u0

        return A,B,d

    def testDiff(self):
        # compare F(x0+dx,u0+du) and A(x0+dx) + B(u0+du) + d
        # read log
        logname = "../log/ethsim/full_state1.p"
        with open(logname, 'rb') as f:
            data = pickle.load(f)
        data = np.array(data)

        # data.shape = time_seq,car_no,data_entry_id
        data = data.squeeze(1)

        t = data[:,0]
        t = t-t[0]
        x = data[:,1]
        y = data[:,2]
        heading = data[:,3]
        # calculate speed from pos, to use as no-bias but noise reference
        dt = 0.01
        dx = np.diff(x)
        dy = np.diff(y)
        dheading = np.diff(heading)
        steering = data[:,4]
        throttle = data[:,5]

        #pick a random frame
        i = 1234
        # use it as x0
        x0 = (x[i],dx[i],y[i],dy[i],heading[i],dheading[i])
        u0 = (throttle[i],steering[i])
        A,B,d = self.diff(x0,u0)

        i = i + 2
        x_a_prior = (x[i],dx[i],y[i],dy[i],heading[i],dheading[i])
        u_a = (throttle[i],steering[i])
        self.sim.states = np.array(x_a_prior)
        self.sim.updateCar(self.dt,None,u_a[0],u_a[1])
        x_a_post_truth = np.array(self.sim.states)

        x_a_post_guess = A @ x_a_prior + B @ u_a + d

        print("x0")
        print(x0)
        print("x")
        print(x_a_prior)
        print("real post x")
        print(x_a_post_truth)
        print("guess post x")
        print(x_a_post_guess)
        print("diff")
        print(x_a_post_guess - x_a_post_truth)


if __name__ == '__main__':
    main = LinearizeDynamics()
    #main.testGetRefTraj()
    main.testDiff()
