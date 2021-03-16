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
from time import time

class LinearizeDynamics():
    def __init__(self,horizon):
        self.dt = 0.01
        self.sim = ethCarSim(0,0,0)
        self.n = 6
        # noise dimension
        self.l = self.n
        self.m = 2
        self.N = horizon
        return 

    # read a log
    # use trajectory of second lap as reference trajectory
    # this sets and saves reference state and trajectory
    def getRefTraj(self, logname, lap_no = 2, show=False):

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
        self.ref_traj = np.vstack([x[i:f],dx[i:f],y[i:f],dy[i:f],heading[i:f],dheading[i:f]]).T
        self.ref_ctrl = np.vstack([throttle[i:f],steering[i:f]]).T

        # plot the lap
        if show:
            plt.plot(x[i:f],y[i:f])
            plt.show()
        print(start_index)
        print(end_index)

        return

    def testGetRefTraj(self):
        self.getRefTraj("../log/ethsim/full_state1.p")
        return

    # differentiate dynamics around nominal state and control
    # return: A, B, d, s.t. x_k+1 = Ax + Bu + d
    def linearize(self, nominal_state, nominal_ctrl):
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

            A[:,i] += (x_post_r - x_post_l) / (2*epsilon)

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

            B[:,i] += (x_post_r - x_post_l) / (2*epsilon)

        x0 = nominal_state.copy()
        u0 = nominal_ctrl.copy()
        self.sim.states = np.array(x0.copy())
        self.sim.updateCar(self.dt,None,nominal_ctrl[0],nominal_ctrl[1])
        x_post = np.array(self.sim.states)
        # d = x_k+1 - Ak*x_k - Bk*u_k
        d = x_post - A @ x0 - B @ u0

        return A,B,d

    def testLinearize(self):
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
        i = 2210 + 100
        # use it as x0
        x0 = (x[i],dx[i],y[i],dy[i],heading[i],dheading[i])
        u0 = (throttle[i],steering[i])
        tic = time()
        A,B,d = self.linearize(x0,u0)
        print("time %.6f s"%(time()-tic))

        i = i
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

    # stolen from jacob's cs_model.py
    def form_long_matrices_LTV(self, A, B, d, D):
        nx = self.n
        nu = self.m
        # noise w dimension
        nl = self.n
        # horizon length
        N = self.N

        AA = np.zeros((nx*N, nx))
        BB = np.zeros((nx*N, nu * N))
        dd = np.zeros((nx*N, 1))
        DD = np.zeros((nx*N, nl * N))
        AA_i_row = np.eye(nx)
        dd_i_row = np.zeros((nx, 1))
        # B_i_row = zeros((nx, 0))
        # D_i_bar = zeros((nx, nx))
        for ii in np.arange(0, N):
            AA_i_row = np.dot(A[:, :, ii], AA_i_row)
            AA[ii*nx:(ii+1)*nx, :] = AA_i_row

            B_i_row = B[:, :, ii]
            D_i_row = D[:, :, ii]
            for jj in np.arange(ii-1, -1, -1):
                B_i_cell = np.dot(A[:, :, ii], BB[(ii-1)*nx:ii*nx, jj*nu:(jj+1)*nu])
                B_i_row = np.hstack((B_i_cell, B_i_row))
                D_i_cell = np.dot(A[:, :, ii], DD[(ii-1)*nx:ii*nx, jj*nl:(jj+1)*nl])
                D_i_row = np.hstack((D_i_cell, D_i_row))
            BB[ii*nx:(ii+1)*nx, :(ii+1)*nu] = B_i_row
            DD[ii*nx:(ii+1)*nx, :(ii+1)*nl] = D_i_row

            dd_i_row = np.dot(A[:, :, ii], dd_i_row) + d[:, :, ii]
            dd[ii*nx:(ii+1)*nx, :] = dd_i_row

        return AA, BB, dd, DD

    def makeBigMatrices(self, As, Bs, ds, Ds):
        n = self.n
        m = self.m
        N = self.N
        l = self.l

        AA = np.zeros((n*N,n))
        BB = np.zeros((n*N, m * N))
        dd = np.zeros((n*N,1))
        DD = np.zeros((n*N, l*N))

        AA_row_i = np.eye(n)
        # fill out row i of each matrix
        for i in range(N):
            AA_row_i = As[i] @ AA_row_i
            AA[n*i:n*(i+1),:] = AA_row_i

            BB_row_i = np.hstack([Bs[i] @ BB_row_i, Bs[i+1]])

        
        return


    # the model in log file is different from log in current model
    # re-generate reference trajectory using same x0 and uu
    def generateTestRefTraj(self):
        self.getRefTraj("../log/ethsim/full_state1.p",show=False)
        for i in range(self.ref_traj.shape[0]-1):
            x = self.ref_traj[i,:]
            self.sim.states = np.array(x)
            self.sim.updateCar(self.dt,None,self.ref_ctrl[i,0],self.ref_ctrl[i,1])
            x_post = np.array(self.sim.states)
            self.ref_traj[i+1,:] = x_post
        return


    def testBigMatrices(self):
        # get ref traj
        self.generateTestRefTraj()

        # linearize dynamics around ref traj
        As = []
        Bs = []
        ds = []
        Ds = []

        D = np.diag([0.1,0.3,0.1,0.3,radians(10),1.0])*self.dt
        # random starting point
        start = 100
        for i in range(start,start+self.N):
            A,B,d = self.linearize(self.ref_traj[i,:],self.ref_ctrl[i,:])
            if i == start:
                A0 = A
                B0 = B
                d0 = d
            if i == start+1:
                A1 = A
                B1 = B
                d1 = d
            if i == start+2:
                A2 = A
                B2 = B
                d2 = d

            As.append(A)
            Bs.append(B)
            ds.append(d)
            Ds.append(D)

        # make big matrices
        As = np.dstack(As)
        Bs = np.dstack(Bs)
        ds = np.dstack(ds).reshape((self.n,1,self.N))
        Ds = np.dstack(Ds)

        # propagate big matrices dynamics
        AA, BB, dd, DD = self.form_long_matrices_LTV(As, Bs, ds, Ds)

        # compare with actual result
        # X = AA x0 + BB u + C d + D noise
        i = start + 2
        #x0 = (x[i],dx[i],y[i],dy[i],heading[i],dheading[i])
        #u0 = (throttle[i],steering[i])
        x0 = self.ref_traj[i,:]
        u0 = self.ref_ctrl[i:i+1,:].flatten()
        uu = self.ref_ctrl[i:i+self.N,:].flatten()
        # actually need + DD @ w
        XX = AA @ x0 + BB @ uu + dd.flatten()

        xx_truth = self.ref_traj[i+1:i+1+self.N,:].flatten()
        x1 = A0 @ x0 + B0 @ u0 + d0
        '''
        print("x0")
        print(x0)
        print("xx_truth")
        print(xx_truth.reshape((self.N,-1)))
        print("x1")
        print(x1)
        '''

        '''
        # test A matrix
        tAA = np.zeros((self.n*self.N, self.n))
        tAA[:self.n,:] = A0
        tAA[self.n:2*self.n,:] = A1 @ A0
        #assert np.linalg.norm(tAA-AA) < 1e-5

        # test B matrix
        tBB = np.zeros((self.n*self.N, self.m*self.N))
        tBB[:self.n,:self.m] = B0
        tBB[self.n:2*self.n,:self.m] = A1 @ B0
        tBB[self.n:2*self.n,self.m:2*self.m] = B1
        #assert np.linalg.norm(tBB-BB) < 1e-5

        # test vector d
        tdd = np.zeros((self.n*self.N,1))
        tdd[:self.n,:] = d0.reshape((-1,1))
        tdd[self.n:2*self.n,:] = A1 @ d0.reshape((-1,1)) + d1.reshape((-1,1))

        #assert np.linalg.norm(tdd-dd) < 1e-5
        '''
        
        print("diff")
        print(np.linalg.norm(XX-xx_truth))
        print(XX[-self.n:])
        print(xx_truth[-self.n:])

        # plot true and linearized traj
        img_track = self.track.drawTrack()
        img_track = self.track.drawRaceline(img=img_track)
        car_state = (x0[0],x0[2],x0[4],0,0,0)
        img = track.drawCar(img_track.copy(), car_state, steering)

        actual_future_traj  = self.ref_traj[i+1:i+1+self.N,(0,2)]
        #actual_future_traj = np.vstack([x[i:i+lookahead_steps],y[i:i+lookahead_steps]]).T
        img = track.drawPolyline(actual_future_traj,lineColor=(255,0,0),img=img.copy())
        plt.imshow(img)
        plt.show()
        


if __name__ == '__main__':
    main = LinearizeDynamics(20)
    #main.testGetRefTraj()
    #main.testLinearize()
    main.testBigMatrices()
