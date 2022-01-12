# CCMPPI for dynamic bicycle model

import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from common import *
from laptimer import Laptimer
from RCPTrack import RCPTrack
from math import pi,radians,degrees,asin,acos,isnan
from ethCarSim import ethCarSim
from time import time
from cs_solver import CSSolver #from cs_solver_covariance_only import CSSolver
import cvxpy as cp
from cvxpy.atoms.affine.trace import trace 
from cvxpy.atoms.affine.transpose import transpose

from RCPTrack import RCPTrack

class CCMPPI():
    def __init__(self):
        # set time horizon
        self.N = 4
        self.n = 4
        self.m = 1
        self.l = self.n

        self.dt = 0.1
        self.Sigma_epsilon = np.array([[1.0]])

        # set up parameters for the model
        self.setupParam()
        # load track 
        self.loadTrack()

    def setupParam(self):
        # dimension
        self.lf = lf = 0.09-0.036
        self.lr = lr = 0.036
        self.L = 0.09
        # basic properties
        self.Iz = 0.00278
        self.mass = m = 0.1667

        # tire model
        self.Df = Df = 3.93731
        self.Dr = Dr = 6.23597
        self.C = C = 2.80646
        self.B = B = 0.51943

        
        self.Caf = Df *  C * B * 9.8 * lr / (lr + lf) * m
        self.Car = Dr *  C * B * 9.8 * lr / (lr + lf) * m

    def loadTrack(self):
        # full RCP track
        # NOTE load track instead of re-constructing
        fulltrack = RCPTrack()
        # for laptimer
        fulltrack.startPos = (0.6*3.5,0.6*1.75)
        fulltrack.startDir = radians(90)
        fulltrack.load()
        self.track = fulltrack
        return 

    def nearest_spd_cholesky(self,A):
        # print(np.linalg.eigvals(A))
        B = (A + A.T)/2
        U, Sigma, V = np.linalg.svd(B)
        H = np.dot(np.dot(V.T, np.diag(Sigma)), V)
        Ahat = (B+H)/2
        Ahat = (Ahat + Ahat.T)/2
        p = 1
        k = 0
        spacing = np.spacing(np.linalg.norm(A))
        I = np.eye(A.shape[0])
        while p != 0:
            k += 1
            try:
                R = np.linalg.cholesky(Ahat)
                p = 0
            except np.linalg.LinAlgError:
                eig = np.linalg.eigvals(Ahat)
                # print(eig)
                mineig = np.min(np.real(eig))
                #print(mineig)
                Ahat = Ahat + I * (-mineig * k**2 + spacing)
        #print(np.linalg.norm(Ahat - A))
        R_old = R.copy()
        R[np.abs(R) < 1e-5] = 1e-5
        np.tril(R)
        #print(np.linalg.norm(R - R_old))
        return R

    # NOTE this gives discretized dynamics
    def calcDynamics(self, v_ref, dpsi_dt_ref):
        lf = self.lf
        lr = self.lr
        Iz = self.Iz
        Caf = self.Caf
        Car = self.Car
        mass = self.mass

        As = []
        Bs = []
        Ds = []
        ds = []

        for i in range(self.N):
            V = v_ref[i]
            A = np.array([[0, 1, 0, 0],
                        [0, -(2*Caf+2*Car)/(mass*V), 0, -V-(2*Caf*lf-2*Car*lr)/(mass*V)],
                        [0, 0, 0, 1],
                        [0, -(2*lf*Caf-2*lr*Car)/(Iz*V),(2*lf*Caf-2*lr*Car)/(Iz), -(2*lf**2*Caf+2*lr**2*Car)/(Iz*V)]])

            B = np.array([[0,2*Caf/mass,0,2*lf*Caf/Iz]]).T
            D = B @ self.Sigma_epsilon
            d = np.array([0, -(2*Caf*lf-2*Car*lr)/(mass*V) - V, 0, -(2*lf**2*Caf+2*lr**2*Car)/(Iz*V)]) * dpsi_dt_ref[i]

            As.append(A * self.dt)
            Bs.append(B * self.dt)
            # NOTE is this right?
            Ds.append(D * self.dt)
            ds.append(d * self.dt)
        return As,Bs,Ds,ds

    # form long matrices, following ji's paper on ccmppi
    def make_batch_dynamics(self, As, Bs, ds, Ds,Sigma_epsilon):
        n = self.n
        l = self.l
        m = self.m
        N = self.N
        assert (Sigma_epsilon.shape == (m,m))
        # A: (N+1)n x n
        I = np.eye(n)
        A = [I]
        # row 1 to row N+1
        # i -> row i+1
        for i in range(N):
            A.append(As[:,:,i] @ A[-1])
        A = np.vstack(A)

        # B (N+1)n x Nm
        row0 = np.zeros((n,m*N))
        B = [row0]
        # row 1 to row N
        for i in range(1,N+1):
            row_i = B[-1].copy()
            row_i = As[:,:,i-1] @ row_i
            row_i[:,(i-1)*m:i*m] = Bs[:,:,i-1]
            B.append(row_i)
        B = np.vstack(B)

        # C: n(N+1) x nN
        '''
        row0 = np.zeros((n, n*N))
        C = [row0]
        for i in range(1,N+1):
            row_i = C[-1].copy()
            row_i = As[:,:,i-1] @ row_i
            row_i[:,(i-1)*n:i*n] = np.eye(n)
            C.append(row_i)
        C = np.vstack(C)
        '''

        # d
        d = ds.reshape((-1,1))

        # take Sigma_epsilon to be 1
        # since D can be used to control effect of gaussian noise on control
        # D (N+1)n x n
        # for scalar Sigma_epsilon
        #D = B.copy() * Sigma_epsilon

        Sigma_epsilon_half = self.nearest_spd_cholesky(Sigma_epsilon)
        row0 = np.zeros((n,m*N))
        D = [row0]
        # row 1 to row N
        for i in range(1,N+1):
            row_i = D[-1].copy() 
            row_i = As[:,:,i-1] @ row_i
            row_i[:,(i-1)*m:i*m] = Bs[:,:,i-1] @ Sigma_epsilon_half
            D.append(row_i)
        D = np.vstack(D)
        return A,B,d,D

    # apply covariance control
    #
    # input:
    #   state: (x,y,heading,v_forward,v_sideway,omega)
    # return: N? K matrices
    def cc(self, state):
        n = self.n
        N = self.N
        m = self.m
        l = self.l

        e_cross, e_heading, v_ref, k_ref, coord_ref, valid = self.track.getRefPoint(state, self.N-1, self.dt, reverse=False)
        # reference angular velocity dpsi_dt = Vx * K
        dpsi_dt_ref = v_ref * k_ref

        # As = [A0..A(N-1)]
        # NOTE this is discretized dynamics
        As, Bs, Ds, ds = self.calcDynamics(v_ref, dpsi_dt_ref)

        # assemble big matrices for batch dynamics
        self.As = As = np.dstack(As)
        self.Bs = Bs = np.dstack(Bs)
        self.ds = ds = np.dstack(ds).reshape((self.n,1,self.N))
        A, B, d, D = self.make_batch_dynamics(As, Bs, ds, None, self.Sigma_epsilon)

        # TODO tune me
        # cost matrix 
        Q = np.eye(n)
        Q_bar = np.kron(np.eye(N+1, dtype=int), Q)
        R = np.eye(m)
        R_bar = np.kron(np.eye(N, dtype=int), R)

        # technically incorrect, but we can just specify R_bar_1/2 instead of R_bar
        R_bar_sqrt = R_bar
        Q_bar_sqrt = Q_bar

        # terminal mean constrain
        # TODO tune me
        sigma_f = np.diag([1e3]*n)

        # setup cvxpy
        I = np.eye(n*(N+1))
        E_N = np.zeros((n,n*(N+1)))
        E_N[:,n*(N):] = np.eye(n)

        # assemble K as a diagonal block matrix with K_0..K_N-1 as var
        Ks = [cp.Variable((m,n)) for i in range(N)]
        # K dim: mN x n(N+1)
        K = cp.hstack([Ks[0], np.zeros((m,(N)*n))])
        for i in range(1,N):
            line = cp.hstack([ np.zeros((m,n*i)), Ks[i], np.zeros((m,(N-i)*n)) ])
            K = cp.vstack([K, line])

        objective = cp.Minimize(cp.norm(cp.vec(R_bar_sqrt @ K @ D)) + cp.norm(cp.vec(Q_bar_sqrt @ (I + B@K) @ D )))

        # TODO verify with Ji
        sigma_y_sqrt = self.nearest_spd_cholesky(D@D.T)
        #sigma_y_sqrt = D@D.T
        #constraints = [cp.bmat([[sigma_f, E_N @(I+B@K)@sigma_y_sqrt], [ sigma_y_sqrt@(I+B @ K).T@E_N.T, I ]]) >= 0]
        constraints = []
        prob = cp.Problem(objective, constraints)

        print("Optimal J = ", prob.solve())
        print("Optimal Ks: ")
        for i in range(N):
            print(Ks[i].value)
        self.Ks = Ks
        return K

    # simulate model with and without cc
    def simulate(self):
        As = self.As
        Bs = self.Bs
        ds = self.ds
        v = np.array([[0.1]])
        x0 = np.array([0.0,0.0,0.0,0.0])
        # without CC
        sim_steps = self.N
        rollout = 100
        states_vec = []
        for j in range(rollout):
            states_vec.append([])
            x_i = x0.copy()
            for i in range(sim_steps):
                # generate random variable epsilon
                mean = [0.0]
                cov = self.Sigma_epsilon
                epsilon = np.random.multivariate_normal(mean, cov)

                x_i = (As[:,:,i] @ x_i + Bs[:,:,i] @ (v+epsilon) + ds[:,:,i]) * self.dt
                states_vec[j].append(x_i.flatten())

        states_vec = np.array(states_vec)
        plt.subplot(2,1,1)
        for i in range(states_vec.shape[0]):
            # crosstrack and heading error
            plt.plot(states_vec[i,:,0],states_vec[i,:,2])
        plt.title("without CC")

        # with CC
        states_vec = []
        for j in range(rollout):
            states_vec.append([])
            y_i = np.zeros(self.n)
            x_i = x0.copy()
            for i in range(sim_steps):
                # generate random variable epsilon
                mean = [0.0]
                cov = self.Sigma_epsilon
                epsilon = np.random.multivariate_normal(mean, cov)

                # TODO test me
                x_i = (As[:,:,i] @ x_i + Bs[:,:,i] @ (v+epsilon + self.Ks[i].value.copy() @ y_i) + ds[:,:,i]) * self.dt
                y_i = (As[:,:,i] @ y_i + Bs[:,:,i] @ epsilon) * self.dt
                states_vec[j].append(x_i.flatten())

        states_vec = np.array(states_vec)
        plt.subplot(2,1,2)
        for i in range(states_vec.shape[0]):
            # position?
            plt.plot(states_vec[i,:,0],states_vec[i,:,2])
        plt.title("with CC")
        plt.show()

if __name__ == "__main__":
    main = CCMPPI()
    state = np.array([0.6*3.5,0.6*1.75,radians(90), 1.0, 0, 0])
    main.cc(state)
    main.simulate()

