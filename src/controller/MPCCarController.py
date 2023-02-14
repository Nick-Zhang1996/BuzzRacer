from CarController import CarController
from PidController import PidController

import numpy as np
from time import time
import cvxopt
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev,CubicSpline,interp1d

class MPCCarController(CarController):
    def __init__(self, car, config):
        super().__init__(car, config)

        self.N = 3 #30 #horizon
        self.look_ahead = 1

        self.dt = self.look_ahead / self.N
        
        # dim of state
        self.n = 3
        # dim of action
        self.m = 1
        # dim of output y
        self.l = 1 #throttle , steering
        # prediction horizon
        self.p = self.N
        # last applied control command, used for smoothing constrain
        self.last_applied_u = [0, 0]

        self.generate_system_matrices()

    def generate_system_matrices(self):
        dt = self.dt
                
        # state transition matrix
        self.A = np.eye(self.n)
        self.A[0,2] = dt

        # control transition matrix
        self.B = np.array([[0,1,0]]).T*dt

        self.E = np.block([[np.linalg.matrix_power(self.A, k+1)] for k in range(self.N)])

        self.F = np.block([[(np.zeros(self.B.shape) if i-j < 0 else (np.linalg.matrix_power(self.A, i-j) @ self.B)) for j in range(self.N)] for i in range(self.N)])

        p = np.eye(3)

        self.P = np.block([[np.zeros(p.shape) if i != j else p for j in range(self.N)] for i in range(self.N)])


        self.Q = np.array([[0.1,   0, 0],   #throttle
                            [0,   0.1, 0],   #throttle
                            [0,     0, 0]])  #unused

    def slice_f(self, index):
        m = np.zeros((self.N, self.N))

        for row in range(0, self.N):
            for col in range(0, self.N):
                m[row][col] = self.F[col][index][row]

        return m

    def control(self):        
        trajectory = self.track.localTrajectory(self.car.states)
        
        if trajectory is None:
            print("no trajectory")
            return (0,0)

        (local_ctrl_pnt, offset, orientation, curvature, v_target) = trajectory
        (x, y, heading, v_forward, _, _) = self.car.states

        dt = self.dt

        # initial state
        x0 = np.atleast_2d(np.array([0, heading - orientation, v_forward])).T

        #print("x0", x0)
        #print("A", self.A)
        #print("B", self.B)
        #print("E", self.E)
        #print("P", self.P)
        #print("F", self.F)
        
        #print("test", self.A @ self.B)

        #x0TETP = np.array([x0.T @ e @ self.P for e in self.E])

        #x0TETPF = np.array([x0TETP[i] @ self.slice_f(i) for i in range(0, self.N)]) #this will be an array of row-vectors. This gets multiplied by u, which is an array of col

        # J(u) = (P_qp)u + uT[Q_qp]u, where:

        ## VV Quadratic form term
        p = self.Q + self.F.T @ self.P @ self.F

        ## VV Linear term
        q = 2 * x0.T @ self.E.T @ self.P @ self.F #TODO: 2x_rTPF - what is x_r?
        
        #print("p", p)
        
        P_qp = cvxopt.matrix(2 * p)
        Q_qp = cvxopt.matrix(q.T)

        sol=cvxopt.solvers.qp(P_qp, Q_qp)

        print(sol)

        raise Exception("error")

        return (0, 0) #throttle, steering