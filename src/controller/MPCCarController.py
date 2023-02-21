from CarController import CarController
from PidController import PidController

import numpy as np
from time import time
import cvxopt
import math
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev,CubicSpline,interp1d

class MPCCarController(CarController):
    def __init__(self, car, config):
        super().__init__(car, config)

        self.N = 3 #30 #horizon
        self.look_ahead = 1

        self.dt = self.look_ahead / self.N
        
        # dim of state
        # distance along track, perpendicular distance from track, heading error from track (backwards - so positive value means its pointed to the right), v_x, v_y (velocities relative to current heading), r (rate of rotation), current steering (rad), current throttle [-1,1]
        self.n = 9 #also have the last state just be one

        # dim of output y
        self.l = 1 #derivative of steering, derivative of throttle
        # prediction horizon
        self.p = self.N
        # last applied control command, used for smoothing constrain
        self.last_applied_u = [0, 0]

        self.generate_system_matrices()

        self.MPC_STATE_INDICES = {
            "s": 0,
            "n": 1,
            "u": 2,
            "vx": 3,
            "vy": 4,
            "r": 5,
            "ds": 6,
            "dt": 7,
            "one": 8
        }

        self.mass = 1
        self.moment_of_inertia = 5

        self.lf = 5
        self.lr = 5.5

    # TODO: give curvature (kappa) of track given distance from start (s)
    def track_curvature(self, s):
        return 0
    
    # simplified pacejka
    def tire_func(self, a):
        B = 1.1
        C = 1.6
        D = 2.3

        return D * math.sin(C * math.atan(B * a))


    def getA(self, ref_state):
        i = self.MPC_STATE_INDICES

        A = np.eye(self.n, self.n)

        dt = self.dt
        mass = self.mass
        i_z = self.moment_of_inertia
        g = 9.81

        # NOTE: If I have A[i.x][i.y] = c, this means that every step, x = ... + cy

        A[i.s][i.vx] = dt * math.cos(ref_state[i.u]) / (1 - ref_state[i.n] * self.track_curvature(ref_state[i.s]))
        A[i.s][i.vy] = -dt * math.sin(ref_state[i.u]) / (1 - ref_state[i.n] * self.track_curvature(ref_state[i.s]))

        A[i.n][i.vx] = dt * math.sin(ref_state[i.u])
        A[i.n][i.vy] = dt * math.cos(ref_state[i.u])

        v = np.array([ref_state[i.vx], ref_state[i.vy]])
        steering_forward = np.array([math.cos(ref_state[i.ds]), math.sin(ref_state[i.ds])])

        alpha_f = math.acos(np.dot(v, steering_forward) / math.sqrt(np.dot(v, v)))
        alpha_r = math.atan2(ref_state(i.vy), ref_state(i.vx))

        F_fy = self.tire_func(alpha_f) * mass * g * (self.lr / (self.lr + self.lf))
        F_ry = 1.15 * self.tire_func(alpha_r) * mass * g * (self.lr / (self.lr + self.lf))

        w = 0 #TODO: figure out what w (omega) is

        A[i.u][i.dt] = dt * 6.17 / mass
        A[i.u][i.vy] = dt * -6.17 / (15.2 * mass) + dt * w
        A[i.u][i.one] = (dt / mass) * (-6.17/3 - F_fy * math.sin(ref_state[i.ds]))
        
        A[i.vy][i.one] = (dt / mass) * (F_ry + F_fy * math.cos(ref_state[i.ds]))
        A[i.vy][i.vx] = -dt * w

        A[i.r][i.one] = (dt / i_z) * (F_fy * self.lf * math.cos(ref_state[i.ds]) - F_fy * self.lr)

        A[i.vx][i.dt] = (dt / mass) * 6.17
        A[i.vx][i.vx] = -(dt / mass) * 6.17 / 15.2
        A[i.vx][i.one] = (dt / mass) * 6.17 / 3
        

        return A


    def generate_system_matrices(self, x0):
        dt = self.dt
                
        # state transition matrix
        self.A = np.eye(self.n)
        self.A[0,2] = dt

        # control transition matrix
        self.B = np.array([[0,1,0], [0,0,0]]).T*dt

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

        self.generate_system_matrices(x0)

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

        print(sol["x"])

        raise Exception("error")

        return (0, 0) #throttle, steering