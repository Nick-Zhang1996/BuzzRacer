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
        self.look_ahead = 0.1

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

        self.steering = 0
        self.throttle = 0

    def blank_state(self):
        arr = np.zeros(self.n)
        arr[self.MPC_STATE_INDICES["one"]] = 1

        return np.atleast_2d(arr).T

    # TODO: give curvature (kappa) of track given distance from start (s)
    def track_curvature(self, s):
        return 0
    
    # simplified pacejka
    def tire_func(self, a):
        B = 1.1
        C = 1.6
        D = 2.3

        return D * math.sin(C * math.atan(B * a))


    # Linearize the car model based on a given reference state. Returns the system state matrix, A
    def linearizeModel(self, ref_state):
        i = self.MPC_STATE_INDICES

        A = np.eye(self.n, self.n)

        dt = self.dt
        mass = self.mass
        i_z = self.moment_of_inertia
        g = 9.81

        # NOTE: If I have A[i["x"]][i["y"]] = c, this means that every step, x = ... + cy

        A[i["s"]][i["vx"]] = dt * math.cos(ref_state[i["u"]]) / (1 - ref_state[i["n"]] * self.track_curvature(ref_state[i["s"]]))
        A[i["s"]][i["vy"]] = -dt * math.sin(ref_state[i["u"]]) / (1 - ref_state[i["n"]] * self.track_curvature(ref_state[i["s"]]))

        A[i["n"]][i["vx"]] = dt * math.sin(ref_state[i["u"]])
        A[i["n"]][i["vy"]] = dt * math.cos(ref_state[i["u"]])

        v = np.array([ref_state[i["vx"]], ref_state[i["vy"]]])
        steering_forward = np.array([math.cos(ref_state[i["ds"]]), math.sin(ref_state[i["ds"]])])

        #print("steering_forward", steering_forward)
        #print("v", v)

        v_mag = math.sqrt(np.dot(v, v))

        alpha_f = 0 if v_mag < 0.001 else math.acos(np.dot(v, steering_forward) / v_mag)

        #print("alpha_f", alpha_f)

        alpha_r = math.atan2(ref_state[i["vy"]], ref_state[i["vx"]])

        F_fy = self.tire_func(alpha_f) * mass * g * (self.lr / (self.lr + self.lf))
        F_ry = 1.15 * self.tire_func(alpha_r) * mass * g * (self.lr / (self.lr + self.lf))

        #print("F_fy", F_fy)

        w = 0 #TODO: figure out what w (omega) is

        A[i["u"]][i["dt"]] = dt * 6.17 / mass
        A[i["u"]][i["vy"]] = dt * -6.17 / (15.2 * mass) + dt * w
        A[i["u"]][i["one"]] = (dt / mass) * (-6.17/3 - F_fy * math.sin(ref_state[i["ds"]]))
        
        A[i["vy"]][i["one"]] = (dt / mass) * (F_ry + F_fy * math.cos(ref_state[i["ds"]]))
        A[i["vy"]][i["vx"]] = -dt * w

        A[i["r"]][i["one"]] = (dt / i_z) * (F_fy * self.lf * math.cos(ref_state[i["ds"]]) - F_fy * self.lr)

        A[i["vx"]][i["dt"]] = (dt / mass) * 6.17
        A[i["vx"]][i["vx"]] = -(dt / mass) * 6.17 / 15.2
        A[i["vx"]][i["one"]] = (dt / mass) * 6.17 / 3
        

        return A


    def generate_system_matrices(self, reference = []):
        # TODO: re-use the reference trajectories
        A_matrices = []
        for i in range(0, self.N):
            ref_state = None

            if i < len(reference):
                ref_state = reference[i].T[0]
            else:
                ref_state = self.blank_state().T[0]

            model = self.linearizeModel(ref_state)

            #print("model", model)

            A_matrices.append(model)

        A_powers = [np.eye(self.n)]

        for i in range(0, self.N):
            A_powers.append(A_matrices[i] @ A_powers[i])

        dt = self.dt

        # control transition matrix
        self.B = np.zeros((self.n, 2))
        self.B[self.MPC_STATE_INDICES["ds"]][0] = dt
        self.B[self.MPC_STATE_INDICES["dt"]][1] = dt

        self.E = np.block([[A_powers[k+1]] for k in range(self.N)])

        self.F = np.block([[(np.zeros(self.B.shape) if i-j < 0 else (A_powers[i-j] @ self.B)) for j in range(self.N)] for i in range(self.N)])

        # State penalty Matrix
        p = np.eye(self.n)

        p[self.MPC_STATE_INDICES["one"]][self.MPC_STATE_INDICES["one"]] = 0

        # Don't penalize for steering
        p[self.MPC_STATE_INDICES["ds"]][self.MPC_STATE_INDICES["ds"]] = 0
        p[self.MPC_STATE_INDICES["dt"]][self.MPC_STATE_INDICES["dt"]] = 0

        self.P = np.block([[np.zeros(p.shape) if i != j else p for j in range(self.N)] for i in range(self.N)])

        # Control Penality Matrix
        q = np.array([[0.1,   0],   #steering
                      [0,   0.1]])  #throttle
        
        self.Q = np.block([[np.zeros(q.shape) if i != j else q for j in range(self.N)] for i in range(self.N)])

    def slice_f(self, index):
        m = np.zeros((self.N, self.N))

        for row in range(0, self.N):
            for col in range(0, self.N):
                m[row][col] = self.F[col][index][row]

        return m
    
    def generate_ref_trajectory(self):
        ref_trajectory = []

        # states = (x,y,theta,vforward,vsideway=0,omega)
        currState = self.car.states

        for i in range(0, self.N):
            (local_ctrl_pnt,offset,orientation,curvature,v_target) = self.track.localTrajectory(currState)

            distance_along = 0 #TODO

            ref_state = np.array([
                distance_along, 0, 0, v_target, 0, 0, 0, 0, 1
            ])

            ref_trajectory.append(ref_state)


            (x,y,theta,vforward,vsideway,omega) = currState

            x += math.cos(orientation) * v_target * self.dt
            y += math.sin(orientation) * v_target * self.dt

            #print(x,y)

            currState = (x,y,theta,vforward,vsideway,omega)

        return np.atleast_2d(np.block(ref_trajectory)).T

    def control(self):        
        trajectory = self.track.localTrajectory(self.car.states)
        
        if trajectory is None:
            print("no trajectory")
            return (0,0)

        (local_ctrl_pnt, offset, orientation, curvature, v_target) = trajectory
        (x, y, heading, v_forward, v_sideways, omega) = self.car.states

        dt = self.dt

        distance_along = 0

        # initial state
        x0 = np.atleast_2d(np.array([distance_along, offset, orientation - heading, v_forward, v_sideways, omega, self.steering, self.throttle, 1])).T
        
        self.generate_system_matrices([x0])

        #print("x0", x0.shape)
        #print("E", self.E.shape)
        #print("P", self.P.shape)
        #print("F", self.F.shape)
        
        #print("test", self.A @ self.B)

        #x0TETP = np.array([x0.T @ e @ self.P for e in self.E])

        #x0TETPF = np.array([x0TETP[i] @ self.slice_f(i) for i in range(0, self.N)]) #this will be an array of row-vectors. This gets multiplied by u, which is an array of col

        # J(u) = (P_qp)u + uT[Q_qp]u, where:

        ## VV Quadratic form term

        #print("fTpf", self.F.T @ self.P @ self.F)

        p = self.Q + self.F.T @ self.P @ self.F

        ref_trajectory = self.generate_ref_trajectory()

        x_r = ref_trajectory

        ## VV Linear term
        q = 2 * (x0.T @ self.E.T @ self.P @ self.F) - 2 * (x_r.T @ self.P @ self.F)

        #print("p", p)
        
        P_qp = cvxopt.matrix(2 * p)
        Q_qp = cvxopt.matrix(q.T)

        sol=cvxopt.solvers.qp(P_qp, Q_qp)

        sol_x = np.array(sol["x"])

        solved_ds = sol_x[0][0]
        solved_dt = sol_x[1][0]

        print("-------------")
        print("DS", solved_ds)
        print("DT", solved_dt)

        self.steering += solved_ds * self.dt
        self.throttle += solved_dt * self.dt

        self.car.throttle = self.throttle
        self.car.steering = self.steering

        return (self.throttle, self.steering) #throttle, steering