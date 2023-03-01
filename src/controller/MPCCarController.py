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

        self.N = 20 #30 #horizon
        self.look_ahead = .5

        self.dt = self.look_ahead / self.N
        
        # dim of state
        # distance along track, perpendicular distance from track, heading error from track (backwards - so positive value means its pointed to the right), v_x, v_y (velocities relative to current heading), r (rate of rotation), current steering (rad), current throttle [-1,1]
        self.n = 9 #also have the last state just be one

        # dim of output y
        self.l = 1 #derivative of steering, derivative of throttle
        # prediction horizon
        self.p = self.N

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
        
        self.mass = 0.041
        self.moment_of_inertia = 417757e-9

        self.lf = 0.04824
        self.lr = 0.04176
        
        self.steering = 0
        self.throttle = 0

        p = np.zeros((self.n, self.n))

        p[self.MPC_STATE_INDICES["one"]][self.MPC_STATE_INDICES["one"]] = 0
        p[self.MPC_STATE_INDICES["ds"]][self.MPC_STATE_INDICES["ds"]] = 0
        p[self.MPC_STATE_INDICES["dt"]][self.MPC_STATE_INDICES["dt"]] = 0
        p[self.MPC_STATE_INDICES["r"]][self.MPC_STATE_INDICES["r"]] = 0
        p[self.MPC_STATE_INDICES["vy"]][self.MPC_STATE_INDICES["vy"]] = 0
        p[self.MPC_STATE_INDICES["vx"]][self.MPC_STATE_INDICES["vx"]] = 2
        p[self.MPC_STATE_INDICES["u"]][self.MPC_STATE_INDICES["u"]] = 1 #0.2
        p[self.MPC_STATE_INDICES["n"]][self.MPC_STATE_INDICES["n"]] = 2
        p[self.MPC_STATE_INDICES["s"]][self.MPC_STATE_INDICES["s"]] = 8

        self.p = p

        self.u_max = [1, 1]

        np.set_printoptions(linewidth=400)

    def print_array_header(self):
        print(["s", "n", "u", "vx", "vy", "r", "ds", "dt", "one"])


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
    def linearizeModel(self, ref_state, curvature):
        i = self.MPC_STATE_INDICES

        A = np.eye(self.n, self.n)

        dt = self.dt
        mass = self.mass
        i_z = self.moment_of_inertia
        g = 9.81

        # NOTE: If I have A[i["x"]][i["y"]] = c, this means that every step, x = ... + cy
        A[i["s"]][i["vx"]] = dt * math.cos(ref_state[i["u"]]) / (1 - ref_state[i["n"]] * curvature)
        A[i["s"]][i["vy"]] = -dt * math.sin(ref_state[i["u"]]) / (1 - ref_state[i["n"]] * curvature)

        A[i["n"]][i["vx"]] = dt * math.sin(ref_state[i["u"]])
        A[i["n"]][i["vy"]] = dt * math.cos(ref_state[i["u"]])

        v = np.array([ref_state[i["vx"]], ref_state[i["vy"]]])
        
        #print("ref v", v)

        steering_forward = np.array([math.cos(ref_state[i["ds"]]), math.sin(ref_state[i["ds"]])])

        v_mag = math.sqrt(np.dot(v, v))

        #-np.arctan((omega*lf + vy)/vx) + steering

        w = curvature * ref_state[i["vx"]] + ref_state[i["r"]]

        alpha_f = math.atan((w * self.lf + ref_state[i["vy"]]) / ref_state[i["vx"]] + ref_state[i["ds"]]) # 0 if v_mag < 0.001 else math.acos(np.dot(v, steering_forward) / v_mag)

        alpha_r = math.atan((w * self.lr - ref_state[i["vy"]]) / ref_state[i["vx"]]) #math.atan2(ref_state[i["vy"]], ref_state[i["vx"]])

        #print("af, ar", alpha_f, alpha_r)

        F_fy = self.tire_func(alpha_f) * mass * g * (self.lr / (self.lr + self.lf))
        F_ry = 1.15 * self.tire_func(alpha_r) * mass * g * (self.lf / (self.lr + self.lf))

        A[i["u"]][i["one"]] = -dt * curvature #* ref_state[i["vx"]]
        A[i["u"]][i["r"]] = dt

        A[i["vy"]][i["one"]] = (dt / mass) * (F_ry + F_fy * math.cos(ref_state[i["ds"]]))
        A[i["vy"]][i["vx"]] = -dt * w

        #print("F_fy", F_fy)

        #A[i["r"]][i["one"]] = (dt / i_z) * (F_fy * self.lf * math.cos(ref_state[i["ds"]]) - F_ry * self.lr)
        #linearize, assume that sin(x) = x
        
        #A[i["r"]][i["ds"]] = (dt / i_z) * F_fy * self.lf
        #A[i["r"]][i["one"]] = (dt / i_z) * (F_fy * self.lf * (3 * math.pi / 2) - F_ry * self.lr)
        A[i["r"]][i["ds"]] = 0.01 * dt / i_z

        A[i["vx"]][i["dt"]] = (dt / mass) * 6.17
        A[i["vx"]][i["vx"]] = -(dt / mass) * 6.17 / 15.2        
        A[i["vx"]][i["one"]] = -(dt) * 6.17 / 3
        
        return A


    def generate_system_matrices(self, reference = [], referenceCurvature = []):
        # TODO: re-use the reference trajectories
        A_matrices = []
        for i in range(0, self.N):
            ref_state = None
            curvature = 0

            if i < len(reference):
                ref_state = reference[i].T[0]
                curvature = referenceCurvature[i]
            else:
                ref_state = self.blank_state().T[0]

            model = self.linearizeModel(ref_state, curvature)

            #print("model", model)

            A_matrices.append(model)

        #for a in A_matrices:
            #self.print_array_header()
            #print("A matrix\n", np.array2string(a, separator=" "))

        A_powers = [np.eye(self.n)]

        for i in range(0, self.N):
            A_powers.append(A_powers[i] @ A_matrices[i])

        
        #for a in A_powers:
        #    self.print_array_header()
        #    print("A power", a)

        dt = self.dt

        # control transition matrix
        self.B = np.zeros((self.n, 2))
        self.B[self.MPC_STATE_INDICES["ds"]][0] = dt
        self.B[self.MPC_STATE_INDICES["dt"]][1] = dt

        self.E = np.block([[A_powers[k+1]] for k in range(self.N)])

        #print("E", self.E)#

        B_0k = []

        def B_jk(j, k):
            result = np.eye(self.n)
            
            for i in range(j, k + 1):
                result = result @ A_matrices[i]
            
            return result @ self.B
        
        #TODO: compute this more efficiently
        for i in range(0, self.N):
            B_0k.append(B_jk(0, i))
                

        self.F = np.block([[(np.zeros(self.B.shape) if i-j < 0 else B_0k[i - j]) for j in range(self.N)] for i in range(self.N)])

        #print("F", self.F)

        # State penalty Matrix
        p = self.p

        self.P = np.block([[np.zeros(p.shape) if i != j else p for j in range(self.N)] for i in range(self.N)])

        # Control Penality Matrix
        q = np.array([[1e-9,   0],   #steering
                      [0,   1e-9]])  #throttle
        
        self.Q = np.block([[np.zeros(q.shape) if i != j else q for j in range(self.N)] for i in range(self.N)])

    def slice_f(self, index):
        m = np.zeros((self.N, self.N))

        for row in range(0, self.N):
            for col in range(0, self.N):
                m[row][col] = self.F[col][index][row]

        return m
    
    def generate_ref_trajectory(self):
        ref_trajectory = []
        ref_trajectory_curvature = []
        ref_not_goal_trajectory = []

        # states = (x,y,theta,vforward,vsideway=0,omega)
        currState = self.car.states

        distance_along = 0 

        for i in range(0, self.N):
            (local_ctrl_pnt,offset,orientation,curvature,v_target) = self.track.localTrajectory(currState)

            #v_target = 0
            #distance_along = 0

            (x,y,theta,vforward,vsideway,omega) = currState

            v_target *= 0.6

            x += - math.cos(orientation + math.pi/2) * offset
            y += - math.sin(orientation + math.pi/2) * offset

            oldX = x
            oldY = y

            x += math.cos(theta) * v_target * self.dt
            y += math.sin(theta) * v_target * self.dt

            #print("ref pos", x, y)

            distance_along += math.sqrt((oldX - x) * (oldX - x) + (oldY - y) * (oldY - y))

            n = 0

            # ["s", "n", "u", "vx", "vy", "r", "ds", "dt", "one"]

            ref_state = np.array([
                distance_along, 0, 0, v_target, 0, 0, 0, 0, 1
            ])

            ref_not_goal_trajectory.append(np.array([
                distance_along, offset, 0, v_target, vsideway, 0, self.steering, self.throttle, 1
            ]))

            ref_trajectory.append(ref_state)
            ref_trajectory_curvature.append(curvature)


            currState = (x,y,theta,v_target,vsideway,omega)

        return (np.atleast_2d(np.block(ref_trajectory)).T, [x for x in map(lambda a: np.atleast_2d(a).T, ref_trajectory)], [x for x in map(lambda a: np.atleast_2d(a).T, ref_not_goal_trajectory)], ref_trajectory_curvature)

    def control(self):        
        trajectory = self.track.localTrajectory(self.car.states)
        
        if trajectory is None:
            print("no trajectory")
            return (0,0)

        (local_ctrl_pnt, offset, orientation, curvature, v_target) = trajectory
        (x, y, heading, v_forward, v_sideways, omega) = self.car.states



        dt = self.dt

        distance_along = 0

        heading_error = heading - orientation

        if (heading_error > math.pi):
            heading_error = heading_error - 2 * math.pi
        elif heading_error < -math.pi:
            heading_error = heading_error + 2 * math.pi

        #print("heading error", heading_error)

        vx = v_forward * math.cos(heading_error) + v_sideways * math.sin(heading_error)
        vy = v_forward * math.sin(heading_error) + v_sideways * math.cos(heading_error)

        #print("vx, vy", vx, vy)

        # initial state
        x0 = np.atleast_2d(np.array([distance_along, offset, heading_error, vx, vy, omega, self.steering, self.throttle, 1])).T
        
        #print("HEADING ERR", x0.T[0][self.MPC_STATE_INDICES["u"]])

        #print("o", orientation)
        #print("h", heading)

        #print("u", orientation - heading)
        
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

        (ref_trajectory, ref_trajectory_arr, ref_not_goal_trajectory, ref_trajectory_curvature) = self.generate_ref_trajectory()

        
        if 0 > 1:
            print("ref_not_goal_trajectory trajectories: ")

            print("s, n, u, vx, vy, r, ds, dt, one & curvature")

            i = 0

            for ref in ref_not_goal_trajectory:
                #print(ref.T, ref_trajectory_curvature[i])
                print(ref_trajectory_curvature[i])

                i += 1
        

        #print("x0:", x0.T)
            
        self.generate_system_matrices(ref_not_goal_trajectory, ref_trajectory_curvature)

        p = self.Q + self.F.T @ self.P @ self.F

        x_r = ref_trajectory

        #print("ref", ref_trajectory_arr)

        ## VV Linear term
        q = 2 * (x0.T @ self.E.T @ self.P @ self.F) - 2 * (x_r.T @ self.P @ self.F)

        #print("part of q")
        #print(x0.T @ self.E.T)

        #print("p", p)
        
        P_qp = cvxopt.matrix(2 * p)
        Q_qp = cvxopt.matrix(q.T)

        u_max = np.atleast_2d(np.array(self.u_max * self.N)).T

        #print(u_max)

        #raise Exception("lol")

        sol=cvxopt.solvers.qp(P_qp, Q_qp)

        sol_x = np.array(sol["x"])

        #print("sol", sol_x)

        solved_ds = sol_x[0][0]
        solved_dt = sol_x[1][0]

        print("-------------")
        #print("Cost", sol["primal objective"])
        print("DS", solved_ds)
        print("DT", solved_dt)

        #raise Exception("lol")

        self.steering += solved_ds * self.dt
        self.throttle += solved_dt * self.dt

        print("S", self.steering)

        #print(self.steering, self.throttle)

        self.car.throttle = self.throttle
        self.car.steering = self.steering

        #raise Exception("lol")

        return (self.throttle, self.steering) #throttle, steering