from CarController import CarController
from PidController import PidController

import numpy as np
from time import time
import cvxopt
import math
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev,CubicSpline,interp1d

class MPCCarController(CarController):
    def __init__(self, car, config):
        super().__init__(car, config)

        cvxopt.solvers.options['show_progress'] = False

        self.N = 8 #horizon
        self.look_ahead = 0.2
        self.v_target_multiplier = 1.3 #0.95 = safe, 1 = aggresive, >1 = spin out

        self.dt = self.look_ahead / self.N
        
        # dim of state
        # distance along track, perpendicular distance from track, heading error from track (backwards - so positive value means its pointed to the right), v_x, v_y (velocities relative to current heading), r (rate of rotation), current steering (rad), current throttle [-1,1]
        self.n = 9 #also have the last state just be one

        # dim of output y
        self.l = 2 #derivative of steering, derivative of throttle
        # prediction horizon
        self.p = self.N

        self.simulation = False

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
        p[self.MPC_STATE_INDICES["vx"]][self.MPC_STATE_INDICES["vx"]] = 0.5
        p[self.MPC_STATE_INDICES["u"]][self.MPC_STATE_INDICES["u"]] = 0
        p[self.MPC_STATE_INDICES["n"]][self.MPC_STATE_INDICES["n"]] = 3
        p[self.MPC_STATE_INDICES["s"]][self.MPC_STATE_INDICES["s"]] = 0

        self.p = p

        self.q = np.array([[0,   0],   #steering
                      [0,   0]])  #throttle

        self.u_max = [1, 1]

        constraint_count = 4

        self.G = np.zeros((constraint_count, self.l)) #Gx <= h (element-wise). x is n x 1, G needs to be q x n, h needs to be q x 1, where q is the number of constraints
        self.H = np.zeros((constraint_count, 1))

        maxSteeringInput = 30
        maxThrottleInput = 20

        # ds/dt < 1
        self.G[0][0] = 1
        self.H[0][0] = maxSteeringInput

        # -ds/dt < 1
        self.G[1][0] = -1
        self.H[1][0] = maxSteeringInput

        #dT/dt < 1
        self.G[2][1] = 1
        self.H[2][0] = maxThrottleInput

        # -dT/dt < 1
        self.G[3][1] = -1
        self.H[3][0] = maxThrottleInput

        # repeat constraint for every timestep

        self.G = np.block([[(self.G if i == j else np.zeros(self.G.shape)) for j in range(0, self.N)] for i in range(0, self.N)])
        self.H = np.block([[self.H] for i in range(0, self.N)])

        #print("G", self.G)
        #print("H", self.H)
        
        #print("G size", self.G.shape )
        #print("H size", self.H.shape)

        self.last_sol = None
        self.last_goal = None

        self.draw_points = []

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
        #print("linearize, ref_state=", ref_state, "curv= ", curvature)
        i = self.MPC_STATE_INDICES

        A = np.eye(self.n, self.n)

        dt = self.dt
        mass = self.mass
        i_z = self.moment_of_inertia
        g = 9.81

        # NOTE: If I have A[i["x"]][i["y"]] = c, this means that every step, x = ... + cy
        A[i["s"]][i["vx"]] = dt * math.cos(ref_state[i["u"]]) / (1 - ref_state[i["n"]] * curvature)
        A[i["s"]][i["vy"]] = -dt * math.sin(ref_state[i["u"]]) / (1 - ref_state[i["n"]] * curvature)
        if abs(ref_state[i["u"]]) < 0.7:
            # sin x ~ x

            A[i["n"]][i["u"]] = dt * ref_state[i["vx"]]
        else:
            A[i["n"]][i["vx"]] = dt * math.sin(ref_state[i["u"]])
        
        A[i["n"]][i["vy"]] = dt * math.cos(ref_state[i["u"]])

        v = np.array([ref_state[i["vx"]], ref_state[i["vy"]]])
        
        #print("ref v", v)
        #-np.arctan((omega*lf + vy)/vx) + steering

        w = curvature * ref_state[i["vx"]] + ref_state[i["r"]]

        alpha_f = math.atan((w * self.lf + ref_state[i["vy"]]) / ref_state[i["vx"]] + ref_state[i["ds"]]) # 0 if v_mag < 0.001 else math.acos(np.dot(v, steering_forward) / v_mag)

        alpha_r = math.atan((w * self.lr - ref_state[i["vy"]]) / ref_state[i["vx"]]) #math.atan2(ref_state[i["vy"]], ref_state[i["vx"]])

        #print("af, ar", alpha_f, alpha_r)

        F_fy = self.tire_func(alpha_f) * mass * g * (self.lr / (self.lr + self.lf))
        F_ry = 1.15 * self.tire_func(alpha_r) * mass * g * (self.lf / (self.lr + self.lf))

        A[i["u"]][i["one"]] = -dt * curvature * ref_state[i["vx"]] * ref_state[i["vx"]]
        A[i["u"]][i["r"]] = dt

        A[i["vy"]][i["one"]] = (dt / mass) * (F_ry + F_fy * math.cos(ref_state[i["ds"]]))
        A[i["vy"]][i["vx"]] = -dt * w * mass

        #print("F_fy", F_fy)

        A[i["r"]][i["one"]] = (dt / i_z) * (F_fy * self.lf * math.cos(ref_state[i["ds"]]) - F_ry * self.lr)
        #linearize, assume that sin(x) = x
        
        #A[i["r"]][i["ds"]] = -(dt / i_z) * F_fy * self.lf
        #A[i["r"]][i["one"]] = (dt / i_z) * (F_fy * self.lf * (3 * math.pi / 2) - F_ry * self.lr)

        #if self.last_sol is None:
        #NOTE: in simulation, this should be 1. Physical is different for some reason? Need to remove this anyways
        A[i["r"]][i["ds"]] = (1 if self.simulation else 10) * dt / i_z #NOT ACCURATE TO THE MODEL, BUT IT COMMUNICATES A LINEAR RELATIONSHIP BETWEEN STEERING AND ROTATIONAL SPEED. having a little bit of this makes it work better!

        A[i["vx"]][i["dt"]] = (dt) * 6.17 * (1 if self.simulation else 4)
        #A[i["vx"]][i["vx"]] = -(dt) * 6.17 / 15.2        THIS SHOULD BE HERE BUT IT BREAKS IT
        A[i["vx"]][i["one"]] = -(dt) * 6.17 / 3
        
        return A
    
    def nonlinear_curv_dynamics(self, state, curvature):
        newState = np.zeros(self.n)

        i = self.MPC_STATE_INDICES
        dt = self.dt
        m = self.mass

        s = state[i["s"]]
        n = state[i["n"]]
        u = state[i["u"]]
        vx = state[i["vx"]]
        vy = state[i["vy"]]
        r = state[i["r"]]
        ds = state[i["ds"]]
        throttle = state[i["dt"]]


        w = curvature * vx + r

        alpha_f = math.atan((w * self.lf + vy) / vx + ds) # 0 if v_mag < 0.001 else math.acos(np.dot(v, steering_forward) / v_mag)
        alpha_r = math.atan((w * self.lr - vy) / vx) #math.atan2(ref_state[i["vy"]], ref_state[i["vx"]])

        F_rx = 6.17 * (throttle - vx / 15.2 - 0.333) * m
        
        F_fy = self.tire_func(alpha_f) * m * 9.81 * (self.lr / (self.lr + self.lf))
        F_ry = 1.15 * self.tire_func(alpha_r) * m * 9.81 * (self.lf / (self.lr + self.lf))

        s_dot = (vx * math.cos(u) - vy * math.sin(u)) / (1 - n * curvature)
        vx_dot = (1 / m) * F_rx #(F_rx - F_fy * math.sin(ds) + m * vy * w)

        newState[i["s"]] = s + s_dot * dt
        newState[i["n"]] = n + (vx * math.sin(u) + vy * math.cos(u)) * dt
        newState[i["u"]] = u + (r - curvature * s_dot * vx_dot) * dt
        newState[i["vx"]] = vx + vx_dot * dt
        newState[i["vy"]] = vy + (1 / m) * (F_ry + F_fy * math.cos(ds) - m * vx * w) * dt
        newState[i["r"]] = r + (1 / self.moment_of_inertia) * (F_fy * self.lf * math.cos(ds) - F_ry * self.lr) * dt
        newState[i["ds"]] = ds
        newState[i["dt"]] = throttle

        newState[i["one"]] = 1

        return newState


    def generate_system_matrices(self, reference = [], referenceCurvature = []):
        #print("generate system matrices, ref=", reference)
        # TODO: re-use the reference trajectories
        A_matrices = []
        for i in range(0, self.N):
            ref_state = None
            curvature = 0

            if i < len(reference):
                ref_state = reference[i]#.T[0]
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


        #For each i, stores A_matrices[0] @ A_matrices[1] @ ... A_matrices[i]
        A_matrices_cumulative = [A_matrices[0]]

        for i in range(1, self.N):
            A_matrices_cumulative.append(A_matrices_cumulative[i - 1] @ A_matrices[i])
        
        B_0k = []

        for i in range(0, self.N):
            B_0k.append(A_matrices_cumulative[i] @ self.B)
                

        self.F = np.block([[(np.zeros(self.B.shape) if i-j < 0 else B_0k[i - j]) for j in range(self.N)] for i in range(self.N)])

        #print("F", self.F)

        # State penalty Matrix
        p = self.p

        self.P = np.block([[np.zeros(p.shape) if i != j else p for j in range(self.N)] for i in range(self.N)])

        # Control Penality Matrix
        q = self.q
        
        self.Q = np.block([[np.zeros(q.shape) if i != j else q for j in range(self.N)] for i in range(self.N)])

    def slice_f(self, index):
        m = np.zeros((self.N, self.N))

        for row in range(0, self.N):
            for col in range(0, self.N):
                m[row][col] = self.F[col][index][row]

        return m
    
    def generate_ref_trajectory(self):
        ref_trajectory = []
        goal_trajectory = []
        ref_trajectory_curvature = []

        # states = (x,y,theta,vforward,vsideway=0,omega)
        currState = self.car.states

        distance_along = 0
        goal_distance_along = 0

        ds = self.steering
        dt = self.throttle

        first_v_target = None

        self.draw_points.clear()
        
        #print("state pos", currState[0], currState[1])
        #print("state vf", currState[3])

        start_time = time()
        total_localTrajectory_time = 0

        for i in range(0, self.N):
            #rint("getting local trajectory...")
            #total_localTrajectory_time -= time()
            (local_ctrl_pnt,offset,orientation,curvature,v_target) = self.track.localTrajectory(currState)
            #total_localTrajectory_time += time()

            #print("offset", offset)

            if first_v_target is None:
                first_v_target = v_target

            solved_dds = 0
            solved_ddt = 0

            solved_index = 2 * (i + 1)

            if self.last_sol is not None and len(self.last_sol) > (solved_index):
                solved_dds = self.last_sol[solved_index][0]
                solved_ddt = self.last_sol[solved_index + 1][0]
            else:
                pass
                #solved_dds = curvature

            ds += solved_dds * self.dt
            dt += solved_ddt * self.dt

            #print("ds, dt = ", ds, dt)

            #v_target = 0
            #distance_along = 0

            (x,y,theta,vforward,vsideway,omega) = currState

            v_target *= 1
            
            #x += math.cos(theta) * v_target * self.dt
            #y += math.sin(theta) * v_target * self.dt
            
            #print("ref pos", x, y)
            
            # ["s", "n", "u", "vx", "vy", "r", "ds", "dt", "one"]

            heading_error = self.heading_error(theta, orientation)

            vx = vforward * math.cos(heading_error) - vsideway * math.sin(heading_error)
            vy = vsideway * math.cos(heading_error) + vforward * math.sin(heading_error)
            
            ref_state = np.array([
                distance_along, offset, heading_error, vx, vy, omega, ds, dt, 1
            ])

            distance_along += vx * self.dt

            ref_trajectory.append(ref_state)
            ref_trajectory_curvature.append(curvature)

            new_curvilinear_state = self.nonlinear_curv_dynamics(ref_state, curvature)

            #print(ref_state, "->", new_curvilinear_state)

            delta_s = new_curvilinear_state[self.MPC_STATE_INDICES["s"]] - ref_state[self.MPC_STATE_INDICES["s"]]
            delta_n = new_curvilinear_state[self.MPC_STATE_INDICES["n"]] - ref_state[self.MPC_STATE_INDICES["n"]]

            #print("orientation", math.degrees(orientation))
            #print("err", heading_error)
            #print("ds", delta_s)
            #print("dn", delta_n)
            
            x += math.cos(orientation) * delta_s# - math.sin(orientation) * delta_n #double check this w delta_n
            y += math.sin(orientation) * delta_s# + math.cos(orientation) * delta_n #double check this w delta_n

            self.draw_points.append([(x, y), (255, 0, 255)])

            #print("refpos2", x, y)

            #print("newv", new_curvilinear_state[self.MPC_STATE_INDICES["vx"]])

            # (x,y,theta,vforward,vsideway=0,omega)
            currState = (x,y,theta + new_curvilinear_state[self.MPC_STATE_INDICES["r"]] * self.dt,
                         new_curvilinear_state[self.MPC_STATE_INDICES["vx"]],0,omega)
            
        #print("reference part of generate_ref_tractory took", time() - start_time)
        start_time = time()

        currState = self.car.states

        goal_distance_along = -self.dt

        for i in range(0, self.N):
            total_localTrajectory_time -= time()
            (local_ctrl_pnt,offset,orientation,curvature,v_target) = self.track.localTrajectory(currState)
            total_localTrajectory_time += time()

            (x,y,theta,vforward,vsideway,omega) = currState

            v_target *= self.v_target_multiplier
            
            correction = 1
            
            x += - math.cos(orientation + math.pi/2) * offset * correction
            y += - math.sin(orientation + math.pi/2) * offset * correction
            
            x += math.cos(theta) * v_target * self.dt
            y += math.sin(theta) * v_target * self.dt

            goal_distance_along += v_target * self.dt
            
            goal_trajectory.append(np.array([
                goal_distance_along, 0, 0, v_target, 0, 0, 0, 0, 1
            ]))

            self.draw_points.append([(x, y), (0, 255, 255)])

            currState = (x,y, orientation, v_target, 0, 0)

        #ref_trajectory = np.atleast_2d(np.block(ref_trajectory)).T
        #goal_trajectory = np.atleast_2d(np.block(goal_trajectory)).T

        self.draw_points.reverse()

        #print("total_localTrajectory_time", total_localTrajectory_time)

        #print("goal part of generate_ref_tractory took", time() - start_time)

        return (ref_trajectory, goal_trajectory, ref_trajectory_curvature)
    
    def draw(self, img):    
        for pt in self.draw_points:
            pt_adjusted = self.track.m2canvas(pt[0])
            img = cv2.circle(img, pt_adjusted, 5, pt[1], -1)
            img = cv2.circle(img, pt_adjusted, 5, (0,0,0), 2)

        return img
    
    def heading_error(self, heading, orientation):
        heading_error = (heading - orientation + math.pi) % (2 * math.pi) - math.pi

        return heading_error

    def control(self):        
        print("----------------------")

        start_control_time = time()

        trajectory = self.track.localTrajectory(self.car.states)
        
        if trajectory is None:
            print("no trajectory")
            return (0,0)

        (local_ctrl_pnt, offset, orientation, curvature, v_target) = trajectory
        (x, y, heading, v_forward, v_sideways, omega) = self.car.states

        #print("got states took", time() - start_control_time, 1/(time() - start_control_time))


        dt = self.dt

        distance_along = 0

        heading_error = self.heading_error(heading, orientation)

        #print("heading error", heading_error)

        vx = v_forward * math.cos(heading_error) + v_sideways * math.sin(heading_error)
        vy = v_forward * math.sin(heading_error) + v_sideways * math.cos(heading_error)

        #print("vx, vy", vx, vy)

        
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

        start_ref_traj_time = time()
        (ref_trajectory, goal_trajectory, ref_trajectory_curvature) = self.generate_ref_trajectory()  
        #print("generate reference trajectory took", time() - start_ref_traj_time, 1 / (time() - start_ref_traj_time))
        
        #self.print_array_header()

        """
        print("refs:")
        for (ref, curv) in zip(ref_trajectory, ref_trajectory_curvature):
            print(ref, curv)

        print("goals:")
        for goal in goal_trajectory:
            print(goal)
        """

        # initial state
        x0 = np.atleast_2d(np.array([0, offset, heading_error, vx, vy, omega, self.steering, self.throttle, 1])).T
        #x0 = ref_trajectory[0]
        

        goal_trajectory_vec = np.atleast_2d(np.block(goal_trajectory)).T    

        #print("x0:", x0.T)
            
        self.generate_system_matrices(ref_trajectory, ref_trajectory_curvature)

        p = self.Q + self.F.T @ self.P @ self.F

        x_r = goal_trajectory_vec

        #print("ref", ref_trajectory_arr)

        ## VV Linear term
        q = 2 * (x0.T @ self.E.T @ self.P @ self.F) - 2 * (x_r.T @ self.P @ self.F)

        #print("part of q")
        #print(x0.T @ self.E.T)

        #print("p", p)
        
        P_qp = cvxopt.matrix(2 * p)
        Q_qp = cvxopt.matrix(q.T)

        G_qp = cvxopt.matrix(self.G)
        H_qp = cvxopt.matrix(self.H)

        #print(u_max)

        #raise Exception("lol")
        time_before_sol = time()
        sol=cvxopt.solvers.qp(P_qp, Q_qp, G_qp, H_qp)
        print("solution took", time() - time_before_sol, 1 / (time() - time_before_sol))

        sol_x = np.array(sol["x"])

        #print("sol", sol_x)

        solved_ds = sol_x[0][0]
        solved_dt = sol_x[1][0]

        #print("Cost", sol["primal objective"])
        print("DS", solved_ds)
        print("DT", solved_dt)

        #raise Exception("lol")

        self.steering += solved_ds * self.dt
        self.throttle += solved_dt * self.dt

        max_throttle = 2

        self.throttle = min(max(self.throttle, -max_throttle), max_throttle)

        #print("S", self.steering)

        #print(self.steering, self.throttle)

        #self.throttle = max(min(self.throttle, 0.4), -0.4)
        #self.steering = max(min(self.steering, 0.3), -0.3)
       

        self.car.throttle = self.throttle
        self.car.steering = self.steering

        self.last_sol = sol_x
        self.last_goal = goal_trajectory

        #raise Exception("lol")

        print("control loop took", time() - start_control_time, 1 / (time() - start_control_time))

        return (self.throttle, self.steering, True) #throttle, steering
