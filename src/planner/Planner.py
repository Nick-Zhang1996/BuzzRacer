# TODO Sept 26
# weird behavior crossing start line (index wrap around?)
# trajectory does not start at vehicle position, maybe p calculation is not as good as before cartesianToCurvilinear

# MPC based trajectory planner
from common import *
import matplotlib.pyplot as plt
import numpy as np
from planner.MPC import MPC
from time import time
from math import floor
from itertools import product
import scipy.optimize
from scipy.linalg import block_diag
from scipy.interpolate import splprep, splev,CubicSpline,interp1d
from bisect import bisect
from scipy.optimize import minimize

# TODO:
# properly fix first step error
# in curvilinear coord conversion, search for shorter range
# drop irrelevant opponent ASAP

class Planner(ConfigObject):
    def __init__(self,config=None):
        super().__init__(config)
        self.config = config
        # NOTE these will be overridded by config file
        # p: prediction horizon
        '''
        self.N = N = horizon = 30
        self.opponent_lookahead_time = 1.0
        # if opponents are closer than this threshold, pass them on same side
        self.same_side_passing_threshold = 0.5
        self.dt = 0.1
        self.skip_count = 0
        '''
        #self.opponent_length = 0.21*2
        #self.opponent_width = 0.12*2
        self.opponent_length = 0.21*2
        self.opponent_width = 0.15*2
        self.best_solution = None
        self.best_plan_traj_points = None
        # replan every x steps
        #self.replan_steps = 3
        # when replan, overlap previous plan x steps
        #self.replan_overlap = 3
        self.no_solution = True
        self.plan_age = 0
        return

    def init(self):
        # state x:[s,n,ds]
        # u: [dn] first derivative of n
        self.n = n = 3
        self.m = m = 1

        self.track = self.main.track
        self.genPath()
        return

    # retrieve information about local trajectory
    # plan() should be called prior, ideally immediately before calling this function
    # However, it's possible to make multiple localTrajectory() inquiries for one planned trajectory
    # generated with plan() for computation efficiency
    # TODO
    def localTrajectory(self,state):
        coord = (state[0],state[1])
        heading = state[2]
        omega = state[5]
        vf = state[3]
        vs = state[4]
        P = self.best_solution[4]

        # find proper index u
        def fun(u):
            val = self.evalBezierSpline(P,u)
            return (val[0][0]-coord[0])**2 + (val[0][1]-coord[1])**2
        retval = minimize(fun,0.1)
        u = retval.x[0]
        p = self.evalBezierSpline(P,u)
        dp = self.evalBezierSpline(P,u,der=1)
        ddp = self.evalBezierSpline(P,u,der=2)
        orientation = np.arctan2(dp[0][1],dp[0][0])


        # offset: negative offset means left steering needed
        vec_path_tangent = dp
        vec_path_to_car = p - np.array(coord)
        offset = np.cross(vec_path_tangent, vec_path_to_car) / np.linalg.norm(vec_path_tangent)


        # calculate velocity based on a lookahead point
        lookahead = 2
        p = self.evalBezierSpline(P,u+lookahead)
        dp = self.evalBezierSpline(P,u+lookahead,der=1)
        ddp = self.evalBezierSpline(P,u+lookahead,der=2)
        curvature = np.cross(dp,ddp)/ np.linalg.norm(dp)**3
        v_target = (0.5*9.8/np.abs(curvature))**0.5


        #(local_ctrl_pnt,offset,orientation,curvature,v_target) = retval
        retval = (None,offset.item(),orientation,None,v_target.item())
        return retval

    def plan(self):
        self.plan_age += 1

        if (self.plan_age < self.skip_count and not self.no_solution):
            return True

        self.plan_age = 0

        vs = 1.3
        #x0 = [0,0,vs]
        coord = self.car.states[0:2]
        car_state_curvi = self.cartesianToCurvilinear(coord,skip_wrap=True)
        x0 = [car_state_curvi[0], car_state_curvi[1], vs]

        self.idx = self.pickRelevantIndex(x0)
        # opponents, static, [s,n]
        #opponent_state_vec = [[1,0],[0.5,-0.1]]
        opponent_state_vec = self.getOpponentState()
        sols = self.solveSingleControl(x0,opponent_state_vec)
        if (self.no_solution):
            return False

        costs = [x[2] for x in sols]
        best_sol_idx = np.argmin(costs)
        #self.print_info('min cost = ',np.min(costs))

        '''
        # visualize
        self.plotTrack()
        # initial state
        self.plotCar(x0)
        # target state
        #self.plotCar(xref)
        for (u_vec, state_traj) in sols:
            self.plotStateTraj(state_traj)
        for opponent in opponent_state_vec:
            self.plotOpponent(opponent)
        plt.axis('equal')
        plt.show()
        '''

        # plot on visualization
        # TODO have this stay on before next replan
        # maybe redraw this every frame, move to localTrajectory
        #self.plotSolutions(sols)
        #self.plotSolutions([sols[best_sol_idx]],color=(100,100,100))
        self.best_solution = sols[best_sol_idx]
        self.solutions = sols
        self.best_solution_index = best_sol_idx
        self.best_plan_traj_points = self.stateTrajToCartesianTraj(self.best_solution[1])
        return True

    def plotAllSolutions(self):
        # plot solutions
        sols = self.solutions
        best_sol_idx = self.best_solution_index
        self.plotSolutions(sols)
        self.plotSolutionsPoint(sols)
        self.plotSolutionsPoint([sols[best_sol_idx]],color=(100,100,100))

    # plot smooth bezier curve
    def plotSolutions(self,sols,color=(255,51,204)):
        img = self.main.visualization.visualization_img
        for sol in sols:
            bezier_coeffs = sol[4]
            u = np.linspace(0,self.N-1)
            traj = self.evalBezierSpline(bezier_coeffs,u)
            img = self.track.drawPolyline(traj,lineColor=color,img=img)
        self.main.visualization.visualization_img = img
        return 

    # plot raw points
    def plotSolutionsPoint(self,sols,color=(255,51,204)):
        img = self.main.visualization.visualization_img
        for sol in sols:
            p = sol[3]
            img = self.track.drawPoints(img,p,color=color)
        self.main.visualization.visualization_img = img
        return 

    def test(self):
        vs = 1.3
        x0 = [0,0,vs]
        # opponents, static, [s,n]
        opponent_state_vec = [[1,0],[0.5,-0.1]]

        sols = self.solveSingleControl(x0,opponent_state_vec)

        self.plotTrack()
        # initial state
        self.plotCar(x0)
        # target state
        #self.plotCar(xref)
        for (u_vec, state_traj) in sols:
            self.plotStateTraj(state_traj)
        for opponent in opponent_state_vec:
            self.plotOpponent(opponent)
        plt.axis('equal')
        plt.show()
        return

    def getOpponentState(self):
        opponent_state_vec = []
        for car in self.main.cars:
            if (car == self.car):
                continue
            #x,y,heading,v_forward,v_sideways,omega
            coord = (car.states[0],car.states[1])
            # NOTE optimization possible
            car_state_curvi = self.cartesianToCurvilinear(coord)
            opponent_state_vec.append(car_state_curvi)
        return opponent_state_vec

    # NOTE optimization possible
    def cartesianToCurvilinear(self,coord,skip_wrap=False):
        dist = np.sum((self.r-coord)**2, axis=1)
        idx = np.argmin(dist)
        s = self.s_vec[idx]
        # ensure s is larger than ego vehicle's s
        if (not skip_wrap):
            if (idx < self.idx[0]):
                s += self.track.raceline_len_m
                #self.print_info("wrap around")

        r = self.r[idx]
        rp = coord - r
        dr = self.dr[idx]
        n = float(np.cross(dr,rp))
        return (s,n)

    # prepare curvilinear path from RCPTrack or like
    def genPath(self):
        # need: x_t_fun, y_t_fun, t_vec
        # r,dr,ddr
        # left,right limit
        self.discretized_raceline_len = 1024
        self.s_vec = s_vec = np.linspace(0,self.track.raceline_len_m,self.discretized_raceline_len)
        self.r = self.ref_path = np.array(splev(s_vec%self.track.raceline_len_m,self.track.raceline_s,der=0)).T
        dr = splev(s_vec%self.track.raceline_len_m,self.track.raceline_s,der=1)
        self.raceline_headings = np.arctan2(dr[1],dr[0])

        self.s_step = s_vec[1]-s_vec[0]
        # TODO use M1 M2
        self.dr,self.ddr = self.calcDerivative(self.ref_path)
        # describe track boundary as offset from raceline
        self.createBoundary()
        self.left_limit = np.array(self.raceline_left_boundary)
        self.right_limit = -np.array(self.raceline_right_boundary)
        # ccw 90 deg
        R = np.array([[0,-1],[1,0]])
        tangent_dir = (R @ self.dr.T)/np.linalg.norm(self.dr,axis=1)
        self.left_boundary = (tangent_dir * self.left_limit).T + self.ref_path
        self.right_boundary = (tangent_dir * self.right_limit).T + self.ref_path
        self.calcArcLen(self.ref_path)
        return

    def createBoundary(self,show=False):
        # construct a (self.discretized_raceline_len * 2) vector
        # to record the left and right track boundary as an offset to the discretized raceline
        left_boundary = []
        right_boundary = []

        left_boundary_points = []
        right_boundary_points = []

        ref_path = self.ref_path.T
        ref_heading = self.raceline_headings

        for i in range(self.discretized_raceline_len):
            # find normal direction
            coord = ref_path[:,i]
            heading = ref_heading[i]

            left, right = self.track.preciseTrackBoundary(coord,heading)
            left_boundary.append(left)
            right_boundary.append(right)

            '''
            # debug boundary points
            left_point = (coord[0] + left * np.cos(heading+np.pi/2),coord[1] + left * np.sin(heading+np.pi/2))
            right_point = (coord[0] + right * np.cos(heading-np.pi/2),coord[1] + right * np.sin(heading-np.pi/2))

            left_boundary_points.append(left_point)
            right_boundary_points.append(right_point)


            # DEBUG
            # plot left/right boundary
            left_point = (coord[0] + left * cos(heading+np.pi/2),coord[1] + left * sin(heading+np.pi/2))
            right_point = (coord[0] + right * cos(heading-np.pi/2),coord[1] + right * sin(heading-np.pi/2))
            img = self.track.drawTrack()
            img = self.track.drawRaceline(img = img)
            img = self.track.drawPoint(img,coord,color=(0,0,0))
            img = self.track.drawPoint(img,left_point,color=(0,0,0))
            img = self.track.drawPoint(img,right_point,color=(0,0,0))
            plt.imshow(img)
            plt.show()
            '''


        self.raceline_left_boundary = left_boundary
        self.raceline_right_boundary = right_boundary

        if (show):
            img = self.track.drawTrack()
            img = self.track.drawRaceline(img = img)
            img = self.track.drawPolyline(left_boundary_points,lineColor=(0,255,0),img=img)
            img = self.track.drawPolyline(right_boundary_points,lineColor=(0,0,255),img=img)
            plt.imshow(img)
            plt.show()
            return  img
        return

    # generate a path
    def genSamplePath(self):
        # create a parametric curve
        x_t_fun = lambda t:t
        y_t_fun = lambda t:np.sin(t)
        t_range = 2*np.pi
        t_vec = np.linspace(0,t_range,1000)

        # given t_start and t_end, calculate the curve length inbetween
        def tToS(t_start,t_end,steps=10):
            t_range = t_end-t_start
            if (t_range < 0):
                return 0
            t = np.linspace(t_start,t_end,steps).flatten()
            s = np.sum(np.sqrt(np.diff(x_t_fun(t))**2 + np.diff(y_t_fun(t))**2))
            return s

        # resample to parameter of curve length r(t) -> r(s)
        self.s_vec = s_vec = np.linspace(0,tToS(0,t_range,steps=10000),1000)
        self.s_step = s_step = s_vec[1] - s_vec[0]
        t_vec = [0]
        for i in range(s_vec.shape[0]-1):
            t = scipy.optimize.root(lambda m:tToS(t_vec[-1],m)-s_step, t_vec[-1])
            t_vec.append(t.x[0])
        x_vec = x_t_fun(t_vec)
        y_vec = y_t_fun(t_vec)
        self.t_vec = np.array(t_vec)

        # curvilinear frame reference curve
        # size: n*2
        self.r = self.ref_path = np.vstack([x_vec,y_vec]).T
        self.dr,self.ddr = self.calcDerivative(self.ref_path)
        self.curvature = self.calcCurvature(self.dr,self.ddr)

        # define track boundary
        # left positive, left negetive
        self.track_width = 0.6
        self.left_limit = np.ones_like(s_vec)*self.track_width/2
        self.right_limit = -np.ones_like(s_vec)*self.track_width/2

        # ccw 90 deg
        R = np.array([[0,-1],[1,0]])
        tangent_dir = (R @ self.dr.T)/np.linalg.norm(self.dr,axis=1)
        self.left_boundary = (tangent_dir * self.left_limit).T + self.ref_path
        self.right_boundary = (tangent_dir * self.right_limit).T + self.ref_path

        self.calcArcLen(self.ref_path)

    # control with state = [s,n,dsdt]
    def demoSingleControl(self):
        # state x:[s,n,dsdt]
        # u: [dn] first derivative of n
        self.n = n = 3
        self.m = m = 1
        self.genSamplePath()

        # m/s
        self.dt = 0.1
        vs = 1.3
        x0 = [1,0,vs]
        # opponents, static, [s,n]
        opponent_state_vec = [[3,0],[5,-0.1]]

        sols = self.solveSingleControl(x0,opponent_state_vec)

        self.plotTrack()
        # initial state
        self.plotCar(x0)
        # target state
        #self.plotCar(xref)
        for (u_vec, state_traj) in sols:
            self.plotStateTraj(state_traj)
        for opponent in opponent_state_vec:
            self.plotOpponent(opponent)
        plt.axis('equal')
        plt.show()

    def verifyCost(self,u):
        mpc = self.mpc
        # calculate progress and curvature cost
        #u = sols[0][0]
        mse = lambda a: np.sum((a)**2)
        x0 = self.x0

        progress_cost = 0.5*u.T @ mpc.old_P @ u + mpc.old_q.T @ u
        cur_cost = 0.5*u.T @ mpc.dP @ u + mpc.dq.T @ u
        total_cost = 0.5*u.T @ mpc.P @ u + mpc.q.T @ u
        print("progress cost = %.2f"%(progress_cost))
        print("curvature cost = %.2f"%(cur_cost))
        print("total cost = %.2f"%(total_cost))
        return

        # base curvature(r' * r'') for r
        k_r = np.cross(self.debug_dr, self.debug_ddr,axis=0)
        #k_r_norm = np.linalg.norm(k_r)
        k_r_norm = mse(k_r)

        # ccw 90 deg
        R = np.array([[0,-1],[1,0]])
        I_2 = np.eye(2)
        # curvature for actual p
        n = sols[0][1][1:,1]
        # ds = ds/dt * dt
        ds = x0[2][0] * self.dt
        M1 = self.getDiff1Matrix(N,ds)
        M2 = self.getDiff2Matrix(N,ds)

        p = self.debug_r + R @ self.debug_dr * n
        dp = (np.kron(M1,I_2) @ p.T.flatten()).reshape((N,2)).T
        ddp = (np.kron(M2,I_2) @ p.T.flatten()).reshape((N,2)).T
        k_p = np.cross(dp, ddp,axis=0)
        k_p_norm = mse(k_p)

        # should agree with k_r
        dr = (np.kron(M1,I_2) @ self.debug_r.T.flatten()).reshape((N,2)).T
        ddr = (np.kron(M2,I_2) @ self.debug_r.T.flatten()).reshape((N,2)).T
        k_r_test = np.cross(dr, ddr,axis=0)
        k_r_norm = mse(k_r_test)
        print("curvature for ref r = %.3f"%(k_r_norm))
        print("curvature for ref p = %.3f"%(k_p_norm))

        # check k_p_norm in direction of n*
        k_p_norm_vec = []
        ratio = np.linspace(-1,2,20)
        for c in ratio:
            p = self.debug_r + R @ self.debug_dr * (c*n)
            dp = (np.kron(M1,I_2) @ p.T.flatten()).reshape((N,2)).T
            ddp = (np.kron(M2,I_2) @ p.T.flatten()).reshape((N,2)).T
            k_p = np.cross(dp, ddp,axis=0)
            k_p_norm = mse(k_p)
        return

    def verifyIniial(self,u):
        x0 = self.x0
        mpc = self.mpc
        # calculate plan tangent at p0
        # current heading:
        heading = self.car.states[2]
        heading_x = np.cos(heading) # = dp x
        heading_y = np.sin(heading) # = dp y
        # dp
        ds = x0[2][0] * self.dt
        idx = self.idx
        r = self.r.T[:,idx]
        dr = self.dr.T[:,idx]
        N = self.N
        I_2 = np.eye(2)
        I_N1 = np.eye(N+1)
        M = np.kron(I_N1, [0,1,0]) @ mpc.F
        K = np.kron(I_N1, [0,1,0]) @ mpc.Ex0
        M1 = self.getDiff1Matrix(N+1,ds)
        def C(i):
            if i==1:
                return np.hstack([np.eye(2),np.zeros((2,(N+1-i)*2))])
            if i==N+1:
                return np.hstack([np.zeros((2,2*(i-1))),np.eye(2)])
            else:
                return np.hstack([np.zeros((2,2*(i-1))),np.eye(2),np.zeros((2,(N+1-i)*2))])
        # ccw 90 deg
        A = np.array([[0,-1],[1,0]])
        Mat = A @ dr
        D_Adr = block_diag(* [Mat[:,[i]] for i in range(N+1)])
        n = (M @ u + K)
        p = (r.T.reshape(-1,1) + D_Adr @ n)
        dp = np.kron(M1,I_2) @ p
        dp0 = C(1) @ dp
        heading = self.car.states[2]
        heading_x = np.cos(heading) # = dp x
        heading_y = np.sin(heading) # = dp y
        dp0_ref = np.array([[heading_x,heading_y]]).T
        print('dp0_ref',dp0_ref)
        print('dp0',dp0)

        return

    def verifyTangentialConstraint(self,x0,u):
        mpc = self.mpc
        # calculate plan tangent at p0
        # current heading:
        heading = self.car.states[2]
        heading_x = np.cos(heading) # = dp x
        heading_y = np.sin(heading) # = dp y
        # dp
        ds = x0[2][0] * self.dt
        idx = self.idx
        r = self.r.T[:,idx]
        dr = self.dr.T[:,idx]
        N = self.N
        I_2 = np.eye(2)
        I_N1 = np.eye(N+1)
        M = np.kron(I_N1, [0,1,0]) @ mpc.F
        K = np.kron(I_N1, [0,1,0]) @ mpc.Ex0
        M1 = self.getDiff1Matrix(N+1,ds)
        def C(i):
            if i==1:
                return np.hstack([np.eye(2),np.zeros((2,(N+1-i)*2))])
            if i==N+1:
                return np.hstack([np.zeros((2,2*(i-1))),np.eye(2)])
            else:
                return np.hstack([np.zeros((2,2*(i-1))),np.eye(2),np.zeros((2,(N+1-i)*2))])
        # ccw 90 deg
        A = np.array([[0,-1],[1,0]])
        Mat = A @ dr
        D_Adr = block_diag(* [Mat[:,[i]] for i in range(N+1)])
        dp0 = C(1) @ np.kron(M1,I_2) @ (r.T.reshape(-1,1) + D_Adr @ (M @ u + K))
        heading = self.car.states[2]
        heading_x = np.cos(heading) # = dp x
        heading_y = np.sin(heading) # = dp y
        dp0_ref = np.array([[heading_x,heading_y]]).T
        print('dp0_ref',dp0_ref)
        print('dp0',dp0)

        # constraints
        G_dp = C(1) @ np.kron(M1,I_2) @ (D_Adr @ M)
        h_dp = C(1) @ np.kron(M1,I_2) @ (r.T.reshape(-1,1) + D_Adr @ K)
        G_dp1 = G_dp
        G_dp2 = -G_dp
        h_dp1 = -h_dp + dp0_ref + 0.05
        h_dp2 = -(-h_dp + dp0_ref - 0.05)
        print('constrain satisfaction',G_dp1 @ u < h_dp1)
        print('constrain satisfaction',G_dp2 @ u < h_dp2)
        return


    def buildTangentialConstraint(self,x0):
        # constrain path tangent to equal vehicle current heading
        mpc = self.mpc
        # current heading:
        heading = self.car.states[2]
        heading_x = np.cos(heading) # = dp x
        heading_y = np.sin(heading) # = dp y
        dp0_ref = np.array([[heading_x,heading_y]]).T
        # dp
        ds = x0[2][0] * self.dt
        idx = self.idx
        r = self.r.T[:,idx]
        dr = self.dr.T[:,idx]
        N = self.N
        I_2 = np.eye(2)
        I_N1 = np.eye(N+1)
        M = np.kron(I_N1, [0,1,0]) @ mpc.F
        K = np.kron(I_N1, [0,1,0]) @ mpc.Ex0
        M1 = self.getDiff1Matrix(N+1,ds)
        def C(i):
            if i==1:
                return np.hstack([np.eye(2),np.zeros((2,(N+1-i)*2))])
            if i==N+1:
                return np.hstack([np.zeros((2,2*(i-1))),np.eye(2)])
            else:
                return np.hstack([np.zeros((2,2*(i-1))),np.eye(2),np.zeros((2,(N+1-i)*2))])
        # ccw 90 deg
        A = np.array([[0,-1],[1,0]])
        Mat = A @ dr
        D_Adr = block_diag(* [Mat[:,[i]] for i in range(N+1)])
        G_dp = C(1) @ np.kron(M1,I_2) @ (D_Adr @ M)
        h_dp = C(1) @ np.kron(M1,I_2) @ (r.T.reshape(-1,1) + D_Adr @ K)
        G_dp1 = G_dp
        G_dp2 = -G_dp
        h_dp1 = -h_dp + dp0_ref + 0.5
        h_dp2 = -(-h_dp + dp0_ref - 0.5)
        G = np.vstack([G_dp1,G_dp2])
        h = np.vstack([h_dp1,h_dp2])
        return (G,h)

    def solveSingleControl(self,x0,opponent_state):
        planner_t0 = time()
        mpc = MPC()
        self.mpc = mpc
        n = self.n
        m = self.m
        N = self.N
        s_step = self.s_step
        # setup mpc 
        mpc.setup(n,m,n,N)
        dt = self.dt
        A = np.eye(n)
        A[0,2] = dt
        B = np.array([[0,1,0]]).T*dt
        P = np.diag([1,1,0])
        Q = np.eye(m)

        x0 = np.array(x0).T.reshape(-1,1)
        self.x0 = x0
        xref = [x0[0,0]+x0[2,0]*(self.N+1),0,x0[2,0]]
        xref = np.array(xref).T.reshape(-1,1)
        xref_vec = xref.repeat(N,1).T.reshape(N,n,1)
        #du_max = np.array([[1,1]]).T
        du_max = None
        #u_max = np.array([[1.5,1.5]]).T
        u_max = None
        mpc.convertLtiPlanner(A,B,P,Q,xref_vec,x0,N,u_max,du_max)
        # add track boundary constraints
        self.constructTrackBoundaryConstraint()
        scenarios = self.constructOpponentConstraint(opponent_state)
        self.addCurvatureNormObjective( weight=1, n_estimate=None )
        self.addDeviationObjective( weight=1 )
        self.scenarios = scenarios
        sols = []

        # u0 is unusually large
        # constrain u0=u1
        '''
        G_u0 = np.zeros((2,N))
        G_u0[0,0] = 1
        G_u0[0,1] = -1
        G_u0[1,0] = -1
        G_u0[1,1] = 1
        h_u0 = np.zeros((2,1))
        h_u0[0,0] = 0.1
        h_u0[1,0] = -0.1
        '''
        # constrain u0=<0.01
        '''
        G_u0 = np.zeros((2,N))
        G_u0[0,0] = 1
        G_u0[1,0] = -1
        h_u0 = np.zeros((2,1))
        h_u0[0,0] = 0.01
        h_u0[1,0] = -0.01
        mpc.G = np.vstack([mpc.G,G_u0])
        mpc.h = np.vstack([mpc.h,h_u0])
        '''

        # add constraint: path start must be tangential to heading
        G_tan, h_tan = self.buildTangentialConstraint(x0)
        mpc.G = np.vstack([mpc.G,G_tan])
        mpc.h = np.vstack([mpc.h,h_tan])

        # save current mpc matrices
        # solve for different scenarios
        G = mpc.G
        h = mpc.h

        # iterate over all scenarios
        t = time()
        for case in scenarios:
            t = time()
            #print("case ---- ")
            mpc.G = np.vstack([G,case[0]])
            mpc.h = np.vstack([h,case[1]])
            success = mpc.solve()
            if (not success):
                print_warning("MPC fail to find solution on this path")
                continue
            if (mpc.h is not None):
                #print("constraints: %d"%(mpc.h.shape[0]))
                pass
            sol = self.buildSolution(mpc,x0)
            # usually cost is -200 to -30
            # if cost too high neglect
            if (sol[2] > 0):
                continue
            sols.append(sol)
            duration = time()-t
            self.print_debug("case freq = %.2fHz"%(1/duration))

        # if there's no opponent in sight
        # or no way to pass them
        # TODO don't crash into opponents if ego can't pass
        if (len(scenarios)==0):
            self.print_debug("no opponent in horizon")
        elif (len(sols)==0):
            self.print_debug("can't pass opponent")
        else:
            self.no_solution = False

        if (len(sols)==0):
            dt = time()-t
            success = mpc.solve()
            if (not success):
                print_warning("MPC fail to find solution")
                self.no_solution = True
                return sols
            else:
                self.no_solution = False

            sol = self.buildSolution(mpc,x0)
            sols.append(sol)

        self.print_debug("found %d valid trajectory from  %d scenarios"%(len(sols),len(scenarios)))
        duration = time() - planner_t0
        self.print_debug("planner step freq = %.2fHz"%(1/duration))


        #self.verifyTangentialConstraint(x0,sols[0][0])
        ctrl = sols[0][0]
        #self.verifyCost(ctrl)
        #self.verifyIniial(ctrl)



        return sols

    def buildSolution(self,mpc,x0):
        n = self.n
        m = self.m
        N = self.N
        # state_traj in curvilinear frame
        state_traj = mpc.F @ mpc.u + mpc.Ex0
        state_traj = state_traj.reshape((N+1,n))

        # trajectory in cartesian frame
        # XXX can we treat plan_xy as p?
        #plan_xy = self.stateTrajToCartesianTraj(state_traj)

        ds = x0[2][0] * self.dt
        idx = self.idx
        r = self.r.T[:,idx]
        dr = self.dr.T[:,idx]
        A = np.array([[0,-1],[1,0]])
        Mat = A @ dr
        D_Adr = block_diag(* [Mat[:,[i]] for i in range(N+1)])

        u = mpc.u
        I_2 = np.eye(2)
        I_N1 = np.eye(N+1)
        M = np.kron(I_N1, [0,1,0]) @ mpc.F
        K = np.kron(I_N1, [0,1,0]) @ mpc.Ex0
        # N+1 * 2
        n = M @ u + K
        # 2*N+1
        p = r.T.reshape(-1,1) + D_Adr @ n
        M1 = self.getDiff1Matrix(N+1,ds)
        M2 = self.getDiff2Matrix(N+1,ds)
        # N+1*2
        dp = (np.kron(M1,I_2) @ p.T.flatten()).reshape((N+1,2))
        ddp = (np.kron(M2,I_2) @ p.T.flatten()).reshape((N+1,2))
        p = p.reshape(-1,2)

        P = self.bezierSpline(p,dp,ddp,ds)
        # DEBUG
        '''
        u = np.linspace(0,N-2)
        xy = self.evalBezierSpline(P,u)
        plt.plot(xy[:,0],xy[:,1],'*')
        plt.xlim([xy[0,0]-0.7,xy[0,0]+0.7])
        plt.ylim([xy[0,1]-0.7,xy[0,1]+0.7])
        plt.show()
        '''

        # sol = [ u (control seq), state_curvi, cost, traj (cartesian), Bezier Spline for plan(u:0:N-2) ]
        sol = (mpc.u,state_traj,mpc.cost,p,P)
        return sol

    # construct the state limits
    def constructTrackBoundaryConstraint(self):
        # create additional lines for Gx<h
        # track boundary limits
        mpc = self.mpc
        N = self.N
        if (self.n==4):
            M = np.kron(np.eye(N),np.array([[0,1,0,0]]))
        elif (self.n==3):
            M = np.kron(np.eye(N),np.array([[0,1,0]]))
        elif (self.n==2):
            M = np.kron(np.eye(N),np.array([[0,1]]))

        # selection matrix, select [x1..xp] from x0..xp
        C = np.eye((N+1)*self.n)[self.n:,:]
        # left is positive
        G1 = M @ C @ mpc.F
        # left
        L = self.left_limit[self.idx[1:]].reshape((N,1))
        #L = np.ones((N,1))*self.track_width/2
        h1 = L - M @ C @ mpc.Ex0

        G2 = -M @ C @ mpc.F
        R = -self.right_limit[self.idx[1:]].reshape((N,1))
        h2 = R + M @ C @ mpc.Ex0

        if (mpc.G is None):
            mpc.G = np.vstack([G1,G2])
            mpc.h = np.vstack([h1,h2])
        else:
            mpc.G = np.vstack([mpc.G,G1,G2])
            mpc.h = np.vstack([mpc.h,h1,h2])
        return

    def constructOpponentConstraint(self,opponent_state):
        # TODO quick check feasibility
        # additional constraints to add
        # (G,h)
        # opponent_count * feasible_direction_count
        # one sublist for each opponent
        # sublist contains a set of OR constraints
        # solution need to satisfy ONE constraint from sublist 0, ONE constraint from sublist 1 ...
        mpc = self.mpc
        opponent_constraints = []
        opponent_idx = 0
        for opponent in opponent_state:
            # treat opponent as a single point
            idx = self.getIndex(opponent[0])
            # +
            left = self.left_limit[idx]
            # -
            right = self.right_limit[idx]
            # which time step to apply constraints
            #step = (opponent[0] - self.x0[0])/self.x0[2]/self.dt
            #print("step = %d"%(step))
            step_begin = (opponent[0] - self.x0[0] - self.opponent_length/2)/self.x0[2]/self.dt
            step_end = (opponent[0] - self.x0[0] + self.opponent_length/2)/self.x0[2]/self.dt
            step_begin = floor(step_begin)
            step_end = floor(step_end) + 1
            # if opponent is too far, skip
            if (step_begin > int(self.opponent_lookahead_time/self.dt)):
                self.print_debug("ignoring opponent at step %d"%(step_begin))
                continue
            opponent_constraints.append([])
            if (left > opponent[1]+self.opponent_width/2):
                # there's space in left for passing
                left_bound = opponent[1]+self.opponent_width/2
                retval1 = self.getGhForN(step_begin,left_bound,False)
                if (retval1 is not None):
                    G,h = retval1
                    retval2 = self.getGhForN(step_end,left_bound,False)
                    if (retval2 is not None):
                        G2,h2 = retval2
                        G = np.vstack([G,G2])
                        h = np.vstack([h,h2])
                        #print("feasible path, oppo %d, left, step %d-%d, n > %.2f"%(opponent_idx, step_begin, step_end,left_bound))
                    opponent_constraints[-1].append((G,h,opponent[0],'L'))

            if (right < opponent[1]-self.opponent_width/2):
                # there's space in right for passing
                right_bound = opponent[1]-self.opponent_width/2
                retval1 = self.getGhForN(step_begin,right_bound,True)
                if (retval1 is not None):
                    G,h = retval1
                    retval2 = self.getGhForN(step_end,right_bound,True)
                    if (retval2 is not None):
                        G2,h2 = retval2
                        G = np.vstack([G,G2])
                        h = np.vstack([h,h2])
                        #print("feasible path, oppo %d, right, step %d-%d, n > %.2f"%(opponent_idx, step_begin, step_end,right_bound))
                    opponent_constraints[-1].append((G,h,opponent[0],'R'))
                
            opponent_idx += 1

        # possible combination of AND constraints
        # e.g. opponent 0 can be passed on left or right
        #      opponent 1 can be passed on left 
        # 2 scenarios, (left,left) (left,right)

        # if no opponent is in sight
        if (len(opponent_constraints)==0):
            return []
        cons_combination = [ cons for cons in product(*opponent_constraints)]

        scenarios = [ (np.vstack( [con[0] for con in cons] ), np.vstack( [con[1] for con in cons]), [con[2] for con in cons],[con[3] for con in cons] ) for cons in cons_combination]
        # screen all scenarios, remove cases that requires trajectory to pass close opponents on different sides
        screened_scenarios = []
        for scenario in scenarios:
            s_vec = scenario[2]
            side_vec = scenario[3]
            add = True
            for i in range(len(s_vec)-1):
                if (np.abs(s_vec[i]-s_vec[i+1])<self.same_side_passing_threshold
                        and side_vec[i] != side_vec[i+1]):
                    add = False
            if (add):
                screened_scenarios.append((scenario[0], scenario[1]))
        return screened_scenarios

    # handle wrap around with index
    def pickRelevantIndex(self,x0):
        x0 = np.array(x0).flatten()
        idx = []
        s_step = self.s_step
        for i in range(self.N+1):
            this_s = x0[0] + i*self.dt*x0[2]
            i = (this_s%self.track.raceline_len_m) / s_step
            idx.append(i)
        idx = np.array(idx,dtype=int).flatten()
        return idx

    # dr:(2,N)
    # ds: float scalar
    # p: (2,N)
    def addCurvatureNormObjective(self, weight, n_estimate=None):
        # x0 = s,n,ds
        N = self.N
        s_step = self.s_step
        idx = self.idx
        x0 = self.x0
        mpc = self.mpc

        # TODO maybe use good ddr approxmation instead of 2nd degree
        r = self.r.T[:,idx]
        dr = self.dr.T[:,idx]
        ddr = self.ddr.T[:,idx]
        dr_norm = np.linalg.norm(dr,axis=0)

        self.debug_r = r
        self.debug_dr = dr
        self.debug_ddr = ddr
        self.debug_dr_norm = dr_norm
        if (n_estimate is None):
            n_estimate = np.zeros(N)
        n_estimate = np.hstack([x0[1],n_estimate])

        # ccw 90 deg
        A = np.array([[0,-1],[1,0]])
        # this is where iterations may be necessary
        # 2*(N+1) x0, p1, .... pN
        p = r + A @ dr/dr_norm * n_estimate

        def cross(A,B):
            return np.cross(A,B,axis=0)
        def C(i):
            if i==1:
                return np.hstack([np.eye(2),np.zeros((2,(N+1-i)*2))])
            if i==N+1:
                return np.hstack([np.zeros((2,2*(i-1))),np.eye(2)])
            else:
                return np.hstack([np.zeros((2,2*(i-1))),np.eye(2),np.zeros((2,(N+1-i)*2))])

        # ccw 90 deg
        A = np.array([[0,-1],[1,0]])
        Mat = A @ dr
        D_Adr = block_diag(* [Mat[:,[i]] for i in range(N+1)])
        # ds = ds/dt * dt
        ds = x0[2][0] * self.dt
        M1 = self.getDiff1Matrix(N+1,ds)
        M2 = self.getDiff2Matrix(N+1,ds)
        I_2 = np.eye(2)
        I_N1 = np.eye(N+1)
        G = dkdn = np.vstack([ cross(C(i) @ np.kron(M1,I_2) @ D_Adr, C(i) @ np.kron(M2,I_2) @ p.T.flatten()) + cross(C(i) @ np.kron(M1,I_2) @ p.T.flatten(), C(i) @ np.kron(M2,I_2) @ D_Adr) for i in range(1,2+N)])
        k_0 = np.cross(dr.T,ddr.T) 
        M = np.kron(I_N1, [0,1,0]) @ mpc.F
        K = np.kron(I_N1, [0,1,0]) @ mpc.Ex0
        #n = (M @ u + K)

        # curvature of reference trajectory r
        k_0 = k_0.reshape((N+1,1))
        dP = 2*(2*M.T @ G.T @ G @ M)
        dq = (2*K.T @ G.T @ G @ M + 2*k_0.T @ G @ M).T
        mpc.old_P = mpc.P.copy()
        mpc.old_q = mpc.q.copy()
        mpc.dP = weight * dP
        mpc.dq = weight * dq
        mpc.P += weight * dP
        mpc.q += weight * dq
        return 

    def addDeviationObjective(self,weight):
        mpc = self.mpc
        x0 = self.x0
        N = self.N
        I_2 = np.eye(2)
        I_N1 = np.eye(N+1)
        M = np.kron(I_N1, [0,1,0]) @ mpc.F
        K = np.kron(I_N1, [0,1,0]) @ mpc.Ex0
        Q = weight * np.eye(N+1)

        mpc.P += weight * M.T @ Q @ M
        mpc.q += weight * K.T @ Q @ K
        return


    # get the constraint matrix G h for a single n constraint
    # Note that the step here does not contain the first step x0
    def getGhForN(self,step,val,is_max_constrain):
        if (step>= self.N):
            return None
        mpc = self.mpc
        idx = step*self.n + 1
        M = np.zeros(self.N*self.n)
        M[idx] = 1
        # selection matrix, select [x1..xp] from x0..xp
        C = np.eye((self.N+1)*self.n)[self.n:,:]
        if (is_max_constrain):
            G = M @ C @ mpc.F
            h = val - M @ C @ mpc.Ex0
        else:
            G = -M @ C @ mpc.F
            h = -val + M @ C @ mpc.Ex0
        return (G,h)

    def calcDerivative(self,curve):
        # find first and second derivative
        dr = []
        ddr = []
        n = curve.shape[0]
        ds = self.s_vec[1]-self.s_vec[0]
        for i in range(1,n-1):
            rl = curve[i-1,:]
            r = curve[i,:]
            rr = curve[i+1,:]
            points = [rl, r, rr]
            ((al,a,ar),(bl,b,br)) = self.lagrangeDer(points,ds=[ds,ds])
            dr.append(al*rl+a*r+ar*rr)
            ddr.append(bl*rl+b*r+br*rr)
        dr = np.array(dr)
        ddr = np.array(ddr)
        dr = np.vstack([dr[0],dr,dr[-1]])
        ddr = np.vstack([ddr[0],ddr,ddr[-1]])
        return (dr,ddr)

    # right turn negative curvature
    def calcCurvature(self,dr_vec,ddr_vec):
        # ccw 90 deg
        A = np.array([[0,-1],[1,0]])
        curvature = np.dot( (A @ dr_vec.T).T, ddr_vec.T).flatten()
        return curvature

    # calculate segment cumulative length
    def calcArcLen(self,path):
        dr = np.vstack([np.zeros((1,2)),np.diff(path, axis=0)])
        ds = np.linalg.norm(dr,axis=1)
        self.ref_path_s = np.cumsum(ds)

    def stateTrajToCartesianTraj(self,traj):
        cartesian_traj = []
        A = np.array([[0,-1],[1,0]])
        for i in range(traj.shape[0]):
            idx = np.searchsorted(self.ref_path_s,traj[i,0]%self.track.raceline_len_m,side='left')
            xy = self.ref_path[idx] + A @ self.dr[idx] * traj[i,1]
            cartesian_traj.append(xy)
        return np.array(cartesian_traj)

    def plotStateTraj(self,traj):
        traj = self.stateTrajToCartesianTraj(traj)
        plt.plot(traj[:,0],traj[:,1])
        return

    def plotTrack(self):
        # get left boundary
        plt.plot(self.ref_path[:,0],self.ref_path[:,1],'--')
        plt.plot(self.left_boundary[:,0],self.left_boundary[:,1],'-')
        plt.plot(self.right_boundary[:,0],self.right_boundary[:,1],'-')

    # given state x =[s,n,...]
    # plot a dot where the car should be
    def plotCar(self,x):
        # ccw 90 deg
        A = np.array([[0,-1],[1,0]])
        idx = self.getIndex(x[0])
        car_pos = self.ref_path[idx] + A @ self.dr[idx] * x[1]
        plt.plot(car_pos[0],car_pos[1],'o')

    def plotOpponent(self,opponent_state):
        sn = opponent_state
        # left
        s_range = np.linspace(sn[0]-self.opponent_length/2,sn[0]+self.opponent_length/2,10)
        n_range = np.ones_like(s_range)*(sn[1]+self.opponent_width/2)
        traj = np.vstack([s_range,n_range]).T
        traj = self.stateTrajToCartesianTraj(traj)
        plt.plot(traj[:,0],traj[:,1],'b-')
        # right
        n_range = np.ones_like(s_range)*(sn[1]-self.opponent_width/2)
        traj = np.vstack([s_range,n_range]).T
        traj = self.stateTrajToCartesianTraj(traj)
        plt.plot(traj[:,0],traj[:,1],'b-')
        # rear
        n_range = np.linspace((sn[1]-self.opponent_width/2),(sn[1]+self.opponent_width/2),10)
        s_range = np.ones_like(n_range)*(sn[0]-self.opponent_length/2)
        traj = np.vstack([s_range,n_range]).T
        traj = self.stateTrajToCartesianTraj(traj)
        plt.plot(traj[:,0],traj[:,1],'b-')
        # front
        n_range = np.linspace((sn[1]-self.opponent_width/2),(sn[1]+self.opponent_width/2),10)
        s_range = np.ones_like(n_range)*(sn[0]+self.opponent_length/2)
        traj = np.vstack([s_range,n_range]).T
        traj = self.stateTrajToCartesianTraj(traj)
        plt.plot(traj[:,0],traj[:,1],'b-')

    def getIndex(self,s):
        idx = np.searchsorted(self.ref_path_s,s%self.track.raceline_len_m,side='left')
        return idx

    # first order numerical differentiation matrix
    # M1: (p,p)
    # r' = M1 @ r, r = [r_1,r_2,...r_k], r_k = r(s_k), ds = s_{k+1}-s_k
    def getDiff1Matrix(self,N,ds):
        I_N_2 = np.eye(N-2)
        middle_1 = np.hstack([-I_N_2,np.zeros((N-2,2))])
        middle_2 = np.hstack([np.zeros((N-2,2)),I_N_2])
        top = np.zeros((1,N))
        top[0,0] = -3
        top[0,1] = 4
        top[0,2] = -1
        bottom = np.zeros((1,N))
        bottom[0,-1] = 3
        bottom[0,-2] = -4
        bottom[0,-3] = 1
        M1 = np.vstack([top,middle_1+middle_2,bottom])/(2.0*ds)
        return M1

    # second order numerical differentiation matrix
    # M2: (p,p)
    # r' = M2 @ r, r = [r_1,r_2,...r_k], r_k = r(s_k), ds = s_{k+1}-s_k
    def getDiff2Matrix(self,N,ds):
        I_N_2 = np.eye(N-2)
        middle_1 = np.hstack([I_N_2,np.zeros((N-2,2))])
        middle_2 = np.hstack([np.zeros((N-2,1)),-2*I_N_2,np.zeros((N-2,1))])
        middle_3 = np.hstack([np.zeros((N-2,2)),I_N_2])
        middle = (middle_1+middle_2+middle_3)
        top = np.zeros((1,N))
        top[0,0] = 2
        top[0,1] = -5
        top[0,2] = 4
        top[0,3] = -1
        bottom = np.zeros((1,N))
        bottom[0,-1] = 2
        bottom[0,-2] = -5
        bottom[0,-3] = 4
        bottom[0,-4] = -1
        M2 = np.vstack([top,middle,bottom])/ds**2
        return M2

    # given three points, calculate first and second derivative as a linear combination of the three points rl, r, rr, which stand for r_(k-1), r_k, r_(k+1)
    # return: 2*3, tuple
    #       ((al, a, ar),
    #        (bl, b, br))
    # where f'@r = al*rl + a*r + ar*rr
    # where f''@r = bl*rl + b*r + br*rr
    # ds, arc length between rl,r and r, rr 
    # if not specified, |r-rl|_2 will be used as approximation
    def lagrangeDer(self,points,ds=None):
        rl,r,rr = points
        dist = lambda x,y:((x[0]-y[0])**2 + (x[1]-y[1])**2)**0.5
        if ds is None:
            sl = -dist(rl,r)
            sr = dist(r,rr)
        else:
            sl = -ds[0]
            sr = ds[1]

        try:
            al = - sr/sl/(sl-sr)
            a = -(sl+sr)/sl/sr
            ar = -sl/sr/(sr-sl)

            bl = 2/sl/(sl-sr)
            b = 2/sl/sr
            br = 2/sr/(sr-sl)
        except Warning as e:
            print(e)

        return ((al,a,ar),(bl,b,br))

    # construct a fifth order bezier curve passing through endpoints r
    # matching first and second derivative dr, ddr
    # r (2,2)
    # dr (2,2),  taken w.r.t. arc length s
    # ddr (2,2), taken w.r.t. arc length s 
    # ds: arc length between the endpoints
    def bezierCurve(self,r,dr,ddr,ds=None):
        rl,rr = r
        drl,drr = dr
        ddrl,ddrr = ddr

        dist = lambda x,y:((x[0]-y[0])**2 + (x[1]-y[1])**2)**0.5
        if ds is None:
            ds = dist(rl,rr)

        # two sets of equations, one for x, one for y
        #bx = np.array([rl[0],rr[0],drl[0],drr[0],ddrl[0],ddrr[0]]).T
        #by = np.array([rl[1],rr[1],drl[1],drr[1],ddrl[1],ddrr[1]]).T

        # dr = dr/ds = dr/dt * dt/ds
        # we want dB/dt = dr/dt = dr(input) * ds/dt = dr * ds(between two endpoints)
        bx = np.array([rl[0],rr[0],drl[0]*ds,drr[0]*ds,ddrl[0]*ds*ds,ddrr[0]*ds*ds]).T
        by = np.array([rl[1],rr[1],drl[1]*ds,drr[1]*ds,ddrl[1]*ds*ds,ddrr[1]*ds*ds]).T
        b = np.vstack([bx,by]).T

        # x_x = P0_x, P1_x ... P5_x
        # x_y = P0_y, P1_y ... P5_y
        A = [[ 1, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 1],
             [-5, 5, 0, 0, 0, 0],
             [ 0, 0, 0, 0,-5, 5],
             [20,-40,20,0, 0, 0],
             [0 , 0, 0,20,-40,20]]
        A = np.array(A)

        try:
            sol = np.linalg.solve(A,b)
        except np.linalg.LinAlgError:
            print_error("can't solve bezier Curve")

        # return the control points
        P = sol
        return P

    # generate a bezier spline matching derivative estimated from lagrange interpolation
    # return: vector function, domain [0,len(points)]
    def bezierSpline(self,p,dp,ddp,ds):
        N = self.N
        assert p.shape == (N+1,2)
        assert dp.shape == (N+1,2)
        assert ddp.shape == (N+1,2)

        P = []
        for i in range(N):
            # generate bezier spline segments
            rl = p[i]
            r  = p[i+1]
            section_P = self.bezierCurve([rl,r],[dp[i],dp[i+1]],[ddp[i],ddp[i+1]],ds=ds)

            # NOTE testing
            # DEBUG
            B = lambda t,p: (1-t)**5*p[0] + 5*t*(1-t)**4*p[1] + 10*t**2*(1-t)**3*p[2] + 10*t**3*(1-t)**2*p[3] + 5*t**4*(1-t)*p[4] + t**5*p[5]
            x_i = B(0,section_P[:,0])
            y_i = B(0,section_P[:,1])
            x_f = B(1,section_P[:,0])
            y_f = B(1,section_P[:,1])
            assert np.isclose(x_i,rl[0],atol=1e-5) and np.isclose(y_i,rl[1],atol=1e-5) and np.isclose(x_f,r[0],atol=1e-5) and np.isclose(y_f,r[1],atol=1e-5)

            P.append(section_P)

        # shape: N * 6 * 2
        return np.array(P)

    # P: array of control points, shape n*2*5
    # u (iterable): parameter, domain [0,n], where n is number of break points in spline generation

    def evalBezierSpline(self,P,u,der=0):
        n = len(P)
        u = np.array(u).flatten()
        u[u<0]=0
        u[u>n]=n
        #assert (u>=0).all()
        #assert (u<=n).all()
        du = 0.001

        B = lambda t,p: (1-t)**5*p[0] + 5*t*(1-t)**4*p[1] + 10*t**2*(1-t)**3*p[2] + 10*t**3*(1-t)**2*p[3] + 5*t**4*(1-t)*p[4] + t**5*p[5]

        if der==0:
            try:
                r = [ [B(uu%1,np.array(P[int(uu),:,0])),B(uu%1,np.array(P[int(uu),:,1]))] for uu in u]
            except Warning as e:
                print(e)
            return np.array(r)
        elif der==1:
            return (self.evalBezierSpline(P,u+du) - self.evalBezierSpline(P,u))/du
        elif der==2:
            return (self.evalBezierSpline(P,u+2*du,der=1) - self.evalBezierSpline(P,u,der=1))/(2*du)

if __name__=='__main__':
    planner = Planner()
    #planner.demoMatrixDiff()
    planner.demoSingleControl()
