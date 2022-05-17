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
        '''
        self.opponent_length = 0.17*2
        self.opponent_width = 0.08*2
        self.best_solution = None
        self.best_local_traj = None
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
    def localTrajectory(self,state):
        coord = (state[0],state[1])
        heading = state[2]
        omega = state[5]
        vf = state[3]
        vs = state[4]
        traj = self.best_local_traj
        dist = np.sqrt((traj[:,0] - coord[0])**2 + (traj[:,1] - coord[1])**2)
        # if plan() was called right before, then idx should be 0
        idx = np.argmin(dist)

        # find orientation
        dy = traj[idx+1,1]-traj[idx,1]
        dx = traj[idx+1,0]-traj[idx,0]
        orientation = np.arctan2(dy,dx)

        # find offset
        vec_path_tangent = (dx,dy)
        vec_path_to_car = (state[0] - traj[idx,0], state[1] - traj[idx,1])
        offset = np.cross(vec_path_tangent, vec_path_to_car) / np.linalg.norm(vec_path_tangent)

        sols = self.solutions
        best_sol_idx = self.best_solution_index
        self.plotSolutions(sols)
        self.plotSolutions([sols[best_sol_idx]],color=(100,100,100))

        # FIXME hacky
        (_,_,_,_,v_target) = self.track.localTrajectory(state)
        # offset: negative offset means left steering needed
        #(local_ctrl_pnt,offset,orientation,curvature,v_target) = retval
        retval = (None,offset,orientation,None,v_target)
        return retval

    def plan(self):
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
        best_sol_idx = np.argmin([x[2] for x in sols])

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
        self.best_local_traj = self.stateTrajToCartesianTraj(self.best_solution[1])
        return

    def plotSolutions(self,sols,color=(255,51,204)):
        img = self.main.visualization.visualization_img
        for (u_vec, state_traj,cost) in sols:
            traj = self.stateTrajToCartesianTraj(state_traj)
            img = self.track.drawPolyline(traj,lineColor=color,img=img)
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
        self.raceline_points = self.ref_path.T

        for i in range(self.discretized_raceline_len):
            # find normal direction
            coord = self.raceline_points[:,i]
            heading = self.raceline_headings[i]

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
            return img
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
        self.constructTrackBoundaryConstraint(mpc)
        scenarios = self.constructOpponentConstraint(mpc,opponent_state)
        self.addCurvatureNormObjective(mpc, x0, weight=1, n_estimate=None )
        self.scenarios = scenarios
        sols = []

        # u0 is unusually large
        # constrain u0=u1
        G_u0 = np.zeros((2,N))
        G_u0[0,0] = 1
        G_u0[0,1] = -1
        G_u0[1,0] = -1
        G_u0[1,1] = 1
        h_u0 = np.zeros((2,1))
        h_u0[0,0] = 0.1
        h_u0[1,0] = -0.1
        # save current mpc matrices
        # solve for different scenarios
        G = mpc.G
        h = mpc.h
        G = np.vstack([G,G_u0])
        h = np.vstack([h,h_u0])


        # iterate over all scenarios
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
            # state_traj in curvilinear frame
            state_traj = mpc.F @ mpc.u + mpc.Ex0
            state_traj = state_traj.reshape((N,n))
            state_traj = np.vstack([x0.T,state_traj])
            sols.append( (mpc.u,state_traj,mpc.cost) )
            duration = time()-t
            self.print_info("case freq = %.2fHz"%(1/duration))

        # if there's no opponent in sight
        # or no way to pass them
        # TODO don't crash into opponents if ego can't pass
        if (len(scenarios)==0):
            self.print_info("no opponent in horizon")
        elif (len(sols)==0):
            self.print_info("can't pass opponent")

        if (len(sols)==0):
            dt = time()-t
            success = mpc.solve()
            if (not success):
                print_warning("MPC fail to find solution")
                # still have to use this
            if (mpc.h is not None):
                #print(mpc.h.shape)
                pass
            # state_traj in curvilinear frame
            state_traj = mpc.F @ mpc.u + mpc.Ex0
            state_traj = state_traj.reshape((N,n))
            state_traj = np.vstack([x0.T,state_traj])
            sols.append( (mpc.u,state_traj,mpc.cost) )

        self.print_info("found %d valid trajectory from  %d scenarios"%(len(sols),len(scenarios)))
        duration = time() - planner_t0
        self.print_info("planner step freq = %.2fHz"%(1/duration))

        '''
        # calculate progress and curvature cost
        u = sols[0][0]
        mse = lambda a: np.sum((a)**2)

        progress_cost = 0.5*u.T @ mpc.old_P @ u + mpc.old_q.T @ u
        cur_cost = 0.5*u.T @ mpc.dP @ u + mpc.dq.T @ u
        total_cost = 0.5*u.T @ mpc.P @ u + mpc.q.T @ u
        print("progress cost = %.2f"%(progress_cost))
        print("curvature cost = %.2f"%(cur_cost))
        print("total cost = %.2f"%(total_cost))

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
            k_p_norm_vec.append(k_p_norm)
        '''

        return sols

    # construct the state limits
    def constructTrackBoundaryConstraint(self,mpc):
        # create additional lines for Gx<h
        # track boundary limits
        N = self.N
        if (self.n==4):
            M = np.kron(np.eye(N),np.array([[0,1,0,0]]))
        elif (self.n==3):
            M = np.kron(np.eye(N),np.array([[0,1,0]]))
        elif (self.n==2):
            M = np.kron(np.eye(N),np.array([[0,1]]))
        # left is positive
        G1 = M @ mpc.F
        # left
        L = self.left_limit[self.idx].reshape((N,1))
        #L = np.ones((N,1))*self.track_width/2
        h1 = L - M @ mpc.Ex0

        G2 = -M @ mpc.F
        R = -self.right_limit[self.idx].reshape((N,1))
        h2 = R + M @ mpc.Ex0

        if (mpc.G is None):
            mpc.G = np.vstack([G1,G2])
            mpc.h = np.vstack([h1,h2])
        else:
            mpc.G = np.vstack([mpc.G,G1,G2])
            mpc.h = np.vstack([mpc.h,h1,h2])
        return

    def constructOpponentConstraint(self,mpc,opponent_state):
        # TODO quick check feasibility
        # additional constraints to add
        # (G,h)
        # opponent_count * feasible_direction_count
        # one sublist for each opponent
        # sublist contains a set of OR constraints
        # solution need to satisfy ONE constraint from sublist 0, ONE constraint from sublist 1 ...
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
                self.print_info("ignoring opponent at step %d"%(step_begin))
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
    def pickRelevantIndex(self,x0):
        x0 = np.array(x0).flatten()
        idx = []
        s_step = self.s_step
        for i in range(self.N):
            this_s = x0[0] + i*self.dt*x0[2]
            i = (this_s%self.track.raceline_len_m) / s_step
            idx.append(i)
        idx = np.array(idx,dtype=int).flatten()
        return idx

    # dr:(2,N)
    # ds: float scalar
    # p: (2,N)
    def addCurvatureNormObjective(self, mpc, x0, weight, n_estimate=None):
        N = self.N
        s_step = self.s_step
        idx = self.idx

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
        # ccw 90 deg
        A = np.array([[0,-1],[1,0]])
        # this is where iterations may be necessary
        p = r + A @ dr/dr_norm * n_estimate

        def cross(A,B):
            return np.cross(A,B,axis=0)
        def C(i):
            if i==1:
                return np.hstack([np.eye(2),np.zeros((2,(N-i)*2))])
            if i==N:
                return np.hstack([np.zeros((2,2*(i-1))),np.eye(2)])
            else:
                return np.hstack([np.zeros((2,2*(i-1))),np.eye(2),np.zeros((2,(N-i)*2))])

        # ccw 90 deg
        A = np.array([[0,-1],[1,0]])
        M = A @ dr
        D_Adr = block_diag(* [M[:,[i]] for i in range(N)])
        # ds = ds/dt * dt
        ds = x0[2][0] * self.dt
        M1 = self.getDiff1Matrix(N,ds)
        M2 = self.getDiff2Matrix(N,ds)
        I_2 = np.eye(2)
        I_N = np.eye(N)
        G = dkdn = np.vstack([ cross(C(i) @ np.kron(M1,I_2) @ D_Adr, C(i) @ np.kron(M2,I_2) @ p.T.flatten()) + cross(C(i) @ np.kron(M1,I_2) @ p.T.flatten(), C(i) @ np.kron(M2,I_2) @ D_Adr) for i in range(1,1+N)])
        k_0 = np.cross(dr.T,ddr.T) 
        M = np.kron(I_N, [0,1,0]) @ mpc.F
        K = np.kron(I_N, [0,1,0]) @ mpc.Ex0


        k_0 = k_0.reshape((N,1))
        dP = 2*(2*M.T @ G.T @ G @ M)
        dq = (2*K.T @ G.T @ G @ M + 2*k_0.T @ G @ M).T
        mpc.old_P = mpc.P.copy()
        mpc.old_q = mpc.q.copy()
        mpc.dP = weight * dP
        mpc.dq = weight * dq
        mpc.P += weight * dP
        mpc.q += weight * dq
        return 


    # get the constraint matrix G h for a single n constraint
    def getGhForN(self,step,val,is_max_constrain):
        if (step>= self.N):
            return None
        mpc = self.mpc
        idx = step*self.n + 1
        M = np.zeros(self.N*self.n)
        M[idx] = 1
        if (is_max_constrain):
            G = M @ mpc.F
            h = val - M @ mpc.Ex0
        else:
            G = -M @ mpc.F
            h = -val + M @ mpc.Ex0
        return (G,h)

    def calcDerivative(self,curve):
        # find first and second derivative
        dr = []
        ddr = []
        n = curve.shape[0]
        ds = self.s_vec[1]-self.s_vec[0]
        for i in range(1,n-1):
            rl = self.ref_path[i-1,:]
            r = self.ref_path[i,:]
            rr = self.ref_path[i+1,:]
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
if __name__=='__main__':
    planner = Planner()
    #planner.demoMatrixDiff()
    planner.demoSingleControl()
