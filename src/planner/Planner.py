# MPC based trajectory planner
import matplotlib.pyplot as plt
import numpy as np
from MPC import MPC
from time import time
from math import floor
from itertools import product
import scipy.optimize
from scipy.linalg import block_diag

class Planner:
    def __init__(self):
        # p: prediction horizon
        self.N = N = horizon = 50
        self.opponent_length = 0.17*2
        self.opponent_width = 0.08*2
        return

    # generate a path
    def genPath(self):
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
        self.ds = ds = s_vec[1] - s_vec[0]
        t_vec = [0]
        for i in range(s_vec.shape[0]-1):
            t = scipy.optimize.root(lambda m:tToS(t_vec[-1],m)-ds, t_vec[-1])
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
        A = np.array([[0,-1],[1,0]])
        tangent_dir = (A @ self.dr.T)/np.linalg.norm(self.dr,axis=1)
        self.left_boundary = (tangent_dir * self.left_limit).T + self.ref_path
        self.right_boundary = (tangent_dir * self.right_limit).T + self.ref_path

        self.calcArcLen(self.ref_path)

    # TODO
    # test numerical differentiation with matrix multiplication
    def demoMatrixDiff(self):
        # generate r(s)
        self.genPath()
        # generate n(s), lateral offset
        t = self.t_vec
        A = np.array([[0,-1],[1,0]])
        n = 0.3*np.sin(5*t)
        dn = 0.3*5*np.cos(5*t)
        ddn = -0.3*25*np.sin(5*t)
        dddr = np.vstack([np.zeros_like(t),-np.cos(t)])
        # p(s)
        r = self.r.T
        dr = self.dr.T
        dr_norm = np.linalg.norm(dr,axis=0)
        ddr = self.ddr.T
        p = r + A @ dr/dr_norm * n
        dp = dr + dn * (A @ dr) + n * (A @ ddr)
        ddp = ddr + ddn * (A @ dr) + dn *(A @ ddr) + dn * (A @ ddr) + n * (A @ dddr)

        N = self.N

        # select a short horizon
        idx = 300
        r = self.r[idx:idx+N,:]
        dr = self.dr[idx:idx+N,:]
        ddr = self.ddr[idx:idx+N,:]
        r_vec = r.flatten()
        n = n[idx:idx+N]
        dn = dn[idx:idx+N]
        ddn = ddn[idx:idx+N]
        p = p.T[idx:idx+N,:]
        dp = dp.T[idx:idx+N,:]
        ddp = ddp.T[idx:idx+N,:]

        k_r = np.cross(dr,ddr) / np.linalg.norm(dr,axis=1)**3

        # test first derivative
        M1 = planner.getDiff1Matrix(N,self.ds)
        M1 = np.kron(M1,np.eye(2))
        # let s_k be first N elements from self.s_vec
        # select {r_k}
        test_dr = M1 @ r_vec
        test_dr = test_dr.reshape((N,2))
        print("M1 result err")
        print(dr - test_dr)
        # SUCCESS !

        # test second derivative
        M2 = planner.getDiff2Matrix(N,self.ds)
        M2 = np.kron(M2,np.eye(2))
        # let s_k be first N elements from self.s_vec
        # select {r_k}
        test_ddr = M2 @ r_vec
        test_ddr = test_ddr.reshape((N,2))
        print("M2 result err")
        print(ddr - test_ddr)
        # first and last value should be different as ddr from self.lagrangeDer() is not done correctly
        # SUCCESS

        # test matrix representation for cross product
        # NOTE this only works in 1D
        r1 = np.array([[6,5]]).T
        r2 = np.array([[3,7]]).T
        c1 = np.cross(r1,r2,axisa=0,axisb=0)
        c2 = (A @ r1).T @ r2
        print(c1)
        print(c2)
        # SUCCESS

        # test p calculation, this requires diag() operation that's subideal
        I_N = np.eye(N)
        I_2N = np.eye(2*N)
        # TODO
        diag = np.kron(I_N,np.array([[1,1]]).T) @ n
        n_diag = np.sum([I_2N[:,[i]].T @ diag * I_2N[:,[i]] @ I_2N[:,[i]].T for i in range(diag.shape[0])],axis=0)
        test_p = r.flatten() +  n_diag @ np.kron(I_N,A) @ dr.flatten()
        print("p")
        print(test_p-p.flatten())
        # SUCCESS

        # test p', or first derivative
        test_dp = M1 @ test_p
        test_dp = test_dp.reshape((N,2))
        print("dp")
        print(test_dp - dp)
        # SUCCESS

        # test p''
        test_ddp = M2 @ test_p
        test_ddp = test_ddp.reshape((N,2))
        print("ddp")
        print(test_ddp - ddp)
        # SUCCESS

        # test curvature calculation
        p = p.T
        dp = dp.T
        ddp = ddp.T
        r = r.T
        dr = dr.T
        ddr = ddr.T

        # plot tangent circles
        idx = 20

        # test curvature calculation with linear approximation
        def cross(A,B):
            return np.cross(A,B,axis=0)
        def C(i):
            if i==1:
                return np.hstack([np.eye(2),np.zeros((2,(N-i)*2))])
            if i==N:
                return np.hstack([np.zeros((2,2*(i-1))),np.eye(2)])
            else:
                return np.hstack([np.zeros((2,2*(i-1))),np.eye(2),np.zeros((2,(N-i)*2))])
        # truth
        # ccw 90 deg
        A = np.array([[0,-1],[1,0]])
        M = A @ dr
        D_Adr = block_diag(* [M[:,[i]] for i in range(N)])
        M1 = planner.getDiff1Matrix(N,self.ds)
        M2 = planner.getDiff2Matrix(N,self.ds)
        I_2 = np.eye(2)
        '''
        dkdn = np.zeros( (N,N) )
        for i in range(N):
            dkdn += cross(C(i) @ np.kron(M1,I_2) @ D_Adr, C(i) @ np.kron(M2,I_2) @ p.T.flatten()) + cross(C(i) @ np.kron(M1,I_2) @ p.T.flatten(), C(i) @ np.kron(M2,I_2) @ D_Adr)
        '''
        dkdn = np.vstack([ cross(C(i) @ np.kron(M1,I_2) @ D_Adr, C(i) @ np.kron(M2,I_2) @ p.T.flatten()) + cross(C(i) @ np.kron(M1,I_2) @ p.T.flatten(), C(i) @ np.kron(M2,I_2) @ D_Adr) for i in range(1,1+N)])

        k_r_unnormal = np.cross(dr.T,ddr.T) # base
        k_p_unnormal = np.cross(dp.T,ddp.T) # ground truth
        mse = lambda a,b: np.sum((a-b)**2)
        print("k_r small", np.sum(k_r_unnormal**2))
        print("k_p large", np.sum(k_p_unnormal**2))
        print("biggest |k - k0| = %.3f"%(mse(k_r_unnormal,k_p_unnormal)))
        k_unnormal = k_r_unnormal + 0.3* dkdn @ n # to test
        print("bigger |k - half k_new| = %.3f"%(mse(k_unnormal , k_p_unnormal)))
        k_unnormal = k_r_unnormal + 0.7* dkdn @ n # to test
        print("smaller |k - half k_new| = %.3f"%(mse(k_unnormal , k_p_unnormal)))
        k_unnormal = k_r_unnormal + dkdn @ n # to test
        print("smallest |k - k_new| = %.3f"%(mse(k_unnormal , k_p_unnormal)))
        breakpoint()


        # reference method
        k_p = np.cross(dp.T,ddp.T) / np.linalg.norm(dr,axis=0)**3
        radius = 1/k_p[idx]
        center = p[:,idx] + A @ dp[:,idx]/np.linalg.norm(dp[:,idx]) * radius
        theta = np.linspace(0,np.pi*2,1000)
        circle_x = np.cos(theta)*radius + center[0]
        circle_y = np.sin(theta)*radius + center[1]
        tangent = p[:,idx] + dp[:,idx]/np.linalg.norm(dp[:,idx])
        normal = p[:,idx] + A @ dp[:,idx]/np.linalg.norm(dp[:,idx])
        plt.plot(circle_x,circle_y,'--',label='circle p')
        plt.plot([p[0,idx],tangent[0]],[p[1,idx],tangent[1]],'--')
        plt.plot([p[0,idx],normal[0]],[p[1,idx],normal[1]],'--')
        plt.plot([p[0,idx],center[0]],[p[1,idx],center[1]],'--')

        # matrix method
        test_p = test_p.reshape((N,2))
        test_p = test_p.T
        test_dp = test_dp.T
        test_ddp = test_ddp.T

        test_k_p = np.cross(test_dp.T,test_ddp.T) / np.linalg.norm(test_dr.T,axis=0)**3
        test_k_p = np.cross(test_dp.T,test_ddp.T)
        radius = 1/test_k_p[idx]
        center = test_p[:,idx] + A @ test_dp[:,idx]/np.linalg.norm(test_dp[:,idx]) * radius
        theta = np.linspace(0,np.pi*2,1000)
        circle_x = np.cos(theta)*radius + center[0]
        circle_y = np.sin(theta)*radius + center[1]
        tangent = test_p[:,idx] + test_dp[:,idx]/np.linalg.norm(test_dp[:,idx])
        normal = test_p[:,idx] + A @ test_dp[:,idx]/np.linalg.norm(test_dp[:,idx])
        plt.plot(circle_x,circle_y,label='circle test p')
        plt.plot([test_p[0,idx],tangent[0]],[test_p[1,idx],tangent[1]])
        plt.plot([test_p[0,idx],normal[0]],[test_p[1,idx],normal[1]])
        plt.plot([test_p[0,idx],center[0]],[test_p[1,idx],center[1]])

        plt.plot(r[0,:],r[1,:],label='r')
        plt.plot(p[0,:],p[1,:],label='p')
        plt.plot(p[0,idx],p[1,idx],'*')
        plt.legend()
        plt.axis('equal')
        plt.show()
        




    # demonstrate curvature calculation, verify with inscribe circle
    def demoCurvature(self):
        # define reference path
        # r = <t, sin(t)>
        # r' = <1, cos(t)>
        # r'' = <0, -sin(t)>
        t = np.linspace(0,2*np.pi,1000)
        r = np.vstack([t,np.sin(t)])
        dr = np.vstack([np.ones_like(t),np.cos(t)])
        ddr = np.vstack([np.zeros_like(t),-np.sin(t)])
        dddr = np.vstack([np.zeros_like(t),-np.cos(t)])
        # alternative reference path
        #t = np.linspace(0,6,101)
        #r = np.vstack([3*t**3+4*t,6*t**2+10])
        #dr = np.vstack([9*t**2+4,12*t])
        #ddr = np.vstack([18*t,12*np.ones_like(t)])

        # calculate curvature for r(t)
        k_r = np.cross(dr.T,ddr.T) / np.linalg.norm(dr,axis=0)**3

        # plot tangent circle
        idx = 250
        # ccw 90 deg
        A = np.array([[0,-1],[1,0]])
        radius = 1/k_r[idx]
        center = r[:,idx] + A @ dr[:,idx]/np.linalg.norm(dr[:,idx]) * radius
        theta = np.linspace(0,np.pi*2,1000)
        circle_x = np.cos(theta)*radius + center[0]
        circle_y = np.sin(theta)*radius + center[1]
        # plotting
        '''
        plt.plot(r[0,:],r[1,:])
        plt.plot(circle_x,circle_y)
        plt.plot(r[0,idx],r[1,idx],'*')
        plt.axis('equal')
        plt.show()
        '''

        # curvature for general path p(t)
        n = 0.1*np.sin(5*t)
        dn = 0.1*5*np.cos(5*t)
        ddn = -0.1*25*np.sin(5*t)
        dr_norm = np.linalg.norm(dr,axis=0)
        p = r + A @ dr/dr_norm * n
        dp_alt = np.diff(p) / (t[1]-t[0])
        dp = dr + dn * (A @ dr) + n * (A @ ddr)
        I = np.eye(2)
        ddp = ddr + ddn * (A @ dr) + dn *(A @ ddr) + dn * (A @ ddr) + n * (A @ dddr)
        # safe to drop third derivative
        #ddp = ddr + ddn * (A @ dr) + dn *(A @ ddr) + dn * (A @ ddr)
        # calculate curvature for r(t)
        #k_p = np.cross(dp.T,ddp.T) / np.linalg.norm(dp,axis=0)**3
        # approximate with constant dr
        k_p = np.cross(dp.T,ddp.T) / np.linalg.norm(dr,axis=0)**3

        # plot tangent circle
        idx = 350
        # ccw 90 deg
        A = np.array([[0,-1],[1,0]])
        radius = 1/k_p[idx]
        center = p[:,idx] + A @ dp[:,idx]/np.linalg.norm(dp[:,idx]) * radius
        theta = np.linspace(0,np.pi*2,1000)
        circle_x = np.cos(theta)*radius + center[0]
        circle_y = np.sin(theta)*radius + center[1]

        tangent = p[:,idx] + dp[:,idx]/np.linalg.norm(dp[:,idx])
        normal = p[:,idx] + A @ dp[:,idx]/np.linalg.norm(dp[:,idx])

        plt.plot(r[0,:],r[1,:])
        plt.plot(p[0,:],p[1,:])
        plt.plot(circle_x,circle_y)
        plt.plot(p[0,idx],p[1,idx],'*')
        plt.plot([p[0,idx],tangent[0]],[p[1,idx],tangent[1]])
        plt.plot([p[0,idx],normal[0]],[p[1,idx],normal[1]])
        plt.plot([p[0,idx],center[0]],[p[1,idx],center[1]])
        plt.axis('equal')
        plt.show()

    # control with state = [s,n,ds]
    def demoSingleControl(self):
        # state x:[s,n,ds]
        # u: [dn] first derivative of n
        self.n = n = 3
        self.m = m = 1
        self.genPath()

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
        mpc = MPC()
        self.mpc = mpc
        n = self.n
        m = self.m
        N = self.N
        ds = self.ds
        # setup mpc 
        mpc.setup(n,m,n,N)
        dt = self.dt
        A = np.eye(n)
        A[0,2] = dt
        B = np.array([[0,1,0]]).T*dt
        P = np.diag([1,1,0])
        Q = np.eye(m)

        t = time()
        x0 = np.array(x0).T.reshape(-1,1)
        self.x0 = x0
        xref = [x0[0,0]+x0[2,0]*(self.N+1),0,ds]
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

        # FIXME hack to force u0 = 0
        G_u0 = np.zeros((2,N))
        G_u0[0,0] = 1
        G_u0[1,0] = -1
        h_u0 = np.zeros((2,1))
        h_u0[0,0] = 0.1
        h_u0[1,0] = -0.1
        # save current mpc matrices
        # solve for different scenarios
        G = mpc.G
        h = mpc.h
        G = np.vstack([G,G_u0])
        h = np.vstack([h,h_u0])

        if (scenarios is None):
            print("no opponent in horizon")
            dt = time()-t
            mpc.solve()
            if (mpc.h is not None):
                print(mpc.h.shape)
            print("freq = %.2fHz"%(1/dt))
            # state_traj in curvilinear frame
            state_traj = mpc.F @ mpc.u + mpc.Ex0
            state_traj = state_traj.reshape((N,n))
            state_traj = np.vstack([x0.T,state_traj])
            sols.append( (mpc.u,state_traj) )
        else:
            #scenarios = [scenarios[3]]
            for case in scenarios:
                print("case ---- ")
                mpc.G = np.vstack([G,case[0]])
                mpc.h = np.vstack([h,case[1]])
                duration = time()-t
                mpc.solve()
                if (mpc.h is not None):
                    print("constraints: %d"%(mpc.h.shape[0]))
                print("freq = %.2fHz"%(1/duration))
                # state_traj in curvilinear frame
                state_traj = mpc.F @ mpc.u + mpc.Ex0
                state_traj = state_traj.reshape((N,n))
                state_traj = np.vstack([x0.T,state_traj])
                sols.append( (mpc.u,state_traj) )

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
        M1 = planner.getDiff1Matrix(N,ds)
        M2 = planner.getDiff2Matrix(N,ds)

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
        G1 = M @ mpc.F
        N = np.ones((N,1))*self.track_width/2
        h1 = N - M @ mpc.Ex0

        G2 = -M @ mpc.F
        h2 = N + M @ mpc.Ex0

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
            if (step_end > self.N):
                continue
            opponent_constraints.append([])
            if (left > opponent[1]+self.opponent_width/2):
                # there's space in left for passing
                left_bound = opponent[1]+self.opponent_width/2
                retval1 = self.getGhForN(step_begin,left_bound,False)
                retval2 = self.getGhForN(step_end,left_bound,False)
                if (retval1 is not None and retval2 is not None):
                    G1,h1 = retval1
                    G2,h2 = retval2
                    G = np.vstack([G1,G2])
                    h = np.vstack([h1,h2])
                    opponent_constraints[-1].append((G,h))
                    print("feasible path, oppo %d, left, step %d-%d, n > %.2f"%(opponent_idx, step_begin, step_end,left_bound))
                else:
                    print("error, out of bound")
                
            if (right < opponent[1]-self.opponent_width/2):
                # there's space in right for passing
                right_bound = opponent[1]-self.opponent_width/2
                retval1 = self.getGhForN(step_begin,right_bound,True)
                retval2 = self.getGhForN(step_end,right_bound,True)
                if (retval1 is not None and retval2 is not None):
                    G1,h1 = retval1
                    G2,h2 = retval2
                    G = np.vstack([G1,G2])
                    h = np.vstack([h1,h2])
                    opponent_constraints[-1].append((G,h))
                    print("feasible path, oppo %d, right, step %d-%d, n < %.2f"%(opponent_idx, step_begin, step_end,right_bound))
                else:
                    print("error, out of bound")
            opponent_idx += 1

        # possible combination of AND constraints
        # e.g. opponent 0 can be passed on left or right
        #      opponent 1 can be passed on left 
        # 2 scenarios, (left,left) (left,right)

        # if no opponent is in sight
        if (len(opponent_constraints)==0):
            return None
        cons_combination = [ cons for cons in product(*opponent_constraints)]

        scenarios = [ (np.vstack( [con[0] for con in cons] ), np.vstack( [con[1] for con in cons] )) for cons in product(*opponent_constraints)]
            
        return scenarios
    # dr:(2,N)
    # ds: float scalar
    # p: (2,N)
    def addCurvatureNormObjective(self, mpc, x0, weight, n_estimate=None):
        N = self.N
        ds = self.ds
        idx = []
        for i in range(N):
            this_s = x0[0,0] + i*self.dt*x0[2,0]
            i = this_s / ds
            idx.append(i)
        idx = np.array(idx,dtype=int).flatten()

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
        M1 = planner.getDiff1Matrix(N,ds)
        M2 = planner.getDiff2Matrix(N,ds)
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
        mpc = self.mpc
        idx = step*self.n + 1
        if (step > self.N):
            return None
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
            idx = np.searchsorted(self.ref_path_s,traj[i,0],side='left')
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

    # given state x =[s,n,ds,dn]
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
        idx = np.searchsorted(self.ref_path_s,s,side='left')
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
