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
from cs_solver import CSSolver #from cs_solver_covariance_only import CSSolver
import cvxpy as cp
from cvxpy.atoms.affine.trace import trace 
from cvxpy.atoms.affine.transpose import transpose


class LinearizeDynamics():
    def __init__(self,horizon):
        self.dt = 0.01
        self.sim = ethCarSim(0,0,0)
        self.n = 6
        # noise dimension
        self.l = self.n
        self.m = 2
        self.N = horizon
        self.setupModel()
        u_min = np.array((-1,-radians(25)))
        u_max = np.array((1,radians(25)))
        self.solver = CSSolver(self.n, self.m, self.l, self.N, u_min, u_max)
        self.getRefTraj("../log/ethsim/full_state1.p",show=False)
        return 


    def covarianceControl_cvxpy(self,state,control):
        n = self.n
        m = self.m
        N = self.N

        x = state[0]
        y = state[2]

        # find where the car is in reference to reference trajectory
        xx = self.ref_traj[:,0]
        yy = self.ref_traj[:,2]
        dist_sqr = (xx-x)**2 + (yy-y)**2
        # start : index of closest ref point to car
        start = ref_traj_index = np.argmin(dist_sqr)
        
        # linearize dynamics around ref traj
        # As = [A0..A(N-1)]
        As = []
        Bs = []
        ds = []
        for i in range(start,start+self.N):
            A,B,d = self.linearize(self.ref_traj[i,:],self.ref_ctrl[i,:])
            As.append(A)
            Bs.append(B)
            ds.append(d)

        # assemble big matrices for batch dynamics
        As = np.dstack(As)
        Bs = np.dstack(Bs)
        ds = np.dstack(ds).reshape((self.n,1,self.N))
        #A, B, d, D = self.form_long_matrices_LTV(As, Bs, ds, Ds)
        Sigma_epsilon = 1.0
        A, B, d, D = self.make_batch_dynamics(As, Bs, ds, None, Sigma_epsilon)

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
        sigma_f = np.diag([3]*n)

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
        constraints = [cp.bmat([[sigma_f, E_N @(I+B@K)@sigma_y_sqrt], [ sigma_y_sqrt@(I+B @ K).T@E_N.T, I ]]) >= 0]
        #constraints = []
        prob = cp.Problem(objective, constraints)

        print("Optimal J = ", prob.solve())
        print("Optimal Ks: ")
        for i in range(N):
            print(Ks[i].value)

        return K


    def testSimpleDynamics(self):
        print("testSimpleDynamics (2d double integrator)")
        # dynamics: 2D double integrator
        dt = self.dt
        n = self.n = 4
        m = self.m = 2
        l = self.l = self.m
        N = self.N = 10

        # double integrator is LTI
        # As = [A0..A(N-1)]
        # assemble big matrices for batch dynamics
        G = [[1,0,0.2,0], [0,1,0,0.2], [0,0,1,0], [0,0,0,1]]
        G = np.array(G)
        As = np.repeat(G[:,:,np.newaxis], N, axis=2)
        H = [[0, 0], [0, 0], [1, 0], [0, 1]]
        H = np.array(H)
        Bs = np.repeat(H[:,:,np.newaxis], N, axis=2)

        ds = np.zeros((n,1))
        ds = np.repeat(ds[:,:,np.newaxis], N, axis=2)

        Sigma_epsilon = np.diag([1.0,1.0])
        A, B, d, D = self.make_batch_dynamics(As, Bs, ds, None, Sigma_epsilon)

        # test G,H dynamics
        # x+ = G x + H u
        x0 = np.array([0.0,0.0,0.0,0.0])
        v = np.array([0, 1])
        states = [x0]
        sim_steps = self.N
        '''
        x_i = x0
        for i in range(sim_steps):
            #print(x_i)
            x_i = G @ x_i + H @ v
            states.append(x_i.flatten())
        states = np.array(states)
        print("step-by-step dynamics")
        plt.subplot(2,1,1)
        plt.plot(states[:,0], states[:,1])
        plt.title("position trajectory")
        plt.subplot(2,1,2)
        plt.plot(states[:,2], states[:,3])
        plt.title("velocity trajectory")
        plt.show()
        old_states = states

        # test batch dynamics
        # x = A x0 + B u + d (d is zero in this system)
        uu = np.array([v[0],v[1]]*self.N)
        xx = A @ x0 + B @ uu
        states = xx.reshape([N+1,4])


        print("batch dynamics")
        plt.subplot(2,1,1)
        plt.plot(states[:,0], states[:,1])
        plt.title("position x,y")
        plt.subplot(2,1,2)
        plt.plot(states[:,2], states[:,3])
        plt.title("velocity x,y")
        plt.show()
        '''

        #Test CCMPPI
        print("test ccmppi")
        print("without K")
        rollout = 100
        states_vec = []
        for j in range(rollout):
            states_vec.append([])
            x_i = x0.copy()
            for i in range(sim_steps):
                # generate random variable epsilon
                mean = [0.0, 0.0]
                cov = Sigma_epsilon
                epsilon = np.random.multivariate_normal(mean, cov)

                x_i = G @ x_i + H @ (v+epsilon)
                states_vec[j].append(x_i.flatten())

        states_vec = np.array(states_vec)
        plt.subplot(2,1,1)
        for i in range(states_vec.shape[0]):
            # position?
            plt.plot(states_vec[i,:,0],states_vec[i,:,1])
        plt.title("without K")

        # cost matrix
        # TODO tune me
        Q = np.eye(n)
        Q_bar = np.kron(np.eye(N + 1, dtype=int), Q)
        R = np.eye(m)
        R_bar = np.kron(np.eye(N, dtype=int), R)

        # technically incorrect, but we can just specify R_bar_1/2 instead of R_bar
        R_bar_sqrt = R_bar
        Q_bar_sqrt = Q_bar

        # terminal mean constrain
        # TODO tune me
        sigma_f = np.diag([1] * n)

        # setup cvxpy
        I = np.eye(n * (N + 1))
        E_N = np.zeros((n, n * (N + 1)))
        E_N[:, n * (N):] = np.eye(n)

        # assemble K as a diagonal block matrix with K_0..K_N-1 as var
        Ks = [cp.Variable((m, n)) for i in range(N)]
        # K dim: mN x n(N+1)
        K = cp.hstack([Ks[0], np.zeros((m, (N) * n))])
        for i in range(1, N):
            line = cp.hstack([np.zeros((m, n * i)), Ks[i], np.zeros((m, (N - i) * n))])
            K = cp.vstack([K, line])

        objective = cp.Minimize(cp.norm(cp.vec(R_bar_sqrt @ K @ D)) + cp.norm(cp.vec(Q_bar_sqrt @ (I + B @ K) @ D)))

        # TODO verify with Ji
        sigma_y_sqrt = self.nearest_spd_cholesky(D @ D.T)
        # sigma_y_sqrt = D@D.T
        constraints = [
            cp.bmat([[sigma_f, E_N @ (I + B @ K) @ sigma_y_sqrt], [sigma_y_sqrt @ (I + B @ K).T @ E_N.T, I]]) >= 0]
        # constraints = []
        prob = cp.Problem(objective, constraints)

        print("Optimal J = ", prob.solve())
        print("Optimal Ks: ")
        print("note value<1e-5 are displayed as 0")
        for i in range(N):
            simple_K = Ks[i].value.copy()
            simple_K[abs(simple_K) < 1e-5] = 0
            print(simple_K)

        print("with K")
        states_vec = []
        for j in range(rollout):
            states_vec.append([])
            y_i = np.zeros(n)
            x_i = x0.copy()
            for i in range(sim_steps):
                # generate random variable epsilon
                mean = [0.0, 0.0]
                cov = Sigma_epsilon
                epsilon = np.random.multivariate_normal(mean, cov)

                # TODO test me
                x_i = G @ x_i + H @ (v+epsilon + Ks[i].value.copy() @ y_i)
                y_i = G @ y_i + H @ epsilon
                states_vec[j].append(x_i.flatten())

        states_vec = np.array(states_vec)
        plt.subplot(2,1,2)
        for i in range(states_vec.shape[0]):
            # position?
            plt.plot(states_vec[i,:,0],states_vec[i,:,1])
        plt.title("with K")
        plt.show()




    # test cc on a simple dynamics
    # warning: this function is standalone and will mess up other functions' paremeters
    def simpleDynamicsCovarianceControl(self):
        # dynamics: 2D double integrator
        dt = self.dt
        n = self.n = 4
        m = self.m = 2
        l = self.l = self.m
        N = self.N

        # double integrator is LTI
        # As = [A0..A(N-1)]
        # assemble big matrices for batch dynamics
        G = [[1,dt,0,0], [0,1,0,0], [0,0,1,dt], [0,0,0,1]]
        G = np.array(G)
        As = np.repeat(G[:,:,np.newaxis], N, axis=2)
        H = [[(dt**2)/2.0, 0], [dt, 0], [0, (dt**2)/2.0], [0, dt]]
        H = np.array(H)
        Bs = np.repeat(H[:,:,np.newaxis], N, axis=2)

        ds = np.zeros((n,1))
        ds = np.repeat(ds[:,:,np.newaxis], N, axis=2)

        Sigma_epsilon = 50.0
        A, B, d, D = self.make_batch_dynamics(As, Bs, ds, None, Sigma_epsilon)

        # cost matrix 
        # TODO tune me
        Q = np.eye(n)
        Q_bar = np.kron(np.eye(N+1, dtype=int), Q)
        R = np.eye(m)
        R_bar = np.kron(np.eye(N, dtype=int), R)

        # technically incorrect, but we can just specify R_bar_1/2 instead of R_bar
        R_bar_sqrt = R_bar
        Q_bar_sqrt = Q_bar

        # terminal mean constrain
        # TODO tune me
        sigma_f = np.diag([1]*n)

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
        constraints = [cp.bmat([[sigma_f, E_N @(I+B@K)@sigma_y_sqrt], [ sigma_y_sqrt@(I+B @ K).T@E_N.T, I ]]) >= 0]
        #constraints = []
        prob = cp.Problem(objective, constraints)

        print("Optimal J = ", prob.solve())
        print("Optimal Ks: ")
        print("note value<1e-5 are displayed as 0")
        for i in range(N):
            simple_K = Ks[i].value.copy()
            simple_K[abs(simple_K) < 1e-5] = 0
            print(simple_K)

        return K

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


    def setupModel(self):
        # dimension
        self.lf = 0.09-0.036
        self.lr = 0.036
        self.L = 0.09
        # basic properties
        self.Iz = 0.00278
        self.mass = 0.1667

        # tire model
        self.Df = 3.93731
        self.Dr = 6.23597
        self.C = 2.80646
        self.B = 0.51943

        # motor/longitudinal model
        self.Cm1 = 6.03154
        self.Cm2 = 0.96769
        self.Cr = -0.20375
        self.Cd = 0.00000

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
        dx = np.diff(x) / dt
        dy = np.diff(y) / dt
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

        return

    def testGetRefTraj(self):
        self.getRefTraj("../log/ethsim/full_state1.p")
        return

    def update_dynamics(self,states,controls,dt):
        lf = self.lf
        lr = self.lr
        L = self.L

        Df = self.Df
        Dr = self.Dr
        B = self.B
        C = self.C
        Cm1 = self.Cm1
        Cm2 = self.Cm2
        Cr = self.Cr
        Cd = self.Cd
        Iz = self.Iz
        m = self.mass

        states = states.reshape((self.n,-1))
        controls = controls.reshape((self.m,-1))


        x = states[0,:]
        y = states[2,:]
        psi = heading = states[4,:]
        omega = states[5,:]
        throttle = controls[0,:]
        steering = controls[1,:]

        d_omega = np.zeros([states.shape[1]])
        d_vx = np.zeros([states.shape[1]])
        d_vy = np.zeros([states.shape[1]])
        slip_f = np.zeros([states.shape[1]])
        slip_r = np.zeros([states.shape[1]])
        Ffy = np.zeros([states.shape[1]])
        Fry = np.zeros([states.shape[1]])
        Ffx = np.zeros([states.shape[1]])
        Frx = np.zeros([states.shape[1]])

        # change ref frame to car frame
        # vehicle longitudinal velocity
        Vx = vx = states[1,:]*np.cos(psi) + states[3,:]*np.sin(psi)
        Vy = vy = -states[1,:]*np.sin(psi) + states[3,:]*np.cos(psi)

        # for small longitudinal velocity use kinematic model
        mask = vx < 0.05
        # kinematic model
        beta_mask = np.arctan(lr/L*np.tan(steering[mask]))
        norm = lambda a,b:(a**2+b**2)**0.5
        # motor model
        d_vx[mask] = (( Cm1 - Cm2 * vx[mask]) * throttle[mask] - Cr - Cd * vx[mask] * vx[mask])
        vx[mask] = vx[mask] + d_vx[mask] * dt
        vy[mask] = norm(vx[mask],vy[mask])*np.sin(beta_mask)
        omega[mask] = vx[mask]/L*np.tan(steering[mask])


        # Dynamic model
        slip_f[~mask] = -np.arctan((omega[~mask]*lf + vy[~mask])/vx[~mask]) + steering[~mask]
        slip_r[~mask] = np.arctan((omega[~mask]*lr - vy[~mask])/vx[~mask])

        Ffy[~mask] = Df * np.sin( C * np.arctan(B *slip_f[~mask])) * 9.8 * lr / (lr + lf) * m
        Fry[~mask] = Dr * np.sin( C * np.arctan(B *slip_r[~mask])) * 9.8 * lf / (lr + lf) * m

        # motor model
        Frx[~mask] = (( Cm1 - Cm2 * vx[~mask]) * throttle[~mask] - Cr - Cd * vx[~mask] * vx[~mask])*m

        # Dynamics
        d_vx[~mask] = 1.0/m * (Frx[~mask] - Ffy[~mask] * np.sin( steering[~mask] ) + m * vy[~mask] * omega[~mask])
        d_vy[~mask] = 1.0/m * (Fry[~mask] + Ffy[~mask] * np.cos( steering[~mask] ) - m * vx[~mask] * omega[~mask])
        d_omega[~mask] = 1.0/Iz * (Ffy[~mask] * lf * np.cos( steering[~mask] ) - Fry[~mask] * lr)

        # discretization
        vx = vx + d_vx * dt
        vy = vy + d_vy * dt
        omega = omega + d_omega * dt 

        '''
        print("vx = %5.2f, vy = %5.2f"%(vx,vy))
        print("slip_f = %5.2f, slip_r = %5.2f"%(degrees(slip_f), degrees(slip_r)))
        print("f_coeff_f = %5.2f, f_coeff_f = %5.2f"%(tireCurve(slip_f), tireCurve(slip_r)))
        '''

        # back to global frame
        vxg = vx*np.cos(heading)-vy*np.sin(heading)
        vyg = vx*np.sin(heading)+vy*np.cos(heading)

        # update x,y, heading
        x += vxg*dt
        y += vyg*dt
        heading += omega*dt + 0.5* d_omega * dt * dt

        states = np.vstack((x,vxg,y,vyg,heading,omega))
        return states

    
    def jacob_linearize(self, states, controls):
        states = states.copy()
        controls = controls.copy()
        nx = self.n
        nu = self.m
        nN = self.N
        dt = 0.01

        delta_x = np.array([0.01,0.01,0.01,0.01,0.01,0.01])
        delta_u = np.array([0.01, 0.01])
        delta_x_flat = np.tile(delta_x, (1, nN))
        delta_u_flat = np.tile(delta_u, (1, nN))
        delta_x_final = np.multiply(np.tile(np.eye(nx), (1, nN)), delta_x_flat)
        delta_u_final = np.multiply(np.tile(np.eye(nu), (1, nN)), delta_u_flat)
        xx = np.tile(states.copy(), (nx, 1)).reshape((nx, nx*nN), order='F')
        # print(delta_x_final, xx)
        ux = np.tile(controls.copy(), (nx, 1)).reshape((nu, nx*nN), order='F')
        x_plus = xx + delta_x_final
        # print(x_plus, ux)
        x_minus = xx - delta_x_final
        fx_plus = self.update_dynamics(x_plus, ux, dt)
        # print(fx_plus)
        fx_minus = self.update_dynamics(x_minus, ux, dt)
        A = (fx_plus - fx_minus) / (2 * delta_x_flat)

        xu = np.tile(states.copy(), (nu, 1)).reshape((nx, nu*nN), order='F')
        uu = np.tile(controls.copy(), (nu, 1)).reshape((nu, nu*nN), order='F')
        u_plus = uu + delta_u_final
        # print(xu)
        u_minus = uu - delta_u_final
        fu_plus = self.update_dynamics(xu, u_plus, dt)
        # print(fu_plus)
        fu_minus = self.update_dynamics(xu, u_minus, dt)
        B = (fu_plus - fu_minus) / (2 * delta_u_flat)

        state_row = np.zeros((nx*nN, nN))
        input_row = np.zeros((nu*nN, nN))
        for ii in range(nN):
            state_row[ii*nx:ii*nx + nx, ii] = states.copy()[:, ii]
            input_row[ii*nu:ii*nu+nu, ii] = controls.copy()[:, ii]
        d = self.update_dynamics(states.copy(), controls.copy(), dt) - np.dot(A, state_row) - np.dot(B, input_row)

        A = A.reshape((self.n, self.n, self.N), order='F')
        B = B.reshape((self.n, self.m, self.N), order='F')
        d = d.reshape((self.n, 1, self.N), order='F')

        return A, B, d

    # differentiate dynamics around nominal state and control
    # return: A, B, d, s.t. x_k+1 = Ax + Bu + d
    def linearize(self, nominal_state, nominal_ctrl):
        nominal_state = np.array(nominal_state).copy()
        nominal_ctrl = np.array(nominal_ctrl).copy()
        epsilon = 1e-2

        # A = df/dx
        A = np.zeros((self.n,self.n),dtype=np.float)
        # find A
        for i in range(self.n):
            # d x / d x_i, ith row in A
            x_l = nominal_state.copy()
            x_l[i] -= epsilon

            x_post_l = self.update_dynamics(x_l, nominal_ctrl, self.dt)

            x_r = nominal_state.copy()
            x_r[i] += epsilon
            x_post_r = self.update_dynamics(x_r, nominal_ctrl, self.dt)

            A[:,i] += (x_post_r.flatten() - x_post_l.flatten()) / (2*epsilon)
            '''
            print("perturbing x%d"%(i))
            print(A[:,i])
            breakpoint()
            print("")
            '''


        # B = df/du
        B = np.zeros((self.n,self.m),dtype=np.float)
        # find B
        for i in range(self.m):
            # d x / d u_i, ith row in B
            x0 = nominal_state.copy()
            u_l = nominal_ctrl.copy()
            u_l[i] -= epsilon
            x_post_l = self.update_dynamics(x0, u_l, self.dt)
            x_post_l = x_post_l.copy()

            x0 = nominal_state.copy()
            u_r = nominal_ctrl.copy()
            u_r[i] += epsilon
            x_post_r = self.update_dynamics(x0, u_r, self.dt)
            x_post_r = x_post_r.copy()

            B[:,i] += (x_post_r.flatten() - x_post_l.flatten()) / (2*epsilon)

        x0 = nominal_state.copy()
        u0 = nominal_ctrl.copy()
        '''
        self.sim.states = np.array(x0.copy())
        self.sim.updateCar(self.dt,None,nominal_ctrl[0],nominal_ctrl[1])
        x_post = np.array(self.sim.states)
        '''
        x_post = self.update_dynamics(x0, u0, self.dt)


        # d = x_k+1 - Ak*x_k - Bk*u_k
        x0 = nominal_state.copy()
        u0 = nominal_ctrl.copy()
        d = x_post.flatten() - A @ x0 - B @ u0

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

        row0 = np.zeros((n,m*N))
        D = [row0]
        # row 1 to row N
        for i in range(1,N+1):
            row_i = D[-1].copy() 
            row_i = As[:,:,i-1] @ row_i
            row_i[:,(i-1)*m:i*m] = Bs[:,:,i-1] @ Sigma_epsilon
            D.append(row_i)
        D = np.vstack(D)
        return A,B,d,D

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


    def testBigMatrices(self,offset = 0):
        # get ref traj
        #self.generateTestRefTraj()
        self.getRefTraj("../log/ethsim/full_state1.p",show=False)

        # linearize dynamics around ref traj
        As = []
        Bs = []
        ds = []
        Ds = []

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
            #Ds.append(D)

        # make big matrices
        As = np.dstack(As)
        Bs = np.dstack(Bs)
        ds = np.dstack(ds).reshape((self.n,1,self.N))
        D = np.zeros((self.n, self.l))
        Ds = np.tile(D.reshape((self.n, self.l, 1)), (1, 1, self.N))
        #Ds = np.dstack(Ds)

        # propagate big matrices dynamics
        AA, BB, dd, DD = self.form_long_matrices_LTV(As, Bs, ds, Ds)

        # compare with actual result
        # X = AA x0 + BB u + C d + D noise
        # start: ref traj, base of linearization
        # test a traj that's slightly offsetted
        i = start + offset
        #x0 = (x[i],dx[i],y[i],dy[i],heading[i],dheading[i])
        #u0 = (throttle[i],steering[i])
        x0 = self.ref_traj[i,:]
        u0 = self.ref_ctrl[i:i+1,:].flatten()
        uu = self.ref_ctrl[i:i+self.N,:].flatten()
        # actually need + DD @ w
        XX = AA @ x0 + BB @ uu + dd.flatten()

        xx_truth = self.ref_traj[i+1:i+1+self.N,:].flatten()
        x1 = A0 @ x0 + B0 @ u0 + d0
        print("x0")
        print(x0)
        print("ref x0")
        print(self.ref_traj[start,:])
        print("diff")
        print(x0-self.ref_traj[start,:])


        print("xx_truth")
        print(xx_truth.reshape((self.N,-1))[:3,:])
        print("x1")

        xx = XX.reshape((self.N,self.n))
        print(xx[:4,:])

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
        print("truth vs prediction diff")
        print(np.linalg.norm(XX-xx_truth))
        print("diff by item")
        print(XX[-self.n:]-xx_truth[-self.n:])
        print("pred xf")
        print(XX[-self.n:])
        print("truth xf")
        print(xx_truth[-self.n:])
        '''

        # plot true and linearized traj
        img_track = self.track.drawTrack()
        img_track = self.track.drawRaceline(img=img_track)
        car_state = (x0[0],x0[2],x0[4],0,0,0)
        img = self.track.drawCar(img_track.copy(), car_state, u0[1])

        actual_future_traj  = self.ref_traj[i+1:i+1+self.N,(0,2)]
        #actual_future_traj  = self.ref_traj[start+1:start+1+self.N,(0,2)]
        img = self.track.drawPolyline(actual_future_traj,lineColor=(255,0,0),img=img.copy())

        predicted_states = XX.reshape((-1,self.n))
        predicted_future_traj  = predicted_states[:,(0,2)]
        img = self.track.drawPolyline(predicted_future_traj,lineColor=(0,255,0),img=img.copy())
        plt.imshow(img)
        plt.show()

        return AA, BB, dd,B0,B1,d0,d1 

        
    def testBigMatricesJacob(self, offset = 0):
        # get ref traj
        #self.generateTestRefTraj()
        self.getRefTraj("../log/ethsim/full_state1.p",show=False)

        # linearize dynamics around ref traj
        As = []
        Bs = []
        ds = []
        Ds = []

        D = np.diag([0.1,0.3,0.1,0.3,radians(10),1.0])*self.dt
        # random starting point
        start = 100
        As,Bs,ds = self.jacob_linearize(self.ref_traj[start:start+self.N,:].T,self.ref_ctrl[start:start+self.N,:].T)

        D = np.zeros((self.n, self.l))
        Ds = np.tile(D.reshape((self.n, self.l, 1)), (1, 1, self.N))

        # propagate big matrices dynamics
        AA, BB, dd, DD = self.form_long_matrices_LTV(As, Bs, ds, Ds)

        # compare with actual result
        # X = AA x0 + BB u + C d + D noise
        # start: ref traj, base of linearization
        # test a traj that's slightly offsetted 
        i = start + offset
        #x0 = (x[i],dx[i],y[i],dy[i],heading[i],dheading[i])
        #u0 = (throttle[i],steering[i])
        x0 = self.ref_traj[i,:]
        u0 = self.ref_ctrl[i:i+1,:].flatten()
        uu = self.ref_ctrl[i:i+self.N,:].flatten()
        # actually need + DD @ w
        XX = AA @ x0 + BB @ uu + dd.flatten()

        xx_truth = self.ref_traj[i+1:i+1+self.N,:].flatten()

        A0 = As[:,:,0]
        A1 = As[:,:,1]
        B0 = Bs[:,:,0]
        B1 = Bs[:,:,1]
        d0 = ds[:,:,0]
        d1 = ds[:,:,1]
        x1 = A0 @ x0.reshape((-1,1)) + B0 @ u0.reshape((-1,1)) + d0
        '''
        print("x0")
        print(x0)
        print("ref x0")
        print(self.ref_traj[start,:])
        print("diff")
        print(x0-self.ref_traj[start,:])

        # compare one step prediction
        print("xx_truth")
        print(xx_truth.reshape((self.N,-1))[:3,:])
        print("x1")
        print(x1)
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
        print("truth vs prediction diff")
        print(np.linalg.norm(XX-xx_truth))
        print("diff by item")
        print(XX[-self.n:]-xx_truth[-self.n:])
        print("pred xf")
        print(XX[-self.n:])
        print("truth xf")
        print(xx_truth[-self.n:])
        '''

        # plot true and linearized traj
        img_track = self.track.drawTrack()
        img_track = self.track.drawRaceline(img=img_track)
        car_state = (x0[0],x0[2],x0[4],0,0,0)
        img = self.track.drawCar(img_track.copy(), car_state, u0[1])

        actual_future_traj  = self.ref_traj[i+1:i+1+self.N,(0,2)]
        img = self.track.drawPolyline(actual_future_traj,lineColor=(255,0,0),img=img.copy())

        predicted_states = XX.reshape((-1,self.n))
        predicted_future_traj  = predicted_states[:,(0,2)]
        img = self.track.drawPolyline(predicted_future_traj,lineColor=(0,255,0),img=img.copy())
        plt.imshow(img)
        plt.show()
        return AA, BB, dd,B0,B1,d0,d1 

    def testK(self):
        self.getRefTraj("../log/ethsim/full_state1.p",show=False)

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
        sigma_0 = np.zeros((self.n, self.n))
        #sigma_N_inv = np.zeros((self.n, self.n))
        sigma_N_inv = np.eye(self.n)



        Q = np.zeros((self.n, self.n))
        # x
        Q[0, 0] = 1
        # vx
        Q[1, 1] = 1
        # y
        Q[2, 2] = 1
        # vy
        Q[3, 3] = 1
        # psi
        Q[4, 4] = 1
        # omega
        Q[5, 5] = 1

        Q_bar = np.kron(np.eye(self.N, dtype=int), Q)
        R = np.zeros((self.m, self.m))
        # throttle
        R[0, 0] = 1  
        # steering
        R[1, 1] = 1 
        R_bar = np.kron(np.eye(self.N, dtype=int), R)

        mu_0 = self.ref_traj[i,:]
        u_0 = self.ref_ctrl[i:i+1,:].flatten()
        x_target = np.tile(np.array([0, 0, 0, 0, 0, 0]).reshape((-1, 1)), (self.N, 1)) 

        self.solver.populate_params( AA, BB, dd, DD, mu_0, sigma_0, sigma_N_inv, Q_bar, R_bar, u_0, x_target, K=None)
        v,k = self.solver.solve()
        return

if __name__ == '__main__':
    main = LinearizeDynamics(3)

    # test CS solver on dynamic bicycle model
    # get x0 and control
    '''
    i = 100
    x0 = main.ref_traj[i,:]
    u0 = main.ref_ctrl[i,:]
    t0 = time()

    print("covariance control : ")
    main.covarianceControl_cvxpy(x0,u0)
    print("time elapsed:")
    print(time()-t0)
    '''

    # test CS solver on double integrator
    #main.simpleDynamicsCovarianceControl()
    main.testSimpleDynamics()

