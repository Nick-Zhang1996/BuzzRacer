# CCMPPI for kinematic bicycle model
# using model in Ji's paper

import os
import sys
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base_dir)
import pickle
import matplotlib.pyplot as plt
import numpy as np
from common import *
from laptimer import Laptimer
from RCPTrack import RCPtrack
from math import pi,radians,degrees,asin,acos,isnan,sin,cos
from ethCarSim import ethCarSim
from time import time
from cs_solver import CSSolver #from cs_solver_covariance_only import CSSolver
import cvxpy as cp
from cvxpy.atoms.affine.trace import trace 
from cvxpy.atoms.affine.transpose import transpose

from RCPTrack import RCPtrack

class CCMPPI_KINEMATIC():
    def __init__(self, N):
        # set time horizon
        self.N = N
        self.n = 4
        self.m = 2
        self.l = self.n

        self.dt = 0.01
        self.Sigma_epsilon = np.diag([0.1,radians(10)])
        # terminal covariance constrain
        self.sigma_f = np.diag([1e-3]*self.n)

        # set up parameters for the model
        self.setupParam()
        # load track 
        #self.loadTrack()
        #self.getRefTraj("../../log/ref_traj/full_state1.p",show=False)
        self.getRefTraj("../log/ref_traj/full_state1.p",show=False)
        np.random.seed()

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
        fulltrack = RCPtrack()
        # for laptimer
        fulltrack.startPos = (0.6*3.5,0.6*1.75)
        fulltrack.startDir = radians(90)
        fulltrack.load()
        self.track = fulltrack
        return 

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
        # NOTE unwrapped
        heading = data[:,3]
        # calculate speed from pos, to use as no-bias but noise reference
        dt = 0.01
        dx = np.diff(x) / dt
        dy = np.diff(y) / dt
        dheading = np.diff(heading)

        heading = (np.pi + heading) % (2*np.pi) - np.pi
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

    # NOTE in place modification on x0, send x0.copy() as argument
    # state x: X,Y,V,heading
    def update_dynamics(self, x0, u0, dt):
        psi = x0[3]
        V = x0[2]
        throttle = u0[0]
        steering = u0[1]
        beta = np.arctan(np.tan(steering)*self.lr/(self.lr+self.lf))
        dX = V*cos(psi + beta) * dt
        dY = V*sin(psi + beta) * dt
        dV = throttle * dt
        dheading = V/self.lr*sin(beta) * dt

        x0[0] += dX
        x0[1] += dY
        x0[2] += dV
        x0[3] += dheading
        return x0

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
        row0 = np.zeros((n, n*N))
        C = [row0]
        for i in range(1,N+1):
            row_i = C[-1].copy()
            row_i = As[:,:,i-1] @ row_i
            row_i[:,(i-1)*n:i*n] = np.eye(n)
            C.append(row_i)
        C = np.vstack(C)

        # d
        d = ds[:,0,:].T.flatten()

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
        return A,B,C,d,D

    # apply covariance control
    #
    # input:
    #   state: (x,y,heading,v_forward,v_sideway,omega)
    # return: N K matrices of size (n,m)
    def cc(self, state):
        n = self.n
        N = self.N
        m = self.m
        l = self.l

        #xy_vec, v_vec, heading_vec = self.track.getRefXYVheading(state, N-1, self.dt)
        # assemble state: X,Y,V,heading
        #ref_state_vec = np.hstack([xy_vec,v_vec[:,np.newaxis],heading_vec[:,np.newaxis]])


        # find where the car is in reference to reference trajectory
        xx = self.ref_traj[:,0]
        yy = self.ref_traj[:,2]
        x = state[0]
        y = state[2]
        dist_sqr = (xx-x)**2 + (yy-y)**2
        # start : index of closest ref point to car
        start = ref_traj_index = np.argmin(dist_sqr)

        # ref_traj: x,dx,y,dy,heading,dheading
        x = self.ref_traj[start:start+self.N,0]
        vx = self.ref_traj[start:start+self.N,1]
        y = self.ref_traj[start:start+self.N,2]
        vy = self.ref_traj[start:start+self.N,3]
        heading = self.ref_traj[start:start+self.N,4]
        v = np.sqrt(vx*vx + vy*vy)

        self.ref_state_vec = ref_state_vec = np.vstack([x,y,v,heading]).T
        self.ref_ctrl_vec = ref_ctrl_vec = self.ref_ctrl[start:start+self.N]

        # find reference throttle and steering

        # ----------
        # As = [A0..A(N-1)]
        # NOTE this is discretized dynamics
        # linearize dynamics around ref traj
        # As = [A0..A(N-1)]
        As = []
        Bs = []
        ds = []
        for i in range(self.N):
            A,B,d = self.linearize(ref_state_vec[i,:],ref_ctrl_vec[i,:])
            As.append(A)
            Bs.append(B)
            ds.append(d)

        # assemble big matrices for batch dynamics
        self.As = As = np.dstack(As)
        self.Bs = Bs = np.dstack(Bs)
        self.ds = ds = np.dstack(ds).reshape((self.n,1,self.N))
        A, B, C, d, D = self.make_batch_dynamics(As, Bs, ds, None, self.Sigma_epsilon)

        # TODO tune me
        # cost matrix 
        Q = np.eye(n)
        Q_bar = np.kron(np.eye(N+1, dtype=int), Q)
        R = np.eye(m)
        R_bar = np.kron(np.eye(N, dtype=int), R)

        # technically incorrect, but we can just specify R_bar_1/2 instead of R_bar
        R_bar_sqrt = R_bar
        Q_bar_sqrt = Q_bar

        # terminal covariance constrain
        sigma_f = self.sigma_f

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

        J = prob.solve()
        #print("Optimal J = ", prob.solve())
        
        print("Optimal Ks: ")
        for i in range(N):
            print(Ks[i].value)
        '''
        print("Optimal Ks: ")
        print(Ks[0].value)
        '''

        self.Ks = Ks

        As = np.swapaxes(As,0,2)
        As = np.swapaxes(As,1,2)

        Bs = np.swapaxes(Bs,0,2)
        Bs = np.swapaxes(Bs,1,2)

        ds = np.swapaxes(ds,0,2)
        ds = np.swapaxes(ds,1,2)

        return [val.value for val in Ks], As, Bs, ds


    # simulate model with and without cc
    def simulate(self):
        As = self.As
        Bs = self.Bs
        ds = self.ds
        x0 = self.ref_state_vec[0,:]
        u0 = self.ref_ctrl_vec[0,:]
        sim_steps = self.N
        rollout = 100
        print(x0)

        # with CC
        states_vec = []
        for j in range(rollout):
            states_vec.append([])
            y_i = np.zeros(self.n)
            x_i = x0.copy()
            for i in range(sim_steps):
                # generate random variable epsilon
                mean = [0.0]*self.m
                cov = self.Sigma_epsilon
                epsilon = np.random.multivariate_normal(mean, cov)
                v = self.ref_ctrl_vec[i,:]

                # TODO test me
                x_i = As[:,:,i] @ x_i + Bs[:,:,i] @ (v+epsilon + self.Ks[i].value.copy() @ y_i) + ds[:,:,i].flatten()
                y_i = As[:,:,i] @ y_i + Bs[:,:,i] @ epsilon
                states_vec[j].append(x_i.flatten())

        states_vec = np.array(states_vec)
        plt.subplot(2,1,2)
        for i in range(states_vec.shape[0]):
            # position?
            plt.plot(states_vec[i,:,0],states_vec[i,:,1])
            #plt.plot(states_vec[i,-1,0],states_vec[i,-1,1],'*')
        plt.title("with CC")
        plt.axis("square")
        left,right = plt.xlim()
        up,down = plt.ylim()

        # without CC
        states_vec = []
        for j in range(rollout):
            states_vec.append([])
            x_i = x0.copy()
            for i in range(sim_steps):
                # generate random variable epsilon
                mean = [0.0]*self.m
                cov = self.Sigma_epsilon
                epsilon = np.random.multivariate_normal(mean, cov)
                v = self.ref_ctrl_vec[i,:]
                x_i = As[:,:,i] @ x_i + Bs[:,:,i] @ (v+epsilon) + ds[:,:,i].flatten()
                states_vec[j].append(x_i.flatten())

        states_vec = np.array(states_vec)
        plt.subplot(2,1,1)
        for i in range(states_vec.shape[0]):
            plt.plot(states_vec[i,:,0],states_vec[i,:,1])
            #plt.plot(states_vec[i,-1,0],states_vec[i,-1,1],'*')
        plt.title("without CC")

        plt.xlim(left,right)
        plt.ylim(up,down)
        plt.axis("square")
        plt.show()

    # compare linearized batch dynamics against
    def testLinearization(self,offset = 0):

        n = self.n
        N = self.N
        m = self.m
        l = self.l

        #xy_vec, v_vec, heading_vec = self.track.getRefXYVheading(state, N-1, self.dt)
        # assemble state: X,Y,V,heading
        #ref_state_vec = np.hstack([xy_vec,v_vec[:,np.newaxis],heading_vec[:,np.newaxis]])


        # find where the car is in reference to reference trajectory
        xx = self.ref_traj[:,0]
        yy = self.ref_traj[:,2]
        x = state[0]
        y = state[2]
        dist_sqr = (xx-x)**2 + (yy-y)**2
        # start : index of closest ref point to car
        start = ref_traj_index = np.argmin(dist_sqr)

        # ref_traj: x,dx,y,dy,heading,dheading
        x = self.ref_traj[start:start+self.N,0]
        vx = self.ref_traj[start:start+self.N,1]
        y = self.ref_traj[start:start+self.N,2]
        vy = self.ref_traj[start:start+self.N,3]
        heading = self.ref_traj[start:start+self.N,4]
        v = np.sqrt(vx*vx + vy*vy)

        ref_state_vec = np.vstack([x,y,v,heading]).T
        ref_ctrl_vec = self.ref_ctrl[start:start+self.N]

        # find reference throttle and steering

        # ----------
        # As = [A0..A(N-1)]
        # NOTE this is discretized dynamics
        # linearize dynamics around ref traj
        # As = [A0..A(N-1)]
        As = []
        Bs = []
        ds = []
        for i in range(self.N):
            A,B,d = self.linearize(ref_state_vec[i,:],ref_ctrl_vec[i,:])
            As.append(A)
            Bs.append(B)
            ds.append(d)

        # assemble big matrices for batch dynamics
        self.As = As = np.dstack(As)
        self.Bs = Bs = np.dstack(Bs)
        self.ds = ds = np.dstack(ds).reshape((self.n,1,self.N))
        A, B, C, d, D = self.make_batch_dynamics(As, Bs, ds, None, self.Sigma_epsilon)

        # compare with actual result
        #x0 = (x[i],dx[i],y[i],dy[i],heading[i],dheading[i])
        #u0 = (throttle[i],steering[i])

        # simulate with batch dynamics
        # X = AA x0 + BB u + C d + D noise
        x0 = ref_state_vec[0,:]
        print(x0)
        u0 = ref_ctrl_vec[0,:]
        #uu = ref_ctrl_vec.flatten() + np.array([0.1,radians(-5)]*self.N)
        uu = ref_ctrl_vec.flatten() 
        # actually need + DD @ w
        xx_linearized = A @ x0 + B @ uu + C @ d

        # simulate with step-by-step simulation
        '''
        xx_linearized = [x0]
        x_i = x0.copy()
        for i in range(self.N):
            u = ref_ctrl_vec[i,:]
            x_i = As[:,:,i] @ x_i + Bs[:,:,i] @ u + ds[:,:,i].flatten()
            xx_linearized.append(x_i)

        xx_linearized = np.array(xx_linearized)
        '''


        # plot true and linearized traj
        img_track = self.track.drawTrack()
        img_track = self.track.drawRaceline(img=img_track)
        car_state = (x0[0],x0[1],x0[3],0,0,0)
        print(car_state)
        img = self.track.drawCar(img_track.copy(), car_state, u0[1])

        '''
        actual_future_traj  = ref_state_vec[:,(0,1)]
        img = self.track.drawPolyline(actual_future_traj,lineColor=(255,0,0),img=img.copy())
        '''

        predicted_states = xx_linearized.reshape((-1,self.n))
        predicted_future_traj  = predicted_states[:,(0,1)]
        img = self.track.drawPolyline(predicted_future_traj,lineColor=(0,255,0),img=img.copy())

        plt.imshow(img)
        plt.show()

        return 

if __name__ == "__main__":
    main = CCMPPI_KINEMATIC(20)
    state = np.array([0.6*3.5,0.6*1.75,radians(90), 1.0, 0, 0])
    # dim: N*m*n
    Ks, As, Bs, ds = main.cc(state)

    m = main.m
    n = main.n

    # K (m*n)
    # Ks_p[i,j] = p*n*m + i*n + j
    Ks = np.array(Ks)
    Ks_flat = np.array(Ks,dtype=np.float32).flatten()
    print(Ks[0,1,2]-Ks_flat[0*n*m + 1*n + 2])

    # A (n*n)
    # As_p[i,j] = p*n*n + i*n + j
    As_flat = np.array(As,dtype=np.float32).flatten()
    print(As[0,1,2]-As_flat[0*n*n + 1*n + 2])

    # B (n*m)
    # Bs_p[i,j] = p*m*n + i*m + j
    Bs_flat = np.array(Bs,dtype=np.float32).flatten()
    print(Bs[0,2,1]-Bs_flat[0*m*n + 2*m + 1])

    # d (n*1)
    # Bs_p[i] = p*n + i*
    ds_flat = np.array(ds,dtype=np.float32).flatten()
    print(ds[1,2]-ds_flat[1*n + 2])

    main.simulate()
    #main.testLinearization()

