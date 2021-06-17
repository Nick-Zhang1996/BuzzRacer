# CCMPPI for kinematic bicycle model
# using model in Ji's paper
import os
import sys
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base_dir)
import pickle
import numpy as np
from time import time

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

from math import pi,radians,degrees,asin,acos,isnan,sin,cos

import cvxpy as cp
from cvxpy.atoms.affine.trace import trace 
from cvxpy.atoms.affine.transpose import transpose


from common import *
from laptimer import Laptimer
from RCPTrack import RCPtrack
from kinematicSimulator import kinematicSimulator

class CCMPPI_KINEMATIC():
    def __init__(self,dt, N, noise_cov, debug_info=None):
        # set time horizon
        self.N = N
        self.n = 4
        self.m = 2
        self.l = self.n
        # (x,y,v,heading)
        self.debug_info = debug_info
        self.rand_vals = None
        self.v = None

        self.dt = dt
        #self.Sigma_epsilon = np.diag([0.2,radians(20)])
        self.Sigma_epsilon = noise_cov
        # terminal covariance constrain
        # not needed with soft constraint
        #self.sigma_f = np.diag([1e-3]*self.n)
        self.control_limit = np.array([[-0.7,0.7],[-radians(27.1), radians(27.1)]])

        # set up parameters for the model
        self.setupParam()
        # load track 
        #self.loadTrack()
        #self.getRefTraj("../../log/ref_traj/full_state1.p",show=False)
        self.getRefTraj("/home/nick/rcvip/log/ref_traj/full_state1.p",show=False)
        
        np.random.seed()
        self.sim = kinematicSimulator(0,0,0,0)

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
        dt = self.dt
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
        print_info(" reference laptime %.2fs "%(self.ref_laptime))

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
    #   state: (x,y,v,heading)
    # return: N K matrices of size (n,m)
    def cc(self, state, return_sx=False, debug=False):
        n = self.n
        N = self.N
        m = self.m
        l = self.l

        # find where the car is in reference to reference trajectory
        ref_xx = self.ref_traj[:,0]
        ref_yy = self.ref_traj[:,2]

        x,y,_,_ = state

        dist_sqr = (ref_xx-x)**2 + (ref_yy-y)**2
        # start : index of closest ref point to car
        start = np.argmin(dist_sqr)

        # ref_traj: x,dx,y,dy,heading,dheading
        # handle wrap around to prevent array out of bound at tail
        ref_traj_wrapped = np.vstack([self.ref_traj, self.ref_traj[:self.N]])
        ref_ctrl_wrapped = np.vstack([self.ref_ctrl, self.ref_ctrl[:self.N]])

        x = ref_traj_wrapped[start:start+self.N,0]
        vx = ref_traj_wrapped[start:start+self.N,1]
        y = ref_traj_wrapped[start:start+self.N,2]
        vy = ref_traj_wrapped[start:start+self.N,3]
        heading = ref_traj_wrapped[start:start+self.N,4]
        v = np.sqrt(vx*vx + vy*vy)

        self.ref_state_vec = ref_state_vec = np.vstack([x,y,v,heading]).T
        self.ref_ctrl_vec = ref_ctrl_vec = ref_ctrl_wrapped[start:start+self.N]

        # find reference throttle and steering

        # As = [A0..A(N-1)]
        # linearize dynamics around ref traj
        # As = [A0..A(N-1)]
        As = []
        Bs = []
        ds = []
        for i in range(self.N):
            # NOTE this gives discretized dynamics
            A,B,d = self.linearize(ref_state_vec[i,:],ref_ctrl_vec[i,:])
            As.append(A)
            Bs.append(B)
            ds.append(d)


        # assemble big matrices for batch dynamics
        self.As = As = np.dstack(As)
        self.Bs = Bs = np.dstack(Bs)
        self.ds = ds = np.dstack(ds).reshape((self.n,1,self.N))

        # NOTE ds, the offset,  is calculated off reference trajectory
        # additional offsert may need to be added to account for difference between
        # actual state and reference state
        #state_diff = state - self.ref_state_vec[0]
        #state_diff = state_diff.reshape(4,1)
        #ds[:3,:,0] += state_diff[:3]
        #ds[:2,:,0] += state_diff[:2]
        #ds[:,:,0] += state_diff

        if (debug):
            print_info("[cc] ref state x0 (x,y,v,heading)")
            print(ref_state_vec[0])
            print_info("[cc] actual state x0")
            print(state)
            print_info("[cc] state diff")
            print(state_diff.flatten())

        A, B, C, d, D = self.make_batch_dynamics(As, Bs, ds, None, self.Sigma_epsilon)

        # cost matrix 
        #Q = np.eye(n)
        #Q_bar = np.kron(np.eye(N+1, dtype=int), Q)
        # soft constraint Q matrix
        Q_bar = np.zeros([(N+1)*self.n, (N+1)*self.n])
        Q_bar[-self.n:, -self.n:] = np.eye(self.n) * 50
        #Q_bar[-self.n:, -self.n:] = np.eye(self.n) * 2000

        R = np.eye(m)
        R_bar = np.kron(np.eye(N, dtype=int), R)

        # technically incorrect, but we can just specify R_bar_1/2 instead of R_bar
        R_bar_sqrt = R_bar
        Q_bar_sqrt = Q_bar

        # terminal covariance constrain
        # not needed with soft constraint
        #sigma_f = self.sigma_f

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
        # hard constraint, cvxpy doesn't respect this for some reasons
        #constraints = [cp.bmat([[sigma_f, E_N @(I+B@K)@sigma_y_sqrt], [ sigma_y_sqrt@(I+B @ K).T@E_N.T, I ]]) >= 0]
        constraints = []
        prob = cp.Problem(objective, constraints)

        J = prob.solve()

        Ks = np.array([val.value for val in Ks])

        if (debug):
            print_info("[cc] Problem status")
            print(prob.status)
            
        # DEBUG veirfy constraint
        '''
        test_mtx = np.block([[sigma_f, E_N @(I+B@K.value)@sigma_y_sqrt], [ sigma_y_sqrt@(I+B @ K.value).T@E_N.T, I ]])
        if not (np.all(np.linalg.eigvals(test_mtx) > 0)):
            print_warning("[cc] constraint not satisfied")
        '''

        self.Ks = Ks

        As = np.swapaxes(As,0,2)
        As = np.swapaxes(As,1,2)

        Bs = np.swapaxes(Bs,0,2)
        Bs = np.swapaxes(Bs,1,2)

        ds = np.swapaxes(ds,0,2)
        ds = np.swapaxes(ds,1,2)

        # return terminal covariance, theoretical values with and without cc
        if (return_sx):
            reconstruct_K = np.hstack([Ks[0], np.zeros((m,(N)*n))])
            for i in range(1,N):
                line = np.hstack([ np.zeros((m,n*i)), Ks[i], np.zeros((m,(N-i)*n)) ])
                reconstruct_K = np.vstack([reconstruct_K, line])
            Sigma_0 = np.zeros([n,n])
            #Sx_cc = (I + B@K.value ) @ (A @ Sigma_0 @ A.T + D @ D.T ) @ (I + B@K.value ).T
            Sx_cc = (I + B@reconstruct_K ) @ (A @ Sigma_0 @ A.T + D @ D.T ) @ (I + B@reconstruct_K ).T
            Sx_nocc = (A @ Sigma_0 @ A.T + D @ D.T )
            return Ks, As, Bs, ds, Sx_cc, Sx_nocc
        else:
            return Ks, As, Bs, ds


    def simulate(self):
        if self.debug_info['model'] =='linear_kinematic':
            return self.simulate_linear()
        elif self.debug_info['model']=='kinematic':
            return self.simulate_kinematic()

    def simulate_kinematic(self):
        As = self.As
        Bs = self.Bs
        ds = self.ds
        x0 = self.debug_info['x0'].copy()
        u0 = self.ref_ctrl_vec[0,:]

        sim_steps = self.N

        # with CC
        cc_states_vec = []

        samples = 100
        for j in range(samples):
            cc_states_vec.append([])
            y_i = np.zeros(self.n)
            x_i = x0.copy()
            for i in range(sim_steps):
                mean = [0.0]*self.m
                # generate random variable epsilon or retrieve from self.rand_val
                if (self.rand_vals is None):
                    cov = self.Sigma_epsilon
                    epsilon = np.random.multivariate_normal(mean, cov)
                else:
                    epsilon = self.rand_vals[j,i,:]
                v = self.ref_ctrl_vec[i,:]
                control = (v+epsilon + self.Ks[i] @ y_i)
                # apply input constraints
                if (self.debug_info['input_constraint']):
                    for k in range(self.m):
                        control[k] = np.clip(control[k], self.control_limit[k,0], self.control_limit[k,1])

                #print("states = %7.4f, %7.4f, %7.4f, %7.4f, ctrl =  %7.4f, %7.4f,"%(x_i[0], x_i[1], x_i[2], x_i[3], control[0], control[1]))
                x_i = self.sim.updateCar(self.dt, control[0], control[1], external_states=x_i)
                y_i = As[:,:,i] @ y_i + Bs[:,:,i] @ epsilon

                cc_states_vec[j].append(x_i.flatten())


        # size: (rollout, n, 2)
        cc_states_vec = np.array(cc_states_vec)

        # without CC
        nocc_states_vec = []
        for j in range(rollout):
            nocc_states_vec.append([])
            x_i = x0.copy()
            for i in range(sim_steps):
                # generate random variable epsilon
                mean = [0.0]*self.m
                cov = self.Sigma_epsilon
                if (self.rand_vals is None):
                    epsilon = np.random.multivariate_normal(mean, cov)
                else:
                    epsilon = self.rand_vals[j,i,:]
                v = self.ref_ctrl_vec[i,:]
                control = v+epsilon
                # apply input constraints
                if (self.debug_info['input_constraint']):
                    for k in range(self.m):
                        control[k] = np.clip(control[k], self.control_limit[k,0], self.control_limit[k,1])
                #x_i = As[:,:,i] @ x_i + Bs[:,:,i] @ control + ds[:,:,i].flatten()
                x_i = self.sim.updateCar(self.dt, control[0], control[1], external_states=x_i)
                nocc_states_vec[j].append(x_i.flatten())

        nocc_states_vec = np.array(nocc_states_vec)

        ret_dict = {'cc_states_vec':cc_states_vec, 'nocc_states_vec':nocc_states_vec}
        return ret_dict
        

    # simulate model with and without cc
    def simulate_linear(self):
        print_info("simulation -- linear model")
        As = self.As
        Bs = self.Bs
        ds = self.ds
        #x0 = self.ref_state_vec[0,:]
        x0 = self.debug_info['x0'].copy()
        u0 = self.ref_ctrl_vec[0,:]
        sim_steps = self.N
        rollout = 100

        # with CC
        cc_states_vec = []
        for j in range(rollout):
            cc_states_vec.append([])
            y_i = np.zeros(self.n)
            x_i = x0.copy()
            for i in range(sim_steps):
                # generate random variable epsilon
                mean = [0.0]*self.m
                cov = self.Sigma_epsilon
                if (self.rand_vals is None):
                    epsilon = np.random.multivariate_normal(mean, cov)
                else:
                    epsilon = self.rand_vals[j,i,:]
                v = self.ref_ctrl_vec[i,:]

                control = (v+epsilon + self.Ks[i] @ y_i)
                # apply input constraints
                if (self.debug_info['input_constraint']):
                    for k in range(self.m):
                        control[k] = np.clip(control[k], self.control_limit[k,0], self.control_limit[k,1])
                x_i = As[:,:,i] @ x_i + Bs[:,:,i] @ control + ds[:,:,i].flatten()
                # TODO is this the right format?
                y_i = As[:,:,i] @ y_i + Bs[:,:,i] @ epsilon
                cc_states_vec[j].append(x_i.flatten())

                '''
                if (j==0):
                    print("step %d, control: %.2f, %.2f"%(i,control[0], control[1]))
                '''

        # size: (rollout, n, 2)
        cc_states_vec = np.array(cc_states_vec)

        # without CC
        nocc_states_vec = []
        for j in range(rollout):
            nocc_states_vec.append([])
            x_i = x0.copy()
            for i in range(sim_steps):
                # generate random variable epsilon
                mean = [0.0]*self.m
                cov = self.Sigma_epsilon
                if (self.rand_vals is None):
                    epsilon = np.random.multivariate_normal(mean, cov)
                else:
                    epsilon = self.rand_vals[j,i,:]
                v = self.ref_ctrl_vec[i,:]
                control = v+epsilon
                # apply input constraints
                if (self.debug_info['input_constraint']):
                    for k in range(self.m):
                        control[k] = np.clip(control[k], self.control_limit[k,0], self.control_limit[k,1])
                x_i = As[:,:,i] @ x_i + Bs[:,:,i] @ control + ds[:,:,i].flatten()
                nocc_states_vec[j].append(x_i.flatten())

        nocc_states_vec = np.array(nocc_states_vec)

        ret_dict = {'cc_states_vec':cc_states_vec, 'nocc_states_vec':nocc_states_vec}
        return ret_dict

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

        # reference state trajectory and control
        ref_state_vec = np.vstack([x,y,v,heading]).T
        ref_ctrl_vec = self.ref_ctrl[start:start+self.N]

        # As = [A0..A(N-1)]
        # linearize dynamics around ref traj
        # As = [A0..A(N-1)]
        As = []
        Bs = []
        ds = []
        for i in range(self.N):
            # NOTE this gives discretized dynamics
            A,B,d = self.linearize(ref_state_vec[i,:],ref_ctrl_vec[i,:])
            As.append(A)
            Bs.append(B)
            ds.append(d)

        # assemble big matrices for batch dynamics
        self.As = As = np.dstack(As)
        self.Bs = Bs = np.dstack(Bs)
        self.ds = ds = np.dstack(ds).reshape((self.n,1,self.N))
        A, B, C, d, D = self.make_batch_dynamics(As, Bs, ds, None, self.Sigma_epsilon)

        # simulate with batch dynamics
        # X = AA x0 + BB u + C d + D noise
        x0 = ref_state_vec[0,:]
        print(x0)
        u0 = ref_ctrl_vec[0,:]
        #uu = ref_ctrl_vec.flatten() + np.array([0.1,radians(-5)]*self.N)
        uu = ref_ctrl_vec.flatten() 
        # actually need + DD @ w
        xx_linearized = A @ x0 + B @ uu + C @ d

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

    # mean: (x_mean, y_mean)
    def plotConfidenceEllipse(self, ax, mean, cov_matrix, color='red'):
        facecolor = 'none'
        # sigma, how large covariance matrix is
        n_std = 3.0
        mean_x, mean_y = mean
        cov = cov_matrix
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        # Using a special case to obtain the eigenvalues of this
        # two-dimensionl dataset.
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                          facecolor=facecolor, edgecolor=color)

        # Calculating the stdandard deviation of x from
        # the squareroot of the variance and multiplying
        # with the given number of standard deviations.
        scale_x = np.sqrt(cov[0, 0]) * n_std

        # calculating the stdandard deviation of y ...
        scale_y = np.sqrt(cov[1, 1]) * n_std

        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)

        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)


    def testSingleFrame(self):
        state = np.array([0.6*3.5,0.6*1.75,radians(90), 1.0, 0, 0])
        # dim: N*m*n
        Ks, As, Bs, ds, = self.cc(state, True)

        m = self.m
        n = self.n

        # K (m*n)
        # Ks_p[i,j] = p*n*m + i*n + j
        Ks = np.array(Ks)
        Ks[0] = 0.0
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

        ret_dict = self.simulate()
        #self.testLinearization()

    def visualizeOnTrack(self):
        state = self.debug_info['x0'].copy()

        # dim: N*m*n
        Ks, As, Bs, ds = self.cc(state, False)

        m = self.m
        n = self.n

        ret_dict = self.simulate()
        cc_states_vec = ret_dict['cc_states_vec']
        nocc_states_vec = ret_dict['nocc_states_vec']

        # prepare track map
        track = RCPtrack()
        track.startPos = (0.6*3.5,0.6*1.75)
        track.startDir = radians(90)
        track.load()

        img = track.drawTrack()
        track.drawRaceline(img=img)
        car_steering = 0.0

        x,y,v,heading = state
        x0 = np.hstack([x,y,heading,0,0,0])
        img = track.drawCar(img, x0, car_steering)

        for i in range(cc_states_vec.shape[0]):
            img = track.drawPolyline(cc_states_vec[i,:,:],img=img,lineColor=(200,200,200),thickness=1)

        '''
        for i in range(nocc_states_vec.shape[0]):
            img = track.drawPolyline(nocc_states_vec[i,:,:],img=img,lineColor=(200,200,200),thickness=1)
        '''
        plt.imshow(img)
        plt.show()

    def visualizeConfidenceEllipse(self):
        # x,y,heading, v
        state = self.debug_info['x0'].copy()

        # dim: N*m*n
        Ks, As, Bs, ds, Sx_cc, Sx_nocc = self.cc(state, return_sx = True , debug=True)
        theory_cc_cov_mtx =  Sx_cc[-4:-2,-4:-2]
        theory_nocc_cov_mtx =  Sx_nocc[-4:-2,-4:-2]

        m = self.m
        n = self.n

        ret_dict = self.simulate()
        cc_states_vec = ret_dict['cc_states_vec']
        nocc_states_vec = ret_dict['nocc_states_vec']


        # visualize
        # with CC
        ax_cc = plt.subplot(2,1,2)
        plt.plot(state[0],state[1], 'ro')
        for i in range(cc_states_vec.shape[0]):
            # position?
            plt.plot(cc_states_vec[i,:,0],cc_states_vec[i,:,1])

        xy_vec = np.vstack([cc_states_vec[:,-1,0], cc_states_vec[:,-1,1]])
        x_mean = np.mean(cc_states_vec[:,-1,0])
        y_mean = np.mean(cc_states_vec[:,-1,1])
        cc_cov_mtx = np.cov(xy_vec)
        self.plotConfidenceEllipse(ax_cc,(x_mean,y_mean), cc_cov_mtx) 
        self.plotConfidenceEllipse(ax_cc,(x_mean,y_mean), theory_cc_cov_mtx, color='blue') 

        plt.title("with CC (%s, input limit= %s)"%(self.debug_info['model'], str(self.debug_info['input_constraint'])))
        plt.axis("square")
        left,right = plt.xlim()
        up,down = plt.ylim()

        # without CC
        ax_nocc = plt.subplot(2,1,1)
        plt.plot(state[0],state[1], 'ro')
        for i in range(nocc_states_vec.shape[0]):
            plt.plot(nocc_states_vec[i,:,0],nocc_states_vec[i,:,1])
            #plt.plot(nocc_states_vec[i,-1,0],nocc_states_vec[i,-1,1],'*')
        xy_vec = np.vstack([nocc_states_vec[:,-1,0], nocc_states_vec[:,-1,1]])
        x_mean = np.mean(nocc_states_vec[:,-1,0])
        y_mean = np.mean(nocc_states_vec[:,-1,1])
        nocc_cov_mtx = np.cov(xy_vec)
        self.plotConfidenceEllipse(ax_nocc,(x_mean,y_mean), nocc_cov_mtx) 
        self.plotConfidenceEllipse(ax_nocc,(x_mean,y_mean), theory_nocc_cov_mtx, color='blue') 

        plt.title("without CC (%s, input limit= %s)"%(self.debug_info['model'], str(self.debug_info['input_constraint'])))
        plt.xlim(left,right)
        plt.ylim(up,down)
        plt.axis("square")

        print("theoretical cc cov")
        print(theory_cc_cov_mtx)
        print("theoretical nocc cov")
        print(theory_nocc_cov_mtx)

        print("actual cc cov")
        print(cc_cov_mtx)
        print("actual no cc cov")
        print(nocc_cov_mtx)
        plt.show()

if __name__ == "__main__":
    state = np.array([0.6*0.7,0.6*0.5, 0.5, radians(130)])
    #state = np.array([0.6*3.5,0.6*1.75, 1.0, radians(-90)])
    #state = np.array([0.6*3.7,0.6*1.75, 1.0, radians(-90)])
    #main = CCMPPI_KINEMATIC(20, x0=state, model = 'linear_kinematic', input_constraint=True)

    debug_info = {'x0':state, 'model':'kinematic', 'input_constraint':True}
    main = CCMPPI_KINEMATIC(0.03,20, debug_info)
    main.visualizeConfidenceEllipse()
    main.visualizeOnTrack()

