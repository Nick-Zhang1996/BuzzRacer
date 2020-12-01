import os
import sys
import numpy as np
from math import sin
import matplotlib.pyplot as plt
from timeUtil import execution_timer
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base_dir)
from common import *
# Model Predictive Path Integral

class MPPI:
    def __init__(self,samples_count, horizon_steps, control_dim, temperature,dt,noise_cov,cuda=False):
        self.K = samples_count
        self.T = horizon_steps
        self.m = control_dim
        self.temperature = float(temperature)
        self.dt = dt
        # variation of noise added to ref control
        self.noise_cov = noise_cov
        self.p = execution_timer(True)
        # whether to use cuda
        self.cuda = cuda
        if cuda:
            print_info("loading cuda module ...")
            from pycuda.compiler import SourceModule
            import pycuda.driver as driver
            with open("./mppi.cu","r") as f:
                code = f.read()
            mod = SourceModule(code)

            self.cuda_evaluate_control_sequence = mod.get_function("evaluate_control_sequence")
            print_info("registers used = %d"%evaluate_control_sequence.num_regs)
            assert evaluate_control_sequence.num_regs * sample_count < 65536
            assert self.K<=1024

        return

    # given state, apply MPPI and find control
    # state: current plant state
    # ref_control: reference control, dim (self.T,self.m)
    # control_limit: min,max for each control element dim self.m*2 (min,max)
    # control_cov: covariance matrix for noise added to ref_control
    def control_single(self,state,ref_control,control_limit,noise_cov=None):
        if noise_cov is None:
            noise_cov = self.noise_cov
        p = self.p
        p.s()
        p.s("prep")
        ref_control = np.array(ref_control).reshape(-1,self.m).astype(np.float32)
        control_limit = np.array(control_limit)

        # generate random noise for control sampling
        epsilon_vec = np.random.multivariate_normal([0.0]*self.m, noise_cov, size=(self.K,self.T))
        '''
        a = epsilon_vec.reshape(-1,2)[:2000,0]
        b = epsilon_vec.reshape(-1,2)[:2000,1]
        plt.plot(a,b,'*')
        plt.show()
        '''

        # assemble control, ref_control is broadcasted along axis 0 (K)
        control_vec = ref_control + epsilon_vec

        # cap control
        for i in range(self.m):
            control_vec[:,i] = np.clip(control_vec[:,i],control_limit[i,0],control_limit[i,1])

        # NOTE which is better
        clipped_epsilon_vec = control_vec - ref_control
        #clipped_epsilon_vec = epsilon_vec


        # NOTE probably doesn't matter
        #control_cost_mtx_inv = np.linalg.inv(noise_cov)
        control_cost_mtx_inv = np.eye(self.m)
        p.e("prep")

        if (self.cuda):
            # NVIDIA YES !!!
            # CUDA implementation
            p.s("cuda sim")
            cost = np.zeros([self.K]).astype(np.float32)
            epsilon = clipped_epsilon_vec.astype(np.float32)
            control = control.astype(np.float32)
            x0 = state.copy()
            # TODO maybe remove these
            control = control.flatten()
            memCount = cost.size*cost.itemsize + x0.size*x0.itemsize + control.size*control.itemsize + epsilon*epsilon.itemsize
            assert np.sum(memCount)<8370061312
            evaluate_control_sequence( 
                    drv.Out(cost),drv.In(x0),drv.In(control), drv.In(epsilon),
                    block=(self.K,1,1), grid=(1,1))
            S_vec = cost
            p.e("cuda sim")
        else:
            p.s("cpu sim")
            # cost value for each simulation
            S_vec = []
            # spawn k simulations
            for k in range(self.K):
                S = 0
                x = state.copy()
                # run each simulation for self.T timesteps
                for t in range(self.T):
                    control = control_vec[k,t,:]
                    x = self.applyDiscreteDynamics(x,control,self.dt)
                    # NOTE correct format for cost as defined in paper
                    #S += self.evaluateCost(x) + self.temperature * ref_control[t,:].T @ control_cost_mtx_inv @ epsilon_vec[k,t]
                    S += self.evaluateCost(x,control) 
                # NOTE missing terminal cost
                S_vec.append(S)
            p.e("cpu sim")

        p.s("post")
        # Calculate statistics of cost function
        S_vec = np.array(S_vec)
        beta = np.min(S_vec)

        # calculate weights
        weights = np.exp(- (S_vec - beta)/self.temperature)
        weights = weights / np.sum(weights)


        # synthesize control signal
        old_ref_control = ref_control.copy()
        for t in range(self.T):
            for i in range(self.m):
                ref_control[t,i] = ref_control[t,i] + np.sum(weights * clipped_epsilon_vec[:,t,i])
        p.s("post")


        # evaluate performance of synthesized control
        '''
        print("best cost in sampled traj   %.2f"%(beta))
        print("worst cost in sampled traj   %.2f"%(np.max(S_vec)))
        print("avg cost in sampled traj    %.2f"%(np.mean(S_vec)))
        print_info("cost of synthesized control %.2f"%(self.evalControl(state,ref_control)))
        print("cost of ref control %.2f"%(self.evalControl(state,old_ref_control)))
        print("cost of no control(0) %.2f"%(self.evalControl(state,old_ref_control)))
        print("cost of const control(-1) %.2f"%(self.evalControl(state,[[-1,-1]]*self.T)))
        print("cost of const control(1) %.2f"%(self.evalControl(state,[[1,1]]*self.T)))
        '''

        #return ref_control[0]
        p.e()
        return ref_control

    def evalControl(self,state,candidate_control):
        candidate_control = np.array(candidate_control).reshape(-1,self.m)
        S = 0
        x = state.copy()
        # run each simulation for self.T timesteps
        for t in range(self.T):
            control = candidate_control[t,:]
            x = self.applyDiscreteDynamics(x,control,self.dt)
            S += self.evaluateCost(x)
        return S

    '''
    # NOTE inverted pendulum
    def applyDiscreteDynamics(self,state,control,dt):
        m = 1
        g = 9.81
        L = 1
        x = state
        u = control
        x[0] += x[1]*dt
        x[0] = (x[0] + np.pi) % (2*np.pi) - np.pi
        x[1] += (u - m*g*L*sin(x[0]))*dt
        return x

    # NOTE inverted pendulum
    def evaluateCost(self,state):
        x = state
        return ((x[0]-np.pi + np.pi)%(2*np.pi)-np.pi)**2 + 0.1*(x[1])**2
    '''

if __name__=="__main__":
    horizon_steps = 20
    noise = 0.1
    dt = 0.02
    mppi = MPPI(1000,horizon_steps,1,1,dt,noise)
    print(mppi.evaluateCost([-np.pi+0,0]))
    print(mppi.evaluateCost([-np.pi+1,0]))
    print(mppi.evaluateCost([-np.pi+0,1]))
    print(mppi.evaluateCost([-np.pi+-1,0]))
    print(mppi.evaluateCost([-np.pi+0,-1]))
