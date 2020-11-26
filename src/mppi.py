import numpy as np
from math import sin
import matplotlib.pyplot as plt
from timeUtil import execution_timer
# Model Predictive Path Integral

class MPPI:
    def __init__(self,samples_count, horizon_steps, control_dim, temperature,dt,noise_cov):
        self.K = samples_count
        self.T = horizon_steps
        self.m = control_dim
        self.temperature = float(temperature)
        self.dt = dt
        # variation of noise added to ref control
        self.noise_cov = noise_cov
        self.p = execution_timer(True)

        return


    # NOTE INOP
    def control_iterative(self,state,ref_control,acceptable_cost=10,max_iter=10):
        i = 1
        projected_cost = acceptable_cost *2
        while i < max_iter and projected_cost > acceptable_cost:
            print("iter = %d"%(i))
            i += 1
            ref_control, projected_cost = self.control_single(state,ref_control)
            print(" ")

        return ref_control


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
        ref_control = np.array(ref_control).reshape(-1,self.m)
        control_limit = np.array(control_limit)

        # generate random noise for control sampling
        # TODO support multiple dimension/covariance matrix for noise generation
        #epsilon_vec = np.random.normal(loc=0.0, scale=self.noise, size=(self.K,self.T,self.m))
        epsilon_vec = np.random.multivariate_normal([0.0]*self.m, noise_cov, size=(self.K,self.T))
        # assemble control, ref_control is broadcasted along axis 0 (K)
        control_vec = ref_control + epsilon_vec

        # cap control
        for i in range(self.m):
            control_vec[:,i] = np.clip(control_vec[:,i],control_limit[i,0],control_limit[i,1])

        clipped_epsilon_vec = control_vec - ref_control

        # cost value for each simulation
        S_vec = []

        # FIXME for speed
        #control_cost_mtx_inv = np.linalg.inv(np.array([[1,0],[1,0]]))
        control_cost_mtx_inv = np.eye(self.m)
        p.e("prep")

        p.s("sim")
        # spawn k simulations
        for k in range(self.K):
            S = 0
            x = state.copy()
            # run each simulation for self.T timesteps
            for t in range(self.T):
                control = control_vec[k,t,:]
                x = self.applyDiscreteDynamics(x,control,self.dt)
                S += self.evaluateCost(x) + self.temperature * ref_control[t,:].T @ control_cost_mtx_inv @ epsilon_vec[k,t]
            # NOTE missing terminal cost
            S_vec.append(S)
        p.e("sim")

        p.s("post")
        # Calculate statistics of cost function
        S_vec = np.array(S_vec)
        beta = np.min(S_vec)

        # calculate weights
        weights = np.exp(- (S_vec - beta)/self.temperature)
        weights = weights / np.sum(weights)


        # synthesize control signal
        for t in range(self.T):
            ref_control[t] = ref_control[t] + np.sum(weights[:,np.newaxis] * clipped_epsilon_vec[:,t,:])
        p.s("post")


        # NOTE
        # evaluate performance of synthesized control

        '''
        print("best cost in sampled traj   %.2f"%(beta))
        print("avg cost in sampled traj    %.2f"%(np.mean(S_vec)))
        print("cost of synthesized control %.2f"%(self.evalControl(state,ref_control)))
        print("cost of ref control(0) %.2f"%(self.evalControl(state,old_ref_control)))
        print("cost of cw control(-1) %.2f"%(self.evalControl(state,[-1]*self.T)))
        print("cost of ccw control(1) %.2f"%(self.evalControl(state,[1]*self.T)))
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
