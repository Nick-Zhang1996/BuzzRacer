import numpy as np
from math import sin
import matplotlib.pyplot as plt
# Model Predictive Path Integral

class MPPI:
    def __init__(self,samples_count, horizon_steps, control_dim, temperature,dt,noise):
        self.K = samples_count
        self.T = horizon_steps
        self.m = control_dim
        self.temperature = float(temperature)
        self.dt = dt
        # variation of noise added to ref control
        self.noise = noise

        return


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
    def control_single(self,state,ref_control):
        ref_control = np.array(ref_control).reshape(-1,self.m)

        # generate random noise for control sampling
        epsilon_vec = np.random.normal(loc=0.0, scale=self.noise, size=(self.K,self.T,self.m))
        #epsilon_vec = np.random.uniform(low=-30, high=30, size=(self.K,self.T,self.m))
        # assemble control, ref_control is broadcasted along axis 0 (K)
        control_vec = ref_control + epsilon_vec
        # cap control
        # NOTE need to make this a setting
        control_vec = np.clip(control_vec,-10,10)
        clipped_epsilon_vec = control_vec - ref_control


        # cost value for each simulation
        S_vec = []

        # FIXME for speed
        #control_cost_mtx_inv = np.linalg.inv(np.array([[1,0],[1,0]]))
        control_cost_mtx_inv = np.eye(self.m)

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

        # Calculate statistics of cost function
        S_vec = np.array(S_vec)
        beta = np.min(S_vec)

        # calculate weights
        weights = np.exp(- (S_vec - beta)/self.temperature)
        weights = weights / np.sum(weights)

        '''
        plt.plot((weights-np.mean(weights))/np.std(weights))
        plt.plot((S_vec-np.mean(S_vec))/np.std(S_vec))
        plt.show()
        '''

        assert np.isclose(1.0,np.sum(weights))

        # synthesize control signal
        old_ref_control = ref_control.copy()
        '''
        for t in range(self.T):
            for k in range(self.K):
                ref_control[t] = ref_control[t] + weights[k] * clipped_epsilon_vec[k,t,:]
        '''
        # above has numerical issues
        for t in range(self.T):
            ref_control[t] = ref_control[t] + np.sum(weights * clipped_epsilon_vec[:,t,:])

        '''
        plt.plot(old_ref_control-ref_control)
        plt.show()
        '''

        # NOTE
        # evaluate performance of synthesized control
        S = 0
        x = state.copy()
        # run each simulation for self.T timesteps
        for t in range(self.T):
            control = ref_control[t,:]
            x = self.applyDiscreteDynamics(x,control,self.dt)
            S += self.evaluateCost(x)

        print("best cost in sampled traj   %.2f"%(beta))
        print("avg cost in sampled traj    %.2f"%(np.mean(S_vec)))
        print("cost of synthesized control %.2f"%(S))

        #return ref_control[0]
        return ref_control, S

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
        return ((x[0]-np.pi + np.pi)%(2*np.pi)-np.pi)**2 + 1*(x[1])**2

