# Covariance Control - Model Predictive Path Integral for kinematic bicycle model
import os
import sys
import numpy as np
from time import sleep,time
from math import sin,cos,radians,degrees,ceil,isnan
import matplotlib.pyplot as plt
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base_dir)

from common import *
from timeUtil import execution_timer
from ccmppi_kinematic import CCMPPI_KINEMATIC

class CCMPPI:
    def __init__(self,samples_count, horizon_steps, state_dim, control_dim, temperature,dt,noise_cov,discretized_raceline,cuda=False,cuda_filename=None):
        self.cc = CCMPPI_KINEMATIC(dt, horizon_steps, noise_cov)
        self.K = samples_count

        self.N = self.T = horizon_steps
        self.m = control_dim
        # state: X,Y,vf, psi
        self.n = self.state_dim = state_dim
        self.temperature = float(temperature)
        self.dt = dt
        # variation of noise added to ref control
        self.noise_cov = noise_cov
        self.p = execution_timer(True)
        # whether to use cuda
        self.cuda = cuda
        self.debug_dict = {}

        self.old_ref_control = np.zeros([self.T,self.m],dtype=np.float32)
        if cuda:
            self.curand_kernel_n = 1024
            print_info("loading cuda module ...")
            import pycuda.autoinit
            global drv
            import pycuda.driver as drv
            from pycuda.compiler import SourceModule
            if cuda_filename is None:
                cuda_filename = "./ccmppi.cu"
            with open(cuda_filename,"r") as f:
                code = f.read()

            # prepare constants
            cuda_code_macros = {"SAMPLE_COUNT":self.K, "HORIZON":self.T, "CONTROL_DIM":self.m,"STATE_DIM":self.state_dim,"RACELINE_LEN":discretized_raceline.shape[0],"TEMPERATURE":self.temperature,"DT":dt, "CC_RATIO":0.8, "ZERO_REF_CTRL_RATIO":0.2}
            # add curand related config
            # new feature for Python 3.9
            #cuda_code_macros = cuda_code_macros | {"CURAND_KERNEL_N":self.curand_kernel_n}
            cuda_code_macros.update({"CURAND_KERNEL_N":self.curand_kernel_n})

            mod = SourceModule(code % cuda_code_macros, no_extern_c=True)

            if (self.K < 1024):
                # if K is small only employ one grid
                self.cuda_block_size = (self.K,1,1)
                self.cuda_grid_size = (1,1)
            else:
                # employ multiple grid,
                self.cuda_block_size = (1024,1,1)
                self.cuda_grid_size = (ceil(self.K/1024.0),1)
            print("cuda block size %d, grid size %d"%(self.cuda_block_size[0],self.cuda_grid_size[0]))

            self.cuda_init_curand_kernel = mod.get_function("init_curand_kernel")
            self.cuda_generate_random_var = mod.get_function("generate_random_normal")
            self.cuda_evaluate_control_sequence = mod.get_function("evaluate_control_sequence")

            seed = np.int32(int(time()*10000))
            self.cuda_init_curand_kernel(seed,block=(1024,1,1),grid=(1,1,1))

            self.rand_vals = np.zeros(self.K*self.T*self.m, dtype=np.float32)
            self.device_rand_vals = drv.to_device(self.rand_vals)

            print_info("registers used each kernel in eval_ctrl= %d"%self.cuda_evaluate_control_sequence.num_regs)
            assert int(self.cuda_evaluate_control_sequence.num_regs * self.cuda_block_size[0]) <= 65536
            assert int(self.cuda_init_curand_kernel.num_regs * self.cuda_block_size[0]) <= 65536
            assert int(self.cuda_generate_random_var.num_regs * self.cuda_block_size[0]) <= 65536

            self.discretized_raceline = discretized_raceline.astype(np.float32)
            self.discretized_raceline = self.discretized_raceline.flatten()
            self.device_discretized_raceline = drv.to_device(self.discretized_raceline)

            sleep(1)

        return

    # given state, apply CCMPPI and find control
    # state: current plant state
    # opponents_prediction: predicted positions of opponent(s) list of n opponents, each of dim (steps, 2)
    # control_limit: min,max for each control element dim self.m*2 (min,max)
    # control_cov: covariance matrix for noise added to ref_control
    # specifically for racecar
    def control(self,state,opponents_prediction,control_limit,noise_cov=None,cuda=None):
        if noise_cov is None:
            noise_cov = self.noise_cov
        if cuda is None:
            cuda = self.cuda
        p = self.p
        p.s()
        opponent_count = len(opponents_prediction)
        opponent_count = np.int32(opponent_count)

        p.s("CC")

        # CCMPPI specific, generate and pack K matrices
        Ks, As, Bs, ds = self.cc.cc(state)

        # effectively disable cc
        '''
        print_warning("CC disabled")
        Ks = np.zeros([self.N*self.m*self.n])
        '''


        Ks_flat = np.array(Ks,dtype=np.float32).flatten()
        device_Ks = drv.to_device(Ks_flat)

        As_flat = np.array(As,dtype=np.float32).flatten()
        device_As = drv.to_device(As_flat)

        Bs_flat = np.array(Bs,dtype=np.float32).flatten()
        p.e("CC")
        device_Bs = drv.to_device(Bs_flat)

        '''
        ds_flat = np.array(ds,dtype=np.float32).flatten()
        device_ds = drv.to_device(ds_flat)
        '''

        # NOTE
        # use zero as reference control
        #ref_control = np.zeros(self.N*self.m, dtype=np.float32)
        # reference control is solution at last timestep
        # TODO try to use this as linearization ref trajectory TODO TODO
        ref_control = np.vstack([self.old_ref_control[1:,:],np.zeros([1,self.m],dtype=np.float32)])
        # use ref raceline control
        #ref_control = np.array(self.cc.ref_ctrl_vec.flatten(), dtype=np.float32)
        

        if (cuda):
            # CUDA implementation
            p.s("prep ref ctrl")
            # assemble limites
            #limits = np.array([-1,1,-2,2],dtype=np.float32)
            control_limit = np.array(control_limit,dtype=np.float32).flatten()
            device_control_limits = drv.to_device(control_limit)

            scales = np.array([np.sqrt(noise_cov[0,0]),np.sqrt(noise_cov[1,1])],dtype=np.float32)
            device_scales = drv.to_device(scales)

            self.cuda_generate_random_var(self.device_rand_vals,device_scales,block=(self.curand_kernel_n,1,1),grid=(1,1,1))
            # DEBUG
            self.rand_vals = drv.from_device(self.device_rand_vals,shape=(self.K*self.T*self.m,), dtype=np.float32)

            p.e("prep ref ctrl")

            p.s("cuda sim")
            cost = np.zeros([self.K]).astype(np.float32)
            control = np.zeros(self.K*self.N*self.m).astype(np.float32)

            # we leave the entry point in api for possible future modification
            x0 = state.copy()
            x0 = x0.astype(np.float32)
            # reference control, in standard mppi this should be zero
            ref_control = ref_control.astype(np.float32)
            ref_control = ref_control.flatten()


            # opponent position
            opponents_prediction = np.array(opponents_prediction).astype(np.float32)
            # shape: opponent_count, prediction_steps, 2(x,y)
            if opponents_prediction.shape[0] > 0:
                assert opponents_prediction.shape[1] == self.T+1
            opponents_prediction = opponents_prediction.flatten()


            # ensure we do not fill GPU memory
            memCount = cost.size*cost.itemsize + control.size*control.itemsize + x0.size*x0.itemsize + ref_control.size*ref_control.itemsize + self.rand_vals.size*self.rand_vals.itemsize + self.rand_vals.size*self.rand_vals.itemsize + self.discretized_raceline.flatten().size*self.rand_vals.flatten().itemsize + Ks_flat.size*Ks_flat.itemsize + As_flat.size*As_flat.itemsize + Bs_flat.size*Bs_flat.itemsize
            assert np.sum(memCount)<8370061312


            if (opponent_count == 0):
                self.cuda_evaluate_control_sequence( 
                        drv.Out(cost), # size: K
                        drv.Out(control), # output control size:(K*N*m)
                        drv.In(x0),  
                        drv.In(ref_control),
                        device_control_limits, 
                        self.device_rand_vals,
                        self.device_discretized_raceline, 
                        np.uint64(0),
                        opponent_count,
                        device_Ks, device_As, device_Bs,
                        block=self.cuda_block_size, grid=self.cuda_grid_size
                        )
            else:
                self.cuda_evaluate_control_sequence( 
                        drv.Out(cost), # size: K
                        drv.Out(control), # output control size:(K*N*m)
                        drv.In(x0),  
                        drv.In(ref_control),
                        device_control_limits, 
                        self.device_rand_vals,
                        self.device_discretized_raceline, 
                        drv.In(opponents_prediction),
                        opponent_count,
                        device_Ks, device_As, device_Bs,
                        block=self.cuda_block_size, grid=self.cuda_grid_size
                        )


            # NOTE rand_vals is updated to respect control limits
            #self.rand_vals = drv.from_device(self.device_rand_vals,shape=(self.K*self.T*self.m,), dtype=np.float32)
            p.e("cuda sim")
            # DEBUG
        else:
            print_error("cpu implementation of CCMPPI is unavailable")


        p.s("post")
        # Calculate statistics of cost function
        cost = np.array(cost)
        beta = np.min(cost)
        #print("overall best cost %.2f"%(beta))
        min_cost_index = np.argmin(cost)

        # calculate weights
        weights = np.exp(- (cost - beta)/self.temperature)
        weights = weights / np.sum(weights)
        #print("best cost %.2f, max weight %.2f"%(beta,np.max(weights)))

        # synthesize control signal
        # NOTE test me
        ref_control = ref_control.reshape([self.T,self.m])
        control = control.reshape([self.K, self.N, self.m])
        for t in range(self.T):
            for i in range(self.m):
                # control: (rollout, timestep, control_var)
                ref_control[t,i] =  np.sum(weights * control[:,t,i])
        self.old_ref_control = ref_control.copy()
        p.e("post")

        # DEBUG
        # simulate same control sequence in CPU to simpler debugging
        # No CC

        '''
        print("CPU sim of min cost control sequence")
        i = min_cost_index
        this_control_seq = control[i]
        state = x0.copy()
        cost = 0.0
        for k in range(self.N):
            print("step = %d, x= %.3f, y=%.3f, v=%.3f, psi=%.3f, T=%.3f, S=%.3f"%(k,state[0],state[1],state[2],state[3], this_control_seq[k,0], this_control_seq[k,1]))
            state = self.forwardKinematics(state,this_control_seq[k])
            step_cost, index, dist = self.evaluateStepCost(state, this_control_seq[k], self.discretized_raceline)
            cost += step_cost
            print(step_cost, index, dist)
        '''

        # DEBUG
        '''
        self.cc.debug_info = {'x0':state, 'model':'kinematic', 'input_constraint':True}
        self.cc.rand_vals = self.rand_vals.reshape([self.K,self.T,self.m])
        self.cc.visualizeOnTrack()
        '''


        # throttle, steering
        #print("control = %7.3f, %7.3f " %(ref_control[0,0], degrees(ref_control[0,1])))

        p.e()
        self.debug_dict = {'sampled_control':control}
        if isnan(ref_control[0,1]):
            print_error("cc-mppi fail to return valid control")
        return ref_control

    def evalControl(self,state,candidate_control):
        candidate_control = np.array(candidate_control).reshape(-1,self.m)
        S = 0
        x0 = state.copy()
        x = state.copy()
        # run each simulation for self.T timesteps
        for t in range(self.T):
            control = candidate_control[t,:]
            x = self.applyDiscreteDynamics(x,control,self.dt)
            S += self.evaluateStepCost(x,control)
        S += self.evaluateTerminalCost(x,x0)
        return S

    def forwardKinematics(self,state, control):
        self.lr = 0.036
        self.l = 0.09

        dt = self.dt
        throttle,steering = control
        x,y,v,heading = state
        beta = np.arctan( np.tan(steering) * self.lr / (self.l))
        dXdt = v * np.cos( heading + beta )
        dYdt = v * np.sin( heading + beta )
        dvdt = throttle
        dheadingdt = v/self.lr*np.sin(beta)

        x += dt * dXdt
        y += dt * dYdt
        v += dt * dvdt
        heading += dt * dheadingdt

        return (x,y,v,heading)


    def findClosestId(self, state, in_raceline):
        x = state[0]
        y = state[1]
        dist2 = (x-in_raceline[:,0])**2 + (y-in_raceline[:,1])**2
        min_index = np.argmin(dist2)
        return min_index, dist2[min_index]**0.5 

    def evaluateStepCost(self, state, control, in_raceline):
        in_raceline = in_raceline.reshape(-1,6)
        # find closest id
        index, dist = self.findClosestId(state, in_raceline)
        vx = state[2]
        dv = vx - in_raceline[index, 3]

        cost = dist + 0.1*dv*dv
        return cost, index, dist

