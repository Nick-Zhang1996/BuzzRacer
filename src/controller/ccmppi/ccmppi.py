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
from util.timeUtil import execution_timer
from ccmppi.ccmppi_kinematic import CCMPPI_KINEMATIC

class CCMPPI:
    def __init__(self,parent,arg_list):
        self.parent = parent

        self.K = arg_list['samples']

        self.N = self.T = arg_list['horizon']
        self.m = arg_list['control_dim']
        # kinematic model:
        # state: X,Y,vf, psi
        # dynamic model:
        # state: x,y,heading,vf,vs,omega
        self.n = self.state_dim = arg_list['state_dim']
        self.temperature = arg_list['temperature']
        self.dt = arg_list['dt']
        # variation of noise added to ref control
        self.noise_cov = arg_list['noise_cov']
        cuda_filename = arg_list['cuda_filename']
        discretized_raceline = arg_list['raceline']
        # kinematic simulator velocity cap
        max_v = arg_list['max_v']
        self.cc = CCMPPI_KINEMATIC(self.dt, self.T, self.noise_cov,arg_list)

        self.p = execution_timer(True)
        self.debug_dict = {}

        self.old_ref_control = np.zeros([self.T,self.m],dtype=np.float32)
        self.curand_kernel_n = 1024
        print_info("loading cuda module ...")
        self.ref_traj = None
        self.ref_ctrl = None


        import pycuda.autoinit
        global drv
        import pycuda.driver as drv
        from pycuda.compiler import SourceModule
        with open(cuda_filename,"r") as f:
            code = f.read()


        car = self.car
        # prepare constants
        cuda_code_macros = {"SAMPLE_COUNT":self.K, "HORIZON":self.T, "CONTROL_DIM":self.m,"STATE_DIM":self.state_dim,"RACELINE_LEN":discretized_raceline.shape[0],"TEMPERATURE":self.temperature,"DT":self.dt, "CC_RATIO":arg_list['cc_ratio'], "ZERO_REF_CTRL_RATIO":arg_list['zero_ref_ratio'], "MAX_V":max_v, "R1":arg_list['R_diag'][0],"R2":arg_list['R_diag'][1] }
        self.cuda_code_macros = cuda_code_macros
        # add curand related config
        # new feature for Python 3.9
        #cuda_code_macros = cuda_code_macros | {"CURAND_KERNEL_N":self.curand_kernel_n}
        cuda_code_macros.update({"CURAND_KERNEL_N":self.curand_kernel_n})
        cuda_code_macros.update({"alfa":arg_list['alfa']})
        cuda_code_macros.update({"beta":arg_list['beta']})
        cuda_code_macros.update({"use_raceline":"true" if arg_list['rcp_track'] else "false"})
        cuda_code_macros.update({"obstacle_radius":arg_list['obstacle_radius']})
        print_info("[ccmppi]:"+str(cuda_code_macros))

        mod = SourceModule(code % cuda_code_macros, no_extern_c=True)

        threads_per_block = 512
        assert(threads_per_block<=1024)
        if (self.K < threads_per_block):
            # if K is small only employ one grid
            self.cuda_block_size = (self.K,1,1)
            self.cuda_grid_size = (1,1)
        else:
            # employ multiple grid,
            self.cuda_block_size = (threads_per_block,1,1)
            self.cuda_grid_size = (ceil(self.K/float(threads_per_block)),1)
        print("cuda block size %d, grid size %d"%(self.cuda_block_size[0],self.cuda_grid_size[0]))

        self.cuda_init_curand_kernel = mod.get_function("init_curand_kernel")
        self.cuda_generate_random_var = mod.get_function("generate_random_normal")
        self.cuda_evaluate_control_sequence = mod.get_function("evaluate_control_sequence")

        seed = np.int32(int(time()*10000))
        self.cuda_init_curand_kernel(seed,block=(self.curand_kernel_n,1,1),grid=(1,1,1))

        self.rand_vals = np.zeros(self.K*self.T*self.m, dtype=np.float32)
        self.device_rand_vals = drv.to_device(self.rand_vals)

        print_info("registers used each kernel in eval_ctrl= %d"%self.cuda_evaluate_control_sequence.num_regs)
        print("total registers used in one block (%d/%d)(used/available)"%(int(self.cuda_evaluate_control_sequence.num_regs * self.cuda_block_size[0]),65536))
        assert int(self.cuda_evaluate_control_sequence.num_regs * self.cuda_block_size[0]) <= 65536
        assert int(self.cuda_init_curand_kernel.num_regs * self.cuda_block_size[0]) <= 65536
        assert int(self.cuda_generate_random_var.num_regs * self.cuda_block_size[0]) <= 65536

        self.discretized_raceline = discretized_raceline.astype(np.float32)
        self.discretized_raceline = self.discretized_raceline.flatten()
        self.device_discretized_raceline = drv.to_device(self.discretized_raceline)

        sleep(0.01)

        return

    # given state, apply CCMPPI and find control
    # state: current plant state
    # opponents_prediction: predicted positions of opponent(s) list of n opponents, each of dim (steps, 2)
    # control_limit: min,max for each control element dim self.m*2 (min,max)
    # control_cov: covariance matrix for noise added to ref_control
    # specifically for racecar
    def control(self,state,opponents_prediction,control_limit):
        noise_cov = self.noise_cov
        p = self.p
        p.s()
        opponent_count = len(opponents_prediction)
        opponent_count = np.int32(opponent_count)

        p.s("CC")

        # CCMPPI specific, generate and pack K matrices
        if (self.cuda_code_macros['CC_RATIO'] > 0.01):
            #Ks, As, Bs, ds = self.cc.cc(state)
            if (self.ref_traj is None):
                Ks, As, Bs, ds, Sx_cc, Sx_nocc = self.cc.old_cc(state, return_sx = True , debug=False)
            else:
                Ks, As, Bs, ds, Sx_cc, Sx_nocc = self.cc.cc(state, self.ref_traj, self.ref_ctrl, return_sx = True , debug=False)
            self.theory_cov_mtx =  Sx_cc[-4:-2,-4:-2]

        else:
            # effectively disable cc
            #print_warning("CC disabled")
            Sx_nocc = self.cc.getNoCcSx(state)
            self.theory_cov_mtx =  Sx_nocc[-4:-2,-4:-2]
            Ks = np.zeros([self.N*self.m*self.n])
            As = np.zeros([self.N*self.n*self.n])
            Bs = np.zeros([self.N*self.n*self.m])


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


        #print("[ccmppi.py: ",x0)
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


        p.s("post")
        # Calculate statistics of cost function
        cost = np.array(cost)
        beta = np.min(cost)
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

        # generate ref_traj by simulating synthesized optimal control sequence 
        self.buildReferenceTrajectory(x0, ref_control)


        # throttle, steering
        #print("control = %7.3f, %7.3f " %(ref_control[0,0], degrees(ref_control[0,1])))

        p.e()
        self.debug_dict = {'sampled_control':control}
        if isnan(ref_control[0,1]):
            print_error("cc-mppi fail to return valid control")
        return ref_control

    # given initial state and control sequence, generate ref_traj and ref_ctrl
    def buildReferenceTrajectory(self, x0, ctrl_sequence):
        assert (ctrl_sequence.shape[0] == self.N)
        assert (ctrl_sequence.shape[1] == self.m)
        state = x0.copy()
        self.ref_traj = [state]
        self.ref_ctrl = ctrl_sequence.copy()
        for k in range(self.N):
            #print("step = %d, x= %.3f, y=%.3f, v=%.3f, psi=%.3f, T=%.3f, S=%.3f"%(k,state[0],state[1],state[2],state[3], this_control_seq[k,0], this_control_seq[k,1]))
            state = self.parent.applyDiscreteDynamics(state,self.ref_ctrl[k],self.dt)
            self.ref_traj.append(state)
        self.ref_traj = np.array(self.ref_traj)
        self.ref_ctrl = np.array(self.ref_ctrl)
        return

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

