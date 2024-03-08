# mppi car controller, with dynamic model
import numpy as np
from time import time,sleep
from math import radians,degrees,cos,sin,ceil,floor,atan,tan
from scipy.interpolate import splprep, splev,CubicSpline,interp1d

global drv
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from controller.CarController import CarController
from extension.simulator.KinematicBicycleFrenetSimulator import KinematicBicycleFrenetSimulator

# DEBUG FIXME
import matplotlib.pyplot as plt

class MppiFrenetCarController(CarController):
    def __init__(self,car,config):

        # reconfigurable parameters
        self.state_dim = 5
        self.control_dim = 2
        self.samples_count = None # to be set in config
        self.horizon = None       # to be set in config
        self.dt = 0.01
        self.temperature = 0.01


        # cost to apply on state
        # state x: s,v,n,phi, beta
        self.Q1 = np.diag(  [ 0,0.00,1.0,1.0, 0.0])
        self.q1 = np.array([[-4,0,0,0, 0.0]]).T
        # aggressiveness: 0->don't care about opponent 1->J = s_i - s_j
        self.Qop1 = 0

        self.Q2 = np.diag(  [ 0,0.00,1.0,1.0,0.0])
        self.q2 = np.array([[-4,0,0,0,0.0]]).T
        self.Qop2 = 0

        # cost on track boundary
        #self.boundary_min_distance = 0.06 * 2
        self.boundary_min_distance = 0.02
        self.boundary_cost = 30.0*3

        # cost on opponent collision
        Kcol = 30.0*3
        self.opponent_min_distance_s = 0.3
        self.opponent_min_distance_n = 0.19
        self.Qcol = np.diag([Kcol,0,Kcol,0,0])

        super().__init__(car,config)
        self.track = self.car.main.track
        self.n =  self.state_dim
        self.m =  self.control_dim

        np.set_printoptions(formatter={'float': lambda x: "{0:7.4f}".format(x)})


        '''
        for key,value_text in config.attributes.items():
            setattr(self,key,eval(value_text))
            #self.print_info(" controller.",key,'=',value_text)
        '''

    def init(self):
        max_ay = self.car.max_ay
        max_ax = self.car.max_ax

        self.control_limit = np.array([[-max_ay,max_ay],[-max_ax,max_ax]])
        # directly sample control
        # FIXME
        self.noise_cov = np.array([(max_ay*0.01)**2,(max_ax*0.01)**2])
        self.noise_mean = np.array([0,0])

        # sample control change rate val/sec
        #self.noise_cov = np.array([(self.car.max_throttle*2/0.4)**2,(radians(27.0)*2/0.2)**2])
        #self.noise_mean = np.array([0.0,0])

        #self.old_ref_control = np.zeros( (self.samples_count,self.control_dim) )
        self.last_control = np.zeros(2,dtype=np.float32)
        self.freq_vec = []

        # TODO do these in track initialization
        track = self.track
        track.prepareDiscretizedRaceline()
        # s, curvature, v, left_bdry, righ_bdry
        self.discretized_raceline = np.vstack([track.ss,track.curvature,track.raceline_velocity,track.raceline_left_boundary, track.raceline_right_boundary]).T

        self.raceline_left_boundary = track.raceline_left_boundary
        self.raceline_right_boundary = track.raceline_right_boundary
        self.initCuda()
        # TODO may need to initialize
        #self.predict()

    def initCuda(self):
        self.curand_kernel_n = 1024

        # prepare constants
        cuda_code_macros = {
                "SAMPLE_COUNT":self.samples_count,
                "HORIZON":self.horizon, 
                "CONTROL_DIM":self.m,
                "STATE_DIM":self.state_dim,
                "RACELINE_LEN":self.discretized_raceline.shape[0],
                "TEMPERATURE":self.temperature,
                "DT":self.dt
                }
        cuda_code_macros.update({"CURAND_KERNEL_N":self.curand_kernel_n})
        # cost parameters
        cuda_code_macros.update({"COST_Q_S":self.Q1[0,0]})
        cuda_code_macros.update({"COST_Q_V":self.Q1[1,1]})
        cuda_code_macros.update({"COST_Q_N":self.Q1[2,2]})
        cuda_code_macros.update({"COST_Q_PHI":self.Q1[3,3]})

        cuda_code_macros.update({"COST_q_S":self.q1[0,0]})
        cuda_code_macros.update({"COST_q_V":self.q1[1,0]})
        cuda_code_macros.update({"COST_q_N":self.q1[2,0]})
        cuda_code_macros.update({"COST_q_PHI":self.q1[3,0]})

        cuda_code_macros.update({"COST_QOP_1":self.Qop1})
        cuda_code_macros.update({"COST_QOP_2":self.Qop2})

        cuda_code_macros.update({"COST_BDRY_MIN":self.boundary_min_distance})
        cuda_code_macros.update({"COST_BDRY":self.boundary_cost})

        cuda_code_macros.update({"COST_OPPO_MIN_S":self.opponent_min_distance_s})
        cuda_code_macros.update({"COST_OPPO_MIN_N":self.opponent_min_distance_n})
        cuda_code_macros.update({"COST_Q_COL":self.Qcol[0][0]})

        cuda_code_macros.update({"PARAM_MAX_AX":self.car.max_ax})
        cuda_code_macros.update({"PARAM_MAX_AY":self.car.max_ay})
        cuda_code_macros.update({"PARAM_MAX_V":self.car.max_v})

        cuda_filename = "./controller/mppi_frenet/mppi_kinematic_bicycle_frenet.cu"
        self.loadCudaFile(cuda_filename, cuda_code_macros)
        self.setBlockGrid()

        self.cuda_init_curand_kernel = self.getFunctionSafe("init_curand_kernel")
        self.cuda_generate_control_noise = self.getFunctionSafe("generate_control_noise")
        self.cuda_evaluate_control_sequence = self.getFunctionSafe("evaluate_control_sequence")
        self.cuda_set_control_limit = self.getFunctionSafe("set_control_limit")
        self.cuda_set_noise_cov = self.getFunctionSafe("set_noise_cov")
        self.cuda_set_noise_mean = self.getFunctionSafe("set_noise_mean")
        self.cuda_set_raceline = self.getFunctionSafe("set_raceline")
        self.initCurand()

        # set control limit
        device_control_limit = self.to_device(self.control_limit)
        self.cuda_set_control_limit(device_control_limit,block=(1,1,1),grid=(1,1,1))
        # set noise variance
        device_noise_cov = self.to_device(self.noise_cov)
        self.cuda_set_noise_cov(device_noise_cov, block=(1,1,1),grid=(1,1,1))
        # set noise mean
        device_noise_mean = self.to_device(self.noise_mean)
        self.cuda_set_noise_mean(device_noise_mean, block=(1,1,1),grid=(1,1,1))
        # set raceline
        device_raceline = self.to_device(self.discretized_raceline)
        self.cuda_set_raceline(device_raceline, block=(1,1,1),grid=(1,1,1))

        sleep(1)

    def initCurand(self):
        seed = np.int32(int(time()*10000))
        self.cuda_init_curand_kernel(seed,block=(self.curand_kernel_n,1,1),grid=(1,1,1))

    def loadCudaFile(self,cuda_filename,macros):
        self.print_info("loading cuda source code ...")
        with open(cuda_filename,"r") as f:
            code = f.read()
        self.mod = SourceModule(code % macros, no_extern_c=True)

    def setBlockGrid(self):
        if (self.samples_count < 1024):
            # if sample count is small only employ one grid
            self.cuda_block_size = (self.samples_count,1,1)
            self.cuda_grid_size = (1,1)
        else:
            # employ multiple grid,
            self.cuda_block_size = (1024,1,1)
            self.cuda_grid_size = (ceil(self.samples_count/1024.0),1)
        self.print_info("cuda block size %d, grid size %d"%(self.cuda_block_size[0],self.cuda_grid_size[0]))
        return

    def getFunctionSafe(self,name):
        fun = self.mod.get_function(name)
        self.print_info("registers used, ",name,"= %d"%(fun.num_regs))
        assert fun.num_regs < 64
        assert int(fun.num_regs * self.cuda_block_size[0]) <= 65536
        return fun

    def getOpponentStatus(self):
        opponent_count = 0
        opponent_traj = []
        for car in self.main.cars:
            if not (car is self.car):
                opponent_count += 1
                predicted_traj = self.predict(car.sim_states)
                opponent_traj.append(car.controller.predicted_traj)
        # dim: no_opponents, horizon, states
        opponent_traj = np.array(opponent_traj)
        return opponent_count, opponent_traj

    # TODO generate predicted trajectory for opponenet by running MPPI
    # for now just repeat current state
    def predict(self,states):
        predicted_traj = [states for i in range(self.horizon)]
        return predicted_traj



#   state: (x,y,heading,v_forward,v_sideway,omega)
# sim_state: (s,v,n,phi,beta)
    def control(self):
        t = time()
        # vf: forward v
        # vs: lateral v, left positive
        # omega: angular velocity
        x,y,heading,vf,vs,omega = self.car.states
        s,v,n,phi,beta = self.car.sim_states
        self.print_info(f'car v={v}')

        # warm start from previous solution
        #ref_control = np.vstack([self.old_ref_control[1:,:],np.zeros([1,self.m],dtype=np.float32)])

        # cold start from zero reference
        ref_control = np.zeros([self.horizon,self.m],dtype=np.float32)

        # generate random var
        random_vals = np.zeros(self.samples_count*self.horizon*self.control_dim,dtype=np.float32) 
        self.cuda_generate_control_noise(block=(self.curand_kernel_n,1,1),grid=(1,1,1))
        #random_vals = random_vals.reshape( (self.samples_count, self.horizon, self.control_dim) )
        #cov0 = np.std(random_vals[:,:,0])
        #cov1 = np.std(random_vals[:,:,1])
        #self.print_info("cov0 %.2f, cov1 %.2f"%(cov0,cov1))

        # prepare opponent info
        opponent_count, opponent_traj = self.getOpponentStatus()
        opponent_count = np.int32(opponent_count)
        if (opponent_count == 0):
            device_opponent_traj = np.uint64(0)
        else:
            device_opponent_traj = self.to_device(opponent_traj)

        # evaluate control sequence
        device_ref_control = self.to_device(ref_control)
        device_initial_state = self.to_device(self.car.sim_states)
        costs = np.zeros((self.samples_count), dtype=np.float32)
        sampled_control = np.zeros( self.samples_count*self.horizon*self.m, dtype=np.float32 )
        device_last_control = self.to_device(self.last_control)


        sampled_trajectory = np.zeros((self.samples_count*self.horizon*self.n), dtype=np.float32)
        # FIXME remove sampled_trajectory
        self.cuda_evaluate_control_sequence(
                device_initial_state, 
                device_last_control,
                device_ref_control, 
                drv.Out(costs),
                drv.Out(sampled_control),
                opponent_count,
                device_opponent_traj,
                drv.Out(sampled_trajectory),
                block=self.cuda_block_size,grid=self.cuda_grid_size
                )

        # copyig sampled trajectory from gpu to cpu has large negative perf impact
        sampled_trajectory = sampled_trajectory.reshape(self.samples_count, self.horizon, self.n)

        # retrieve cost
        sampled_control = sampled_control.reshape(self.samples_count,self.horizon,self.m)
        # FIXME
        control = self.synthesizeControlMin(costs, sampled_control)


        self.last_ref_control = control.copy()

        #self.car.throttle += control_rate[0,0]*self.dt
        #self.car.steering += control_rate[0,1]*self.dt

        self.car.steering = control[0,0]
        self.car.throttle = control[0,1]

        #self.print_info("T: %.2f, S: %.2f"%(self.car.throttle, degrees(self.car.steering)))
        self.last_control = [self.car.steering,self.car.throttle]
        dt = time() - t
        self.freq_vec.append(1.0/dt)
        #self.print_info("mean freq = %.2f Hz"%(np.mean(self.freq_vec)))

        # DEBUG
        str_mean = np.mean(sampled_control[:,:,0])
        str_std = np.std(sampled_control[:,:,0])
        self.print_info("steering mean %.2f std %.2f"%(str_mean,str_std))
        th_mean = np.mean(sampled_control[:,:,1])
        th_std = np.std(sampled_control[:,:,1])
        self.print_info("throttle mean %.2f std %.2f"%(th_mean, th_std))
        '''
        #FIXME DEBUG plot sampled trajectory
        for i in range(sampled_trajectory.shape[0]):
            self.plotTrajectory(sampled_trajectory[i])

        # display expected trajectory, perf impact
        expected_trajectory = self.getDynamicTrajectory( self.car.states, control )
        self.expected_trajectory = expected_trajectory
        self.plotTrajectory(expected_trajectory)
        '''

        # FIXME verify GPU against cpu
        index = 50
        cpu_control = sampled_control[index,:,:]
        gpu_trajectory = sampled_trajectory[index,:]
        cpu_trajectory = self.getTrajectory(self.car.sim_states, cpu_control)
        #breakpoint()

        cpu_trajectory = self.getTrajectory(self.car.sim_states, cpu_control)
        self.print_info("diff = %.2f"%(np.linalg.norm(cpu_trajectory-gpu_trajectory)))


        # FIXME
        if (self.main.breakpoint.is_set()):
            breakpoint()
            self.main.breakpoint.clear()
        return True

    # select min cost control
    def synthesizeControlMin(self, cost_vec, sampled_control):
        min_index = np.argmin(cost_vec)
        return sampled_control[min_index]

    # given cost and sampled control, return optimal control per MPPI algorithm
    # control_vec: samples * horizon * m
    # cost_vec: samples
    def synthesizeControl(self, cost_vec, sampled_control_rate):
        cost_vec = np.array(cost_vec)
        beta = np.min(cost_vec)
        cost_mean = np.mean(cost_vec-beta)

        # calculate weights
        weights = np.exp(- (cost_vec - beta)/cost_mean/self.temperature)
        weights = weights / np.sum(weights)
        #self.print_info("best cost %.2f, max weight %.2f"%(beta,np.max(weights)))

        synthesized_control_rate = np.zeros((self.horizon,self.m))
        for t in range(self.horizon):
            for i in range(self.m):
                synthesized_control_rate[t,i] = np.sum(weights * sampled_control_rate[:,t,i])
        return synthesized_control_rate

    def to_device(self,data):
        return drv.to_device(np.array(data,dtype=np.float32).flatten())
    def from_device(self,data,shape,dtype=np.float32):
        return drv.from_device(data,shape,dtype)

    # plot trajectory
    def plotTrajectory(self,curv_traj):
        if (self.main.visualization.update_visualization.is_set()):
            img = self.main.visualization.visualization_img
            cart_traj = np.array([KinematicBicycleFrenetSimulator.curv2CartTrack(coord,self.track) for coord in curv_traj])
            img = self.track.drawPolyline(cart_traj[:,:2], img=img, thickness=1)
            self.main.visualization.visualization_img = img

    def getTrajectory(self, x0, control_vec):
        x = x0
        traj = []
        for i in range(self.horizon):
            traj.append(x)
            new_x = KinematicBicycleFrenetSimulator.advanceKinematicBicycleDynamics(x, control_vec[i], self.dt, self.track)
            ddt = (new_x-x)/self.dt
            print(ddt)
            x = new_x
        return traj



