# mppi car controller, with dynamic model
from controller.CarController import CarController
import numpy as np
from time import time,sleep
from math import radians,degrees,cos,sin,ceil,floor,atan,tan
from scipy.interpolate import splprep, splev,CubicSpline,interp1d
import pycuda.autoinit
global drv
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt

class MppiCarController(CarController):
    def __init__(self,car):
        super().__init__(car)
        np.set_printoptions(formatter={'float': lambda x: "{0:7.4f}".format(x)})

        # these setting should be handled in config file
        self.n = self.state_dim = 6
        self.m = self.control_dim = 2
        self.samples_count = 4096
        self.horizon = 30
        self.track = self.car.main.track
        self.dt = 0.02
        self.discretized_raceline_len = 1024
        self.temperature = 0.01
        self.control_limit = np.array([[-1.0,1.0],[-radians(27.1),radians(27.1)]])
        # directly sample control
        self.print_ok("max throttle = %.2f"%(self.car.max_throttle))
        #self.noise_cov = np.array([(self.car.max_throttle*1.5)**2,radians(30.0)**2])
        #self.noise_mean = np.array([0.207,0])

        # sample control change rate val/sec
        self.noise_cov = np.array([(self.car.max_throttle*2/0.4)**2,(radians(27.0)*2/0.4)**2])
        self.noise_mean = np.array([0.0,0])

        #self.old_ref_control = np.zeros( (self.samples_count,self.control_dim) )
        self.last_control = np.zeros(2,dtype=np.float32)
        self.freq_vec = []


        self.prepareDiscretizedRaceline()
        self.createBoundary()
        self.initCuda()

    def prepareDiscretizedRaceline(self):
        ss = np.linspace(0,self.track.raceline_len_m,self.discretized_raceline_len)
        rr = splev(ss%self.track.raceline_len_m,self.track.raceline_s,der=0)
        drr = splev(ss%self.track.raceline_len_m,self.track.raceline_s,der=1)
        heading_vec = np.arctan2(drr[1],drr[0])
        vv = self.track.sToV(ss) 
        vv *= 0.8
        vv[vv>1.8] = 1.8

        # parameter, distance along track
        self.ss = ss
        self.raceline_points = np.array(rr)
        self.raceline_headings = heading_vec
        self.raceline_velocity = vv

        # describe track boundary as offset from raceline
        self.createBoundary()
        self.discretized_raceline = np.vstack([self.raceline_points,self.raceline_headings,vv, self.raceline_left_boundary, self.raceline_right_boundary]).T
        return

    def createBoundary(self,show=False):
        # construct a (self.discretized_raceline_len * 2) vector
        # to record the left and right track boundary as an offset to the discretized raceline
        left_boundary = []
        right_boundary = []

        left_boundary_points = []
        right_boundary_points = []

        for i in range(self.discretized_raceline_len):
            # find normal direction
            coord = self.raceline_points[:,i]
            heading = self.raceline_headings[i]

            left, right = self.track.preciseTrackBoundary(coord,heading)
            left_boundary.append(left)
            right_boundary.append(right)

            # debug boundary points
            left_point = (coord[0] + left * cos(heading+np.pi/2),coord[1] + left * sin(heading+np.pi/2))
            right_point = (coord[0] + right * cos(heading-np.pi/2),coord[1] + right * sin(heading-np.pi/2))

            left_boundary_points.append(left_point)
            right_boundary_points.append(right_point)


            # DEBUG
            # plot left/right boundary
            '''
            left_point = (coord[0] + left * cos(heading+np.pi/2),coord[1] + left * sin(heading+np.pi/2))
            right_point = (coord[0] + right * cos(heading-np.pi/2),coord[1] + right * sin(heading-np.pi/2))
            img = self.track.drawTrack()
            img = self.track.drawRaceline(img = img)
            img = self.track.drawPoint(img,coord,color=(0,0,0))
            img = self.track.drawPoint(img,left_point,color=(0,0,0))
            img = self.track.drawPoint(img,right_point,color=(0,0,0))
            plt.imshow(img)
            plt.show()
            '''


        self.raceline_left_boundary = left_boundary
        self.raceline_right_boundary = right_boundary

        if (show):
            img = self.track.drawTrack()
            img = self.track.drawRaceline(img = img)
            img = self.track.drawPolyline(left_boundary_points,lineColor=(0,255,0),img=img)
            img = self.track.drawPolyline(right_boundary_points,lineColor=(0,0,255),img=img)
            plt.imshow(img)
            plt.show()
            return img
        return

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
        cuda_filename = "./controller/mppi/mppi_racecar.cu"
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

        # TODO:
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
        #self.rand_vals = np.zeros(self.samples_count*self.horizon*self.m, dtype=np.float32)
        #self.device_rand_vals = drv.to_device(self.rand_vals)

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

#   state: (x,y,heading,v_forward,v_sideway,omega)
# Note the difference between control_rate and actual control. Since we sample the time rate of change on control it's a bit confusing
    def control(self):
        t = time()
        # vf: forward v
        # vs: lateral v, left positive
        # omega: angular velocity
        x,y,heading,vf,vs,omega = self.car.states
        self.print_info("v = %.2f"%(vf))

        #ref_control = np.vstack([self.old_ref_control[1:,:],np.zeros([1,self.m],dtype=np.float32)])
        ref_control_rate = np.zeros([self.horizon,self.m],dtype=np.float32)

        # generate random var
        random_vals = np.zeros(self.samples_count*self.horizon*self.control_dim,dtype=np.float32) 
        self.cuda_generate_control_noise(block=(self.curand_kernel_n,1,1),grid=(1,1,1))
        #random_vals = random_vals.reshape( (self.samples_count, self.horizon, self.control_dim) )
        #cov0 = np.std(random_vals[:,:,0])
        #cov1 = np.std(random_vals[:,:,1])
        #self.print_info("cov0 %.2f, cov1 %.2f"%(cov0,cov1))
        # evaluate control sequence
        device_ref_control_rate = self.to_device(ref_control_rate)
        device_initial_state = self.to_device(self.car.states)
        costs = np.zeros((self.samples_count), dtype=np.float32)
        sampled_control_rate = np.zeros( self.samples_count*self.horizon*self.m, dtype=np.float32 )
        device_last_control = self.to_device(self.last_control)

        sampled_trajectory = np.zeros((self.samples_count*self.horizon*self.n), dtype=np.float32)
        self.cuda_evaluate_control_sequence(
                device_initial_state, 
                device_last_control,
                device_ref_control_rate, 
                drv.Out(costs),
                drv.Out(sampled_control_rate),
                #drv.Out(sampled_trajectory),
                block=self.cuda_block_size,grid=self.cuda_grid_size
                )
        # sampled trajectory overhead with GPU has 10Hz impact
        #sampled_trajectory = sampled_trajectory.reshape(self.samples_count, self.horizon, self.n)

        # retrieve cost
        sampled_control_rate = sampled_control_rate.reshape(self.samples_count,self.horizon,self.m)
        control_rate = self.synthesizeControl(costs, sampled_control_rate)
        #self.print_info("steering rate: %.2f"%(degrees(control_rate[0,1])))

        # display expected trajectory
        # 5Hz impact
        control = self.last_control + np.cumsum( control_rate, axis=0)*self.dt
        expected_trajectory = self.getTrajectory( self.car.states, control )
        self.expected_trajectory = expected_trajectory
        self.plotTrajectory(expected_trajectory)

        #self.last_ref_control = control.copy()
        self.last_ref_control = np.zeros_like(control)

        self.car.throttle += control_rate[0,0]*self.dt
        self.car.steering += control_rate[0,1]*self.dt

        #self.print_info("T: %.2f, S: %.2f"%(self.car.throttle, degrees(self.car.steering)))
        self.last_control = [self.car.throttle,self.car.steering]
        dt = time() - t
        self.freq_vec.append(1.0/dt)
        #self.print_info("mean freq = %.2f Hz"%(np.mean(self.freq_vec)))

        '''
        display_trajectory = sampled_trajectory[:,:,0:2]
        for i in range(display_trajectory.shape[0]):
            self.plotTrajectory(display_trajectory[i])
        self.print_info("steering std %.2f deg"%(180.0/np.pi*np.std(sampled_control[:,:,1])))
        '''

        # verify GPU against cpu
        '''
        x0 = self.car.states
        index = 50
        cpu_control = sampled_control[index,:,:]
        cpu_trajectory = self.getTrajectory(x0, cpu_control)
        gpu_trajectory = sampled_trajectory[index,:]
        self.print_info("diff = %.2f"%(np.linalg.norm(cpu_trajectory-gpu_trajectory)))
        '''
        return True

    def getTrajectory(self, x0, control):
        trajectory = []
        state = x0
        for i in range(control.shape[0]):
            state = self.advanceDynamics( state, control[i] )
            trajectory.append(state)
        return np.array(trajectory)

    def advanceDynamics(self, state, control, dt=0.01):
        # constants
        lf = 0.09-0.036
        lr = 0.036
        L = 0.09

        Df = 3.93731
        Dr = 6.23597
        C = 2.80646
        B = 0.51943
        Iz = 0.00278*0.5
        m = 0.1667

        # convert to local frame
        #x,vxg,y,vyg,heading,omega = tuple(state)
        x,y,heading,vx,vy,omega = tuple(state)
        throttle,steering = tuple(control)

        # for small velocity, use kinematic model 
        if (vx<0.05):
            beta = atan(lr/L*tan(steering))
            norm = lambda a,b:(a**2+b**2)**0.5
            # motor model
            d_vx = 0.425*(15.2*throttle - vx - 3.157)

            vx = vx + d_vx * dt
            vy = norm(vx,vy)*sin(beta)
            d_omega = 0.0
            omega = vx/L*tan(steering)

            slip_f = 0
            slip_r = 0
            Ffy = 0
            Fry = 0

        else:
            slip_f = -np.arctan((omega*lf + vy)/vx) + steering
            slip_r = np.arctan((omega*lr - vy)/vx)

            Ffy = Df * np.sin( C * np.arctan(B *slip_f)) * 9.8 * lr / (lr + lf) * m
            Fry = Dr * np.sin( C * np.arctan(B *slip_r)) * 9.8 * lf / (lr + lf) * m

            # motor model
            #Frx = (1.8*0.425*(15.2*throttle - vx - 3.157))*m
            # Dynamics
            #d_vx = 1.0/m * (Frx - Ffy * np.sin( steering ) + m * vy * omega)
            d_vx = 1.8*0.425*(15.2*throttle - vx - 3.157)

            d_vy = 1.0/m * (Fry + Ffy * np.cos( steering ) - m * vx * omega)
            d_omega = 1.0/Iz * (Ffy * lf * np.cos( steering ) - Fry * lr)

            # discretization
            vx = vx + d_vx * dt
            vy = vy + d_vy * dt
            omega = omega + d_omega * dt 

        # back to global frame
        vxg = vx*cos(heading)-vy*sin(heading)
        vyg = vx*sin(heading)+vy*cos(heading)

        # apply updates
        # TODO add 1/2 a t2
        x += vxg*dt
        y += vyg*dt
        heading += omega*dt + 0.5* d_omega * dt * dt

        retval = x,y,heading,vx,vy,omega
        return retval

    def plotTrajectory(self,trajectory):
        if (not self.car.main.visualization.update_visualization.is_set()):
            return
        img = self.car.main.visualization.visualization_img
        for coord in trajectory:
            img = self.car.main.track.drawCircle(img,coord, 0.02, color=(0,0,0))
        self.car.main.visualization.visualization_img = img
        return

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
