from common import *
from extension import Extension
from time import time,sleep
import numpy as np
# base class for all simulators
# contains code for aligning simulator time with real time

# it is required that:
# car.states = x,y,heading,v_forward,v_sideway,omega
# however simulator can establish a property car.sim_states
# that use different state representation
class Simulator(Extension,PrintObject):
    def __init__(self,main):
        super().__init__(main)
        self.match_time = None
        self.state_noise_enabled = None
        self.state_noise_magnitude = None
        self.state_noise_type = None
        self.state_noise_probability = None

        self.t0 = None
        self.real_sim_time_ratio = 1.0
        self.print_info("real/sim time ratio = %.1f "%(self.real_sim_time_ratio))

    def init(self):
        if (self.main.experiment_type != ExperimentType.Simulation):
            self.print_error("Experiment type is not Simulation but a Simulator is loaded")
        self.main.sim_t = 0
        self.print_info( "match_time: " + str(self.match_time))

        if self.state_noise_enabled:
            assert (self.state_noise_type is not None)
            assert (self.state_noise_magnitude is not None)
            self.state_noise_magnitude = np.array(self.state_noise_magnitude)
            if (self.state_noise_type == 'normal'):
                self.addStateNoise = self.addStateNoiseNormal
            elif (self.state_noise_type == 'uniform'):
                self.addStateNoise = self.addStateNoiseUniform
            elif (self.state_noise_type == 'impulse'):
                assert (self.state_noise_probability is not None)
                self.addStateNoise = self.addStateNoiseImpulse
            else:
                self.print_error('unknown noise type ',self.state_noise_type)


    def matchRealTime(self):
        if (not self.match_time):
            return
        if (self.t0 is None):
            self.t0 = time()
        time_to_reach = self.main.sim_t * self.real_sim_time_ratio + self.t0
        #print("sim_t = %.3f, time = %.3f, expected= %.3f, delta = %.3f"%(self.main.sim_t, time()-self.t0, self.main.sim_t*self.real_sim_time_ratio, time_to_reach-time() ))
        if (time_to_reach-time() < 0):
            pass
            #print_warning("algorithm can't keep up ..... %.3f s"%(time()-time_to_reach))

        sleep(max(0,time_to_reach - time()))

    def addStateNoiseNormal(self):
        for car in self.cars:
            car.states += np.random.normal(size=car.states.shape) * self.state_noise_magnitude * self.main.dt

    def addStateNoiseUniform(self):
        for car in self.cars:
            car.states += np.random.uniform(low=-1.0,high=1.0,size=car.states.shape) * self.state_noise_magnitude * self.main.dt

    def addStateNoiseImpulse(self):
        for car in self.cars:
            val = np.random.uniform()
            if val < self.state_noise_probability:
                # to stay consistent with cvar_racecar.cu
                '''
                new_val = np.random.uniform()
                if (new_val < 0.5):
                    car.states += (1/self.state_noise_probability) * self.state_noise_magnitude * self.main.dt
                else:
                    car.states -= (1/self.state_noise_probability) * self.state_noise_magnitude * self.main.dt
                '''
                car.states += (1/self.state_noise_probability) * self.state_noise_magnitude * self.main.dt
