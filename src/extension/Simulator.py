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
        self.state_noise_std = None

        self.t0 = None
        self.real_sim_time_ratio = 1.0
        self.print_info("real/sim time ratio = %.1f "%(self.real_sim_time_ratio))

    def init(self):
        if (self.main.experiment_type != ExperimentType.Simulation):
            self.print_error("Experiment type is not Simulation but a Simulator is loaded")
        self.main.sim_t = 0
        self.print_info( "match_time: " + str(self.match_time))
        self.state_noise_std = np.array(self.state_noise_std)

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

    def addStateNoise(self):
        for car in self.cars:
            car.states += np.random.normal(size=car.states.shape) * self.state_noise_std * self.main.dt
