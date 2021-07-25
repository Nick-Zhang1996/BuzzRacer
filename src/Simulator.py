from common import *
from Extension import Extension
from time import time,sleep
# base class for all simulators
# contains code for aligning simulator time with real time
class Simulator(Extension):
    def __init__(self,main):
        super().__init__(main)
        self.t0 = None
        self.real_sim_time_ratio = 1.0
        print_ok("[%s]: real/sim time ratio = %.1f "%(self.__class__.__name__, self.real_sim_time_ratio))

    def init(self):
        # ensure experiment_type hasn't been initialized
        flag_is_unique = False
        try:
            self.main.experiment_type != ExperimentType.Simulation
        except (AttributeError):
            flag_is_unique = True
        if (not flag_is_unique):
            print_error("[%s]: another state update source has been initialized"%(self.__class__.__name__))

        self.main.experiment_type = ExperimentType.Simulation
        self.main.sim_t = 0

    def matchRealTime(self):
        if (self.t0 is None):
            self.t0 = time()
        time_to_reach = self.main.sim_t * self.real_sim_time_ratio + self.t0
        #print("sim_t = %.3f, time = %.3f, expected= %.3f, delta = %.3f"%(self.main.sim_t, time()-self.t0, self.main.sim_t*self.real_sim_time_ratio, time_to_reach-time() ))
        if (time_to_reach-time() < 0):
            pass
            #print_warning("algorithm can't keep up ..... %.3f s"%(time()-time_to_reach))

        sleep(max(0,time_to_reach - time()))

