from common import *
from planner.Planner import Planner
from controller.iLQGameCarController import iLQGameCarController
from simulator.CurvilinearSimulator import CurvilinearSimulator

class iLQGamePlanner(Planner,iLQGameCarController):
    def __init__(self,config=None):
        self.config = config
        self.car = None
        self.main = None
        Planner.__init__(self,config)



    def init(self):
        iLQGameCarController.__init__(self,self.car,self.config)
        iLQGameCarController.preInit(self)
        iLQGameCarController.init(self)

        self.simulator = CurvilinearSimulator(self.main)
        self.simulator.init()
        return
    # create a plan, store states internally
    def plan(self):
        # set all car sim_states
        x0 = self.simulator.cart2Curv(self.ego_car.states)
        x1 = self.simulator.cart2Curv(self.oppo_car.states)
        iLQGameCarController.lqControl(self,x0,x1)
        # propagate control forward
        breakpoint()

        return

    def plotDebug(self):
        #plot debug information
        return

    def localTrajectory(self,state):
        return


