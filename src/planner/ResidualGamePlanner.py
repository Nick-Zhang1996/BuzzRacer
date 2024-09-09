from common import *
from planner.Planner import Planner
from controller.ResidualGameCarController import ResidualGameCarController
from simulator.CurvilinearSimulator import CurvilinearSimulator

class ResidualGamePlanner(Planner,ResidualGameCarController):
    def __init__(self,config=None):
        self.config = config
        self.car = None
        self.main = None
        Planner.__init__(self,config)
        self.ego_traj = None
        self.oppo_traj = None


    def init(self):
        ResidualGameCarController.__init__(self,self.car,self.config)
        ResidualGameCarController.preInit(self)
        ResidualGameCarController.init(self)

        self.simulator = CurvilinearSimulator(self.main)
        self.simulator.init()
        return
    # create a plan, store states internally
    def plan(self):
        # set all car sim_states
        x0 = self.simulator.cart2Curv(self.ego_car.states)
        x1 = self.simulator.cart2Curv(self.oppo_car.states)
        # TODO start here
        xi_ref, xj_ref = ResidualGameCarController.solveGame(self,x0,x1)

        # convert to cartesian coord
        xx_i_cart = [self.simulator.curv2Cart(val) for val in xi_ref]
        xx_j_cart = [self.simulator.curv2Cart(val) for val in xj_ref]

        # store for use in localTrajectory
        self.ego_traj = xx_i_cart
        self.oppo_traj = xx_j_cart
        return

    def plotDebug(self):
        #plot debug information
        if (self.main.visualization.update_visualization.is_set()):
            img = self.main.visualization.visualization_img
            ego_path = [val[:2] for val in self.ego_traj]
            oppo_path = [val[:2] for val in self.oppo_traj]
            img = self.main.track.drawPolyline(ego_path,img)
            img = self.main.track.drawPolyline(oppo_path,img)
            self.main.visualization.visualization_img = img
        return

    def localTrajectory(self,state):
        # TODO
        return


