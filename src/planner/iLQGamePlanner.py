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
        self.ego_traj = None
        self.oppo_traj = None


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
        alpha1s, P1s, alpha2s, P2s = iLQGameCarController.lqControl(self,x0,x1)

        # propagate control forward
        xx_i =[x0.reshape((self.n,1))]
        xx_j =[x1.reshape((self.n,1))]
        for t in range(self.horizon):
            #for ego agent i
            # x+ = A x + B u + d, for x~x_ref
            # ~x = x - x_ref
            # ~x+ = A~x + B~u
            dx_i = xx_i[-1] - self.x_i_ref[t]
            dx_j = xx_j[-1] - self.x_j_ref[t]
            dx = np.vstack([dx_i,dx_j])

            # NOTE ignoring control constraint
            #for ego agent i
            u = self.u_i_ref[t] - P1s[t] @ dx + alpha1s[t]
            u = np.array(u).reshape(-1,1)
            new_x = self.update_dynamics(xx_i[-1],u)
            xx_i.append(new_x.reshape(4,1))

            # for agent j
            u = self.u_j_ref[t] - P2s[t] @ dx + alpha2s[t]
            u = np.array(u).reshape(-1,1)
            new_x = self.update_dynamics(xx_j[-1],u)
            xx_j.append(new_x.reshape(4,1))
        # convert to cartesian coord
        xx_i_cart = [self.simulator.curv2Cart(val) for val in xx_i]
        xx_j_cart = [self.simulator.curv2Cart(val) for val in xx_j]

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


