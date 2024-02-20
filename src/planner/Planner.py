from common import *
# base class for Planner
# Planner is a layer between a global trajectory and a local controller
# it can run asynchronously, updating a local trajectory for controller to follow
class Planner(ConfigObject):
    def __init__(self,config=None):
        self.config = config
        self.car = None
        self.main = None
        # N*2 cartesian points
        self.plan_traj = None
        ConfigObject.__init__(self,config)

    # create a plan, store states internally
    # return True/False, indicating planner success/failure
    def plan(self):
        return True

    def plotDebug(self):
        #plot debug information
        return

    def localTrajectory(self,state):
        #(local_ctrl_pnt,offset,orientation,curvature,v_target) = retval
        return


