from common import *
# base class for Planner
# Planner is a layer between a global trajectory and a local controller
# it can run asynchronously, updating a local trajectory for controller to follow
class Planner(ConfigObject):
    def __init__(self,config=None):
        self.config = config
        self.car = None
        self.main = None
        self.no_solution = False

        # N*2 cartesian points
        self.plan_traj = None
        # replan every [skip_count] steps
        self.skip_count = 3

        ConfigObject.__init__(self,config)
        # age of a plan in timesteps, use for asynchronous updates
        # init to skip_count so plan would happen on first step
        self.plan_age = self.skip_count

    def preInit(self):
        return
    def postInit(self):
        return
    def init(self):
        return

    def needReplan(self):
        self.plan_age += 1
        if (self.plan_age < self.skip_count and not self.no_solution):
            return False

        self.plan_age = 0
        return True

    # create a plan, store states internally
    # return True/False, indicating planner success/failure
    def plan(self):
        if (not self.needReplan()):
            return True
        # make new plan
        return True

    def plotDebug(self):
        #plot debug information
        return

    def localTrajectory(self,state):
        #(local_ctrl_pnt,offset,orientation,curvature,v_target) = retval
        return


