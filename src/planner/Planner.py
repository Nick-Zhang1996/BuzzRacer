from common import *
# base class for Planner
# Planner is a layer between a global trajectory and a local controller
# it can run asynchronously, updating a local trajectory for controller to follow
class Planner(ConfigObject):
    def __init__(self,config=None):
        self.config = config
        self.car = None
        self.main = None
        ConfigObject.__init__(self,config)

    def init(self):
        return
    # create a plan, store states internally
    def plan(self):
        return

    def plotDebug(self):
        #plot debug information
        return

    def localTrajectory(self,state):
        return


