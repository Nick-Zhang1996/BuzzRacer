# Base class for RCPtrack and Skidpad
# this class provides API for interacting with a Track object
# a track object provides information on the trajectory and provide access for drawing the track
class Track(object):
    def __init__(self,resolution=200):
        # resolution : pixels per grid length
        self.resolution = resolution

    # draw a picture of the track
    def drawTrack(self):
        pass

    def localTrajectory(self,state):
        pass

    def setResolution(self,res):
        self.resolution = res
        return
