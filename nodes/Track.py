# Parent class for Track, children include RCPtrack and Skidpad
class Track(object):
    def __init__(self,resolution=200):
        # resolution : pixels per grid length
        self.resolution = resolution

    # draw a picture of the track
    def drawTrack(self):
        pass

    def localTrajectory(self,state):
        pass

