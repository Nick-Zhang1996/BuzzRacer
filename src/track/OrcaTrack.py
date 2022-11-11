# Orca track from ETH Zurich
from common import *
from track.Track import Track

class OrcaTrack(Track,PrintObject):
    def __init__(self):
        self.obstacle = False
        self.obstacles = []
        pass

    # draw a picture of the track
    def drawTrack(self):
        pass

    # draw a raceline
    def drawRaceline(self,img=None):

    def drawCricle(self,img,pos,radius,color):
        return img

    def isInObstacle(self,state,get_obstacle_id=False):
        return has_collided, obstacle_id

    def m2canvas(self,coord):
        return (0,0)
    
    def drawArrow(self):
        return

