# Base class for RCPTrack and Skidpad
# this class provides API for interacting with a Track object
# a track object provides information on the trajectory and provide access for drawing the track
import numpy as np
from scipy.interpolate import splprep, splev,CubicSpline,interp1d
from math import radians,degrees,cos,sin,ceil,floor,atan,tan
import cv2
class Track(object):
    def __init__(self,config=None):
        self.config = config


        # the following variables need to be overriden in subclass initilization
        # pixels per meter
        self.resolution = None
        self.discretized_raceline_len = 1024

        # obstacles
        self.obstacle=False
        self.obstacle_count=0
        self.obstacle_filename=None
        self.obstacle_radius=None

        # track dimension, in meters
        self.x_limit = None
        self.y_limit = None


    # need to move to this file TODO
    # draw a circle on canvas at coord
    def drawCircle(self, img, coord, radius_m, color = (0,0,0)):
        src = self.m2canvas(coord)
        radius_pix = int(radius_m * self.resolution)
        img = cv2.circle(img, src, radius_pix, color,-1)
        return img

    # check if vehicle is currently in collision with obstacle
    # only give index of the first obstacle if multiple obstacle is in collision
    # TODO optimize
    def isInObstacle(self, state):
        if (not self.obstacle):
            return (False,-1)
        dist = self.obstacle_radius
        x,y,heading,vf,vs,omega = state
        min_dist = 100.0
        for i in range(self.obstacles.shape[0]):
            obs = self.obstacles[i]
            dist = ((x-obs[0])**2+(y-obs[1])**2)**0.5 
            if (dist<min_dist):
                min_dist = dist
            if (dist < self.obstacle_radius):
                return (True,i)
        return (False,-1)

    def m2canvas(self,coord):
        x_new = int(np.clip(coord[0],0,self.x_limit) * self.resolution)
        y_new = int((self.y_limit-np.clip(coord[1],0,self.y_limit)) * self.resolution)
        return (x_new,y_new)


    def drawArrow(self):
        return


    # NOTE need to be overridden in each subclass Track
    # draw a raceline
    def drawRaceline(self,img=None):
        raise NotImplementedError

    # draw a picture of the track
    def drawTrack(self, img=None,show=False):
        pass

    def localTrajectory(self,state):
        raise NotImplementedError


    # universal function for all Track classes
    def setResolution(self,res):
        self.resolution = res
        return

    def prepareDiscretizedRaceline(self):
        ss = np.linspace(0,self.raceline_len_m,self.discretized_raceline_len)
        rr = splev(ss%self.raceline_len_m,self.raceline_s,der=0)
        drr = splev(ss%self.raceline_len_m,self.raceline_s,der=1)
        heading_vec = np.arctan2(drr[1],drr[0])
        vv = self.sToV(ss) 
        top_speed = 10
        vv[vv>top_speed] = top_speed

        # parameter, distance along track
        self.ss = ss
        self.raceline_points = np.array(rr)
        self.raceline_headings = heading_vec
        self.raceline_velocity = vv

        # describe track boundary as offset from raceline
        self.createBoundary()
        self.discretized_raceline = np.vstack([self.raceline_points,self.raceline_headings,vv, self.raceline_left_boundary, self.raceline_right_boundary]).T
        '''
        left = np.array(self.raceline_left_boundary)
        right = np.array(self.raceline_right_boundary)
        plt.plot(left+right)
        plt.show()
        breakpoint()
        '''
        return

    def createBoundary(self,show=False):
        # construct a (self.discretized_raceline_len * 2) vector
        # to record the left and right track boundary as an offset to the discretized raceline
        left_boundary = []
        right_boundary = []

        left_boundary_points = []
        right_boundary_points = []

        for i in range(self.discretized_raceline_len):
            # find normal direction
            coord = self.raceline_points[:,i]
            heading = self.raceline_headings[i]

            left, right = self.preciseTrackBoundary(coord,heading)
            left_boundary.append(left)
            right_boundary.append(right)

            # debug boundary points
            left_point = (coord[0] + left * cos(heading+np.pi/2),coord[1] + left * sin(heading+np.pi/2))
            right_point = (coord[0] + right * cos(heading-np.pi/2),coord[1] + right * sin(heading-np.pi/2))

            left_boundary_points.append(left_point)
            right_boundary_points.append(right_point)


            # DEBUG
            # plot left/right boundary
            '''
            left_point = (coord[0] + left * cos(heading+np.pi/2),coord[1] + left * sin(heading+np.pi/2))
            right_point = (coord[0] + right * cos(heading-np.pi/2),coord[1] + right * sin(heading-np.pi/2))
            img = self.drawTrack()
            img = self.drawRaceline(img = img)
            img = self.drawPoint(img,coord,color=(0,0,0))
            img = self.drawPoint(img,left_point,color=(0,0,0))
            img = self.drawPoint(img,right_point,color=(0,0,0))
            plt.imshow(img)
            plt.show()
            '''


        self.raceline_left_boundary = left_boundary
        self.raceline_right_boundary = right_boundary

        if (show):
            img = self.drawTrack()
            img = self.drawRaceline(img = img)
            img = self.drawPolyline(left_boundary_points,lineColor=(0,255,0),img=img)
            img = self.drawPolyline(right_boundary_points,lineColor=(0,0,255),img=img)
            plt.imshow(img)
            plt.show()
            return img
        return
