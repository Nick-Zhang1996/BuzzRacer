import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt

from common import *
from util.timeUtil import execution_timer
from extension.Extension import Extension
from track.RCPTrack import RCPTrack

# helper functions for 2d vector
cross = lambda a,b : a[0]*b[1]-a[1]*b[0]
inv = lambda M: np.array([[M[1,1],-M[0,1]],[-M[1,0],M[0,0]]])/(M[0,0]*M[1,1]-M[0,1]*M[1,0])

class TofSensorSimulation(Extension):
    def __init__(self,main):
        Extension.__init__(self,main)
        self.t = execution_timer(False)

        # Configurable param
        # cars to simulate ToF readings for
        # If enabled, a car will have this variable set
        # car.tof_measurement = [front, left, right, rear], in meter
        self.car_id = [0]

    def init(self):
        track = self.main.track
        assert(isinstance(track,RCPTrack))
        # [X size, Y size] in meter
        self.size = (x,y) = (track.scale*track.gridsize[1],track.scale*track.gridsize[0])
        self.max_range = np.linalg.norm(self.size)*2
        self.edges = [Edge((0,0),(0,y)), Edge((0,0),(x,0)), Edge((0,y),(x,y)), Edge((x,0),(x,y))]
        inf = float('inf')
        for i in self.car_id:
            self.main.cars[i].tof_measurement = (inf,inf,inf,inf)

    def update(self):
        for i in self.car_id:
            car = self.main.cars[i]
            self.t.s()
            front = self.getTofReading(car.states,car.states[2])
            left = self.getTofReading(car.states,car.states[2]+np.pi/2)
            right = self.getTofReading(car.states,car.states[2]-np.pi/2)
            rear = self.getTofReading(car.states,car.states[2]+np.pi)
            car.tof_measurement = (front, left, right, rear)
            self.t.e()
            #self.plotTof(car)

    def plotTof(self,car):
        if (self.main.visualization.update_visualization.is_set()):
            img = self.main.visualization.visualization_img
            # front
            if (not car.tof_measurement[0] is None):
                x = car.states[0] + car.tof_measurement[0] * np.cos(car.states[2])
                y = car.states[1] + car.tof_measurement[0] * np.sin(car.states[2])
                self.main.track.drawCircle(img, (x,y), 0.05)
            # left
            if (not car.tof_measurement[1] is None):
                x = car.states[0] + car.tof_measurement[1] * np.cos(car.states[2] + np.pi/2)
                y = car.states[1] + car.tof_measurement[1] * np.sin(car.states[2] + np.pi/2)
                self.main.track.drawCircle(img, (x,y), 0.05)
            # right
            if (not car.tof_measurement[2] is None):
                x = car.states[0] + car.tof_measurement[2] * np.cos(car.states[2] - np.pi/2)
                y = car.states[1] + car.tof_measurement[2] * np.sin(car.states[2] - np.pi/2)
                self.main.track.drawCircle(img, (x,y), 0.05)
            # rear
            if (not car.tof_measurement[3] is None):
                x = car.states[0] + car.tof_measurement[3] * np.cos(car.states[2] + np.pi)
                y = car.states[1] + car.tof_measurement[3] * np.sin(car.states[2] + np.pi)
                self.main.track.drawCircle(img, (x,y), 0.05)
            self.main.visualization.visualization_img = img

    def getTofReading(self,coord,direction):
        x0 = coord[0]; y0 = coord[1]
        x1 = x0 + self.max_range*np.cos(direction)
        y1 = y0 + self.max_range*np.sin(direction)
        e0 = Edge((x0,y0),(x1,y1))
        tof_range = float('inf')
        dist = lambda a,b: ( (a[0]-b[0])**2 + (a[1]-b[1])**2 )**0.5
        for edge in self.edges:
            p = Edge.getIntersection(e0,edge)
            if (p is None):
                continue
            edge_dist = dist(coord,p)
            if (edge_dist < tof_range):
                tof_range = edge_dist
        return tof_range
    def final(self):
        self.t.summary()





class Edge:
    # A,B: Edge endpoint in format (x,y)
    def __init__(self,A,B):
        self.A = A = np.array(A)
        self.B = B = np.array(B)
        x1 = A[0];y1 = A[1];x2 = B[0];y2 = B[1]
        '''
        # line: Ax + By = C
        # x1 A + y1 B = x2 A + y2B = C, A+B+C=1
        # P [A,B,C].T = [0,1,0].T = v
        P = np.array([[x1-x2,y1-y2,0],[1,1,1],[x1,y1,-1]])
        v = np.array([0,1,0])
        breakpoint()
        self.AB_coeff = np.linalg.inv(P) @ v
        '''
        inf = float('inf')
        # method 1: x + By = C
        if(not y1==y2 ):
            A1 = 1
            B1 = -(x1-x2)/(y1-y2)
            C1 = x1 + B1*y1
        else:
            A1 = inf
            B1 = inf
            C1 = inf

        # method 2: Ax + y = C
        if(not x1==x2 ):
            A2 = -(y1-y2)/(x1-x2)
            B2 = 1
            C2 = A2*x1 + y1
        else:
            A2 = inf
            B2 = inf
            C2 = inf

        # use the smaller one for numerical stability
        if (A2 == inf and B1 == inf):
            raise Exception(f'{self.__class__.__name__} cant make edge between identical points')
        if (np.abs(A2) < np.abs(B1)):
            self.coeff = np.array([A2,B2,C2])
        else:
            self.coeff = np.array([A1,B1,C1])


    # C,D: endpoints for input line segment
    # return intersection position if AB intersects CD
    # elsewise return None
    # FIXME
    @staticmethod
    def getIntersection(edge1,edge2):
        A = edge1.A
        B = edge1.B
        C = edge2.A
        D = edge2.B
        # check for intersection
        AD = D-A
        AB = B-A
        BC = C-B
        AC = C-A
        CD = D-C
        c1 = cross(AD,AB) * cross(AB,AC) > 0
        c2 = cross(-AC,CD) * cross(CD,-BC) > 0
        if (c1 and c2):
            P = np.array([[edge1.coeff[0], edge1.coeff[1]],[edge2.coeff[0], edge2.coeff[1]]])
            v = np.array([edge1.coeff[2], edge2.coeff[2]])
            intersection = inv(P) @ v
            return intersection
        else:
            return None

if __name__ == '__main__':
    e1 = Edge([0,3],[0,2])
    e2 = Edge([1,2],[-1,2.3])
    retval = Edge.getIntersection(e1,e2)
    print(retval)
    plt.plot( [e1.A[0],e1.B[0]], [e1.A[1],e1.B[1]] )
    plt.plot( [e2.A[0],e2.B[0]], [e2.A[1],e2.B[1]] )
    if (not retval is None):
        plt.plot( retval[0], retval[1], '*' )
    plt.show()
