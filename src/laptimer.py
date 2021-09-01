# laptimer
from threading import Thread,Event
from math import sin,cos
import numpy as np
from time import time

class Laptimer:
    def __init__(self,finish,orientation,voice=False):
        # coordinate
        self.finish = np.array(finish)
        # in rad
        self.orientation = orientation
        self.finish_vec = np.array([cos(orientation),sin(orientation)])

        self.voice = voice
        if (voice):
            global pyttsx3
            import pyttsx3
            self.engine = pyttsx3.init()


        self.last_coord = np.array([0,0])
        self.last_lap_ts = 0
        self.last_laptime = 0

        self.new_lap = Event()
        self.lap_count = 0
        self.p1dist = lambda a,b:abs(a[0]-b[0])+abs(a[1]-b[1])
        self.p1norm = lambda a:abs(a[0])+abs(a[1])

        # freeze laptimer for a certain time after a new lap to prevent immediate recounting
        self.timeout = 0.5
        self.hotzone_radius = 0.5

    def update(self,coord,current_time=None):
        # let the finish location be O
        # last position/coord be A
        # current position be B
        if current_time is None:
            current_time = time()
        if ( current_time<self.last_lap_ts+self.timeout):
            return False
        coord = np.array(coord)
        OB = coord - self.finish
        if (self.p1norm(OB) > self.hotzone_radius):
            self.last_coord = coord
            return False
        OA = self.last_coord - self.finish
        if (np.dot(OA,self.finish_vec)*np.dot(OB,self.finish_vec) < 0):
            self.last_laptime = current_time - self.last_lap_ts
            self.last_lap_ts = current_time

            if (self.lap_count == 0 ):
                self.lap_count += 1
                return False

            self.lap_count += 1
            self.new_lap.set()
            return True

    def announce(self):
        self.thread = Thread(target = self.__announceThread)
        self.thread.start()

    def __announceThread(self):
        self.engine.say("%.2f"%self.last_laptime)
        self.engine.runAndWait()
        return


if __name__ == '__main__':
    lp = Laptimer((0,0),3,voice=True)
    lp.last_laptime = 9.882424
    lp.announce()
