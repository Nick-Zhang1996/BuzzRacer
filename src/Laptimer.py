# laptimer
from Extension import *
from threading import Thread,Event
from math import sin,cos
import numpy as np
from time import time
from common import *

class Laptimer(Extension):
    def __init__(self, main):
        Extension.__init__(self,main)

    def init(self):
        for car in self.main.cars:
            car.enableLaptimer = True
            if car.enableLaptimer:
                car.laptimer = _Laptimer(self.main.track.startPos, self.main.track.startDir)
                car.laptime_vec = []

    def update(self):
        for car in self.main.cars:
            if (car.enableLaptimer):
                retval = car.laptimer.update((car.states[0],car.states[1]),current_time=self.main.sim_t)
                if retval:
                    #car.laptimer.announce()
                    print_info("[Laptimer]: car%d, new laptime: %.4f s"%(car.id, car.laptimer.last_laptime))
                    car.laptime_vec.append(car.laptimer.last_laptime)
                    self.showStats()

    def final(self):
        self.showStats()

    def showStats(self):
        car_laptime_mean = []
        for car in self.main.cars:
            if (car.enableLaptimer and len(car.laptime_vec) > 0):
                mean = np.mean(car.laptime_vec)
                stddev = np.std(car.laptime_vec)
                laps = len(car.laptime_vec)
                print_info("[Laptimer]: car%d, %d laps, mean %.4f, stddev %.4f (sec)"%(car.id,laps,mean,stddev))
                car_laptime_mean.append(mean)
        self.main.car_laptime_mean = car_laptime_mean


class _Laptimer:
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

        # indicator that this lap is new
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
            self.new_lap.clear()
            return False
        coord = np.array(coord)
        OB = coord - self.finish
        if (self.p1norm(OB) > self.hotzone_radius):
            self.last_coord = coord
            self.new_lap.clear()
            return False
        OA = self.last_coord - self.finish
        if (np.dot(OA,self.finish_vec)*np.dot(OB,self.finish_vec) < 0):
            self.last_laptime = current_time - self.last_lap_ts
            self.last_lap_ts = current_time

            if (self.lap_count == 0 ):
                self.lap_count += 1
                self.new_lap.clear()
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
