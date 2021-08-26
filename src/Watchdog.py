from common import *
from Extension import Extension

class Watchdog(Extension):
    def __init__(self,main):
        Extension.__init__(self,main)
        self.track = main.track
        self.triggered = False
        for car in self.main.cars:
            car.in_track = True

    def postUpdate(self):
        for car in self.main.cars:
            x = car.states[0]
            y = car.states[1]
            vf = car.states[3]
            if (self.track.isOutside((x,y)) or vf < 0.05):
                car.in_track = False
                self.triggered = True
                self.main.exit_request.set()
                print_warning(self.prefix()+"car outside track, terminating experiment")

