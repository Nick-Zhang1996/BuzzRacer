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
            # if car is outside track, halt
            x = car.states[0]
            y = car.states[1]
            vf = car.states[3]
            if (self.track.isOutside((x,y)) or vf < 0.05):
                car.in_track = False
                self.triggered = True
                self.main.exit_request.set()
                print_warning(self.prefix()+"car outside track, terminating experiment")
            # if laptime is unreasonable, halt
            if (car.laptimer.new_lap.is_set()):
                if(car.laptimer.last_laptime < 2.0 or self.main.sim_t - car.laptimer.last_lap_ts > 20):
                    self.triggered = True
                    self.main.exit_request.set()
                    print_warning(self.prefix()+"unreasonable laptime: %.2f, terminating experiment"%(car.laptimer.last_laptime))



