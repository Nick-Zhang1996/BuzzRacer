# lapcounter to set quit flag after a certain number of laps
from Extension import *
from threading import Thread,Event
from common import *

class LapCounter(Extension):
    def __init__(self, main):
        Extension.__init__(self,main)

    def init(self):
        self.total_laps = 100
        print_ok("[LapCounter]: total %d laps"%(self.total_laps))
        self.main.car_total_laps = []
        for car in self.main.cars:
            car.laps_remaining = self.total_laps
            car.critical_lap = Event()
            self.main.car_total_laps.append(self.total_laps)

    def update(self):
        for car in self.main.cars:
            if car.laptimer.new_lap.is_set():
                if (not car.critical_lap.is_set()):
                    car.critical_lap.set()
                    print_ok("[LapCounter]: car%d critical lap start, total = %d laps"%(car.id, car.laps_remaining))
                    continue

                car.laps_remaining -= 1
                print_ok("[LapCounter]: car%d, %d laps remaining"%(car.id, car.laps_remaining))

                if (car.laps_remaining == 0):
                    print_ok("[LapCounter]: car%d critical lap end"%(car.id))
                    car.critical_lap.clear()
                    # should we wait for next time step to st exit flag?
                    self.main.exit_request.set()
