# lapcounter to set quit flag after a certain number of laps
from Extension import *
from threading import Thread,Event
from common import *
import cv2

class LapCounter(Extension):
    def __init__(self, main):
        Extension.__init__(self,main)
        self.plotLapCountFlag = True

    def init(self):
        self.total_laps = 20
        print_ok("[LapCounter]: total %d laps"%(self.total_laps))
        self.main.car_total_laps = []
        for car in self.main.cars:
            car.laps_remaining = self.total_laps
            car.lap_count = 0
            car.critical_lap = Event()
            self.main.car_total_laps.append(self.total_laps)

    def update(self):
        for car in self.main.cars:
            if (self.plotLapCountFlag):
                self.plotLapCount(car)
            if car.laptimer.new_lap.is_set():
                if (not car.critical_lap.is_set()):
                    car.critical_lap.set()
                    print_ok("[LapCounter]: car%d critical lap start, total = %d laps"%(car.id, car.laps_remaining))
                    continue

                car.laps_remaining -= 1
                car.lap_count += 1
                print_ok("[LapCounter]: car%d, %d laps remaining"%(car.id, car.laps_remaining))



                if (car.laps_remaining == 0):
                    print_ok("[LapCounter]: car%d critical lap end"%(car.id))
                    car.critical_lap.clear()
                    # should we wait for next time step to st exit flag?
                    self.main.exit_request.set()

    def plotLapCount(self,car):
        if (not self.main.visualization.update_visualization.is_set()):
            return
        img = car.main.visualization.visualization_img
        text = "Lap: %d"%(car.lap_count)

        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # org
        org = (20, 80)
        # fontScale
        fontScale = 1
        # Blue color in BGR
        color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 2
        # Using cv2.putText() method
        img = cv2.putText(img, text, org, font,
                           fontScale, color, thickness, cv2.LINE_AA)
        car.main.visualization.visualization_img = img
