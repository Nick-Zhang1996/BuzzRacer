''' Extension Lapcounter sets quit flag after a certain number of laps'''
from threading import Event
import cv2

from extension.Extension import Extension
from car.Car import Car


class LapCounter(Extension):
    ''' Extension to terminate experiment after a certain number of laps. '''

    def __init__(self):
        Extension.__init__(self, 'lap_counter')
        self.plotLapCountFlag = True
        self.total_laps = None
        ''' Laps to run before termination, set in config files'''
        self.lap_count: dict[Car, int] = {car: 0 for car in self.main.cars}
        ''' Laps completed for each car'''
        self.is_hot_lap: dict[Car, Event] = {
            car: Event()
            for car in self.main.cars
        }
        ''' Car has started a hot-lap where laptime is expected to be representative'''

        self.print_ok('[LapCounter]: total %d laps' % (self.total_laps))

    def update(self):
        for car in self.main.cars:
            if self.plotLapCountFlag:
                self.plot_lap_count(car)
            if car.laptimer.new_lap.is_set():
                if not self.is_hot_lap[car].is_set():
                    # car crossed finishing line for first time
                    # warm-up completed, hot-lap started
                    # laptimes from now on are representative
                    self.is_hot_lap[car].set()
                    self.print_ok('car%d critical lap start, total = %d laps' %
                                  (car.id, self.total_laps))
                    continue

                car.laps_remaining -= 1
                car.lap_count += 1
                self.print_ok('[LapCounter]: car%d, %d laps remaining' %
                              (car.id, car.laps_remaining))
                if car.laps_remaining <= 0:
                    self.print_ok('[LapCounter]: car%d critical lap end' %
                                  (car.id))
                    car.critical_lap.clear()
                    # should we wait for next time step to st exit flag?
                    self.main.exit_request.set()

    def plot_lap_count(self, car):
        if not self.main.visualization.update_visualization.is_set():
            return
        img = car.main.visualization.visualization_img
        text = 'Lap: %d' % (car.lap_count)

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
        img = cv2.putText(img, text, org, font, fontScale, color, thickness,
                          cv2.LINE_AA)
        car.main.visualization.visualization_img = img
