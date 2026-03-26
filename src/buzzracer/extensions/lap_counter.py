''' Extension Lapcounter sets quit flag after a certain number of laps'''
from __future__ import annotations
from typing import TYPE_CHECKING
from threading import Event
import cv2

from buzzracer.extensions.extension import Extension, ExtensionConfig, ExtensionState
if TYPE_CHECKING:
    from buzzracer.cars.car import Car


class LapCounterConfig(ExtensionConfig):
    def __init__(self, main_config):
        super().__init__(main_config)
        self.total_laps = 0
        ''' Laps to run before termination'''
        self.plot_lap_count = False


@Extension.register('lap_counter', LapCounterConfig, ExtensionState)
class LapCounter(Extension):
    ''' Extension to terminate experiment after a certain number of laps.
     If multiple car exist, then experiment stops after any car finishes'''

    def __init__(self, config, state):
        Extension.__init__(self, config, state)
        self.lap_count: dict[Car, int] = {car: 0 for car in self.main.cars}
        ''' Laps completed for each car'''
        self.is_hot_lap: dict[Car, Event] = {
            car: Event()
            for car in self.main.cars
        }
        ''' Car has started a hot-lap where laptime is expected to be representative'''

        self.print_ok('total %d laps' % (self.config.total_laps))

    def update(self):
        for car in self.main.cars:
            if self.config.plot_lap_count:
                self.plot_lap_count(car)
            if self.main.laptimer.laptimer_by_car[car].new_lap.is_set():
                if not self.is_hot_lap[car].is_set():
                    # car crossed finishing line for first time
                    # warm-up completed, hot-lap started
                    # laptimes from now on are representative
                    self.is_hot_lap[car].set()
                    self.print_ok('car%d critical lap start, total = %d laps',
                                  car.id, self.config.total_laps)
                    continue

                self.lap_count[car] += 1
                laps_remaining = self.config.total_laps - self.lap_count[car]
                self.print_ok('car%d, %d laps remaining', car.id, laps_remaining)
                if self.lap_count[car] >= self.config.total_laps:
                    self.print_ok(f'car{car.id} critical lap end')
                    self.is_hot_lap[car].clear()
                    self.main.exit_request.set()

    def plot_lap_count(self, car):
        ''' Plot lap count onto visualization'''
        if not self.main.visualization.update_visualization.is_set():
            return
        img = car.main.visualization.visualization_img
        text = f'Lap: {self.lap_count[car]}'

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
