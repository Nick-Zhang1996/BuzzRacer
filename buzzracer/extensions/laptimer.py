''' Track laptimes of cars'''

import os
from time import time
from threading import Thread, Event
from math import sin, cos

import pickle
import numpy as np

from buzzracer.common import BASEDIR
from buzzracer.extensions.extension import Extension
from buzzracer.cars.car import Car


class Laptimer(Extension):
    ''' Laptimer for simulation and experiments.'''

    def __init__(self):
        Extension.__init__(self, 'laptimer')

        cars = self.main.cars
        self.laptime_mean_by_car: dict[Car, float] = {car: 0 for car in cars}
        ''' Mean laptime for each car '''
        self.laptime_stddev_by_car: dict[Car, float] = {car: 0 for car in cars}
        ''' Stddev for each car '''
        self.total_laps_by_car: dict[Car, int] = {car: 0 for car in cars}
        ''' Total laps run for each car '''
        self.laptime_vec_by_car: dict[Car, float] = {car: [] for car in cars}
        ''' Laptime history for each car '''
        self.laptimer_by_car: dict[Car, _Laptimer] = {
            car: _Laptimer(self.main.track.start_pos,
                           self.main.track.start_dir)
            for car in cars
        }
        ''' Laptimer object for each car '''

    def update(self):
        for car in self.main.cars:
            is_new_lap = self.laptimer_by_car[car].update(
                (car.states[0], car.states[1]), current_time=self.main.time)
            if is_new_lap:
                # Audio announcement with text-to-voice
                # car.laptimer.announce()
                self.print_info('car%d, Lap %d laptime: %.4f s' %
                                (car.id, len(self.laptime_vec_by_car[car]),
                                 self.laptimer_by_car[car].last_laptime))
                self.laptime_vec_by_car[car].append(
                    self.laptimer_by_car[car].last_laptime)

    def final(self):
        for car in self.main.cars:
            car.debug_dict.update(
                {'laptime_vec': self.laptime_vec_by_car[car]})
        self.show_stats()
        self.log_laptime()

    def log_laptime(self):
        try:
            logname = os.path.join(self.main.logger.log_folder,
                f'laptime{self.main.logger.log_no}.p')
        except AttributeError:
            logname = os.path.join(BASEDIR, 'log', 'laptime_latest.p')
            self.print_warning(
                f"Logger extension wasn't enabled, saving to {logname}")
        with open(logname, 'wb') as f:
            laptime_vec_by_car_id = {
                car.id: self.laptime_vec_by_car[car]
                for car in self.main.cars
            }
            pickle.dump(laptime_vec_by_car_id, f)
            self.print_ok('Saved laptime vec to ' + logname)

    def show_stats(self):
        for car in self.main.cars:
            if len(self.laptime_vec_by_car[car]) > 0:
                # Ignore first warm up lap
                mean = np.mean(self.laptime_vec_by_car[car][1:])
                stddev = np.std(self.laptime_vec_by_car[car][1:])
                laps = len(self.laptime_vec_by_car[car][1:])
                self.print_info(
                    'car%d, %d laps, mean %.4f, stddev %.4f (sec)' %
                    (car.id, laps, mean, stddev))
            else:
                mean = -1
                stddev = -1
                self.print_warning(f'car{car.id} has no laps')
            self.laptime_mean_by_car[car] = mean
            self.laptime_stddev_by_car[car] = stddev


class _Laptimer:

    def __init__(self, finish: np.array, orientation: float, voice=False):
        ''' Internal Laptimer
        Args:
            finish: np.array dim (2,)
            orientation: float Orientation in rad for track orientation at finish 
        '''
        self.finish = np.array(finish)
        ''' Coordinate dim(2,) for finishing line '''
        self.orientation = orientation
        ''' Orientation in rad for track orientation at finish '''
        self.finish_vec = np.array([cos(orientation), sin(orientation)])

        self.voice = voice
        if voice:
            global pyttsx3
            import pyttsx3
            self.engine = pyttsx3.init()

        self.last_coord = np.array([0, 0])
        self.last_lap_ts = 0
        self.last_laptime = 0

        # indicator that this lap is new
        self.new_lap = Event()
        self.lap_count = 0
        self.p1dist = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])
        self.p1norm = lambda a: abs(a[0]) + abs(a[1])

        # freeze laptimer for a certain time after a new lap to prevent immediate recounting
        self.timeout = 1.0
        self.hotzone_radius = 0.5
        self.thread = None

    def update(self, coord, current_time=None):
        # let the finish location be O
        # last position/coord be A
        # current position be B
        if current_time is None:
            current_time = time()
        if current_time < self.last_lap_ts + self.timeout:
            self.new_lap.clear()
            return False
        coord = np.array(coord)
        OB = coord - self.finish
        if self.p1norm(OB) > self.hotzone_radius:
            self.last_coord = coord
            self.new_lap.clear()
            return False
        OA = self.last_coord - self.finish
        if np.dot(OA, self.finish_vec) * np.dot(OB, self.finish_vec) < 0:
            self.last_laptime = current_time - self.last_lap_ts
            self.last_lap_ts = current_time
            self.lap_count += 1
            self.new_lap.set()
            return True

    def announce(self):
        self.thread = Thread(target=self.__announceThread)
        self.thread.start()

    def __announceThread(self):
        self.engine.say('%.2f' % self.last_laptime)
        self.engine.runAndWait()
        return
