from ctrlMppiWrapper import ctrlMppiWrapper
from RCPTrack import RCPtrack
from Track import Track
import matplotlib.pyplot as plt
from math import radians
from ctrlMppiWrapper import ctrlMppiWrapper


if __name__=="__main__":
    track = RCPtrack()
    track.startPos = (0.6*3.5,0.6*1.75)
    track.startDir = radians(90)
    track.load()

    porsche_setting = {'wheelbase':90e-3,
                     'max_steer_angle_left':radians(27.1),
                     'max_steer_pwm_left':1150,
                     'max_steer_angle_right':radians(27.1),
                     'max_steer_pwm_right':1850,
                     'serial_port' : '/dev/ttyUSB0',
                     'max_throttle' : 0.55}
    car_setting = porsche_setting
    car_setting['serial_port'] = None
    car = ctrlMppiWrapper(car_setting,0.01)
    car.discretized_raceline_len = 1024
    car.track = track
    car.prepareDiscretizedRaceline()
    car.createBoundary(show=True)




