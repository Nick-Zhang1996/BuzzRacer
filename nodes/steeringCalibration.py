# quick script to hold a steering for calibration
from car import Car
from math import radians
from time import sleep
porsche_setting = {'wheelbase':90e-3,
                 'max_steer_angle_left':radians(27.1),
                 'max_steer_pwm_left':1150,
                 'max_steer_angle_right':radians(27.1),
                 'max_steer_pwm_right':1850,
                 'serial_port' : '/dev/ttyUSB0',
                 'max_throttle' : 0.5}

lambo_setting = {'wheelbase':98e-3,
                 'max_steer_angle_left':arcsin(2*98e-3/0.52),
                 'max_steer_pwm_left':1100,
                 'max_steer_angle_right':arcsin(2*98e-3/0.47),
                 'max_steer_pwm_right':1850,
                 'serial_port' : '/dev/ttyUSB1',
                 'max_throttle' : 0.5}

car2 = Car(lambo_setting)
# 1100-1850
steering = 1100
throttle = 1500
while True:
    car2.actuatePWM(steering,throttle)
    sleep(0.05)
