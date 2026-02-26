''' Defines the interface for working with physical and simulated cars'''
from __future__ import annotations
from math import radians, degrees, asin
from enum import Enum
from typing import NamedTuple

from buzzracer.common import PrintObject, LogObject, ExperimentType, get_logger
from buzzracer.controllers.car_controller import CarController
from buzzracer.types import CartesianState, Control


_logger = get_logger('Car')


class CarParams(NamedTuple):
    # Physical properties
    # default values are for the MR03 chassis with Porsche 911 GT3 RS body
    wheelbase: float = 0.09
    ''' Wheelbase, front to rear axle'''
    lf: float = 0.04824
    ''' CG to front axle'''
    lr: float = 0.09 - 0.04824
    ''' CG to rear axle'''
    width: float = 0.0461
    ''' Track width '''

    # Iz = 417757e-9
    m: float = 0.1667
    ''' Mass in kg'''
    Iz: float = 1/12 * 0.1667 * (0.15**2 + 0.1 ** 2)
    ''' Rotational inertia in kg*m*m'''

    # Tire model
    # Ffy = Df * sin(C * arctan(B * slip_f)) * 9.8 * lr / (lr + lf) * m
    Df: float = 3.93731
    Dr: float = 6.23597
    C: float = 2.80646
    B: float = 0.51943

    # Motor/longitudinal model
    # d_vx_dt = ((Cm1 - Cm2 * vx) * throttle - Cr - Cd * vx * vx)
    Cm1: float = 6.03154
    Cm2: float = 0.96769
    Cr: float = -0.20375
    Cd: float = 0.00000
    max_throttle: float = 1.0
    min_throttle: float = -1.0

    max_steering_left: float = radians(30.0)
    max_steer_pwm_left: int = 1000
    max_steering_right: float = radians(30.0)
    max_steer_pwm_right: int = 2000

    serial_port: str = '/dev/ttyUSB0'
    car_ip: str = '0.0.0.0'
    optitrack_id: int = -1
    rendering: str = ''
    ''' path to rendering image e.g. "data/porsche_orange.png" '''


class CarConfig(Enum):


    orca = CarParams(wheelbase=0.029+0.033,
                     width=0.03,
                     rendering='data/porsche_green.png')

    # TODO render audi
    audi_11 = CarParams(optitrack_id=998,
                        car_ip='192.168.10.11',
                        max_steering_left=radians(26.1),
                        max_steering_right=radians(26.1),
                        rendering='data/porsche_green.png')

    audi_12 = CarParams(optitrack_id=1005,
                        car_ip='192.168.10.12',
                        max_steering_left=radians(26.1),
                        max_steering_right=radians(26.1),
                        max_throttle=1.0,
                        min_throttle=-1.0,
                        rendering='data/porsche_orange.png')

    porsche_16 = CarParams(wheelbase=90e-3,
                        car_ip='192.168.10.16',
                        max_steering_left=radians(27.1),
                        max_steering_right=radians(27.1),
                        optitrack_id=1006,
                        rendering='data/porsche_orange.png')

    lambo_13 = CarParams(wheelbase=98e-3,
                      car_ip='192.168.10.13',
                      max_steering_left=asin(2*98e-3/0.52),
                      max_steering_right=asin(2*98e-3/0.47),
                      optitrack_id=15,
                      rendering='data/porsche_green.png')

class Car(PrintObject, LogObject):
    ''' Base class for various types of cars,
    Subclasses should implement communication details to interact with different
    types of physical car.'''
    car_count = 0
    ''' Total number of cars'''
    cars = []
    ''' List of cars '''

    def __init__(self, main):
        LogObject.__init__(self)
        self.main = main
        self.controller: CarController = None
        self._throttle = 0.0
        self._steering = 0.0
        self.state = CartesianState(0, 0, 0, 0, 0, 0)

        # default values, will be overridden in config
        self.max_throttle = 1.0
        self.min_throttle = -1.0
        self.max_steering_left = radians(26.1)
        self.max_steering_right = radians(26.1)
        self.debug_dict = {}

    @property
    def throttle(self):
        return self._throttle

    @throttle.setter
    def throttle(self, val):
        val = val if val < self.max_throttle else self.max_throttle
        val = val if val > self.min_throttle else self.min_throttle
        self._throttle = val

    @property
    def steering(self):
        return self._steering

    @steering.setter
    def steering(self, val):
        val = val if val < self.max_steering_left else self.max_steering_left
        val = val if val > -self.max_steering_right else -self.max_steering_right
        self._steering = val

    def pre_init(self):
        self.controller.pre_init()

    def post_init(self):
        self.controller.post_init()

    def init(self):
        ''' Initialization for cars.
        This will be run after initialization for all other extensions have concluded
        '''
        if self.main.experiment_type is ExperimentType.Realworld:
            self.init_hardware()
        self.controller.init()

    def init_hardware(self):
        ''' Initialize code that require hardware initializations here '''

    def actuate(self):
        ''' Send control commands to the car'''

    def control(self):
        if self.controller is None:
            self.throttle = 0.0
            self.steering = 0.0
        else:
            self.controller.control()
            _logger.debug('T=%4.1f, S=%4.1f deg' % (self.throttle, degrees(self.steering)))

        if self.main.slowdown.is_set():
            self.throttle = 0.0
        if self.main.experiment_type == ExperimentType.Realworld:
            self.actuate()

    @classmethod
    def reset(cls):
        cls.cars = []
        cls.car_count = 0

    @classmethod
    def Factory(cls, main, config):
        try:
            hardware_class_text = config.getElementsByTagName(
                'hardware')[0].firstChild.nodeValue
            # pylint: disable-next=exec-used
            exec('from buzzracer.cars import '+hardware_class_text)
        except IndexError:
            _logger.warning('no hardware specified')

        config_controller = config.getElementsByTagName('controller')[0]
        controller_class_text = config_controller.getElementsByTagName('type')[
            0].firstChild.nodeValue

        try:
            init_states_text = config.getElementsByTagName('init_states')[0].firstChild.nodeValue
            init_states = eval(init_states_text)
        except IndexError:
            _logger.warning('Car: no initial state specified, using track default')
            init_states = (*main.track.start_pos, main.track.start_dir, 0.1)

        config_name = config.getElementsByTagName('config_name')[0].firstChild.nodeValue
        # pylint: disable-next=exec-used
        exec(f'from buzzracer.controllers import {controller_class_text}')
        controller = eval(controller_class_text)

        car = eval(hardware_class_text)(main)

        # (x,y,theta,vforward,vsideway=0,omega)
        x, y, heading, v_forward = init_states
        car.state = CartesianState(x=x, y=y, heading=heading, v_forward=v_forward, v_sideway=0, omega=0)

        car.params = eval(f'CarConfig.{config_name}.value')

        if not controller is None:
            car.controller = controller(car, config_controller)

        car.init_param()

        car.id = Car.car_count
        Car.cars.append(car)
        Car.car_count += 1

        return car
