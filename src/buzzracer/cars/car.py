''' Defines the interface for working with physical and simulated cars'''
from __future__ import annotations
import logging
from math import radians, degrees
from enum import Enum
from typing import NamedTuple

from buzzracer.common import PrintObject, LogObject, ExperimentType
from buzzracer.controllers.car_controller import CarController
from buzzracer.types import CartesianState, Control


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class CarParam(NamedTuple):
    name: str = ''
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

    ss_throttle_p0: float = 0.06246385
    """ steady state throttle = v * p0 + p1 default for MR03 Offboard"""
    ss_throttle_p1: float = 0.19171776
    """ steady state throttle = v * p0 + p1 """

    max_throttle: float = 0.8
    min_throttle: float = -1.0
    max_steer_left: float = radians(27)
    """ Max steering angle in radians, left, positive"""
    max_steer_right: float = radians(27)
    """ Max steering angle in radians, right, positive"""

    # For A7105 transmitter controlled cars
    max_steer_pwm_left: int = 1100
    max_steer_pwm_right: int = 2000

    # For Arduino 33 IoT Controlled Cars
    steer_ratio: float = 1.0
    ''' command = desired_angle * ratio + offset (unit:rad)'''
    steer_offset: float = 0.0
    ''' command = desired_angle * ratio + offset (unit:rad)'''

    serial_port: str = '/dev/ttyUSB0'
    car_ip: str = '0.0.0.0'
    optitrack_id: int = -1
    fhss_modem_id: int = -1
    """ Modem number in fhss binding"""
    rendering: str = ''
    ''' path to rendering image e.g. "car_imgs/porsche_orange.png" '''


class CarConfig(Enum):
    # TODO render audi
    audi_11 = CarParam(
        name='audi_11',
        m=172e-3,
        wheelbase=97e-3,
        lr=50e-3,
        lf=97e-3-50e-3,
        steer_ratio=1.1363636363636365,
        steer_offset=0.03014659617081118,
        optitrack_id=11,
        car_ip='192.168.10.11',
        rendering='car_imgs/audi_12.png')

    # NOTE no calibration, using audi_11 value
    audi_12 = CarParam(
        name='audi_12',
        m=172e-3,
        wheelbase=97e-3,
        lr=50e-3,
        lf=97e-3-50e-3,
        max_steer_left=radians(27),
        max_steer_right=radians(27),
        steer_ratio=1.1363636363636365,
        steer_offset=0.03014659617081118,
        optitrack_id=12,
        car_ip='192.168.10.12',
        max_throttle=1.0,
        min_throttle=-1.0,
        rendering='car_imgs/audi_12.png')

    porsche_16 = CarParam(
        name='porsche_16',
        m=172e-3,
        wheelbase=90e-3,
        lr=41e-3,
        lf=90e-3-41e-3,
        steer_ratio=1.2113055181695829,
        steer_offset=-0.006130968166494196,
        max_steer_left=radians(25),  # FIXME forgot to calibrate this
        max_steer_right=radians(25),
        car_ip='192.168.10.16',
        optitrack_id=16,
        rendering='car_imgs/porsche_18.png')

    lambo_13 = CarParam(
        name='lambo_13',
        m=192e-3,
        wheelbase=98e-3,
        lr=48e-3,
        lf=98e-3-48e-3,
        steer_ratio=1.0638297872340425,
        steer_offset=0.028408018676077906,
        max_steer_left=radians(23.85),
        max_steer_right=radians(26.91),
        car_ip='192.168.10.13',
        optitrack_id=13,
        rendering='car_imgs/lambo_13.png')

    corvette_17 = CarParam(
        name='corvette_17',
        m=174e-3,
        wheelbase=98e-3,
        lr=47e-3,
        lf=98e-3-47e-3,
        max_steer_right=radians(29.77),
        max_steer_left=radians(23.21),
        optitrack_id=17,
        fhss_modem_id=0,
        rendering='car_imgs/corvette_17.png'
    )
    porsche_18 = CarParam(
        name='porsche_18',
        m=165e-3,
        wheelbase=90e-3,
        lr=40e-3,
        lf=90e-3-40e-3,
        max_steer_right=radians(28.13),
        max_steer_left=radians(23.17),
        optitrack_id=18,
        fhss_modem_id=1,
        rendering='car_imgs/porsche_18.png'
    )
    porsche_19 = CarParam(
        name='porsche_19',
        m=165e-3,
        wheelbase=90e-3,
        lr=40e-3,
        lf=90e-3-40e-3,
        max_steer_right=radians(30.24),
        max_steer_left=radians(22.33),
        optitrack_id=19,
        fhss_modem_id=2,
        rendering='car_imgs/porsche_19.png'
    )

    audi_20 = CarParam(
        name='audi_20',
        m=166e-3,
        wheelbase=98e-3,
        lr=41e-3,
        lf=98e-3-41e-3,
        max_steer_right=radians(29.08),
        max_steer_left=radians(23.95),
        optitrack_id=20,
        fhss_modem_id=3,
        rendering='car_imgs/audi_12.png'
    )

    mclaren_21 = CarParam(
        name='mclaren_21',
        m=168e-3,
        wheelbase=98e-3,
        lr=44e-3,
        lf=98e-3-44e-3,
        max_steer_right=radians(27.69),
        max_steer_left=radians(21.14),
        optitrack_id=21,
        fhss_modem_id=4,
        rendering='car_imgs/mclaren_21.png'
    )

    mclaren_22 = CarParam(
        name='mclaren_22',
        m=169e-3,
        wheelbase=98e-3,
        lr=46e-3,
        lf=98e-3-46e-3,
        max_steer_right=radians(28.10),
        max_steer_left=radians(22.69),
        optitrack_id=22,
        fhss_modem_id=5,
        rendering='car_imgs/mclaren_22.png'
    )


class Car(PrintObject, LogObject):
    ''' Base class for various types of cars,
    Subclasses should implement communication details to interact with different
    types of physical car.'''
    car_count = 0
    ''' Total number of cars'''
    cars = []
    """ All cars, this include cars of different subclass.
    If a subclass needs a list of cars of that specific subclass, it must overwrite this attribute"""
    main = None
    """ Access to main"""
    registry = {}
    """ Registry of all Car subclasses"""

    def __init__(self):
        LogObject.__init__(self)
        self.controller: CarController = None
        self._throttle = 0.0
        self._steering = 0.0
        self.state = CartesianState(0, 0, 0, 0, 0, 0)
        self.param: CarParam

        # default values, will be overridden in config
        self.max_throttle = 1.0
        self.min_throttle = -1.0
        self.debug_dict = {}

    @staticmethod
    def register(cls):
        """Decorator to add a car class to the registry."""
        name = cls.__name__
        Car.registry[name] = cls
        _logger.debug('Registered %s' % {name})
        return cls

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
        val = val if val < self.param.max_steer_left else self.param.max_steer_left
        val = val if val > -self.param.max_steer_right else -self.param.max_steer_right
        self._steering = val

    def pre_init(self):
        self.controller.pre_init()

    def post_init(self):
        self.controller.post_init()

    def init(self):
        ''' Initialization for cars.
        This will be run after initialization for all other extensions have concluded
        '''
        if Car.main.config.experiment_type is ExperimentType.Realworld:
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
            _logger.debug('T=%4.1f, S=%4.1f deg' %
                          (self.throttle, degrees(self.steering)))

        if Car.main.state.slowdown.is_set():
            self.throttle = 0.0
        if Car.main.config.experiment_type == ExperimentType.Realworld:
            self.actuate()

    @classmethod
    def reset(cls, main):
        Car.main = main
        Car.cars = []
        Car.car_count = 0

    @classmethod
    def Factory(cls, config_minidom):
        try:
            car_cls_text = config_minidom.getElementsByTagName(
                'hardware')[0].firstChild.nodeValue
            car_cls = Car.registry[car_cls_text]
        except IndexError:
            _logger.warning('No hardware specified')

        config_controller = config_minidom.getElementsByTagName('controller')[0]
        controller_class_text = config_controller.getElementsByTagName('type')[
            0].firstChild.nodeValue

        try:
            init_states_text = config_minidom.getElementsByTagName(
                'init_states')[0].firstChild.nodeValue
            init_states = eval(init_states_text)
        except IndexError:
            _logger.warning(
                'Car: no initial state specified, using track default')
            init_states = (*Car.main.track.start_pos, Car.main.track.start_dir, 0.1)

        config_name = config_minidom.getElementsByTagName(
            'config_name')[0].firstChild.nodeValue

        controller, controller_config, controller_state = CarController.factory(
            controller_class_text, Car.main.config, config_controller)

        car = car_cls()
        # (x,y,theta,vforward,vsideway=0,omega)
        x, y, heading, v_forward = init_states
        car.state = CartesianState(
            x=x, y=y, heading=heading, v_forward=v_forward, v_sideway=0, omega=0)

        car.param = eval(f'CarConfig.{config_name}.value')

        if not controller is None:
            car.controller = controller

        car.id = Car.car_count
        Car.cars.append(car)
        Car.car_count += 1

        return car
