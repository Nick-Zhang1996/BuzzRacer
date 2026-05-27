""" Parameter settings for specific car chassis"""
# pylint: disable=invalid-name
from math import radians
from enum import Enum
from typing import NamedTuple


class CarParam(NamedTuple):
    """ Parameters for a specific chassis, including necessary tuning params"""
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
    lookahead: float = 7e-2
    """ Lookahead for stanley """

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

    drift_wheel_radius: float = 12e-3
    """ Wheel radius used by the simplified drift model, in meters. """
    drift_motor_torque_coeff: float = 11-3
    """ Effective motor torque coefficient in the drift drivetrain model. """
    drift_motor_back_emf: float = 4e-3
    """ Effective back-EMF coefficient for the quasi-steady wheel-speed solve. """
    drift_front_tire_A: float = 0.85
    """ Front combined-slip tire force scale in Newtons for the drift model. """
    drift_front_tire_B: float = 8.0
    """ Front combined-slip tire shape coefficient for the drift model. """
    drift_rear_tire_A: float = 1.00
    """ Rear combined-slip tire force scale in Newtons for the drift model. """
    drift_rear_tire_B: float = 6.0
    """ Rear combined-slip tire shape coefficient for the drift model. """

    serial_port: str = '/dev/ttyUSB0'
    car_ip: str = '0.0.0.0'
    optitrack_id: int = -1
    fhss_modem_id: int = -1
    """ Modem number in fhss binding"""
    rendering: str = ''
    ''' path to rendering image e.g. "car_imgs/porsche_orange.png" '''


class CarConfig(Enum):
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
        lookahead=70e-3,
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

    drift_rx7_30 = CarParam(
        name='drift_rx7_30',
        m=169e-3,
        wheelbase=98e-3,
        lr=46e-3,
        lf=98e-3-46e-3,
        max_steer_right=radians(30.0),
        max_steer_left=radians(30.0),
        max_throttle=1.0,
        min_throttle=-1.0,
        optitrack_id=30,
        fhss_modem_id=6,
        car_ip='192.168.10.30',
        rendering='car_imgs/mclaren_22.png'
    )
