''' Sanity check for the physical parameters for cars'''
import os
from math import radians

import numpy as np

from buzzracer.scripts.run import Main
from buzzracer.common import BASEDIR
from buzzracer.sysid.tire import tire_curve


def get_dummy_main():
    ''' Build a dummy main with .cars and .track'''
    test_config_folder = os.path.join(
        BASEDIR, 'buzzracer', 'tests', 'test_configs')
    config_filename = os.path.join(
        test_config_folder, 'test_minimum.xml')
    if not os.path.exists(config_filename):
        raise FileNotFoundError
    main = Main(config_filename)
    return main


def get_reactions(slip_f, slip_r, car):
    Ffy = tire_curve(slip_f) * car.params.m * 9.8 * car.params.lr / (car.params.lr + car.params.lf)
    Fry = 1.15 * tire_curve(slip_r) * car.params.m * 9.8 * \
        car.params.lf / (car.params.lr + car.params.lf)
    angular_acc = 1.0 / car.params.Iz * (Ffy * car.params.lf - Fry * car.params.lr)
    lateral_acc = (Ffy + Fry) / car.params.m
    return lateral_acc, angular_acc


def test_tire_curve():
    main = get_dummy_main()
    car = main.cars[0]

    # No slip angle, no lateral acceleration
    a_y, a_psi = get_reactions(0, 0, car)
    assert np.isclose(a_y, 0)
    assert np.isclose(a_psi, 0)

    # ~10 deg slip angle, acc is around 1g
    a_y, a_psi = get_reactions(radians(10), radians(10), car)
    assert a_y > 5.0
    assert a_y < 20.0
    a_y, a_psi = get_reactions(radians(-10), radians(-10), car)
    assert a_y < -5.0
    assert a_y > -20.0

    # Saturates at 40 deg
    a_y, a_psi = get_reactions(radians(40), radians(40), car)
    assert a_y > 5.0
    assert a_y < 30.0
    a_y, a_psi = get_reactions(radians(-40), radians(-40), car)
    assert a_y < -5.0
    assert a_y > -30.0
