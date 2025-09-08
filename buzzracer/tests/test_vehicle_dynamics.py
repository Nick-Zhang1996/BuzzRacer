import os
from math import radians

import pytest
import numpy as np
import matplotlib.pyplot as plt

from buzzracer.types import CurvilinearState, CartesianState, Control
from buzzracer.scripts.run import Main
from buzzracer.common import BASEDIR
from buzzracer.sysid.kinematic_bicycle_model import KinematicBicycleModelCartesian, KinematicBicycleModelFrenet


def get_dummy_main():
    ''' Build a dummy main with .cars and .track'''
    test_config_folder = os.path.join(
        BASEDIR, 'buzzracer', 'tests', 'test_configs')
    config_filename = os.path.join(
        test_config_folder, 'test_stanley.xml')
    if not os.path.exists(config_filename):
        raise FileNotFoundError
    main = Main(config_filename)
    return main


@pytest.mark.parametrize(
    'config_filename',
    [
        pytest.param(
            'test_kinematic_bicycle.xml', id='test_kinematic_bicycle'
        ),
        pytest.param(
            'test_dynamic_bicycle.xml', id='test_dynamic_bicycle'
        ),
    ],
)
def test_run_with_config(config_filename):
    config_filename = os.path.join(
        BASEDIR, 'buzzracer', 'tests', 'test_configs', config_filename)

    if not os.path.exists(config_filename):
        raise FileNotFoundError

    experiment = Main(config_filename)
    experiment.run()


def test_curv_to_from_cart():
    ''' Transform CurvilinearState to/from CartesianState'''
    np.random.seed(0)
    main = get_dummy_main()
    track = main.track

    for _ in range(10):
        # make random curv state
        s = np.random.uniform(0, track.raceline_len_m)
        n = np.random.uniform(-0.3, 0.3)
        rel_heading = np.random.uniform(radians(-30), radians(30))
        curv = CurvilinearState(progress=s,
                                lateral_err=n,
                                rel_heading=rel_heading,
                                v_forward=1.0,
                                v_sideway=0.0,
                                rel_omega=0.0)
        cart = track.curv_to_cart(curv)
        remake_curv = track.cart_to_curv(cart)
        print(f'curv {curv}')
        print(f'cart {cart}')
        print(f'remake_curv {remake_curv}')
        np.testing.assert_allclose(curv, remake_curv, atol=1e-5, rtol=1e-5)


def test_kinematic_bicycle_frenet():
    main = get_dummy_main()
    car = main.cars[0]

    state = CurvilinearState(progress=0,
                             lateral_err=1.0,
                             rel_heading=np.pi/6,
                             v_forward=1.0,
                             v_sideway=0.0,
                             rel_omega=0.0
                             )
    control = Control(steering=0, throttle=1.0)
    state_vec = [state]
    for i in range(10):
        state_vec.append(
            KinematicBicycleModelFrenet.advance_dynamics(
                state_vec[-1], control, car, 0.01, 0)
        )

    def plot(field):
        plt.plot([getattr(val, field) for val in state_vec], label=field)
    for field in CurvilinearState._fields:
        plot(field)
    plt.legend()
    plt.show()
