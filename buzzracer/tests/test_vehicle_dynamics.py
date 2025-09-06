import os

import pytest
import numpy as np
import matplotlib.pyplot as plt

from buzzracer.types import CurvilinearState, CartesianState, Control
from buzzracer.scripts.run import Main
from buzzracer.common import BASEDIR
from buzzracer.sysid.kinematic_bicycle_model import KinematicBicycleModelCartesian, KinematicBicycleModelFrenet


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
    config_filename = os.path.join(BASEDIR, 'buzzracer','tests', 'test_configs', config_filename)

    if not os.path.exists(config_filename):
        raise FileNotFoundError

    experiment = Main(config_filename)
    experiment.run()

def test_kinematic_bicycle_frenet():

    # just to set up a dummy car
    test_config_folder = os.path.join(BASEDIR, 'buzzracer','tests', 'test_configs')
    config_filename = os.path.join(test_config_folder, 'test_kinematic_bicycle.xml')
    if not os.path.exists(config_filename):
        raise FileNotFoundError
    experiment = Main(config_filename)
    car = experiment.cars[0]

    state = CurvilinearState(progress=0,
                             lateral_err=1.0,
                             rel_heading=np.pi/6,
                             v_forward=1.0,
                             v_sideway=0.0,
                             rel_omega=0.0
                             )
    control = Control(steering=0, throttle = 1.0)
    state_vec = [state]
    for i in range(10):
        state_vec.append(
            KinematicBicycleModelFrenet.advance_dynamics(state_vec[-1], control, car, 0.01, 0)
        )
    def plot(field):
        plt.plot([getattr(val,field) for val in state_vec], label=field)
    for field in CurvilinearState._fields:
        plot(field)
    plt.legend()
    plt.show()

