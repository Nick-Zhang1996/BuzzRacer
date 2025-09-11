''' Test simulators '''
import os

import pytest
import numpy as np

from buzzracer.types import CurvilinearState, CartesianState, Control
from buzzracer.scripts.run import Main
from buzzracer.common import BASEDIR

VISUALIZE = False
''' If True, plot visualizations. Some tests need human visual checking'''


def get_main_from_test_config(config_name: str):
    ''' Build a dummy main with .cars and .track'''
    test_config_folder = os.path.join(
        BASEDIR, 'buzzracer', 'tests', 'test_configs')
    config_filename = os.path.join(
        test_config_folder, config_name)
    if not os.path.exists(config_filename):
        raise FileNotFoundError(config_filename)
    main = Main(config_filename)
    return main


@pytest.mark.parametrize(
    'config_name',
    [
        pytest.param(
            'test_kinematic_bicycle_cartesian_simulator.xml',
            id='test_kinematic_bicycle_cartesian_simulator'
        ),
        pytest.param(
            'test_dynamic_bicycle_cartesian_simulator.xml',
            id='test_dynamic_bicycle_cartesian_simulator'
        ),
        pytest.param(
            'test_kinematic_bicycle_curvilinear_simulator.xml',
            id='test_kinematic_bicycle_curvilinear_simulator'
        ),
        pytest.param(
            'test_dynamic_bicycle_curvilinear_simulator.xml',
            id='test_dynamic_bicycle_curvilinear_simulator'
        ),
    ],
)
def test_kinematic_bicycle_cartesian_simulator(config_name: str):
    main = get_main_from_test_config(config_name)
    main.run()
