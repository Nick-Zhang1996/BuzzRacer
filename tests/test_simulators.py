''' Test simulators '''
import os

import pytest

from buzzracer.scripts.run import Main
from buzzracer.common import BASEDIR
from buzzracer.extensions.visualization import Visualization
from buzzracer.extensions.extension import Extension

VISUALIZE = False
''' If True, plot visualizations. Some tests need human visual checking'''


@pytest.fixture(autouse=True)
def reset_car_class_variable():
    """
    This fixture runs automatically before each test function,
    ensuring the class variable is reset.
    """
    Extension.extensions = []
    # 'yield' allows teardown code to run after the test, though none is needed here.
    yield


def get_main_from_test_config(config_name: str):
    ''' Build a dummy main with .cars and .track'''
    test_config_folder = os.path.join(
        BASEDIR, 'tests', 'test_configs')
    config_filename = os.path.join(
        test_config_folder, config_name)
    if not os.path.exists(config_filename):
        raise FileNotFoundError(config_filename)
    main = Main(config_filename)
    return main


@pytest.mark.parametrize(
    'config_name',
    [
        # pytest.param(
        #     'test_kinematic_bicycle_cartesian_simulator.xml',
        #     id='test_kinematic_bicycle_cartesian_simulator'
        # ),
        # pytest.param(
        #     'test_dynamic_bicycle_cartesian_simulator.xml',
        #     id='test_dynamic_bicycle_cartesian_simulator'
        # ),
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
    main.simulator.match_time = True
    main.visualization = Visualization()
    main.visualization.car_graphics = True
    main.visualization.init()
    # pylint: disable-next=no-member
    main.step_counter.total_count = 100
    main.run()
