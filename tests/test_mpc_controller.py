"""Smoke tests for the CasADi MPC controller."""
import os

import pytest

from buzzracer.common import BASEDIR
from buzzracer.extensions.extension import Extension
from buzzracer.main import Main


@pytest.fixture(autouse=True)
def reset_extension_registry():
    Extension.extensions = []
    yield


def test_mpc_controller_config_runs():
    """Controller should instantiate from XML and step the simulator."""
    config_filename = os.path.join(
        BASEDIR, 'tests', 'test_configs', 'test_mpc_controller.xml')
    experiment = Main(config_filename)
    experiment.run()
