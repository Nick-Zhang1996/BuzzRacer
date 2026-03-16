
import os

from buzzracer.scripts.run import Main
from buzzracer.common import BASEDIR


def test_stanley_controller():
    config_filename = os.path.join(BASEDIR, 'tests',
                                   'test_configs', 'test_stanley.xml')

    if not os.path.exists(config_filename):
        raise FileNotFoundError

    experiment = Main(config_filename)
    experiment.run()
