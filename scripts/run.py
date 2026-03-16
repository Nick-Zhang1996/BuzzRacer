""" Main entry point for launching experiments """
import os
import sys
import logging

from buzzracer.common import BASEDIR
from buzzracer.main import Main

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':

    # Run default.xml config if none is provided
    name = sys.argv[1] if len(sys.argv) == 2 else 'default'
    config_fn = os.path.join(BASEDIR, 'configs', f'{name}.xml')

    if os.path.exists(config_fn):
        logger.info('using config %s', config_fn)
    else:
        logger.error('%s  does not exist!', config_fn)

    experiment = Main(config_fn)
    experiment.run()
    experiment.timer.summary()
    # experiment.cars[0].controller.p.summary()

    logger.info('program complete')
