''' Calibrate car kinematics and dynamics parameters '''

import os
import logging
import pickle

from buzzracer.common import BASEDIR
from buzzracer.extensions.extension import Extension, ExtensionConfig, ExtensionState

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class StateMachine():
    RAMPUP = 1
    IN_PROGRESS = 2
    RAMPDOWN = 3

    LOG_SLOW_DRIVING = 4
    STOP = 5

    def __init__(self, host, main):
        self.main = main
        self.car = self.main.cars[0]
        self.host = host

        self.state = self.LOG_SLOW_DRIVING
        self.substate = self.RAMPUP

        self.slow_driving_state_log = []
        self.slow_driving_control_log = []

    def act(self):
        """ Act based on current state, update state if needed"""
        if self.state == self.LOG_SLOW_DRIVING:
            if self.substate == self.RAMPUP:
                target_v = 0.5
                self.car.controller.v_override = target_v
                if abs(self.car.state.v_forward - target_v) < 0.05:
                    self.substate = self.IN_PROGRESS
                    logger.info('Start logging')
            elif self.substate == self.IN_PROGRESS:
                self.slow_driving_state_log.append(self.car.state)
                self.slow_driving_control_log.append((self.car.steering, self.car.throttle))
                if len(self.slow_driving_control_log) > 40*100:
                    # log 10 seconds
                    self.substate = self.RAMPDOWN
                    logger.info('End logging')
            elif self.substate == self.RAMPDOWN:
                target_v = 0.0
                self.car.controller.v_override = target_v
                if abs(self.car.state.v_forward - target_v) < 0.05:
                    self.state = self.STOP
                    data = {'state_log': self.slow_driving_state_log,
                            'control_log': self.slow_driving_control_log}
                    with open(os.path.join(BASEDIR, 'outputs', 'logs', 'sysid_slow.p'), 'wb') as f:
                        pickle.dump(data, f)
                    logger.info('Saved')
                    self.host.main.exit_request.set()


@Extension.register('sysid', ExtensionConfig, ExtensionState)
class SysId(Extension):
    def __init__(self, config, state):
        super().__init__(config, state)
        self.sm = StateMachine(self, self.main)

    def update(self):
        self.sm.act()

    def final(self):
        pass
