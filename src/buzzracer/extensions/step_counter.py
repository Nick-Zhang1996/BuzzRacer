''' Terminate experiment after a number of time steps'''
from buzzracer.extensions.extension import Extension, ExtensionConfig, ExtensionState


class StepCounterConfig(ExtensionConfig):
    def __init__(self, main_config):
        super().__init__(main_config)
        self.total_steps = 100
        """ Total steps to run before termination"""


@Extension.register('step_counter', StepCounterConfig, ExtensionState)
class StepCounter(Extension):
    ''' Terminate experiment after a number of time steps'''

    def __init__(self, config, state):
        Extension.__init__(self, config, state)
        self.count = 0

    def update(self):
        self.count += 1
        if self.count >= self.config.total_steps:
            self.main.exit_request.set()
