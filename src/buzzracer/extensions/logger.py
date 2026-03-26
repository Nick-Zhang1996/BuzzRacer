''' Extension to build car states logs '''
import os.path
from datetime import date
from time import time

import pickle

from buzzracer.common import BASEDIR, LogObject, ExperimentType
from buzzracer.extensions.extension import Extension, ExtensionConfig, ExtensionState


class LoggerConfig(ExtensionConfig):
    def __init__(self, main_config):
        super().__init__(main_config)
        log_no, log_folder, log_filename, log_dict_filename = Logger.resolve_logname(main_config)
        self.log_no = log_no
        self.log_folder = log_folder
        self.log_filename = log_filename
        self.log_dict_filename = log_dict_filename


class LoggerState(ExtensionState):
    def __init__(self, config):
        super().__init__(config)
        self.full_state_log = []
        ''' Full state log that's written to pickle file full_state{log_no}.p '''
        self.debug_dict_log = []
        ''' Debug dict that's written to pickle file debug_dict{log_no}.p '''


@Extension.register('logger', LoggerConfig, LoggerState)
class Logger(Extension):
    ''' Logger for car states.
    Log file records state of the vehicle for later analysis
     '''

    def __init__(self, config, state):
        Extension.__init__(self, config, state)

    @staticmethod
    def resolve_logname(main_config):
        ''' Find log file name. '''
        # create a folder using date and type of experiment
        today = date.today()
        suffix_map = {ExperimentType.Simulation: 'sim',
                      ExperimentType.Realworld: 'exp'}
        suffix = suffix_map[main_config.experiment_type]

        try:
            log_folder = os.path.join(BASEDIR, 'outputs', 'logs', main_config.experiment_name)
        except AttributeError:
            log_folder = os.path.join(
                BASEDIR, 'outputs', 'logs',
                '%d_%d_%d_' % (today.year, today.month, today.day) + suffix)

        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        log_prefix = 'full_state'
        log_suffix = '.p'
        log_no = 1
        log_filename = os.path.join(log_folder, log_prefix + str(log_no) + log_suffix)
        while os.path.isfile(log_filename):
            log_no += 1
            log_filename = os.path.join(log_folder, log_prefix + str(log_no) + log_suffix)
        log_prefix = 'debug_dict'
        log_dict_filename = os.path.join(log_folder, log_prefix + str(log_no) + log_suffix)
        return log_no, log_folder, log_filename, log_dict_filename

    def post_update(self):
        # x,y,theta are in track frame
        # v_forward in vehicle frame, forward positive
        # v_sideway in vehicle frame, left positive
        # omega in vehicle frame, axis pointing upward
        log_entry = []
        for car in self.main.cars:
            (x, y, theta, v_forward, v_sideway, omega) = car.state
            # (time, x,y,theta, vforward,vsideway=0,omega)
            log_entry.append([
                time(), x, y, theta, v_forward, v_sideway, omega, car.steering,
                car.throttle
            ])

        self.state.full_state_log.append(log_entry)

        # debug_dict
        logged = set()
        debug_dict = LogObject.populate_log(self.main, logged)
        debug_dict['cars'] = []
        for car in self.main.cars:
            debug_dict['cars'].append(LogObject.populate_log(car, logged))
        self.state.debug_dict_log.append(debug_dict)

    def post_final(self):
        config = self.config
        state = self.state
        self.print_ok('saving full_state log at ' + config.log_filename)

        output = open(config.log_filename, 'wb')
        pickle.dump(state.full_state_log, output)
        output.close()

        self.print_ok('saving debugDict log at ' + config.log_dict_filename)
        output = open(config.log_dict_filename, 'wb')
        pickle.dump(state.debug_dict_log, output)
        output.close()
