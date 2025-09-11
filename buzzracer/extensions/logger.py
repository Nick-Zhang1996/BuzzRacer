''' Extension to build car states logs '''
import os.path
from datetime import date
from time import time

import pickle

from buzzracer.common import BASEDIR, LogObject
from buzzracer.extensions.extension import Extension


class Logger(Extension):
    ''' Logger for car states.

    Log file records state of the vehicle for later analysis
     '''

    def __init__(self):
        Extension.__init__(self, 'logger')
        # if log is enabled this will be updated
        # if log is not enabled this will be used as gif image name
        self.log_no = 0
        self.resolve_logname()
        self.full_state_log = []
        ''' Full state log that's written to pickle file full_state{log_no}.p '''
        self.debug_dict_log = []
        ''' Debug dict that's written to pickle file debug_dict{log_no}.p '''

    def resolve_logname(self, ):
        ''' Find log file name '''

        # create a folder using date and type of experiment
        today = date.today()
        suffix = 'sim' if hasattr(self.main, 'simulator') else 'exp'

        try:
            log_folder = os.path.join(BASEDIR, 'log',
                                      self.main.experiment_name)
        except AttributeError:
            log_folder = os.path.join(
                BASEDIR, 'log',
                '%d_%d_%d_' % (today.year, today.month, today.day) + suffix)

        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        log_prefix = 'full_state'
        log_suffix = '.p'
        no = 1
        while os.path.isfile(log_folder + log_prefix + str(no) + log_suffix):
            no += 1

        self.log_no = no
        self.log_filename = os.path.join(log_folder, log_prefix + str(no) + log_suffix)

        log_prefix = 'debug_dict'
        self.log_dict_filename = os.path.join(log_folder, log_prefix + str(no) + log_suffix)
        self.log_folder = log_folder

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

        self.full_state_log.append(log_entry)

        # debug_dict
        logged = set()
        debug_dict = LogObject.populate_log(self.main, logged)
        debug_dict['cars'] = []
        for car in self.main.cars:
            debug_dict['cars'].append(LogObject.populate_log(car, logged))
        self.debug_dict_log.append(debug_dict)

    def post_final(self):
        self.print_ok('saving full_state log at ' + self.log_filename)

        output = open(self.log_filename, 'wb')
        pickle.dump(self.full_state_log, output)
        output.close()

        self.print_ok('saving debugDict log at ' + self.log_dict_filename)
        output = open(self.log_dict_filename, 'wb')
        pickle.dump(self.debug_dict_log, output)
        output.close()
