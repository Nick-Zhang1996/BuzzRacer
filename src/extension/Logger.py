# handles logging
import numpy as np
from common import *
from extension.Extension import Extension
from datetime import date

import os.path
from time import time
import pickle

class Logger(Extension):
    def __init__(self,main):
        Extension.__init__(self,main)
        # if log is enabled this will be updated
        # if log is not enabled this will be used as gif image name
        self.log_no = 0
        self.resolveLogname()
        # the vector that's written to pickle file
        # this is updated frequently, use the last line
        # (t(s), x (m), y, heading(rad, ccw+, x axis 0), steering(rad, right+), throttle (-1~1), kf_x, kf_y, kf_v,kf_theta, kf_omega )
        self.full_state_log = []

    def resolveLogname(self,):
        # setup log file
        # log file will record state of the vehicle for later analysis
        #logFolder = "../log/jan12/"

        # create a folder using date and type of experiment
        today = date.today()
        try: 
            self.main.simulator
            suffix = 'sim'
        except AttributeError:
            suffix = 'exp'
        logFolder = '../log/' + '%d_%d_%d_'%(today.year,today.month,today.day) + suffix + '/'
        

        if not os.path.exists(logFolder):
            os.makedirs(logFolder)
        logPrefix = "full_state"
        logSuffix = ".p"
        no = 1
        while os.path.isfile(logFolder+logPrefix+str(no)+logSuffix):
            no += 1

        self.log_no = no
        self.logFilename = logFolder+logPrefix+str(no)+logSuffix

        logPrefix = "debug_dict"
        self.logDictFilename = logFolder+logPrefix+str(no)+logSuffix

    def update(self):
        # x,y,theta are in track frame
        # v_forward in vehicle frame, forward positive
        # v_sideway in vehicle frame, left positive
        # omega in vehicle frame, axis pointing upward
        log_entry = []
        for i in range(len(self.main.cars)):
            car = self.main.cars[i]
            (x,y,theta,v_forward,v_sideway,omega) = car.states

            # (x,y,theta,vforward,vsideway=0,omega)
            log_entry.append([time(),x,y,theta,v_forward,v_sideway,omega, car.steering,car.throttle])

        self.full_state_log.append(log_entry)

    def postFinal(self):
        print_ok("[Logger]: saving full_state log at " + self.logFilename)

        output = open(self.logFilename,'wb')
        pickle.dump(self.full_state_log,output)
        output.close()

        print_ok("[Logger]: saving debugDict log at " + self.logDictFilename)
        self.debug_dict = []
        for car in self.main.cars:
            self.debug_dict.append(car.debug_dict)
        output = open(self.logDictFilename,'wb')
        pickle.dump(self.debug_dict,output)
        output.close()
