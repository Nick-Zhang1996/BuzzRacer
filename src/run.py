# universal entry point for running the car
from common import *
from threading import Event,Lock
from math import pi,radians,degrees
from time import time,sleep

from util.timeUtil import execution_timer
from track import TrackFactory

from car.Car import Car

from xml.dom import minidom
import xml.etree.ElementTree as ET

import sys
import os.path
import os
os.environ["PATH"] = os.environ["PATH"]+":/usr/local/cuda/bin/" # enables cuda

class Main(PrintObject):
    def __init__(self,config_filename):
        self.basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_filename = config_filename

    def init(self):
        self.print_ok(" loading settings")
        config = minidom.parse(self.config_filename)
        config_settings = config.getElementsByTagName('settings')[0]
        self.print_ok(" setting main attributes")
        for key,value_text in config_settings.attributes.items():
            setattr(self,key,eval(value_text))
            self.print_info(" main.",key,'=',value_text)

        config_experiment_text = config_settings.getElementsByTagName('experiment_type')[0].firstChild.nodeValue
        self.experiment_type = eval('ExperimentType.'+config_experiment_text)

        # prepare track
        #config_track_text = config_settings.getElementsByTagName('track')[0].firstChild.nodeValue
        config_track= config.getElementsByTagName('track')[0]
        self.track = TrackFactory(self,config_track)
        self.track.init()

        # prepare cars
        Car.reset()
        config_cars = config.getElementsByTagName('cars')[0]
        for config_car in config_cars.getElementsByTagName('car'):
            Car.Factory(self,config_car)
        self.cars = Car.cars
        self.print_info(" total cars: %d"%(len(self.cars)))

        self.timer = execution_timer(True)
        self.new_state_update = Event()
        # flag to quit all child threads gracefully
        self.exit_request = Event()
        # if set, continue to follow trajectory but set throttle to -0.1
        # so we don't leave car uncontrolled at max speed
        # currently this is ignored and pressing 'q' the first time will cut motor
        # second 'q' will exit program
        self.slowdown = Event()
        self.slowdown_ts = 0

        # --- Extensions ---
        self.print_ok("setting up extensions...")
        self.extensions = []
        config_extensions = config.getElementsByTagName('extensions')[0]
        for config_extension in config_extensions.getElementsByTagName('extension'):
            extension_class_name = config_extension.firstChild.nodeValue
            exec('from extension import '+extension_class_name)
            #ext = eval(extension_class_name+'(self)')
            ext = eval(extension_class_name)(self)
            handle_name = ''
            for key,raw in config_extension.attributes.items():
                if key == 'handle':
                    handle_name = raw
                    setattr(self,handle_name,ext)
                    self.print_info('main.'+handle_name+' = '+ext.__class__.__name__)
                else:
                    try:
                        value = eval(raw)
                    except NameError:
                        value = raw
                    # all other attributes will be set to extension
                    setattr(ext,key,value)
                    self.print_info('main.'+handle_name+'.'+key+' = '+str(value))

        for item in self.extensions:
            item.init()

        for car in self.cars:
            car.init()

        for item in self.extensions:
            item.postInit()

    # run experiment until user press q in visualization window
    def run(self):
        self.print_info("running ... press q to quit")
        while not self.exit_request.is_set():
            ts = time()
            self.update()
        # exit point
        self.print_info("Exiting ...")
        for item in self.extensions:
            item.preFinal()
        for item in self.extensions:
            item.final()
        for item in self.extensions:
            item.postFinal()

    def time(self):
        if self.experiment_type == ExperimentType.Simulation:
            return self.sim_t
        else:
            return time()


    # run the control/visualization update
    # this should be called in a loop(while not self.exit_request.isSet()) continuously, without delay

    # in simulation, this is called with evenly spaced time
    # in real experiment, this is called after a new vicon update is pulled
    # when a new vicon/optitrack state is available, vi.newState.isSet() will be true
    # client (this function) need to unset that event
    def update(self,):
        t = self.timer
        # -- Extension update -- 
        t.s()
        for item in self.extensions:
            t.s(item.name)
            item.preUpdate()
            t.e(item.name)

        self.new_state_update.wait()
        self.new_state_update.clear()

        t.s('control')
        for car in self.cars:
            # call controller, send command to car in real experiment
            car.control()
        t.e('control')

        # -- Extension update -- 
        t.s('update')
        for item in self.extensions:
            item.update()
        t.e('update')
        t.s('post')
        for item in self.extensions:
            item.postUpdate()
        t.e('post')
        t.e()
        

    # call before exiting
    def stop(self,):
        for car in self.cars:
            car.stopStateUpdate(car)



if __name__ == '__main__':
    if (len(sys.argv) == 2):
        name = sys.argv[1]
    else:
        name = 'default'

    config_filename = './configs/'+name+'.xml'
    if (os.path.exists(config_filename)):
        print_ok('using config '+config_filename)
    else:
        print_error(config_filename + '  does not exist!')

    experiment = Main(config_filename)
    experiment.init()
    experiment.run()
    experiment.timer.summary()
    #experiment.cars[0].controller.p.summary()

    print_info("program complete")
