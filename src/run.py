# universal entry point for running the car
from common import *
from threading import Event,Lock
from math import pi,radians,degrees
from time import time,sleep

# Extensions
import extension
from extension import KinematicSimulator,DynamicSimulator
from extension import Gifsaver, Laptimer,Optitrack,Logger
#from extension import Gifsaver, Laptimer,CrosstrackErrorTracker,Logger,LapCounter,CollisionChecker, Optitrack,Visualization, PerformanceTracker, Watchdog

from util.timeUtil import execution_timer
from track import TrackFactory

from Car import Car
from controller import StanleyCarController
from controller import CcmppiCarController
from controller import MppiCarController

class Main():
    def __init__(self,params={}):
        self.timer = execution_timer(True)
        self.dt = 0.01
        self.params = params
        self.algorithm = params['algorithm']
        self.new_state_update = Event()

        self.track = TrackFactory(name='full')

        Car.reset()
        car0 = Car.Factory(self, "porsche", controller=MppiCarController,init_states=(3.7*0.6,1.75*0.6, radians(-90), 1.0))
        car1 = Car.Factory(self, "lambo", controller=StanleyCarController,init_states=(3.7*0.6,1.75*0.6, radians(-90), 1.0))
        #car0 = Car.Factory(self, "porsche", controller=CcmppiCarController,init_states=(3.7*0.6,1.75*0.6, radians(-90),1.0))

        self.cars = Car.cars
        print_info("[main] total cars: %d"%(len(self.cars)))

        # flag to quit all child threads gracefully
        self.exit_request = Event()
        # if set, continue to follow trajectory but set throttle to -0.1
        # so we don't leave car uncontrolled at max speed
        # currently this is ignored and pressing 'q' the first time will cut motor
        # second 'q' will exit program
        self.slowdown = Event()
        self.slowdown_ts = 0

        # --- Extensions ---
        self.extensions = []
        self.visualization = extension.Visualization(self)
        #Optitrack(self)
        self.simulator = DynamicSimulator(self)
        self.simulator.match_time = False

        #Gifsaver(self)

        # Laptimer
        Laptimer(self)
        # save experiment as a gif, this provides an easy to use visualization for presentation
        #Logger(self)

        # steering rack tracker
        #SteeringTracker(self)

        for item in self.extensions:
            item.init()

        for car in self.cars:
            car.init()

        for item in self.extensions:
            item.postInit()

    # run experiment until user press q in visualization window
    def run(self):
        print_info("running ... press q to quit")
        while not self.exit_request.is_set():
            ts = time()
            self.update()
        # exit point
        print_info("Exiting ...")
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
    # alfa: progress
    #params = {'samples':4096, 'algorithm':'ccmppi','alfa':0.8,'beta':2.5}
    params = {'samples':4096, 'algorithm':'mppi-experiment','alfa':50.0,'beta':0.0}
    experiment = Main(params)
    experiment.run()
    experiment.timer.summary()
    #experiment.cars[0].controller.p.summary()

    print_info("program complete")
