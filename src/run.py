# universal entry point for running the car
import cv2
import sys
import os.path
import pickle
from threading import Event,Lock
from time import sleep,time
from PIL import Image

from common import *
from vicon import Vicon
from car import Car
from track import RCPtrack
from skidpad import Skidpad

from enum import Enum, auto
class VisualTrackingSystem(Enum):
    vicon = auto()
    optitrack = auto()
    simulator = auto()

class VehiclePlatform(Enum):
    offboard = auto()
    onboard = auto() # NOT IMPLEMENTED
    simulator = auto()

class Main():
    def __init__(self,):
        self.enableLog = False

        # set visual tracking system to be used
        # Indoor Flight Laboratory: vicon
        # G13: optitrack
        # simulation: simulator
        self.visualTrackingSystem = VisualTrackingSystem.simulator
        # set target platform
        # if running simulation set this to simulator
        self.vehiclePlatform = VehiclePlatform.simulator

        if self.visualTrackingSystem == VisualTrackingSystem.optitrack:
            self.initVisualTracking = self.initOptitrack
            self.updateVisualTracking = self.updateOptitrack
            self.stopVisualTracking = self.stopOptitrack
        else:
            self.initVisualTracking = self.initVicon
            self.updateVisualTracking = self.updateVicon
            self.stopVisualTracking = self.stopVicon

        # flag to quit all child threads gracefully
        self.exit_request = Event()

        if self.visualTrackingSystem != VisualTrackingSystem.simulator:
            # initialize visual tracking system
            self.initVisualTracking()
        else:
            # TODO add simulation init code here
            self.simulation_dt = 0.02

        # prepare log
        if (self.enableLog):
            self.resolveLogname()
            self.state_log = []

        # TODO: handle simulator case
        self.car = self.prepareCar()

        self.track = self.prepareSkidpad()
        # or, use RCP track
        #self.track = self.prepareRcpTrack()

        # verify that a valid track subclass is sued
        if not issubclass(type(self.track),Track):
            print_error(" self.track is not a subclass of Track")
            exit(1)

        self.prepareVisualization()

        # prepare save gif, this provides an easy to use visualization for presentation
        self.saveGif = True
        self.prepareGif()

    # run experiment until user press q in visualization window
    def run(self):
        while not self.exit_request.isSet():
            self.update()

        # exit point
        cv2.destroyAllWindows()
        self.vi.stopUpdateDaemon()
        if self.saveGif:
            self.gifimages[0].save(fp="./gifs/mk103exp"+str(no)+".gif",format='GIF',append_images=gifimages,save_all=True,duration = 50,loop=0)

        if self.enableLog:
            output = open(self.logFilename,'wb')
            pickle.dump(state_vec,output)
            output.close()

    def updateSimulation(self,):
        pass

    # update local state
    # TODO use new KF
    # OBSOLETE
    def updateVicon(self,):
        # retrieve vehicle state
        retval = self.vi.getState2d(car.vicon_id)
        if retval is None:
            return
        (x,y,heading) = retval
        kf_x,vx,ax,kf_y,vy,ay,kf_heading,omega = self.vi.getKFstate(self.car.vicon_id)

        # state_car = (x,y,heading, vf, vs, omega)
        # assume no lateral velocity
        vf = (vx**2+vy**2)**0.5

        # low pass filter on vf
        vf_lf, z_vf = signal.lfilter(b,a,[vf],zi=z_vf)
        vf = vf_lf[0]

        self.car_state = (x,y,vf,heading,omega)
        return

    def updateVisualization(self,):
        # restrict update rate to 0.1s/frame, a rate higher than this can lead to frozen frames
        if (time()-self.visualization_ts>0.1):
            img = track.drawCar(self.img_track.copy(), self.car_state[0], self.car.steering)

            self.visualization_ts = time()
            cv2.imshow('experiment',img)
            if self.saveGif:
                self.gifimages.append(Image.fromarray(cv2.cvtColor(img.copy(),cv2.COLOR_BGR2RGB)))
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                self.exit_request.set()

    # run the control/visualization update
    # this should be called in a loop(while not self.exit_request.isSet()) continuously, without delay

    # in simulation, this is called with evenly spaced time
    # in real experiment, this is called after a new vicon update is pulled
    # when a new vicon/optitrack state is available, vi.newState.isSet() will be true
    # client (this function) need to unset that event
    def update(self,):
        if (self.visualTrackingSystem == VisualTrackingSystem.simulator):
            sleep(self.simulation_dt)
            self.updateSimulation()
        else:
            self.new_visual_update.wait()
            self.new_visual_update.clear()

            # retrieve car state
            self.updateVisualTracking()
        
        throttle,steering,valid,debug_dic = self.car.ctrlCar(self.car_state,self.track,reverse=False)
        
        car.steering = steering
        car.throttle = throttle
        if (self.visualTrackingSystem == VisualTrackingSystem.simulator):
            #
        else:
            car.actuate(steering,throttle)
            # TODO implement throttle model
            self.vi.updateAction(car.steering, car.getExpectedAcc())


        self.state_log.append(self.car_state)
        self.updateVisualization()
        

    def prepareGif():
        if self.saveGif:
            self.gifimages = []
            self.gifimages.append(Image.fromarray(cv2.cvtColor(self.img_track.copy(),cv2.COLOR_BGR2RGB)))

    def prepareVisualization(self,):
        self.visualization_ts = time()
        self.img_track = self.track.drawTrack()
        self.img_track = self.track.drawRaceline(img=self.img_track)
        cv2.imshow('experiment',self.img_track)
        cv2.waitKey(1)


    # call before exiting
    def stop(self,):
        self.stopVisualTracking()

    def initVicon(self,):
        print_info("Initializing Vicon...")
        self.vi = Vicon()
        self.new_visual_update = self.vi.newState
        self.vicon_dt = 0.01
        # wait for vicon to find objects
        sleep(0.05)
        self.car.vicon_id = self.vi.getItemID('nick_mr03_porsche')
        if car.vicon_id is None:
            print_error("error, can't find car in vicon")
            exit(1)
        return

    def stopVicon(self,):
        self.vi.stopUpdateDaemon()

    def initOptitrack(self,)vi
        print_info("Initializing Optitrack...")
        self.vi = Optitrack()
        # TODO use acutal optitrack id for car
        self.car.internal_id = self.vi.getInternalId(1)
        self.new_visual_update = self.vi.newState

    def stopOptitrack(self,):
        # the optitrack destructor should handle things properly
        pass

    def updateOptitrack(self,):
        # update for eachj car
        (x,y,v,theta,omega) = self.vi.getKFstate(self.car.internal_id)
        self.car_state = (x,y,v,theta,omega)
        return
        


    def prepareSkidpad(self,):
        sp = Skidpad()
        sp.initSkidpad(radius=2,velocity=target_velocity)
        return sp

    def prepareRcpTrack(self,):
        # current track setup in mk103, L shaped
        # width 0.563, length 0.6
        mk103 = RCPtrack()
        mk103.initTrack('uuruurddddll',(5,3),scale=0.57)
        # add manual offset for each control points
        adjustment = [0,0,0,0,0,0,0,0,0,0,0,0]
        adjustment[4] = -0.5
        adjustment[8] = -0.5
        adjustment[9] = 0
        adjustment[10] = -0.5
        mk103.initRaceline((2,2),'d',4,offset=adjustment)
        return mk103

    def prepareCar(self,):
        porsche_setting = {'wheelbase':90e-3,
                         'max_steer_angle_left':radians(27.1),
                         'max_steer_pwm_left':1150,
                         'max_steer_angle_right':radians(27.1),
                         'max_steer_pwm_right':1850,
                         'serial_port' : '/dev/ttyUSB0',
                         'max_throttle' : 0.5}

        lambo_setting = {'wheelbase':98e-3,
                         'max_steer_angle_left':asin(2*98e-3/0.52),
                         'max_steer_pwm_left':1100,
                         'max_steer_angle_right':asin(2*98e-3/0.47),
                         'max_steer_pwm_right':1850,
                         'serial_port' : '/dev/ttyUSB1',
                         'max_throttle' : 0.5}

        # porsche 911
        car = Car(porsche_setting)
        return car

    def resolveLogname(self,):

        # setup log file
        # log file will record state of the vehicle for later analysis
        #   state: (x,y,heading,v_forward,v_sideway,omega)
        logFolder = "./log/"
        logPrefix = "exp_state"
        logSuffix = ".p"
        no = 1
        while os.path.isfile(logFolder+logPrefix+str(no)+logSuffix):
            no += 1
        self.logFilename = logFolder+logPrefix+str(no)+logSuffix


if __name__ == '__main__':
    experiment = Main()
    experiment.run()


