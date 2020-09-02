# universal entry point for running the car
import cv2
import sys
import os.path
import pickle
from threading import Event,Lock
from time import sleep,time
from PIL import Image
from math import pi,radians,degrees,asin,acos

from common import *
from vicon import Vicon
from car import Car
from Track import Track
from RCPTrack import RCPtrack
from skidpad import Skidpad
from Optitrack import Optitrack

from enum import Enum, auto
class StateUpdateSource(Enum):
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
        # if log is enabled this will be updated
        # if log is not enabled this will be used as gif image name
        self.log_no = 0

        # set visual tracking system to be used
        # Indoor Flight Laboratory: vicon
        # G13: optitrack
        # simulation: simulator
        self.stateUpdateSource = StateUpdateSource.optitrack
        #self.stateUpdateSource = StateUpdateSource.simulator
        # set target platform
        # if running simulation set this to simulator
        self.vehiclePlatform = VehiclePlatform.offboard
        #self.vehiclePlatform = VehiclePlatform.simulator

        if self.stateUpdateSource == StateUpdateSource.optitrack:
            self.initStateUpdate = self.initOptitrack
            self.updateState = self.updateOptitrack
            self.stopStateUpdate = self.stopOptitrack
        elif self.stateUpdateSource == StateUpdateSource.vicon:
            self.initStateUpdate = self.initVicon
            self.updateState = self.updateVicon
            self.stopStateUpdate = self.stopVicon
        elif self.stateUpdateSource == StateUpdateSource.simulator:
            self.initStateUpdate = self.initSimulation
            self.updateState = self.updateSimulation
            self.stopStateUpdate = self.stopSimulation
        else:
            print_error("unknown state update source")
            exit(1)

        # flag to quit all child threads gracefully
        self.exit_request = Event()

        # TODO: handle simulator case
        self.car = self.prepareCar()

        self.initStateUpdate()

        # prepare log
        if (self.enableLog):
            self.resolveLogname()
            self.state_log = []

        #self.track = self.prepareSkidpad()
        # or, use RCP track
        self.track = self.prepareRcpTrack()

        # verify that a valid track subclass is sued
        if not issubclass(type(self.track),Track):
            print_error(" self.track is not a subclass of Track")
            exit(1)

        self.prepareVisualization()

        # prepare save gif, this provides an easy to use visualization for presentation
        self.saveGif = False
        self.prepareGif()

    # run experiment until user press q in visualization window
    def run(self):
        while not self.exit_request.isSet():
            self.update()

        # exit point
        cv2.destroyAllWindows()
        self.stopStateUpdate()

        if self.saveGif:
            gif_filename = "../gifs/sim"+str(self.log_no)+".gif"
            # TODO better way of determining duration
            self.gifimages[0].save(fp=gif_filename,format='GIF',append_images=self.gifimages,save_all=True,duration = 30,loop=0)
            print_info("gif saved at "+gif_filename)

        if self.enableLog:
            output = open(self.logFilename,'wb')
            pickle.dump(state_vec,output)
            output.close()


    def updateVisualization(self,):
        # restrict update rate to 0.01s/frame, a rate higher than this can lead to frozen frames
        if (time()-self.visualization_ts>0.02 or self.stateUpdateSource == StateUpdateSource.simulator):
            img = self.track.drawCar(self.img_track.copy(), self.car_state, self.car.steering)

            self.visualization_ts = time()
            cv2.imshow('experiment',img)
            if self.saveGif:
                self.gifimages.append(Image.fromarray(cv2.cvtColor(img.copy(),cv2.COLOR_BGR2RGB)))

            if (self.stateUpdateSource == StateUpdateSource.simulator):
                k = cv2.waitKey(int(self.sim_dt/0.001)) & 0xFF
            else:
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
        # in simulation, sleep() between updates
        # in real experiment, wait on next state update
        if (self.stateUpdateSource == StateUpdateSource.simulator):
            # delay handled in cv2.waitKey()
            #sleep(self.sim_dt)
            self.updateSimulation()
        else:
            # wait on next state update
            self.new_visual_update.wait()
            # manually clear the Event()
            self.new_visual_update.clear()
            # retrieve car state from visual tracking update
            self.updateState()
        
        # get control signal
        throttle,steering,valid,debug_dict = self.car.ctrlCar(self.car_state,self.track,reverse=False)
        # TODO debug only
        self.v_target = debug_dict['v_target']
        
        self.car.steering = steering
        self.car.throttle = throttle

        if (self.vehiclePlatform == VehiclePlatform.offboard):
            throttle = 0.3
            self.car.actuate(steering,throttle)
            # TODO implement throttle model
            # do not use EKF for now
            #self.vi.updateAction(car.steering, car.getExpectedAcc())
        elif (self.vehiclePlatform == VehiclePlatform.simulator):
            # update is done in updateSimulation()
            pass
        elif (self.vehiclePlatform == VehiclePlatform.onboard):
            raise NotImplementedError


        self.updateVisualization()
        
# ---- Short Routine ----
    def prepareGif(self):
        if self.saveGif:
            self.gifimages = []
            self.gifimages.append(Image.fromarray(cv2.cvtColor(self.img_track.copy(),cv2.COLOR_BGR2RGB)))

    def prepareVisualization(self,):
        self.visualization_ts = time()
        self.img_track = self.track.drawTrack()
        self.img_track = self.track.drawRaceline(img=self.img_track)
        cv2.imshow('experiment',self.img_track)
        cv2.waitKey(1)

    def prepareSkidpad(self,):
        sp = Skidpad()
        sp.initSkidpad(radius=2,velocity=target_velocity)
        return sp

    def prepareRcpTrackSmall(self,):
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

    def prepareRcpTrack(self,):
        # current track setup in mk103, L shaped
        # TODO change dimension
        # width 0.563, square length 0.6

        # full RCP track
        # row, col
        fulltrack = RCPtrack()
        track_size = (6,4)
        #fulltrack.initTrack('uuurrullurrrdddddluulddl',track_size, scale=0.565)
        fulltrack.initTrack('uuurrullurrrdddddluulddl',track_size, scale=0.6)
        # add manual offset for each control points
        adjustment = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

        adjustment[0] = -0.2
        adjustment[1] = -0.2
        #bottom right turn
        adjustment[2] = -0.2
        adjustment[3] = 0.5
        adjustment[4] = -0.2

        #bottom middle turn
        adjustment[6] = -0.2

        #bottom left turn
        adjustment[9] = -0.2

        # left L turn
        adjustment[12] = 0.5
        adjustment[13] = 0.5

        adjustment[15] = -0.5
        adjustment[16] = 0.5
        adjustment[18] = 0.5

        adjustment[21] = 0.35
        adjustment[22] = 0.35

        # start coord, direction, sequence number of origin
        # pick a grid as the starting grid, this doesn't matter much, however a starting grid in the middle of a long straight helps
        # to find sequence number of origin, start from the start coord(seq no = 0), and follow the track, each time you encounter a new grid it's seq no is 1+previous seq no. If origin is one step away in the forward direction from start coord, it has seq no = 1
        fulltrack.initRaceline((3,3),'d',10,offset=adjustment)
        return fulltrack

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

        if (self.stateUpdateSource == StateUpdateSource.simulator):
            porsche_setting['serial_port'] = None
            lambo_setting['serial_port'] = None

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

        self.log_no = no
        self.logFilename = logFolder+logPrefix+str(no)+logSuffix

    # call before exiting
    def stop(self,):
        self.stopStateUpdate()


# ---- VICON ----
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

    def stopVicon(self,):
        self.vi.stopUpdateDaemon()

# ---- Optitrack ----
    def initOptitrack(self,):
        print_info("Initializing Optitrack...")
        self.vi = Optitrack(wheelbase=self.car.wheelbase)
        # TODO use acutal optitrack id for car
        # porsche: 2
        self.car.internal_id = self.vi.getInternalId(2)
        self.new_visual_update = self.vi.newState

    def updateOptitrack(self,):
        # update for eachj car
        # not using kf state for now
        #(x,y,v,theta,omega) = self.vi.getKFstate(self.car.internal_id)


        (x,y,theta) = self.vi.getState2d(self.car.internal_id)
        # (x,y,theta,vforward,vsideway=0,omega)
        self.car_state = (x,y,theta,0,0,0)
        return

    def stopOptitrack(self,):
        # the optitrack destructor should handle things properly
        pass

# ---- Simulation ----
# TODO encapsulate this in a different class/file
    def initSimulation(self):
        coord = (0.5*0.565,1.7*0.565)
        x,y = coord
        heading = pi/2
        omega = 0
        v = 0

        self.car.steering = steering = 0
        self.car.throttle = throttle = 0
        self.v_target = 0

        self.car_state = (x,y,heading,v,0,omega)
        self.sim_states = {'coord':coord,'heading':heading,'vf':throttle,'vs':0,'omega':0}
        self.sim_dt = 0.01

    def updateSimulation(self):
        # update car
        sim_states = self.sim_states = self.track.updateCar(self.sim_dt,self.sim_states,self.car.throttle,self.car.steering,v_override=self.v_target)
        self.car_state = np.array([sim_states['coord'][0],sim_states['coord'][1],sim_states['heading'],sim_states['vf'],0,sim_states['omega']])

    def stopSimulation(self):
        return




if __name__ == '__main__':
    experiment = Main()
    experiment.run()


