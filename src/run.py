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
from joystick import Joystick
from laptimer import Laptimer
from advCarSim import advCarSim
from kinematicSimulator import kinematicSimulator
from ctrlMpcWrapper import ctrlMpcWrapper
from ctrlStanleyWrapper import ctrlStanleyWrapper
from ctrlMppiWrapper import ctrlMppiWrapper

from enum import Enum, auto

# for cpu/ram analysis
#import psutil
class StateUpdateSource(Enum):
    vicon = auto()
    optitrack = auto()
    simulator = auto()
    dynamic_simulator = auto()

class VehiclePlatform(Enum):
    offboard = auto()
    onboard = auto() # NOT IMPLEMENTED
    simulator = auto()
    # no controller, this means out of loop control
    empty = auto()
    dynamic_simulator = auto()


class Controller(Enum):
    stanley = auto()
    purePursuit = auto() # NOT IMPLEMENTED
    joystick = auto()
    dynamicMpc = auto()
    mppi = auto()

    # no controller, this means out of loop control
    empty = auto()

class Main():
    def __init__(self,):
        # state update rate
        self.dt = 0.01

        # noise in simulation
        self.sim_noise = False
        # EXTREME noise
        self.sim_noise_cov = 10*np.diag([0.1,0.1,0.1,0.1,0.1,0.1])

        # CONFIG
        # whether to record control command, car state, etc.
        self.enableLog = False
        # save experiment as a gif, this provides an easy to use visualization for presentation
        self.saveGif = False
        # enable Laptime Voiceover, if True, will read out lap time after each lap
        self.enableLaptimer = False

        # run the track in reverse direction
        self.reverse = False


        # set visual tracking system to be used
        # Indoor Flight Laboratory (MK101/103): vicon
        # MK G13: optitrack
        # simulation: simulator
        #self.stateUpdateSource = StateUpdateSource.optitrack
        #self.stateUpdateSource = StateUpdateSource.simulator
        self.stateUpdateSource = StateUpdateSource.dynamic_simulator

        # real time/sim_time
        # larger value result in slower simulation
        self.real_sim_time_ratio = 2.0

        # set target platform
        # if running simulation set this to simulator
        #self.vehiclePlatform = VehiclePlatform.offboard
        #self.vehiclePlatform = VehiclePlatform.simulator
        self.vehiclePlatform = VehiclePlatform.dynamic_simulator

        # set control pipeline
        #self.controller = Controller.stanley
        self.controller = Controller.dynamicMpc
        self.controller = Controller.mppi

        if (self.controller == Controller.joystick):
            self.joystick = Joystick()
        if self.stateUpdateSource == StateUpdateSource.optitrack:
            self.initStateUpdate = self.initOptitrack
            self.updateState = self.updateOptitrack
            self.stopStateUpdate = self.stopOptitrack
        elif self.stateUpdateSource == StateUpdateSource.vicon:
            print_error("vicon not ready")
            self.initStateUpdate = self.initVicon
            self.updateState = self.updateVicon
            self.stopStateUpdate = self.stopVicon
        elif self.stateUpdateSource == StateUpdateSource.simulator:
            self.initStateUpdate = self.initSimulation
            self.updateState = self.updateSimulation
            self.stopStateUpdate = self.stopSimulation
        elif self.stateUpdateSource == StateUpdateSource.dynamic_simulator:
            self.initStateUpdate = self.initAdvSimulation
            self.updateState = self.updateAdvSimulation
            self.stopStateUpdate = self.stopAdvSimulation
        else:
            print_error("unknown state update source")
            exit(1)


        # if log is enabled this will be updated
        # if log is not enabled this will be used as gif image name
        self.log_no = 0

        # flag to quit all child threads gracefully
        self.exit_request = Event()
        # if set, continue to follow trajectory but set throttle to -0.1
        # so we don't leave car uncontrolled at max speed
        self.slowdown = Event()
        self.slowdown_ts = 0

        self.car = self.prepareCar()

        self.initStateUpdate()

        #self.track = self.prepareSkidpad()
        # or, use RCP track
        self.track = self.prepareRcpTrack()

        if (self.controller == Controller.dynamicMpc):
            if (self.stateUpdateSource == StateUpdateSource.dynamic_simulator):
                self.car.initMpcSim(self.simulator)
            elif (self.stateUpdateSource == StateUpdateSource.optitrack):
                self.car.initMpcReal()
        elif (self.controller == Controller.mppi):
            if (self.stateUpdateSource == StateUpdateSource.dynamic_simulator):
                self.car.init(self.track,self.simulator)
            elif (self.stateUpdateSource == StateUpdateSource.optitrack):
                print_error("unimplemented")
            

        # log with undetermined format
        self.debug_dict = {'target_v':[],'actual_v':[],'throttle':[],'p':[],'i':[],'d':[],'crosstrack_error':[],'heading_error':[]}

        # prepare log
        if (self.enableLog):
            self.resolveLogname()
            # the vector that's written to pickle file
            # this is updated frequently, use the last line
            # (t(s), x (m), y, heading(rad, ccw+, x axis 0), steering(rad, right+), throttle (-1~1), kf_x, kf_y, kf_v,kf_theta, kf_omega )
            self.full_state_log = []


        # verify that a valid track subclass is sued
        if not issubclass(type(self.track),Track):
            print_error(" self.track is not a subclass of Track")
            exit(1)

        self.prepareVisualization()

        # prepare save gif, this provides an easy to use visualization for presentation
        self.prepareGif()

    # run experiment until user press q in visualization window
    def run(self):
        print_info("running ... press q to quit")
        while not self.exit_request.isSet():
            self.update()
            # x,y,theta are in track frame
            # v_forward in vehicle frame, forward positive
            # v_sideway in vehicle frame, left positive
            # omega in vehicle frame, axis pointing upward
            (x,y,theta,v_forward,_,_) = self.car_state

            # (x,y,theta,vforward,vsideway=0,omega)
            self.debug_dict['target_v'].append(self.v_target)
            self.debug_dict['actual_v'].append(v_forward)
            self.debug_dict['throttle'].append(self.car.throttle)
            p,i,d = self.car.throttle_pid.getDebug()
            self.debug_dict['p'].append(p)
            self.debug_dict['i'].append(i)
            self.debug_dict['d'].append(d)

            if self.stateUpdateSource == StateUpdateSource.optitrack \
                    or self.stateUpdateSource == StateUpdateSource.vicon:
                (kf_x,kf_y,kf_v,kf_theta,kf_omega) = self.vi.getKFstate(self.car.internal_id)
            else:
                # in simulation there's no need for kf states, just use ground truth
                (kf_x,kf_y,kf_theta,kf_v,_,kf_omega) = self.car_state

            if self.enableLog:
                self.full_state_log.append([time(),x,y,theta,self.car.steering,self.car.throttle, kf_x, kf_y, kf_v, kf_theta, kf_omega])

        # exit point
        print_info("Exiting ...")
        cv2.destroyAllWindows()
        self.stopStateUpdate()

        if (self.controller == Controller.joystick):
            print_info("exiting joystick... move joystick a little")
            self.joystick.quit()

        if self.saveGif:
            print_info("saving gif.. This may take a while")
            gif_filename = "../gifs/sim"+str(self.log_no)+".gif"
            # TODO better way of determining duration
            self.gifimages[0].save(fp=gif_filename,format='GIF',append_images=self.gifimages,save_all=True,duration = 30,loop=0)
            print_info("gif saved at "+gif_filename)


        if self.enableLog:
            print_info("saving log...")
            print_info(self.logFilename)

            output = open(self.logFilename,'wb')
            pickle.dump(self.full_state_log,output)
            output.close()

            print_info("saving log...")
            print_info(self.logDictFilename)
            output = open(self.logDictFilename,'wb')
            pickle.dump(self.debug_dict,output)
            output.close()



    def updateVisualization(self,):
        # we want a real-time simulation, waiting sim_dt between each simulation step
        # however, since we cannot update visualization at sim_dt, we need to keep track of how much time has passed in simulation and match visualization accordingly
        if (self.stateUpdateSource == StateUpdateSource.dynamic_simulator \
                or self.stateUpdateSource == StateUpdateSource.simulator):
            sleep(max(0,self.real_sim_dt-(time()-self.simulator.t*self.real_sim_time_ratio)))

        # restrict update rate to 0.02s/frame, a rate higher than this can lead to frozen frames
        if (time()-self.visualization_ts>0.02 or self.stateUpdateSource == StateUpdateSource.simulator):
            img = self.track.drawCar(self.img_track.copy(), self.car_state, self.car.steering)
            # DEBUG
            # draw the range where solver is seeking answer
            #img = self.track.drawPointU(img,[self.track.debug['seq']-0.6,self.track.debug['seq']+0.6])

            # draw lookahead points x_ref
            '''
            x_ref = self.debug_dict['x_ref_l']
            for coord in x_ref:
                x,y = coord
                img = self.track.drawPoint(img,(x,y),color=(0,0,0))

            x_ref = self.debug_dict['x_ref_r']
            for coord in x_ref:
                x,y = coord
                img = self.track.drawPoint(img,(x,y),color=(0,0,0))
            '''

            x_ref = self.debug_dict['x_ref']
            for coord in x_ref:
                x,y = coord
                img = self.track.drawPoint(img,(x,y),color=(255,0,0))


            # draw projected state
            '''
            x_project = self.debug_dict['x_project']
            for coord in x_project:
                x,y = coord
                img = self.track.drawPoint(img,(x,y))
            '''

            self.visualization_ts = time()

            cv2.imshow('experiment',img)

            if self.saveGif:
                self.gifimages.append(Image.fromarray(cv2.cvtColor(img.copy(),cv2.COLOR_BGR2RGB)))
            '''
            ram = psutil.virtual_memory().percent
            cpu = psutil.cpu_percent()
            print("ram = %.2f, cpu = %.2f"%(ram,cpu))
            '''

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                # first time q is presed, slow down
                if not self.slowdown.isSet():
                    print("slowing down, press q again to shutdown")
                    self.slowdown.set()
                    self.slowdown_ts = time()
                else:
                    # second time, shut down
                    self.exit_request.set()

    # run the control/visualization update
    # this should be called in a loop(while not self.exit_request.isSet()) continuously, without delay

    # in simulation, this is called with evenly spaced time
    # in real experiment, this is called after a new vicon update is pulled
    # when a new vicon/optitrack state is available, vi.newState.isSet() will be true
    # client (this function) need to unset that event
    def update(self,):
        # wait on next state update
        self.new_state_update.wait()
        # manually clear the Event()
        self.new_state_update.clear()
        # retrieve car state from visual tracking update
        self.updateState()
        
        if (self.enableLaptimer):
            retval = self.laptimer.update((self.car_state[0],self.car_state[1]))
            if retval:
                self.laptimer.announce()
                print(self.laptimer.last_laptime)

        # apply controller
        if (self.controller == Controller.stanley):
            throttle,steering,valid,debug_dict = self.car.ctrlCar(self.car_state,self.track,reverse=self.reverse)
            if not valid:
                print_warning("ctrlCar invalid retval")
                exit(1)
            if self.slowdown.isSet():
                throttle = 0.0

            # DEBUG
            debug_dict['v_target'] =0
            self.v_target = debug_dict['v_target']
        elif (self.controller == Controller.dynamicMpc):
            throttle,steering,valid,debug_dict = self.car.ctrlCar(self.car_state,self.track,reverse=self.reverse)
            #self.debug_dict['x_project'] = debug_dict['x_project']
            self.debug_dict['x_ref'] = debug_dict['x_ref']
            self.debug_dict['crosstrack_error'].append(debug_dict['crosstrack_error'])
            self.debug_dict['heading_error'].append(debug_dict['heading_error'])
            if not valid:
                print_warning("ctrlCar invalid retval")
                exit(1)
            if self.slowdown.isSet():
                throttle = 0.0
            # DEBUG
            debug_dict['v_target'] =0
            self.v_target = debug_dict['v_target']
        elif (self.controller == Controller.joystick):
            throttle = self.joystick.throttle
            # just use right side for both ends
            steering = self.joystick.steering*self.car.max_steering_right
            self.v_target = throttle
        elif (self.controller == Controller.mppi):
            # TODO debugging...
            throttle,steering,valid,debug_dict = self.car.ctrlCar(self.car_state,self.track,reverse=self.reverse)
            print("T = %.2f, S = %.2f"%(throttle,steering))
            self.debug_dict['x_ref_l'] = debug_dict['x_ref_l']
            self.debug_dict['x_ref_r'] = debug_dict['x_ref_r']
            self.debug_dict['x_ref'] = debug_dict['x_ref']
            self.debug_dict['crosstrack_error'].append(debug_dict['crosstrack_error'])
            self.debug_dict['heading_error'].append(debug_dict['heading_error'])

        elif (self.controller == Controller.empty):
            throttle = 0
            steering = 0
            
        
        self.car.steering = steering
        self.car.throttle = throttle

        if (self.vehiclePlatform == VehiclePlatform.offboard):
            self.car.actuate(steering,throttle)
            # TODO implement throttle model
            # do not use EKF for now
            #self.vi.updateAction(car.steering, car.getExpectedAcc())
        elif (self.vehiclePlatform == VehiclePlatform.simulator):
            # update is done in updateSimulation()
            pass
        elif (self.vehiclePlatform == VehiclePlatform.dynamic_simulator):
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
        # width 0.563, square tile side length 0.6

        # full RCP track
        # NOTE load track instead of re-constructing
        fulltrack = RCPtrack()
        if self.enableLaptimer:
            self.laptimer = Laptimer((0.6*3.5,0.6*1.75),radians(90))
        fulltrack.load()
        return fulltrack

        # row, col
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
                         'max_throttle' : 0.7}

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
            # NOTE discrepancy with actual experiment
            print_warning("using different max_throttle setting")
            porsche_setting['max_throttle'] = 1.0
        if (self.stateUpdateSource == StateUpdateSource.dynamic_simulator):
            porsche_setting['serial_port'] = None
            lambo_setting['serial_port'] = None
            # NOTE discrepancy with actual experiment
            print_warning("using different max_throttle setting")
            porsche_setting['max_throttle'] = 1.0

        if (self.controller == Controller.dynamicMpc):
            # porsche 911
            car = ctrlMpcWrapper(porsche_setting,self.dt)
        elif (self.controller == Controller.stanley):
            car = ctrlStanleyWrapper(porsche_setting,self.dt)
        elif (self.controller == Controller.mppi):
            car = ctrlMppiWrapper(porsche_setting,self.dt)
        return car

    def resolveLogname(self,):

        # setup log file
        # log file will record state of the vehicle for later analysis
        logFolder = "../log/sim/"
        logPrefix = "full_state"
        logSuffix = ".p"
        no = 1
        while os.path.isfile(logFolder+logPrefix+str(no)+logSuffix):
            no += 1

        self.log_no = no
        self.logFilename = logFolder+logPrefix+str(no)+logSuffix

        logPrefix = "debug_dict"
        self.logDictFilename = logFolder+logPrefix+str(no)+logSuffix

    # call before exiting
    def stop(self,):
        self.stopStateUpdate()


# ---- VICON ----
    def initVicon(self,):
        print_info("Initializing Vicon...")
        self.vi = Vicon()
        self.new_state_update = self.vi.newState
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
        self.new_state_update = self.vi.newState

    def updateOptitrack(self,):
        # update for eachj car
        # not using kf state for now
        (x,y,v,theta,omega) = self.vi.getKFstate(self.car.internal_id)


        #(x,y,theta) = self.vi.getState2d(self.car.internal_id)
        # (x,y,theta,vforward,vsideway=0,omega)
        self.car_state = (x,y,theta,v,0,omega)
        return

    def stopOptitrack(self,):
        # the optitrack destructor should handle things properly
        self.vi.quit()
        pass

# ---- Simulation ----
# TODO encapsulate this in a different class/file
    def initSimulation(self):
        self.new_state_update = Event()
        self.new_state_update.set()
        self.simulator = kinematicSimulator()
        self.real_sim_dt = time()-self.simulator.t
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
        sim_states = self.sim_states = self.simulator.updateCar(self.sim_dt,self.sim_states,self.car.throttle,self.car.steering)
        self.car_state = np.array([sim_states['coord'][0],sim_states['coord'][1],sim_states['heading'],sim_states['vf'],0,sim_states['omega']])
        self.new_state_update.set()

    def stopSimulation(self):
        return

    # new dynamic simulator
    def initAdvSimulation(self):
        self.new_state_update = Event()
        self.new_state_update.set()
        coord = (0.3*0.565,1.7*0.565)
        x,y = coord
        heading = pi/2
        self.simulator = advCarSim(x,y,heading,self.sim_noise,self.sim_noise_cov)
        self.real_sim_dt = time()-self.simulator.t

        self.car.steering = steering = 0
        self.car.throttle = throttle = 0
        self.v_target = 0

        self.car_state = (x,y,heading,0,0,0)
        self.sim_states = {'coord':coord,'heading':heading,'vf':throttle,'vs':0,'omega':0}
        self.sim_dt = 0.01

    def updateAdvSimulation(self):
        # update car
        sim_states = self.sim_states = self.simulator.updateCar(self.sim_dt,self.sim_states,self.car.throttle,self.car.steering)
        self.car_state = np.array([sim_states['coord'][0],sim_states['coord'][1],sim_states['heading'],sim_states['vf'],sim_states['vs'],sim_states['omega']])
        #print(self.car_state)
        print("v = %.2f"%(sim_states['vf']))
        self.new_state_update.set()

    def stopAdvSimulation(self):
        return



if __name__ == '__main__':
    experiment = Main()
    experiment.run()
    #experiment.simulator.debug()
    print("mppi")
    experiment.car.mppi.p.summary()
    print("\noverall")
    experiment.car.p.summary()
    print_info("program complete")


