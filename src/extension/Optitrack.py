# interface for Optitrack Motive stream via NatNet SDK library
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from third_party.NatNetClient import NatNetClient

from time import time,sleep
from threading import Event,Lock
from common import *
from util.kalmanFilter import KalmanFilter
import numpy as np
from math import pi,degrees,atan2
from scipy.spatial.transform import Rotation
from extension.Extension import Extension

class Optitrack(Extension):
    def __init__(self, main):
        super().__init__(main)
        # ensure experiment_type hasn't been initialized
        flag_is_unique = False
        try:
            main.experiment_type != ExperimentType.Simulation
        except (AttributeError):
            flag_is_unique = True
        if (not flag_is_unique):
            print_error("[Optitrack]: another state update source has been initialized")

        main.experiment_type = ExperimentType.Realworld

    def init(self):
        self.vi = _Optitrack(self)
        self.main.vi = self.vi
        for car in self.main.cars:
            car.internal_id = self.vi.getInternalId(car.optitrack_id)
            print_ok("[Optitrack]: Optitrack ID: %d, Internal ID: %d"%(car.optitrack_id, car.internal_id))

    def updateCarStates(self):
        for car in self.main.cars:
            # update for eachj car
            # not using kf state for now
            (x,y,v,theta,omega) = self.vi.getKFstate(car.internal_id)
            #(x,y,theta) = self.vi.getState2d(self.car.internal_id)
            # (x,y,theta,vforward,vsideway=0,omega)
            car.states = (x,y,theta,v,0,omega)
        self.main.new_state_update.set()

    def final(self):
        self.vi.quit()


# ---- Optitrack ---- old
    def initOptitrack(self,car,unused=None):
        print_info("Initializing Optitrack...")
        car.vi = Optitrack(wheelbase=car.wheelbase)
        # TODO use acutal optitrack id for car
        # porsche: 2
        car.internal_id = car.vi.getInternalId(car.optitrack_id)
        car.new_state_update = car.vi.newState

    def updateOptitrack(self,car):
        # update for eachj car
        # not using kf state for now
        (x,y,v,theta,omega) = car.vi.getKFstate(car.internal_id)

        #(x,y,theta) = self.vi.getState2d(self.car.internal_id)
        # (x,y,theta,vforward,vsideway=0,omega)
        car.states = (x,y,theta,v,0,omega)
        return

    def stopOptitrack(self,car):
        # the optitrack destructor should handle things properly
        car.vi.quit()
        pass


class _Optitrack:
    def __init__(self,base,enableKF=True):
        self.base = base
        self.newState = Event()
        self.enableKF = Event()
        self.callback = self.emptyCallback()
        if enableKF:
            self.action = (0,0)
            self.kf = []
            self.enableKF.set()

        # to be used in Kalman filter update
        # action = (steering in rad left positive, longitudinal acc (m/s2))
        self.action = (0,0)
        self.wheelbase = 102e-3

        # This will create a new NatNet client
        self.streamingClient = NatNetClient()
        # set up callback functions later
        self.streamingClient.run()

        # set the relation between Optitrack world frame and our track frame
        # describe the rotation needed to rotate the world frame to track frame
        # in sequence Z, Y, X, each time using intermediate frame axis (intrinsic)
        self.R = Rotation.from_euler("ZYX",[180,0,90],degrees=True).inv()
        # use the rotation matrix
        # vector_track_frame = self.R.apply(vector_world_frame)


        # a list of state tuples, state tuples take the form: (x,y,z,rx,ry,rz), in meters and radians, respectively
        # note that rx,ry,rz are euler angles in XYZ convention, this is different from the ZYX convention commonly used in aviation
        self.state_list = []
        # this is converted 2D state (x,y,heading) in track space
        self.state2d_list = []
        self.kf_state_list = []
        self.state_lock = Lock()

        # a mapping from internal id to optitrack id
        # self.optitrack_id_lookup[internal_id] = optitrack_id
        self.optitrack_id_lookup = []

        self.obj_count = 0

        if self.enableKF.isSet():
            # set callback for rigid body state update, this will create a new KF instance for each object
            # and set up self.optitrack_id_lookup table
            self.streamingClient.rigidBodyListener = self.receiveRigidBodyFrameInit
            # wait for all objects to be detected
            sleep(0.1)
            # switch to regular callback now that everything is initialized
            self.streamingClient.rigidBodyListener = self.receiveRigidBodyFrame



    def __del__(self):
        self.streamingClient.requestQuit()

    def emptyCallback(self,*args):
        pass

    def quit(self):
        self.streamingClient.requestQuit()

    # there are two sets of id
    # Optitrack ID: like object name in vicon, each object has a unique ID that can be any integer value
    # internal ID within this class, like object id in vicon, each object has a unique id, id will be assigned starting from zero
    # for example, the Optitrack ID for two objects may be 7,9, while their corresponding internal ID will be 0,1
    # this is to facilitate easier indexing 
    def getOptitrackId(self,internal_id):
        # hard code since we only have a handful of models
        try:
            return self.optitrack_id_lookup[internal_id]
        except IndexError:
            print_error("can't find internal ID %d"%internal_id)
            return None

    # find internal id from optitrack id
    def getInternalId(self,optitrack_id):
        try:
            return self.optitrack_id_lookup.index(optitrack_id)
        except ValueError:
            print_error("can't find optitrack ID %d"%optitrack_id)
            return None

    # optitrack callback for item discovery
    # this differs from receiveRigidBodyFrame in that
    # 1. does not include kalman filter update
    # 2. if an unseen id is found, it will be added to id list and an KF instance will be created for it
    def receiveRigidBodyFrameInit( self, optitrack_id, position, rotation ):
        if not (optitrack_id in self.optitrack_id_lookup):
            self.obj_count +=1
            self.optitrack_id_lookup.append(optitrack_id)

            x,y,z = position
            qx, qy, qz, qw = rotation
            r = Rotation.from_quat([qx,qy,qz,qw])
            rz, ry, rx = r.as_euler('ZYX',degrees=False)

            # get body pose in track frame
            # x,y,z in track frame
            x_local, y_local, z_local = self.R.apply([x,y,z])
            # x in car frame is forward direction, get that in world frame
            heading_world = r.apply([1,0,0])
            # now convert that to track frame
            heading_track = self.R.apply(heading_world)
            # heading in 2d world is the Z component
            theta_local = atan2(heading_track[1],heading_track[0])

            if self.enableKF.isSet():
                self.kf.append(KalmanFilter(wheelbase=self.wheelbase))
            # get body pose in track/local frame
            # current setup in G13
            x_local = -x
            y_local = z
            theta_local = ry + pi/2
            if self.enableKF.isSet():
                self.kf[-1].init(x_local,y_local,theta_local)

            self.state_lock.acquire(timeout=0.01)
            self.state_list.append((x,y,z,rx,ry,rz))
            self.state2d_list.append((x_local,y_local,theta_local))
            if self.enableKF.isSet():
                # (x,y,v,theta,omega)
                self.kf_state_list.append((x_local,y_local,0,theta_local,0))
            self.state_lock.release()

    # regular callback for state update
    def receiveRigidBodyFrame(self, optitrack_id, position, rotation ):
        #print( "Received frame for rigid body", id )
        internal_id = self.getInternalId(optitrack_id)
        x,y,z = position
        qx, qy, qz, qw = rotation
        r = Rotation.from_quat([qx,qy,qz,qw])
        rz, ry, rx = r.as_euler('ZYX',degrees=False)

        # get body pose in track frame
        # x,y,z in track frame
        x_local, y_local, z_local = self.R.apply([x,y,z])
        # x in car frame is forward direction, get that in world frame
        heading_world = r.apply([1,0,0])
        # now convert that to track frame
        heading_track = self.R.apply(heading_world)
        # heading in 2d world is the Z component
        theta_local = atan2(heading_track[1],heading_track[0])

        if self.enableKF.isSet():
            self.kf[internal_id].predict(self.action)
            observation = np.matrix([[x_local,y_local,theta_local]]).T
            self.kf[internal_id].update(observation)

        self.state_lock.acquire(timeout=0.01)
        self.state_list[internal_id] = (x,y,z,rx,ry,rz)
        self.state2d_list[internal_id] = (x_local,y_local,theta_local)

        if self.enableKF.isSet():
            # kf.getState() := (x,y,v,theta,omega)
            self.kf_state_list[internal_id] = self.kf[internal_id].getState()
        self.state_lock.release()
        if not self.base is None:
            self.base.updateCarStates()
        self.newState.set()
        #print("Internal ID: %d \n Optitrack ID: %d"%(i,op_id))
        #print("World coordinate: %0.2f,%0.2f,%0.2f"%(x,y,z))
        #print("local state: %0.2f,%0.2f, heading= %0.2f"%(x_local,y_local,theta_local))
        #(kf_x,kf_y,kf_v,kf_theta,kf_omega) = self.getKFstate(i)
        #print("kf 2d state: %0.2f,%0.2f, heading= %0.2f"%(kf_x,kf_y,kf_theta))
        #print("\n")
        self.callback(optitrack_id, position, rotation)
        return
    

    # get state by internal id
    def getState(self, internal_id):
        if internal_id>=self.obj_count:
            print_error("can't find internal id %d"%(internal_id))
            return None
        self.state_lock.acquire(timeout=0.01)
        retval = self.state_list[internal_id]
        self.state_lock.release()
        return retval

    def getState2d(self,internal_id):
        if internal_id>=self.obj_count:
            print_error("can't find internal id %d"%(internal_id))
            return None
        try:
            self.state_lock.acquire(timeout=0.01)
            retval = self.state2d_list[internal_id]
        except IndexError as e:
            print_error("can't find internal id %d"%(internal_id))
            print_error(str(e))
            print_error("obj count "+str(self.obj_count))
            print_error("state2d list len "+str(len(self.state2d_list)))
            print_error("state list len "+str(len(self.state_list)))
        finally:
            self.state_lock.release()
            
        return retval

    # get KF state by internal id
    def getKFstate(self,internal_id):
        self.kf[internal_id].predict(self.action)
        # (x,y,v,theta,omega)
        return self.kf[internal_id].getState()

    # update action used in KF prediction
    # this should be called right after a new command is sent to the vehicles
    # action = (steering in rad left positive, longitudinal acc (m/s2))
    def updateAction(self,action):
        self.action = action
        return


# test functionality
if __name__ == '__main__':
    op = _Optitrack(None)
    print("obj count = %d"%(op.obj_count))
    while True:
        for i in range(op.obj_count):
            op_id = op.getOptitrackId(i)
            i = op.getInternalId(op_id)
            x2d,y2d,theta2d = op.getState2d(i)
            x,y,z,rx,ry,rz = op.getState(i)
            (kf_x,kf_y,kf_v,kf_theta,kf_omega) = op.getKFstate(i)
            print("Internal ID: %d \n Optitrack ID: %d"%(i,op_id))
            print("World coordinate: %0.2f,%0.2f,%0.2f"%(x,y,z))
            print("2d state: %0.2f,%0.2f, heading= %0.2f"%(x2d,y2d,degrees(theta2d)))
            print("rx: %0.2f, ry: %0.2f, rz: %0.2f"%(degrees(rx),degrees(ry),degrees(rz)))
            #print("kf 2d state: %0.2f,%0.2f, heading= %0.2f"%(kf_x,kf_y,kf_theta))
            print("\n")
            sleep(0.05)

    input("press enter to stop\n")
    op.quit()


