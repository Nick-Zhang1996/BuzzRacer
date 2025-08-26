''' interface for Optitrack Motive stream via NatNet SDK library'''
# have a lot of x,y,z redefined in local scope
# pylint: disable=redefined-outer-name
from math import pi, degrees, atan2
from threading import Event, Lock
from time import sleep

import numpy as np
from scipy.spatial.transform import Rotation

from common import ExperimentType, PrintObject
from extension.Extension import Extension
from buzzracer.util.kalmanFilter import KalmanFilter
from buzzracer.third_party.NatNetClient import NatNetClient

class Optitrack(Extension):
    ''' interface for Optitrack Motive stream via NatNet SDK library'''
    def __init__(self):
        super().__init__(handle_name='vi')
        if Extension.main.experiment_type != ExperimentType.Realworld:
            self.print_error(
                'Experiment type is not Realworld but Optitrack is loaded')
        self.vi = None
        ''' Internal _Optitrack instance '''

    def init(self):
        self.vi = _Optitrack(self)
        for car in self.main.cars:
            car.internal_id = self.vi.get_internal_id(car.optitrack_id)
            self.print_ok(' Optitrack ID: %d, Internal ID: %d' %
                          (car.optitrack_id, car.internal_id))

    def update_car_states(self):
        for car in self.main.cars:
            # update for eachj car
            # not using kf state for now
            (x, y, v, theta, omega) = self.vi.get_k_fstate(car.internal_id)
            # (x,y,theta) = self.vi.get_state2d(self.car.internal_id)
            # (x,y,theta,vforward,vsideway=0,omega)
            car.states = (x, y, theta, v, 0, omega)
        self.main.new_state_update.set()

    def final(self):
        self.vi.quit()

class _Optitrack(PrintObject):
    def __init__(self, base, enableKF=True):
        self.base = base
        self.newState = Event()
        self.enableKF = Event()
        self.callback = self.empty_callback
        if enableKF:
            self.action = (0, 0)
            self.kf = []
            self.enableKF.set()

        # to be used in Kalman filter update
        # action = (steering in rad left positive, longitudinal acc (m/s2))
        self.action = (0, 0)
        self.wheelbase = 102e-3

        # This will create a new NatNet client
        self.streamingClient = NatNetClient()
        # set up callback functions later
        self.streamingClient.run()

        # set the relation between Optitrack world frame and our track frame
        # describe the rotation needed to rotate the world frame to track frame
        # in sequence Z, Y, X, each time using intermediate frame axis (intrinsic)
        self.R = Rotation.from_euler('ZYX', [180, 0, 90], degrees=True).inv()
        # use the rotation matrix
        # vector_track_frame = self.R.apply(vector_world_frame)

        # a list of state tuples,
        # state tuples take the form: (x,y,z,rx,ry,rz), in meters and radians, respectively
        # note that rx,ry,rz are euler angles in XYZ convention,
        #  this is different from the ZYX convention commonly used in aviation
        self.state_list = []
        # this is converted 2D state (x,y,heading) in track space
        self.state2d_list = []
        self.kf_state_list = []
        self.state_lock = Lock()

        # a mapping from internal id to optitrack id
        # self.optitrack_id_lookup[internal_id] = optitrack_id
        self.optitrack_id_lookup = []

        self.obj_count = 0

        if self.enableKF.is_set():
            # set callback for rigid body state update,
            # this will create a new KF instance for each object
            # and set up self.optitrack_id_lookup table
            self.streamingClient.rigidBodyListener = self.receive_rigid_body_frame_init
            # wait for all objects to be detected
            sleep(0.1)
            # switch to regular callback now that everything is initialized
            self.streamingClient.rigidBodyListener = self.receive_rigid_body_frame

    def __del__(self):
        self.streamingClient.request_quit()

    def empty_callback(self, *args):
        pass

    def quit(self):
        self.streamingClient.request_quit()

    # there are two sets of id
    # Optitrack ID: like object name in vicon,
    # each object has a unique ID that can be any integer value
    # internal ID within this class, like object id in vicon,
    # each object has a unique id, id will be assigned starting from zero
    # for example, the Optitrack ID for two objects may be 7,9,
    # while their corresponding internal ID will be 0,1
    # this is to facilitate easier indexing
    def get_optitrack_id(self, internal_id):
        # hard code since we only have a handful of models
        try:
            return self.optitrack_id_lookup[internal_id]
        except IndexError:
            self.print_error("can't find internal ID %d" % internal_id)
            return None

    # find internal id from optitrack id
    def get_internal_id(self, optitrack_id):
        try:
            return self.optitrack_id_lookup.index(optitrack_id)
        except ValueError:
            self.print_error("can't find optitrack ID %d" % optitrack_id)
            return None

    # optitrack callback for item discovery
    # this differs from receive_rigid_body_frame in that
    # 1. does not include kalman filter update
    # 2. if an unseen id is found,
    # it will be added to id list and an KF instance will be created for it
    def receive_rigid_body_frame_init(self, optitrack_id, position, rotation):
        if not optitrack_id in self.optitrack_id_lookup:
            self.obj_count += 1
            self.optitrack_id_lookup.append(optitrack_id)

            x, y, z = position
            qx, qy, qz, qw = rotation
            r = Rotation.from_quat([qx, qy, qz, qw])
            rz, ry, rx = r.as_euler('ZYX', degrees=False)

            # get body pose in track frame
            # x,y,z in track frame
            x_local, y_local, _ = self.R.apply([x, y, z])
            # x in car frame is forward direction, get that in world frame
            heading_world = r.apply([1, 0, 0])
            # now convert that to track frame
            heading_track = self.R.apply(heading_world)
            # heading in 2d world is the Z component
            theta_local = atan2(heading_track[1], heading_track[0])

            if self.enableKF.is_set():
                self.kf.append(KalmanFilter(wheelbase=self.wheelbase))
            # get body pose in track/local frame
            # current setup in G13
            x_local = -x
            y_local = z
            theta_local = ry + pi/2
            if self.enableKF.is_set():
                self.kf[-1].init(x_local, y_local, theta_local)

            self.state_lock.acquire(timeout=0.01)
            self.state_list.append((x, y, z, rx, ry, rz))
            self.state2d_list.append((x_local, y_local, theta_local))
            if self.enableKF.is_set():
                # (x,y,v,theta,omega)
                self.kf_state_list.append(
                    (x_local, y_local, 0, theta_local, 0))
            self.state_lock.release()

    # regular callback for state update
    def receive_rigid_body_frame(self, optitrack_id, position, rotation):
        # print( "Received frame for rigid body", id )
        internal_id = self.get_internal_id(optitrack_id)
        x, y, z = position
        qx, qy, qz, qw = rotation
        r = Rotation.from_quat([qx, qy, qz, qw])
        rz, ry, rx = r.as_euler('ZYX', degrees=False)

        # get body pose in track frame
        # x,y,z in track frame
        x_local, y_local, _ = self.R.apply([x, y, z])
        # x in car frame is forward direction, get that in world frame
        heading_world = r.apply([1, 0, 0])
        # now convert that to track frame
        heading_track = self.R.apply(heading_world)
        # heading in 2d world is the Z component
        theta_local = atan2(heading_track[1], heading_track[0])

        if self.enableKF.is_set():
            self.kf[internal_id].predict(self.action)
            observation = np.matrix([[x_local, y_local, theta_local]]).T
            self.kf[internal_id].update(observation)

        self.state_lock.acquire(timeout=0.01)
        self.state_list[internal_id] = (x, y, z, rx, ry, rz)
        self.state2d_list[internal_id] = (x_local, y_local, theta_local)

        if self.enableKF.is_set():
            # kf.get_state() := (x,y,v,theta,omega)
            self.kf_state_list[internal_id] = self.kf[internal_id].get_state()
        self.state_lock.release()
        if not self.base is None:
            self.base.update_car_states()
        self.newState.set()
        # print("Internal ID: %d \n Optitrack ID: %d"%(i,op_id))
        # print("World coordinate: %0.2f,%0.2f,%0.2f"%(x,y,z))
        # print("local state: %0.2f,%0.2f, heading= %0.2f"%(x_local,y_local,theta_local))
        # (kf_x,kf_y,kf_v,kf_theta,kf_omega) = self.get_k_fstate(i)
        # print("kf 2d state: %0.2f,%0.2f, heading= %0.2f"%(kf_x,kf_y,kf_theta))
        # print("\n")
        self.callback(optitrack_id, position, rotation)
        return

    # get state by internal id

    def get_state(self, internal_id):
        if internal_id >= self.obj_count:
            self.print_error("can't find internal id %d" % (internal_id))
            return None
        self.state_lock.acquire(timeout=0.01)
        retval = self.state_list[internal_id]
        self.state_lock.release()
        return retval

    def get_state2d(self, internal_id):
        if internal_id >= self.obj_count:
            self.print_error("can't find internal id %d" % (internal_id))
            return None
        try:
            self.state_lock.acquire(timeout=0.01)
            retval = self.state2d_list[internal_id]
        except IndexError as e:
            self.print_error("can't find internal id %d" % (internal_id))
            self.print_error(str(e))
            self.print_error('obj count '+str(self.obj_count))
            self.print_error('state2d list len '+str(len(self.state2d_list)))
            self.print_error('state list len '+str(len(self.state_list)))
        finally:
            self.state_lock.release()

        return retval

    # get KF state by internal id
    def get_k_fstate(self, internal_id):
        self.kf[internal_id].predict(self.action)
        # (x,y,v,theta,omega)
        return self.kf[internal_id].get_state()

    # update action used in KF prediction
    # this should be called right after a new command is sent to the vehicles
    # action = (steering in rad left positive, longitudinal acc (m/s2))
    def update_action(self, action):
        self.action = action
        return


# test functionality
if __name__ == '__main__':
    op = _Optitrack(None)
    print('obj count = %d' % (op.obj_count))
    while True:
        for i in range(op.obj_count):
            op_id = op.get_optitrack_id(i)
            i = op.get_internal_id(op_id)
            x2d, y2d, theta2d = op.get_state2d(i)
            x, y, z, rx, ry, rz = op.get_state(i)
            (kf_x, kf_y, kf_v, kf_theta, kf_omega) = op.get_k_fstate(i)
            print('Internal ID: %d \n Optitrack ID: %d' % (i, op_id))
            print('World coordinate: %0.2f,%0.2f,%0.2f' % (x, y, z))
            print('2d state: %0.2f,%0.2f, heading= %0.2f' %
                  (x2d, y2d, degrees(theta2d)))
            print('rx: %0.2f, ry: %0.2f, rz: %0.2f' %
                  (degrees(rx), degrees(ry), degrees(rz)))
            # print("kf 2d state: %0.2f,%0.2f, heading= %0.2f"%(kf_x,kf_y,kf_theta))
            print('\n')
            sleep(0.05)

    input('press enter to stop\n')
    op.quit()
