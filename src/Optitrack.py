# interface for Optitrack Motive stream via NatNet SDK library

from NatNetClient import NatNetClient
from time import time,sleep
from threading import Event
from common import *



class Optitrack:
    def __init__(self,wheelbase,enableKF=True):
        self.newState = Event()
        self.enableKF = Event()
        if enableKF:
            self.enableKF.set()

        # to be used in Kalman filter update
        # action = (steering in rad left positive, longitudinal acc (m/s2))
        self.action = (0,0)
        self.wheelbase = wheelbase

        # This will create a new NatNet client
        self.streamingClient = NatNetClient()
        # set up callback functions later
        self.streamingClient.run()


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

        # create a TF() object so we can internalize frame transformation
        # go from 3D vicon space -> 2D track space
        self.tf = TF()
        # items related to tf
        # for upright origin
        q_t = self.tf.euler2q(0,0,0)
        self.T = np.hstack([q_t,np.array([-0.0,-0.0,0])])

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
    def receiveRigidBodyFrameInit( optitrack_id, position, rotation ):
        if not (optitrack_id in self.optitrack_id_lookup):
            self.obj_count +=1
            self.optitrack_id_lookup.append(id)

            # TODO verify this
            x,y,z = position
            # TODO verify
            rx,ry,rz = rotation

            # TODO add correct wheelbase here
            if self.enableKF.isSet():
                self.kf.append(KalmanFilter(wheelbase=self.wheelbase))
            # get body pose in track frame
            # (x,y,heading)
            # (z_x,z_y,z_theta)
            x,y,theta = self.tf.reframeR(self.T,x,y,z,self.tf.euler2Rxyz(rx,ry,rz))
            if self.enableKF.isSet():
                self.kf[-1].init(x,y,theta)

            self.state_lock.acquire()
            self.state_list.append((x,y,z,rx,ry,rz))
            self.state2d_list.append((x,y,theta))
            if self.enableKF.isSet():
                # (x,y,v,theta,omega)
                self.kf_state_list.append((x,y,0,theta,0))
            self.state_lock.release()


    # regular callback for state update
    def receiveRigidBodyFrame( optitrack_id, position, rotation ):
        #print( "Received frame for rigid body", id )
        internal_id = self.getInternalId(optitrack_id)
        x,y,z = position
        rx,ry,rz = rotation
        (z_x,z_y,z_theta) = self.tf.reframeR(self.T,x,y,z,self.tf.euler2Rxyz(rx,ry,rz))

        if self.enableKF.isSet():
            self.kf[internal_id].predict(self.action)
            z = np.matrix([[z_x,z_y,z_theta]]).T
            self.kf[internal_id].update(z)

        self.state_lock.acquire()
        self.state_list[internal_id] = (x,y,z,rx,ry,rz)
        self.state2d_list[internal_id] = (z_x, z_y, z_theta)
        if self.enableKF.isSet():
            # kf.getState() := (x,y,v,theta,omega)
            self.kf_state_list[internal_id] = self.kf[internal_id].getState()
        self.state_lock.release()
        self.newState.set()

    # get state by internal id
    def getState(self, internal_id):
        if internal_id>=self.obj_count:
            print_error("can't find internal id %d"%(internal_id))
            return None
        self.state_lock.acquire()
        retval = self.state_list[internal_id]
        self.state_lock.release()
        return retval

    def getState2d(self,internal_id):
        if internal_id>=self.obj_count:
            print_error("can't find internal id %d"%(internal_id))
            return None
        try:
            self.state_lock.acquire()
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
        self.kf[internal_id].predict()
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
    op = Optitrack()
    for i in range(op.obj_count):
        op_id = op.getOptitrackId(i)
        x2d,y2d,theta2d = op.getState2d(i)
        x,y,z,rx,ry,rz = op.getState(i)
        (kf_x,kf_y,kf_v,kf_theta,kf_omega) = op.getKFstate(i)
        print("Internal ID: %d \n Optitrack ID: %d"%(i,op_id))
        print("World coordinate: %0.2f,%0.2f,%0.2f"%(x,y,z))
        print("2d state: %0.2f,%0.2f, heading= %0.2f"(x2d,y2d,theta2d))
        print("kf 2d state: %0.2f,%0.2f, heading= %0.2f"(kf_x,kf_y,kf_theta))
        print("\n")


