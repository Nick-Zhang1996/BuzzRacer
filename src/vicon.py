#!/usr/bin/env python
# Interface for VICON UDB Object Stream
# Parse Vicon UDP Object Stream, supports multiple objects on SINGLE port
import socket
from time import time,sleep
from struct import unpack
from math import degrees,radians
from threading import Lock
from kalmanFilter import KalmanFilter
import numpy as np
from tf import TF
import threading


class Vicon:
    # IP : IP address to listen on, if you only have a single network card, default should work
    # Port : Port number to listen on, default is vicon's default port
    # daemon : whether to spawn a daemon update thread
    #     If user prefer to get vicon update manually, this can be set to false
    #     However, user need to make sure to call getViconUpdate() frequently enough to prevent 
    #     the incoming network buffer from filling up, which would result in self.getViconUpdate
    #     gettting stacked up outdated vicon frames
    #
    #     In applications where getViconUpdate() cannot be called frequently enough,
    #     or when the user code can't afford to wait for getViconUpdate() to complete.
    #     User may choose to set daemon to True, in which case a new thread would be spawned 
    #     dedicated to receiving vicon frames and maintaining a local copy of the most recent states

    # NOTE If the the set of objects picked up by vicon changes as the program is running, each object's ID may change
    # It is recommended to work with a fixed number of objects that can be readily tracked by vicon
    # thoughout the use of the this class
    # One object per port would solve this issue since each object would have a unique ID, whether or not it is active/detected
    # However it is not currently supported. If you have a strong need for this feature please contact author
    def __init__(self,IP=None,PORT=None,daemon=True,enableKF=True):
        if IP is None:
            IP = "0.0.0.0"
        if PORT is None:
            PORT = 51001

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.05)
        self.sock.bind((IP, PORT))
        # number of currently published tracked objects
        self.obj_count = None
        # lock for accessing member variables since they are updated in a separate thread
        self.state_lock = Lock()
        self.state2d_lock = Lock()
        # contains a list of names of tracked objects
        self.obj_names = []
        # a list of state tuples, state tuples take the form: (x,y,z,rx,ry,rz), in meters and radians, respectively
        # note that rx,ry,rz are euler angles in XYZ convention, this is different from the ZYX convention commonly used in aviation
        self.state_list = []
        # this is converted 2D state (x,y,heading) in track space
        self.state2d_list = []

        # flag used to stop update daemon thread
        self.quit_thread = False

        # create a TF() object so we can internalize frame transformation
        # go from 3D vicon space -> 2D track space
        self.tf = TF()
        # items related to tf
        # for upright origin
        q_t = self.tf.euler2q(0,0,0)
        self.T = np.hstack([q_t,np.array([-0.0,-0.0,0])])

        if enableKF:
            # temporarily unset enableKF to trick getViconUpdate() to ignore kf before it's inited
            if not daemon:
                print("Warning: Kalman Filter is enabled but Vicon Update Daemon is not")
            self.enableKF = False
            retval = self.getViconUpdate()
            self.enableKF = True
            if retval is None:
                print("Vicon not ready, can't determine obj count for Kalman Filter")
                exit(1)

            self.kf = [KalmanFilter() for i in range(self.obj_count)]
            #self.kf_state = []
            for i in range(self.obj_count):
                (x,y,theta) = self.getState2d(i)
                self.kf[i].init(x,y,theta)
                #self.kf_state.append(self.kf.getState())

        if daemon:
            self.thread =  threading.Thread(name="vicon",target=self.viconUpateDaemon)
            self.thread.start()
        else:
            self.thread = None

        
    def __del__(self):
        if not (self.thread is None):
            self.quit_thread = True
            self.thread.join()
        self.sock.close()

    # get name of an object given its ID. This can be useful for verifying ID <-> object relationship
    def getItemName(self,obj_id):
        self.state_lock.acquire()
        local_name = self.obj_names[obj_id]
        self.state_lock.release()
        return local_name

    # get item id from name
    def getItemID(self,obj_name):
        self.state_lock.acquire()
        local_names = self.obj_names
        self.state_lock.release()
        try:
            obj_id = local_names.index(obj_name)
        except ValueError:
            obj_id = None
            print("Error, item :"+str(obj_name)+" not found")
        finally:
            return obj_id

    # get state by id
    def getState(self,inquiry_id):
        if inquiry_id>=self.obj_count:
            print("error: invalid id : "+str(inquiry_id))
            return None
        self.state_lock.acquire()
        retval = self.state_list[inquiry_id]
        self.state_lock.release()
        return retval

    def viconUpateDaemon(self):
        while not self.quit_thread:
            self.getViconUpdate()

    # stop the update thread
    def stopUpdateDaemon(self):
            if not (self.thread is None):
                self.quit_thread = True
                self.thread.join()
                self.thread = None

    def getViconUpdate(self,debugData=None):
        # the no of bytes here must agree with length of a vicon packet
        # typically 256,512 or 1024
        try:
            if debugData is None:
                data, addr = self.sock.recvfrom(256)
                # in python 2 data is of type str
                #data = data.encode('ascii')
            else:
                data = debugData
            local_obj_names = []
            local_state_list = []

            self.obj_count = itemsInBlock = data[4]
            itemID = data[5] # always 0, not very useful
            itemDataSize = unpack('h',data[6:8])
            for i in range(itemsInBlock):
                offset = i*75
                itemName = data[offset+8:offset+32].rstrip(b'\0').decode()
                # raw data in mm, convert to m
                x = unpack('d',data[offset+32:offset+40])[0]/1000
                y = unpack('d',data[offset+40:offset+48])[0]/1000
                z = unpack('d',data[offset+48:offset+56])[0]/1000
                # euler angles,rad, rotation order: rx,ry,rz, using intermediate frame
                rx = unpack('d',data[offset+56:offset+64])[0]
                ry = unpack('d',data[offset+64:offset+72])[0]
                rz = unpack('d',data[offset+72:offset+80])[0]
                local_obj_names.append(itemName)
                local_state_list.append((x,y,z,rx,ry,rz))
                #print(i,itemName)
                #print(x,y,z,degrees(rx),degrees(ry),degrees(rz))
            self.state_lock.acquire()
            self.obj_names = local_obj_names
            self.state_list = local_state_list
            self.state_lock.release()
        except socket.timeout:
            return None

        local_state2d_list = []
        for i in range(self.obj_count):
            # get body pose in track frame
            # (x,y,heading)
            x,y,z,rx,ry,rz = local_state_list[i]
            # (z_x,z_y,z_theta)
            (z_x,z_y,z_theta) = self.tf.reframeR(self.T,x,y,z,self.tf.euler2Rxyz(rx,ry,rz))
            local_state2d_list.append((z_x,z_y,z_theta))

            if self.enableKF:
                self.kf[i].predict()
                z = np.matrix([[z_x,z_y,z_theta]]).T
                self.kf[i].update(z)
                #self.kf_state[i] = self.kf[i].getState()

        self.state2d_lock.acquire()
        self.state2d_list = local_state2d_list
        self.state2d_lock.release()
        return local_state_list

    def getState2d(self,inquiry_id):
        if inquiry_id>=self.obj_count:
            return None
        try:
            self.state2d_lock.acquire()
            retval = self.state2d_list[inquiry_id]
        except IndexError as e:
            print(str(e))
            print("obj count "+str(self.obj_count))
            print("state2d list len "+str(len(self.state2d_list)))
            print("state list len "+str(len(self.state_list)))
            exit(0)
        finally:
            self.state2d_lock.release()
            
        return retval

    # get KF state by id
    def getKFstate(self,inquiry_id):
        self.kf[inquiry_id].predict()
        # (x,dx,-,y,dy,-,theta,dtheta)
        return self.kf[inquiry_id].getState()


    def testFreq(self,packets=100):
        # test actual frequency of vicon update, with PACKETS number of state updates
        tic = time()
        for i in range(packets):
            self.getViconUpdate()
        tac = time()
        return packets/(tac-tic)

    # for debug
    def saveFile(self,data,filename):
        newFile = open(filename, "wb")
        newFile.write(data)
        newFile.close()

    # for debug
    def loadFile(self,filename):
        newFile = open(filename, "rb")
        data = bytearray(newFile.read())
        newFile.close()
        return data

if __name__ == '__main__':
    vi = Vicon(daemon=True)
    vi.getViconUpdate()
    sleep(0.1)

    for i in range(vi.obj_count):
        print("ID: "+str(i)+", Name: "+vi.getItemName(i))

    wand_id = vi.getItemID('Wand')
    print("Wand id "+str(wand_id))
    sleep(1)

    # debug speed estimation
    while False:
    #for i in range(10):
        (kf_x,dx,_,kf_y,dy,_,theta,dtheta) = vi.getKFstate(wand_id)
        (x,y,z,rx,ry,rz) = vi.getState(wand_id)
        #print(x,dx,degrees(dtheta))
        #print(x,y,degrees(rz))
        print(kf_x-x)
        sleep(0.02)



    vi.stopUpdateDaemon()


    # test freq
    if False:
        for i in range(3):
            print("Freq = "+str(vi.testFreq())+"Hz")

        
    

