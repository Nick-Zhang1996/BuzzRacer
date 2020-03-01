#!/usr/bin/env python
# Parse Vicon UDP Object Stream, supports multiple objects on SINGLE port
import socket
from time import time
from struct import unpack
from math import degrees,radians
from threading import Lock
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
    def __init__(self,IP=None,PORT=None,daemon=True):
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
        # contains a list of names of tracked objects
        self.obj_names = []
        # a list of state tuples, state tuples take the form: (x,y,z,rx,ry,rz), in meters and radians, respectively
        # note that rx,ry,rz are euler angles in XYZ convention, this is different from the ZYX convention commonly used in aviation
        self.state_list = []

        # flag used to stop update daemon thread
        self.quit_thread = False
        if daemon:
            self.thread =  threading.Thread(name="vicon",target=self.viconUpateDaemon)
        
    def __del__(self):
        if not (self.threading is None):
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

    def viconUpateDaemon(self):
        while not self.quit_thread:
            self.getViconUpdate()

    def getViconUpdate(self,debugData=None):
        # the no of bytes here must agree with length of a vicon packet
        # typically 256,512 or 1024
        try:
            if debugData is None:
                data, addr = self.sock.recvfrom(256)
            else:
                data = debugData
            local_obj_names = []
            local_state_list = []

            self.obj_count = itemsInBlock = data[4]
            itemID = data[5] # always 0, not very useful
            itemDataSize = unpack('h',data[6:8])
            for i in range(itemsInBlock):
                offset = i*75
                itemName = str(data[8:32]).rstrip('\0')
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
                print(itemsInBlock,itemID,itemName)
                print(x,y,z,degrees(rx),degrees(ry),degrees(rz))
            self.state_lock.acquire()
            self.obj_names = local_obj_names
            self.state_list = local_state_list
            self.state_lock.release()
        except socket.timeout:
            return None
        return local_state_list

    def testFreq(self,packets=100):
        # test actual frequency of vicon update, with PACKETS number of state updates
        tic = time()
        for i in range(packets):
            self.getViconUpdate()
        tac = time()
        return packets/(tac-tic)

    # for debug
    def __save2File(self,data,filename):
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
    #f = open('samplevicon.bin','br')
    #data = f.read()
    vi = Vicon()
    testdata = vi.loadFile("./twoobj.vicon")
    vi.getViconUpdate(testdata)
    print(vi.getItemID('nick_mr03_lambo'))
    print(vi.getItemName(0))

    while False:
        #print(vi.getViconUpdate())
        vi.getViconUpdate()
    # test freq
    if False:
        for i in range(3):
            print("Freq = "+str(vi.testFreq())+"Hz")

        
    

