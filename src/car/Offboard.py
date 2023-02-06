# code to communicate with offboard miniz
# An updated version of this file can be found at:
# https://github.com/Nick-Zhang1996/miniz-board/blob/main/Offboard.py
from common import *
import socket
from struct import pack, unpack
import numpy as np
#from time import clock_gettime_ns, CLOCK_REALTIME,time,sleep
from time import  time,sleep,time_ns
from math import degrees,radians

from threading import Thread,Event,Lock
import select
import queue
from .Car import Car

# NOTE ideas to try for performance
# different sockets for incoming/outgoing messages

class OffboardPacket(PrintObject):
    out_seq_no = 0
    packet_size = 64
    def __init__(self):
        #self.print_debug_enable()
        # actual whole packet
        self.seq_no = None
        self.type = None
        self.subtype = None
        self.dest_addr = None
        self.src_addr = None
        self.packet = None
        self.payload = None
        # package encoded ts
        self.ts = None
        return

    def emptyPayload(self):
        self.payload = b''

    # encode all fields into .packet
    def makePacket(self):
        self.seq_no = OffboardPacket.out_seq_no
        self.ts = int(time_ns() / 1000) % 4294967295
        # B: uint8_t
        # H: uint16_t
        # I: uint32_t
        # f: float (4 Byte)
        # d: double (8 Byte)
        # x: padding (1 Byte)
        header = pack('IIBBBB',self.seq_no,self.ts,self.dest_addr,self.src_addr,self.type,self.subtype)
        padding_size = OffboardPacket.packet_size - len(header) - len(self.payload)
        padding = pack('x'*padding_size)
        self.packet = header+self.payload+padding

        OffboardPacket.out_seq_no += 1

    def parsePacket(self):
        packet = self.packet
        header = packet[:12]
        self.seq_no,self.ts,self.dest_addr, self.src_addr,self.type,self.subtype = unpack('IIBBBB',header)
        if (self.type == 0):
            # ping packet
            if (self.subtype == 0):
                #ping request
                pass
            elif (self.subtype == 1):
                # ping response
                pass
        if (self.type == 1):
            self.throttle,self.steering = unpack('ff',packet[12:20])

        # sensor update
        if (self.type == 2):
            self.steering_requested,self.steering_measured = unpack('ff',packet[12:20])
            #self.print_info('sensor update',self.steering_requested, self.steering_measured)

        # parameter
        if (self.type == 3):
            if (self.subtype == 0):
                sensor_update,steering_P,steering_I,steering_D = unpack('?fff',packet[12:12+4+3*4])
                self.print_info('parameter response')
                self.print_info('sensor_update ', sensor_update)
                self.print_info('steering_P ', steering_P)
                self.print_info('steering_I ', steering_I)
                self.print_info('steering_D ', steering_D)
                self.sensor_update = sensor_update
                self.steering_P = steering_P
                self.steering_I = steering_I
                self.steering_D = steering_D
        return self.type

class Offboard(Car):
    available_local_port = 58998
    def __init__(self,main):
        #self.print_debug_enable()
        Car.__init__(self,main)

    # parameter initialization, this will run immediately after self.params is set
    # put all parameters here. 
    def initParam(self):
        self.car_ip = self.params['ip']
        # default physics properties
        # used when a specific car subclass is not speciied
        self.L = 0.09
        self.lf = 0.04824
        self.lr = self.L - self.lf

        self.Iz = 417757e-9
        self.m = 0.1667

        # ethCarsim moved for ccmppi

        # tire model
        self.Df = 3.93731
        self.Dr = 6.23597
        self.C = 2.80646
        self.B = 0.51943
        # motor/longitudinal model
        self.Cm1 = 6.03154
        self.Cm2 = 0.96769
        self.Cr = -0.20375
        self.Cd = 0.00000

        self.width = self.params['width']
        self.wheelbase = self.params['wheelbase']
        self.max_throttle = self.params['max_throttle']
        self.max_steering_left = self.params['max_steer_angle_left']
        self.max_steering_right = self.params['max_steer_angle_right']
        self.max_throttle = self.params['max_throttle']
        self.optitrack_id = self.params['optitrack_streaming_id']

    def initHardware(self):
        self.car_port = 2390
        self.initSocket()
        self.initLog()

        # threading
        self.child_threads = []
        # ready to take new command
        self.ready = Event()
        self.flag_quit = Event()
        self.out_queue = queue.Queue(maxsize=8)

        self.throttle = 0.0
        self.steering = 0.0

        # Create a separate thread for handling data packets
        comm_thread = Thread( target = self.__commThreadFunction, args = (None, ))
        comm_thread.daemon = True
        comm_thread.start()
        self.child_threads.append(comm_thread)
        self.setup()

    def initSocket(self):
        self.local_ip = "192.168.0.101"
        self.local_port = Offboard.available_local_port
        Offboard.available_local_port += 1
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # non-blocking
        sock.setblocking(0)
        sock.bind((self.local_ip, self.local_port))
        self.sock = sock

    def initLog(self):
        self.log_t_vec = []
        self.steering_requested_vec = []
        self.steering_measured_vec = []


    # one-time process
    def setup(self):
        # steering servo PID
        #self.setParam(400.0,0,40)
        self.setParam(300.0,0,30)

    def __commThreadFunction( self, arg ):
        self.print_debug('commThread started')
        while not self.flag_quit.is_set():
            # send control command
            packet = self.prepareCommandPacket(self.throttle,self.steering)
            packet.makePacket()
            try:
                select.select([],[self.sock],[])
                self.sendPacket(packet)
            except BlockingIOError:
                self.print_warning('resource unavailable')

            # send all pending packets
            # packets in queue may be from another thread
            try:
                while True:
                    packet = self.out_queue.get_nowait()
                    # makePacket() fills in timestamp and seq_no, call right before send
                    packet.makePacket()
                    self.sendPacket(packet)
            except queue.Empty:
                pass

            # read all packets from buffer
            try:
                # wait for at least one packet before sending new commands
                select.select([self.sock],[],[],0.1)
                while True:
                    data, addr = self.sock.recvfrom( OffboardPacket.packet_size ) # read 1 packet
                    if( len( data ) > 0 ):
                        assert len( data ) == OffboardPacket.packet_size
                        self.parseResponse( data )
                        #self.print_debug('got packet')
            except BlockingIOError:
                pass

            # ready to take new commands
            self.ready.set()
        self.print_debug('comm thread quit')


    def quit(self):
        self.throttle = 0.0
        self.steering = 0.0
        self.flag_quit.set()
        # TODO collect thread
        self.print_info('quitting, waiting for threads to complete')
        for thread in self.child_threads:
            thread.join()
        self.print_info('quit success')


    def getParam(self):
        packet = self.prepareParameterRequestPacket()
        self.out_queue.put_nowait(packet)
        # TODO add Event to wait for response

    def setParam(self,p,i,d):
        self.print_info('setting parameters')
        packet = OffboardPacket()
        packet.type = 3
        packet.subtype = 3
        packet.dest_addr = 1
        packet.src_addr = 0
        packet.payload = pack('?fff',True,p,i,d)
        self.out_queue.put_nowait(packet)
        return packet

    def sendPacket(self,packet):
        sent_size = self.sock.sendto(packet.packet, (self.car_ip, self.car_port))
        self.last_sent_ts = packet.ts

    def parseResponse(self,data):
        packet = OffboardPacket()
        packet.packet = data
        packet_type = packet.parsePacket()
        self.last_response_ts = int(time_ns() / 1000) % 4294967295

        # sensor update
        if (packet_type == 2):
            self.log_t_vec.append(time())
            self.steering_requested_vec.append(packet.steering_requested)
            self.steering_measured_vec.append(packet.steering_measured)
            
        return packet

    def preparePingPacket(self):
        packet = OffboardPacket()
        packet.type = 0
        packet.subtype = 0
        packet.dest_addr = 1
        packet.src_addr = 0
        packet.emptyPayload()
        return packet

    def prepareCommandPacket(self,throttle=0.0,steering=0.0):
        packet = OffboardPacket()
        packet.type = 1
        packet.subtype = 5
        packet.dest_addr = 1
        packet.src_addr = 0
        packet.payload = pack('ff',throttle,steering)
        return packet

    def prepareParameterRequestPacket(self):
        packet = OffboardPacket()
        packet.type = 3
        packet.subtype = 2
        packet.dest_addr = 1
        packet.src_addr = 0
        packet.emptyPayload()
        return packet
        

