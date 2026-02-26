''' Subclass of Car for the offboard miniz (audi 11, 12) equipped with Nano 33 IoT '''
from __future__ import annotations

import socket
import select
import queue
from time import time, time_ns
from struct import pack, unpack
from threading import Thread, Event

from buzzracer.common import PrintObject
from buzzracer.cars.car import Car, CarParams
from buzzracer.types import CartesianState

# NOTE ideas to try for performance
# different sockets for incoming/outgoing messages


class OffboardPacket(PrintObject):
    ''' UDP packet structure for comms between car and PC'''
    out_seq_no = 0
    packet_size = 64

    def __init__(self):
        self.seq_no = None
        self.type = None
        self.subtype = None
        self.dest_addr = None
        self.src_addr = None
        self.packet = None
        self.payload = None
        # package encoded ts
        self.ts = None

        self.steering = 0
        self.throttle = 0
        self.steering_requested = 0
        self.steering_measured = 0

        self.sensor_update = 0
        self.steering_P = 0
        self.steering_I = 0
        self.steering_D = 0

    def empty_payload(self):
        self.payload = b''

    def make_packet(self):
        ''' Encode all fields into .packet '''
        self.seq_no = OffboardPacket.out_seq_no
        self.ts = int(time_ns() / 1000) % 4294967295
        # B: uint8_t
        # H: uint16_t
        # I: uint32_t
        # f: float (4 Byte)
        # d: double (8 Byte)
        # x: padding (1 Byte)
        # NOTE if anything changes here, the car firmware needs to be updated too
        header = pack('IIBBBB', self.seq_no, self.ts,
                      self.dest_addr, self.src_addr, self.type, self.subtype)
        padding_size = OffboardPacket.packet_size - \
            len(header) - len(self.payload)
        padding = pack('x'*padding_size)
        self.packet = header+self.payload+padding

        OffboardPacket.out_seq_no += 1

    def parse_packet(self):
        packet = self.packet
        header = packet[:12]
        self.seq_no, self.ts, self.dest_addr, self.src_addr, self.type, self.subtype = unpack(
            'IIBBBB', header)
        if self.type == 0:
            # ping packet
            if self.subtype == 0:
                # ping request
                pass
            elif self.subtype == 1:
                # ping response
                pass
        if self.type == 1:
            self.throttle, self.steering = unpack('ff', packet[12:20])

        # sensor update
        if self.type == 2:
            self.steering_requested, self.steering_measured = unpack(
                'ff', packet[12:20])
            # self.print_info('sensor update',self.steering_requested, self.steering_measured)

        # parameter
        if self.type == 3:
            if self.subtype == 0:
                sensor_update, steering_P, steering_I, steering_D = unpack(
                    '?fff', packet[12:12+4+3*4])
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
    ''' Subclass of Car to handle communication with Offboard Cars'''
    available_local_port = 58998

    def __init__(self, main):
        Car.__init__(self, main)

        # Network related attributes
        self.car_port = 2390
        ''' Network port on the car'''
        self.local_ip = '192.168.10.3'
        self.car_ip = None
        ''' To be set by parameters '''
        self.local_port = Offboard.available_local_port
        Offboard.available_local_port += 1
        self.sock = None
        self.last_sent_ts = 0
        ''' Timestamp for last packet sent, unit:us'''
        self.last_response_ts = 0
        ''' Timestamp for last packet received, unit:us'''

        self.child_threads = []
        self.ready = Event()
        ''' ready to take new command '''
        self.flag_quit = Event()
        self.out_queue = queue.Queue(maxsize=8)

        # Log related attributes
        self.log_t_vec = []
        self.steering_requested_vec = []
        self.steering_measured_vec = []

        # Car parameters
        self.params: CarParams | None = None
        self.optitrack_id: int = -1

    def init_param(self):
        ''' Parameter initialization, this will run immediately after self.params is set
        put all parameters here. '''

        # Rename some params
        self.car_ip = self.params.car_ip
        self.optitrack_id = self.params.optitrack_id

    def init_hardware(self):
        self.init_socket()

        # Create a separate thread for handling data packets
        comm_thread = Thread(target=self.__comm_thread_function, daemon=True)
        comm_thread.start()
        self.child_threads.append(comm_thread)
        self.setup()

    def init_socket(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # non-blocking
        sock.setblocking(0)
        sock.bind((self.local_ip, self.local_port))
        self.sock = sock

    def setup(self):
        ''' One-time packets to send for initialization '''
        # steering servo PID
        # old firmware
        # self.set_param(300.0,0,30)
        # new firmware
        # self.car.set_param(1.5,0,0.05)

    def __comm_thread_function(self):
        self.print_debug('comm trhead started')
        while not self.flag_quit.is_set():
            # send control command
            steering_cmd = self.steering * self.params.steer_ratio + self.params.steer_offset
            packet = self.prepare_command_packet(self.throttle, steering_cmd)
            packet.make_packet()
            try:
                select.select([], [self.sock], [])
                self.send_packet(packet)
            except BlockingIOError:
                self.print_warning('resource unavailable')

            # send all pending packets
            # packets in queue may be from another thread
            try:
                while True:
                    packet = self.out_queue.get_nowait()
                    # make_packet() fills in timestamp and seq_no, call right before send
                    packet.make_packet()
                    self.send_packet(packet)
            except queue.Empty:
                pass

            # read all packets from buffer
            try:
                # wait for at least one packet before sending new commands
                select.select([self.sock], [], [], 0.1)
                while True:
                    data, addr = self.sock.recvfrom(
                        OffboardPacket.packet_size)  # read 1 packet
                    if self.car_ip != addr[0]:
                        self.print_warning('Packet source ip != expected car ip'
                                           f'expected car_ip {self.car_ip}'
                                           f'actual {addr}')
                    if len(data) > 0:
                        assert len(data) == OffboardPacket.packet_size
                        self.parse_response(data)
                        # self.print_debug('got packet')
            except BlockingIOError:
                pass

            # ready to take new commands
            self.ready.set()
        self.print_debug('comm thread quit')

    def quit(self):
        self.throttle = 0.0
        self.steering = 0.0
        self.flag_quit.set()
        self.print_info('Quitting, waiting for threads to complete')
        for thread in self.child_threads:
            thread.join()
        self.print_info('Quit success')

    def get_param(self):
        packet = self.prepare_parameter_request_packet()
        self.out_queue.put_nowait(packet)

    def set_param(self, p, i, d):
        self.print_info('setting parameters')
        packet = OffboardPacket()
        packet.type = 3
        packet.subtype = 3
        packet.dest_addr = 1
        packet.src_addr = 0
        packet.payload = pack('?fff', True, p, i, d)
        self.out_queue.put_nowait(packet)
        return packet

    def send_packet(self, packet):
        sent_size = self.sock.sendto(
            packet.packet, (self.car_ip, self.car_port))
        self.print_debug('Sent packet of size %d', sent_size)
        self.last_sent_ts = packet.ts

    def parse_response(self, data):
        packet = OffboardPacket()
        packet.packet = data
        packet_type = packet.parse_packet()
        self.last_response_ts = int(time_ns() / 1000) % 4294967295

        # sensor update
        if packet_type == 2:
            self.log_t_vec.append(time())
            self.steering_requested_vec.append(packet.steering_requested)
            self.steering_measured_vec.append(packet.steering_measured)

        return packet

    def prepare_ping_packet(self):
        packet = OffboardPacket()
        packet.type = 0
        packet.subtype = 0
        packet.dest_addr = 1
        packet.src_addr = 0
        packet.empty_payload()
        return packet

    def prepare_command_packet(self, throttle=0.0, steering=0.0):
        packet = OffboardPacket()
        packet.type = 1
        packet.subtype = 5
        packet.dest_addr = 1
        packet.src_addr = 0
        packet.payload = pack('ff', throttle, steering)
        return packet

    def prepare_parameter_request_packet(self):
        packet = OffboardPacket()
        packet.type = 3
        packet.subtype = 2
        packet.dest_addr = 1
        packet.src_addr = 0
        packet.empty_payload()
        return packet
