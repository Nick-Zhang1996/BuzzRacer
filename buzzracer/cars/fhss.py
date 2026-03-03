''' Subclass of Car for radio-controlled Mini-z'''
from __future__ import annotations
from typing import TYPE_CHECKING

import logging
import struct
from threading import Thread
from time import sleep

import serial

from buzzracer.cars.car import Car
if TYPE_CHECKING:
    from buzzracer.scripts.run import MainState


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@Car.register
class FHSS(Car):
    car_count = 0
    cars:FHSS = []
    serial_port = None
    pwm_values = [1500] * 12 # 6 cars, 2 val each (steering, throttle)
    frame_header = bytes([0xAA, 0x55])

    def __init__(self):
        Car.__init__(self)
        self.index = FHSS.car_count
        FHSS.car_count += 1
        FHSS.cars.append(self)
        FHSS.child_threads = []

    def init(self):
        # All FHSS cars share a hardware interface, calling init() once suffices
        if FHSS.serial_port is None:
            serial_port = '/dev/ttyUSB0'
            try:
                FHSS.serial_port = serial.Serial(
                    serial_port, 115200, timeout=0.001, writeTimeout=0)
            except (FileNotFoundError, serial.serialutil.SerialException):
                logger.error('Interface {} not found'%(serial_port))
                raise
            # Create a separate thread for handling data packets
            comm_thread = Thread(target=self.__comm_thread_function, daemon=True)
            comm_thread.start()
            FHSS.child_threads.append(comm_thread)


    def actuate(self):
        # Car.actuate(self)
        steering_pwm = int(self.mapdata(self.steering,
                                        self.param.max_steer_left,
                                        -self.param.max_steer_right,
                                        self.param.max_steer_pwm_left,
                                        self.param.max_steer_pwm_right))
        throttle_pwm = int(self.mapdata(self.throttle, -1.0, 1.0, 1100, 1900))
        FHSS.pwm_values[2*self.param.fhss_modem_id] = steering_pwm
        FHSS.pwm_values[2*self.param.fhss_modem_id + 1] = throttle_pwm


    @classmethod
    def send_pwm_array(cls) -> bool:
        if len(FHSS.pwm_values) != 12:
            raise ValueError("PWM array must contain exactly 12 elements")
            
        try:
            # Pack 10 unsigned 16-bit integers (Little-Endian)
            # Result is exactly 20 bytes
            payload = struct.pack('<12H', *FHSS.pwm_values)
            
            # Calculate CRC over the payload
            crc = FHSS.calculate_crc8(payload)
            # print(f'payload {payload} crc: {hex(crc)}')
            
            # Construct the final 27-byte frame
            frame = bytearray(FHSS.frame_header)
            frame.extend(payload)
            frame.append(crc)
            
            count = FHSS.serial_port.write(frame)
            return count == 27
            
        except serial.SerialException as e:
            print(f"Serial write error: {e}")
            return False


    def mapdata(self, x, a, b, c, d):
        y = (x-a)/(b-a)*(d-c)+c
        return int(y)

    @classmethod
    def calculate_crc8(cls, data: bytes) -> int:
        """Calculates CRC-8-CCITT (Poly: 0x07) for a given byte array."""
        crc = 0x00
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x80:
                    crc = (crc << 1) ^ 0x07
                else:
                    crc <<= 1
                crc &= 0xFF
        return crc

    @classmethod
    def read_serial_monitor(cls):
        """
        Reads any waiting bytes from the Arduino and prints them to the console.
        Non-blocking: returns immediately if there's nothing to read.
        """
        if FHSS.serial_port.in_waiting > 0:
            try:
                # Read everything sitting in the OS buffer
                raw_bytes = FHSS.serial_port.read(FHSS.serial_port.in_waiting)
                
                # Decode as ASCII. We use errors='replace' so that if a random 
                # corrupted byte or binary artifact comes through, it prints a '?' 
                # instead of crashing the Python script with a UnicodeDecodeError.
                text = raw_bytes.decode('ascii', errors='replace')
                
                # Print without adding an extra newline, since Arduino's println 
                # already sends \r\n
                # print(text, end='', flush=True)
                logger.info(text)
                
            except serial.SerialException as e:
                logger.info("\n[Serial Read Error]: %s"%{e})

    def __comm_thread_function(self):
        while not Car.main.state.exit_request.is_set():
            for car in FHSS.cars:
                car.actuate()
            FHSS.send_pwm_array()
            sleep(0.01)
            FHSS.read_serial_monitor()
