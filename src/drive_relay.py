#!/usr/bin/env python

"""
drive_relay.py:
	Subscribe to /throttle and /steer messages
	and drive the car using these values
"""

import rospy
from std_msgs.msg import Float64 as float_msg
import serial
import time

# callbacks happen in separate threads. Use class to access data across threads
class CarState:
	def __init__(self):
		self.updated = False
		self.throttle = 0
		self.steer_angle = 0

cs = CarState()

def throttle_callback(data):
	cs.throttle = data.data
	cs.updated = True

def steer_callback(data):
	cs.steer_angle = data.data
	cs.updated = True

rospy.init_node('drive_relay')
rospy.Subscriber("/throttle", float_msg, throttle_callback)
rospy.Subscriber("/steer_angle", float_msg, steer_callback)

ser = serial.Serial("/dev/arduino", 9600)
#read #ready from Serial
ser.readline()

rate = rospy.Rate(20)
while not rospy.is_shutdown():
	#When the car states changes, send changes to Arduino
	if cs.updated:
		cs.updated = False
		send_str = "$%.3f,%.3f\n" % (cs.throttle, cs.steer_angle)
		#Print and send data to arduino
		print send_str.replace("\n","")
		ser.write(send_str)

		#Receive confirmation from arduino and print it
		in_line = ser.readline().replace("\r\n","")
		data_rcv = in_line
		print data_rcv

	rate.sleep()

ser.close()

