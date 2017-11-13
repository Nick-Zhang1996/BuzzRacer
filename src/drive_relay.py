#!/usr/bin/env python

"""
drive_relay.py:
	Subscribe to /throttle and /steer messages
	and drive the car using these values
"""

import rospy
from std_msgs.msg import Float64 as float_msg
import serial

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

rate = rospy.Rate(20)
while not rospy.is_shutdown():
	if cs.updated:
		print "updated throttle = %.2f, steer = %.2f" % (cs.throttle, cs.steer_angle)
		cs.updated = False

	ser.write("$%d\n" % cs.throttle)
	new_line_from_arduino = ser.readline()
	print new_line_from_arduino	

	rate.sleep()

ser.close()

