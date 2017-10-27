#!/usr/bin/env python

"""
drive_relay.py:
	Subscribe to /speed and /steer messages
	and drive the car using these values
"""

import rospy
from std_msgs.msg import Float64 as float_msg

# callbacks happen in separate threads. Use class to access data across threads
class CarState:
	def __init__(self):
		self.updated = False
		self.speed = 0
		self.steer = 0

cs = CarState()

def speed_callback(data):
	cs.speed = data.data
	cs.updated = True

def steer_callback(data):
	cs.steer = data.data
	cs.updated = True

rospy.init_node('drive_relay')
rospy.Subscriber("/speed", float_msg, speed_callback)
rospy.Subscriber("/steer", float_msg, steer_callback)

rate = rospy.Rate(20)
while not rospy.is_shutdown():
	if cs.updated:
		print "updated speed = %.2f, steer = %.2f" % (cs.speed, cs.steer)
		cs.updated = False

	# TODO add pwm things to drive car here

	rate.sleep()

