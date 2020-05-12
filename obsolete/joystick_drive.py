#!/usr/bin/env python

"""
joystick_drive.py:
	Publish to /throttle and /steer values
	from the joystick controller.
"""

import rospy
from sensor_msgs.msg import Joy
from std_msgs.msg import Float64 as float_64

trigger_right_unclicked = True
trigger_left_unclicked = True

def joy_callback(data):
	axes = data.axes
	buttons = data.buttons
	
	# publish steering
	steering_val = -1 * axes[0]
	steering_angle_pub.publish(float_64(steering_val))

	# publish throttle
	throttle_val = axes[5]
	if trigger_right_unclicked and throttle_val == 0:
		if throttle_val == 0:
			throttle_val = 1 # this is set to one, because joystick defines all the way up on right trigger as 1
		else:
			throttle_right_unclicked = False
	throttle_val = (throttle_val - 1) / -2	
	throttle_pub.publish(float_64(throttle_val))

rospy.init_node('joy_drive')
throttle_pub = rospy.Publisher("/throttle", float_64, queue_size=1)
steering_angle_pub = rospy.Publisher("/steering_angle", float_64, queue_size=1)
rospy.Subscriber("/joy", Joy, joy_callback)
rospy.spin()
