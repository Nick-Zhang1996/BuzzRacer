#!/usr/bin/env python

"""
driveInCircle.py:
	Publish to /throttle and /steer topic
	and drive the car in a circle
"""

import rospy
from std_msgs.msg import Float64 as float_msg

pubThrottle = rospy.Publisher("/throttle", float_msg, queue_size = 3)
pubAngle = rospy.Publisher("/steer_angle", float_msg, queue_size = 3)
rospy.init_node('driveInCircle')

pubThrottle.publish(0.3)
pubAngle.publish(0.7)
