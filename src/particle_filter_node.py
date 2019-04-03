"""
filter.py
ROS node to filter the vehicle's position using a particle filter.

Inputs:
- car speed, m/s  (from wheel speed sensor)
- yaw rate, rad/s  (from IMU)
- observed features  (from perception layer)
- track description: list of {left, right, straight} segments where each of those segments
    has a predefined size and shape

Output:
- robot pose (x, y, heading) in track coordinate frame
"""

import rospy
# from rc_vip.msg import CarSensors
from geometry_msgs.msg import PoseStamped

# todo implementation
