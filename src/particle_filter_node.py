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
- robot pose in track coordinate frame
"""

from operator import attrgetter

import rospy
from rc_vip.msg import CarSensors, CameraPerception
from geometry_msgs.msg import PoseStamped

import particle_filter


def perception_callback(msg):
    # each feature is {x:int, y:int}


if __name__ == '__main__':
    rospy.init_node("particle_filter_node")

    perception_topic = rospy.get_param("~perception_topic")
    sensors_topic = rospy.get_param("~car_sensors_topic")
    filter_latency = rospy.get_param("~filter_latency")

    perception_sub = rospy.Subscriber(perception_topic, CameraPerception, queue_size=1)
    sensors_sub = rospy.Subscriber(sensors_topic, CarSensors, queue_size=1)
