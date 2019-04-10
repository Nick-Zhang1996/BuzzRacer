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
import time

import rospy
from rc_vip.msg import CarSensors, CameraPerception
from geometry_msgs.msg import PoseStamped

from particle_filter import CameraModel, Track, ParticleFilter


imu_offset = 0
imu_scale = 1024

odometry_updates = []

def odometry_callback(msg):
    odometry_updates.append((
        msg.header.stamp.to_sec(),
        (float(msg.imu_gz) - imu_offset) / imu_scale,

    ))

def perception_callback(msg):
    # each feature is {x:int, y:int}
    callback_start = time.time()

    closest_feature = max(msg.features, key=attrgetter('y'))
    obs = (closest_feature.x, closest_feature.y)




if __name__ == '__main__':
    rospy.init_node("particle_filter")

    perception_topic = rospy.get_param("~perception_topic")
    sensors_topic = rospy.get_param("~car_sensors_topic")
    filter_latency = rospy.get_param("~filter_latency")
    track_string = rospy.get_param("~track_string")

    filt = ParticleFilter(1000, noise_vec, track, camera, target_latency=0.02)

    perception_sub = rospy.Subscriber(perception_topic, CameraPerception, queue_size=0)
    sensors_sub = rospy.Subscriber(sensors_topic, CarSensors, queue_size=1)
