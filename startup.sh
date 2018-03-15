#!/bin/sh

roscore & 1>>./log/roscore.stdout 2>>./log/roscore.stderr
rosrun RC-VIP drive_relay.py & 1>>./log/drive_relay.stdout 2>>./log/drive_relay.stderr
rosrun uvc_camera uvc_camera_node _device:=/dev/video0 & 1>>./log/camera.stdout 2>>./log/camera.stderr

# put your code in here
