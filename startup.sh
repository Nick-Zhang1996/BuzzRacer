#!/bin/sh

roscore & 1>>./log/roscore.stdout 2>>./log/roscore.stderr
sleep 5
rosrun uvc_camera uvc_camera_node _device:=/dev/video0 & 1>>./log/camera.stdout 2>>./log/camera.stderr

rosrun rosserial_python serial_node.py /dev/ttyUSB0 & 1>>./log/drive_relay.stdout 2>>./log/drive_relay.stderr
# put your code in here
