#!/bin/bash

roscore & 1>>./log/roscore.stdout 2>>./log/roscore.stderr
sleep 3
rosrun uvc_camera uvc_camera_node _device:=/dev/video0 & 1>>./log/camera.stdout 2>>./log/camera.stderr
source /home/odroid/catkin_ws/install/setup.bash

rosrun rosserial_python serial_node.py /dev/ttyUSB0 & 1>>./log/drive_relay.stdout 2>>./log/drive_relay.stderr
sleep 2

python /home/odroid/carkin_ws/src/rc-vip/src/onelane.py
# put your code in here
