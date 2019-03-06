#!/bin/bash

source /home/ubuntu/catkin_ws/install/setup.bash
roslaunch raspicam_node camerav1_1280x720.launch& 1>>./log/drive_relay.stdout 2>>./log/drive_relay.stderr

rosrun rosserial_python serial_node.py _port:=/dev/ttyAMA0 _baud:=57600 & 1>>./log/drive_relay.stdout 2>>./log/drive_relay.stderr
sleep 2

python /home/ubuntu/catkin_ws/src/rc_vip/src/onelane.py
# put your code in here
