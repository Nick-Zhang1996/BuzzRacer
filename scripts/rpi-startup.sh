#!/bin/bash

source /home/ubuntu/catkin_ws/install/setup.bash

rosrun rosserial_python serial_node.py _port:=/dev/ttyAMA0 _baud:=57600 &
roslaunch raspicam_node camerav1_1280x720.launch &
