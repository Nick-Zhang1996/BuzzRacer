# Car Info

The car runs on an Odroid XU4 and an Arduino Nano

  * 16.04 Ubuntu Mate w/ ROS Kinetic
  * Username: odroid, Password: galanti5
  * Using Mayhul’s GTWifi login
  * IP address: https://autorc.herokuapp.com/
    * Whenever the odroid starts up and connects to the wifi, the IP address and timestamp is published to that website 

## Building
Run `catkin_make` from ~/catkin_ws

## Starting the car

1. `roscore` (Start ros)
2. `rosrun RC-VIP drive_relay.py` (Node that communicates to arduino)
3. `rosrun uvc_camera uvc_camera_node _device:=/dev/video0` (Start camera)
    1. If video0 is not there, use ls /dev/video* and use the lowest one
4. Use `rqt` -> Plugins -> Topics -> Message Publisher to change values manually

## Important Files
  * drive_relay.py -  Reads from /throttle and /streer_angle and sends to Arduino
  * image_processor.py - reads from /image_raw and converts to OpenCV image 
  * drive.ino - reads from Serial and sends PWM signal to ESC/servo (runs on Arduino)

## ROS Topic APIs
  * /throttle - Accepts values between -1 (reverse), 0 (stopped) and 1 (forward)
  * /steer_angle - Accepts values between -1 (left), 0 (straight) and 1 (right)
  * /image_raw - Raw images from camera

## Arduino Details
  * Servo PWM goes from 60 (right) to 150 (left)
  * Pin 3 connected to ESC, Pin 5 connected to Serivo





	

