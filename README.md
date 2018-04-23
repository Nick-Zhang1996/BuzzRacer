# Car Info

The car runs on an Odroid XU4 and an Arduino Nano

  * 16.04 Ubuntu Mate w/ ROS Kinetic
  * Username: odroid, Password: galanti5
  * Network: TP-Link_8DDA Static IP: 192.168.0.2

## Starting the car (Recommended workflow)

1. Run startup.sh, wait for startup sequence to finish (~15sec, until it says listening on /throttle and /steer_angle)

or, alternatively:

1. `roscore` (Start ros)
2. `rosrun uvc_camera uvc_camera_node _device:=/dev/video0` (Start camera)
    1. If video0 is not there, use `ls /dev/video*` and use the lowest one
3. `rosrun rosserial_python serial_node.py /dev/ttyUSB0`

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

## Setting up a new car
This is very incomplete for now.
"""
1) Setup Arduino software on the car.
2) apt-get packages: ros-kinetic-rosserial-arduino ros-kinetic-rosserial and many more
3) rosrun rosserial_arduino make_libraries.py ~/Arduino/libraries assuming Arduino's folder is in home
"""

## Connecting to the router
SSID: TP_Link_8DDA
password: 90040948
This information can also be found on the stickers underneath the router. Do NOT change these.
Odroid information:
odroid	74-DA-38-C7-F8-3D	192.168.0.2

## Core algorithm / functions

Nick: I'll write more detailed description over the summer, ask me if you have questions 

findCenterline() attempts to find a 2nd degree polynomial that describes the centerline/desired trajectory of the car. 
This polynomial transformed near the end of the function to a real-world coordinate system.(transform() performs a point to point transformation from projected points(image) to original(real-world)) The x axis coincides with the rear axle, pointing to the right. The y axis coincides with the vehicle's centerline, pointing forward.This is called a car reference frame, as it is stational with respect to the car. unit is in cm.

purePursuit() takes the polynomial and find on it a point that is one lookahead distance away from the origin. Lookahead is currently a constant value determined from experiments, although future work may be done to automatically adjust the lookahead distance on the flow. It then calculates the steering angle necessary to reach that point. To better understand pure pursuit algorithm, check the papers in the Google Drive. The one by CMU is particularly clear and friendly to beginners. There's also another article about derivation of pure pursuit that describes basic car dynamics. If you are reading that article keep in mind our L_fw is zero. 

Finally, calcSteer() calculate the actual value to send to the ros_topic to obtain the requested steering angle. 

You may notice that transform() and calcSteer() are super simple functions drawn out of thin air. These transformation relations are obtained by performing a linear regression analysis on measured data with MATLAB. Related .m files can be found on Google drive as well. 

In the very likely case that I forget to update this when you are reading my code. Ask me via email or groupme any question you may have before you give up trying to understand the code and curse me for lack of comments in source files. 

