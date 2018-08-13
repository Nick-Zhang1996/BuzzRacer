# RC-VIP project

Some of this README may be outdated, refer to wiki for updated info.

# Car Info (Odroid XU4 based)

  * 16.04 Ubuntu Mate w/ ROS Kinetic
  * Odroid XU4 Username: odroid, Password: galanti5
  * Network: TP-Link_8DDA Static IP: 192.168.0.2

# Car Info (RPi 3B based)

  * Custom OS from ROS: ubiquityrobots
  * Username: ubuntu, Password: ubuntu
  * hostname: ubiquityrobots

## Starting the car for XU4 (Recommended workflow)

1. Run startup.sh, wait for startup sequence to finish (~15sec, until it says listening on /throttle and /steer_angle)

or, alternatively:

1. `roscore` (Start ros)
2. `rosrun uvc_camera uvc_camera_node _device:=/dev/video0` (Start camera)
    1. If video0 is not there, use `ls /dev/video*` and use the lowest one
3. `rosrun rosserial_python serial_node.py /dev/ttyUSB0`

## Start a RPI 3B based car

1. `cd ~/carkin_ws/src/rc_vip`
2. `./rpi-startup`
3. In case `./rpi-startup` does not already include this. `python src/onelane.py`

## Core Files
    * `onelane.py`
    * Startup scripts
    

## ROS Topics used
  * /rc_vip/CarControl - Throttle (range:0~1) and Steering angle (in degrees)
  * /rc_vip/CarSensors - Sensors from car, IMU, battery voltage, etc.
  * /image_raw or /raspicam_node/image/compressed - video feed from camera

## Connecting to the router in MK101
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

