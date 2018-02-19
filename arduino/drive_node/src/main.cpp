#include <Arduino.h>
#include <ros.h>
#include <std_msgs/Float64.h>
#include <Servo.h>

//drive variables
const int pinDrive = 3;
const int pinServo = 5;
Servo drive;
Servo steer;
const int steeringRightLimit = 60;
const int steeringLeftLimit = 150;
const int steeringCenter = (steeringRightLimit + steeringLeftLimit) / 2;
int throttleServoVal = 0;
int steeringServoVal = steeringCenter; 

//ros variables
ros::NodeHandle nh;

void readThrottleTopic(const std_msgs::Float64 &throttleVal) {
    int tempThrottle = (int) ((throttleVal.data)*180.0*.3);
    if ((tempThrottle < 0) || (tempThrottle > 180)) {
        throttleServoVal = 0;
    } else {
        throttleServoVal = tempThrottle;
    }
}

void readSteeringTopic(const std_msgs::Float64 &steeringVal) {
    int tempSteering = (int) ((steeringVal.data) * ((steeringRightLimit - steeringLeftLimit)/2.0) + steeringCenter);
    if (tempSteering < steeringRightLimit || tempSteering > steeringLeftLimit) {
        steeringServoVal = steeringCenter;
    } else {
        steeringServoVal = tempSteering;
    }
}

ros::Subscriber<std_msgs::Float64> subThrottle("throttle", &readThrottleTopic);

ros::Subscriber<std_msgs::Float64> subSteering("steering", &readSteeringTopic);

void setup() {
    nh.initNode();
    nh.subscribe(subThrottle);
    nh.subscribe(subSteering);

    pinMode(pinDrive, OUTPUT);
    pinMode(pinServo, OUTPUT);
    drive.attach(pinDrive);
    steer.attach(pinServo);
   
    //setup ESC
    drive.write(0);
    delay(5000);
}

void loop() {
    nh.spinOnce();
       
    drive.write(throttleServoVal);
    steer.write(steeringServoVal);
    delay(10);
}
