#include <Arduino.h>
#include <ros.h>
#include <std_msgs/Float64.h>
#include <Server.h>

//drive variables
const int drivePin = 3;
const int servoPin = 5;
Servo drive;
Servo steer;
const int steeringRightLimit = 60;
const int steeringLeftLimit = 150;
const int steeringCenter = (steeringRightLimit + steeringLeftLimit) / 2;
int servoThrottleVal = 0;
int servoSteeringVal = steeringCenter; 

//ros variables



ros::NodeHandle nh;

void readThrottleTopic(const std_msgs::Float64 &throttleVal) {
}

ros::Subscriber<std_msgs::Float64> subscriber("throttle", &readThrottleTopic);

void setup() {
    nh.initNode();
    nh.subscribe(subscriber);
}

void loop() {
    nh.spinOnce();
    delay(1);
}
