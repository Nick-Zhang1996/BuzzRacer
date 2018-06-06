#include <Arduino.h>
#include <ros.h>
#include <std_msgs/Float64.h>
#include <Servo.h>

//throttle variables
const int pinServo = 3;
const int pinDrive = 5;
Servo throttle;
Servo steer;

// values are in us (microseconds)
const float steeringRightLimit = 30.0;
const float steeringLeftLimit = -30.0;
int throttleServoVal = 1500;
int steeringServoVal = 1550;

//ros variables
ros::NodeHandle nh;

void readThrottleTopic(const std_msgs::Float64 &throttleVal) {
    if (throttleVal.data < 0.05) {
        throttleServoVal = 1500;
    } else if (throttleVal.data > 1.01) {
        throttleServoVal = 1500;
    } else {
        throttleServoVal = (int) map( throttleVal.data, 0.0, 1.0, 1460, 1450);
    }
    // COMMENT THIS OUT for moving motor
    throttleServoVal = 1500;
    return;
}

void readSteeringTopic(const std_msgs::Float64 &steeringVal) {
    float tempSteering = constrain(steeringVal.data, steeringLeftLimit, steeringRightLimit);
    if ( tempSteering > 0.0 ){
        steeringServoVal = (int) map(tempSteering, 0.0, steeringRightLimit, 1550, 1900);
    } else if ( tempSteering < 0.0 ){
        steeringServoVal = (int) map(tempSteering, 0.0, steeringLeftLimit, 1550, 1150);
    } else {
        steeringServoVal = 1550;
    }

    return;
}



ros::Subscriber<std_msgs::Float64> subThrottle("throttle", &readThrottleTopic);

ros::Subscriber<std_msgs::Float64> subSteering("steer_angle", &readSteeringTopic);

void setup() {
    Serial.begin(9600);
    nh.initNode();
    nh.subscribe(subThrottle);
    nh.subscribe(subSteering);

    pinMode(pinDrive, OUTPUT);
    pinMode(pinServo, OUTPUT);
    throttle.attach(pinDrive);
    steer.attach(pinServo);
   
    //setup ESC
    throttle.writeMicroseconds(1500);
    delay(5000);
}

void loop() {
    nh.spinOnce();
       
    throttle.writeMicroseconds(throttleServoVal);
    steer.writeMicroseconds(steeringServoVal);
    //Serial.println("Hello");
    delay(10);
}
