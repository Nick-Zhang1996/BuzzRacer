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
const int minThrottleVal = 1500;

static int throttleServoVal = 1500;
static int steeringServoVal = 1550;

static unsigned long throttleTimestamp = 0;
static unsigned long steeringTimestamp = 0;

//ros variables
ros::NodeHandle nh;

void readThrottleTopic(const std_msgs::Float64 &msg_throttle) {
    throttleTimestamp = millis();
    if (msg_throttle.data < 0.05) {
        throttleServoVal = minThrottleVal;
    } else if (msg_throttle.data > 1.01) {
        throttleServoVal = minThrottleVal;
    } else {
        throttleServoVal = (int) map( msg_throttle.data, 0.0, 1.0, 1460, 1450);
    }
    // COMMENT THIS OUT for moving motor
    //throttleServoVal = minThrottleVal;
    return;
}

void readSteeringTopic(const std_msgs::Float64 &msg_steerAngle) {
    steeringTimestamp = millis();
    float tempSteering = constrain(msg_steerAngle.data, steeringLeftLimit, steeringRightLimit);
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
    throttle.writeMicroseconds(minThrottleVal);
    delay(5000);
}

void loop() {
    nh.spinOnce();
    if ( millis() - throttleTimestamp > 500 || millis() - steeringTimestamp > 500 ){
        throttle.writeMicroseconds(minThrottleVal);
    }   


    
       
    throttle.writeMicroseconds(throttleServoVal);
    steer.writeMicroseconds(steeringServoVal);
    delay(10);
}
