#include <Arduino.h>
#include <ros.h>
#include <std_msgs/Float64.h>
#include <rc_vip/CarSensors.h>
#include <rc_vip/CarControl.h>
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

unsigned long carControlTimestamp = 0;

//ros variables
ros::NodeHandle nh;

void readCarControlTopic(const rc_vip::CarControl& msg_CarControl) {
    carControlTimestamp = millis();

    if (msg_CarControl.throttle < 0.005) {
        throttleServoVal = minThrottleVal;
    } else if (msg_CarControl.throttle > 1.0001) {
        throttleServoVal = minThrottleVal;
    } else {
        //needs a re-calibration
        throttleServoVal = (int) map( msg_CarControl.throttle, 0.0, 1.0, 1460, 1450);
    }
    // COMMENT THIS OUT for moving motor
    //throttleServoVal = minThrottleVal;

    float tempSteering = constrain(msg_CarControl.steer_angle, steeringLeftLimit, steeringRightLimit);
    if ( tempSteering > 0.0 ){
        steeringServoVal = (int) map(tempSteering, 0.0, steeringRightLimit, 1550, 1900);
    } else if ( tempSteering < 0.0 ){
        steeringServoVal = (int) map(tempSteering, 0.0, steeringLeftLimit, 1550, 1150);
    } else {
        steeringServoVal = 1550;
    }

    return;
}


ros::Subscriber<rc_vip::CarControl> subCarControl("/carControl", &readCarControlTopic);

void setup() {
    Serial.begin(115200);
    nh.initNode();
    nh.subscribe(subCarControl);

    pinMode(pinDrive, OUTPUT);
    pinMode(pinServo, OUTPUT);
    throttle.attach(pinDrive);
    steer.attach(pinServo);
   
    //ESC requires a low signal durin poweron to prevent accidental input
    throttle.writeMicroseconds(minThrottleVal);
    delay(5000);
}

void loop() {
    nh.spinOnce();
    //failsafe, if there's no new message for over 500ms, halt the motor
    if ( millis() - carControlTimestamp > 500 ){
        throttle.writeMicroseconds(minThrottleVal);
    } else {  
        throttle.writeMicroseconds(throttleServoVal);
        steer.writeMicroseconds(steeringServoVal);
    }

    // a loop rate too high may mess with Servo class's operation
    delay(10);
}
