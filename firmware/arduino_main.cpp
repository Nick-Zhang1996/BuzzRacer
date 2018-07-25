// Arduino firmware for RC-VIP
//
//  Updates:
// // June 2018: merged throttle and steering topic
// // June 2018: added IMU support
// // June 2018: added battery voltage sensor(30k/7.5k resistor voltage divider)
//
//
//
// // MPU6050 code from Jeff Rowberg <jeff@rowberg.net>
// // Updates should (hopefully) always be available at https://github.com/jrowberg/i2cdevlib
// //
// //
// //
// /* ============================================
// I2Cdev device library code is placed under the MIT license
// Copyright (c) 2012 Jeff Rowberg
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
// ===============================================
// */
//
#include <Arduino.h>
#include <ros.h>
#include <std_msgs/Float64.h>
#include <rc_vip/CarSensors.h>
#include <rc_vip/CarControl.h>
#include <Servo.h>

// I2Cdev and MPU6050 must be installed as libraries, or else the .cpp/.h files
// for both classes must be in the include path of your project
#include "I2Cdev.h"
#include "MPU6050.h"

// Arduino Wire library is required if I2Cdev I2CDEV_ARDUINO_WIRE implementation
// is used in I2Cdev.h
#if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
    #include "Wire.h"
#endif

#define LED_PIN 13
#define VOLTAGEDIVIDER_PIN A3
// class default I2C address is 0x68
// specific I2C addresses may be passed as a parameter here
// AD0 low = 0x68 (default for InvenSense evaluation board)
// AD0 high = 0x69
MPU6050 accelgyro;
//MPU6050 accelgyro(0x69); // <-- use for AD0 high

int16_t ax, ay, az;
int16_t gx, gy, gz;


//throttle variables
const int pinServo = 4;
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
unsigned long voltageUpdateTimestamp = 0;
bool newCarControlMsg = false;

//ros variables
//pub,sub, in_buf, out_buf
ros::NodeHandle_<ArduinoHardware, 2, 2, 128, 300 > nh;

void readCarControlTopic(const rc_vip::CarControl& msg_CarControl) {
    carControlTimestamp = millis();
    newCarControlMsg = true;

    if (msg_CarControl.throttle < 0.001) {
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


ros::Subscriber<rc_vip::CarControl> subCarControl("rc_vip/CarControl", &readCarControlTopic);
rc_vip::CarSensors carSensors_msg;
ros::Publisher pubCarSensors("rc_vip/CarSensors", &carSensors_msg);

void setup() {
    pinMode(LED_PIN, OUTPUT);
    pinMode(VOLTAGEDIVIDER_PIN, INPUT);
    digitalWrite(LED_PIN, LOW);
    // join I2C bus (I2Cdev library doesn't do this automatically)
    #if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
        Wire.begin();
    #elif I2CDEV_IMPLEMENTATION == I2CDEV_BUILTIN_FASTWIRE
        Fastwire::setup(400, true);
    #endif

    accelgyro.initialize();
    digitalWrite(LED_PIN, accelgyro.testConnection());


    nh.initNode();
    nh.advertise(pubCarSensors);
    nh.subscribe(subCarControl);

    while (!nh.connected())
        nh.spinOnce();

    pinMode(pinDrive, OUTPUT);
    pinMode(pinServo, OUTPUT);
    throttle.attach(pinDrive);
    steer.attach(pinServo);
   
    //ESC requires a low signal durin poweron to prevent accidental input
    throttle.writeMicroseconds(minThrottleVal);
    delay(300);
}

void loop() {

    //failsafe, if there's no new message for over 500ms, halt the motor
    if ( millis() - carControlTimestamp > 500 ){
        throttle.writeMicroseconds(minThrottleVal);
    } else {  
        if (newCarControlMsg){
            newCarControlMsg = false;
            throttle.writeMicroseconds(throttleServoVal);
            steer.writeMicroseconds(steeringServoVal);
        }
    }

    if ( millis()-voltageUpdateTimestamp>100 ){
        float voltage = (float)analogRead(VOLTAGEDIVIDER_PIN);
        voltage /= 16.27;
        carSensors_msg.voltage = voltage;
    }


    // read raw accel/gyro measurements from device
    // XXX this needs offsetting and scaling
    accelgyro.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    carSensors_msg.imu_ax = ax;
    carSensors_msg.imu_ay = ay;
    carSensors_msg.imu_az = az;

    carSensors_msg.imu_gx = gx;
    carSensors_msg.imu_gy = gy;
    carSensors_msg.imu_gz = gz;

    // TODO maybe add some throttling stuff?
    pubCarSensors.publish(&carSensors_msg);
    nh.spinOnce();

    // a loop rate too high may mess with Servo class's operation
    delay(10);
}
