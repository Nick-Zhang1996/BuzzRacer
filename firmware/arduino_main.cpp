// Arduino firmware for RC-VIP. Initially written by Binit and Katie
// June 2018: merged throttle and steering topic
// June 2018: added IMU support
// MPU6050 code from Jeff Rowberg <jeff@rowberg.net>
// Updates should (hopefully) always be available at https://github.com/jrowberg/i2cdevlib
//
//
//
/* ============================================
I2Cdev device library code is placed under the MIT license
Copyright (c) 2012 Jeff Rowberg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
===============================================
*/



#include <Arduino.h>
#include <ros.h>
#include <std_msgs/Float64.h>
#include <rc_vip/CarSensors.h>
#include <rc_vip/CarControl.h>
#include <Servo.h>

// I2Cdev and MPU6050 must be installed as libraries, or else the .cpp/.h files
// for both classes must be in the include path of your project
#include "I2Cdev.h"

#include "MPU6050_6Axis_MotionApps20.h"
//#include "MPU6050.h" // not necessary if using MotionApps include file

// Arduino Wire library is required if I2Cdev I2CDEV_ARDUINO_WIRE implementation
// is used in I2Cdev.h
#if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
    #include "Wire.h"
#endif

// class default I2C address is 0x68
// specific I2C addresses may be passed as a parameter here
// AD0 low = 0x68 (default for SparkFun breakout and InvenSense evaluation board)
// AD0 high = 0x69
MPU6050 mpu;
//MPU6050 mpu(0x69); // <-- use for AD0 high


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
bool newCarControlmsg = false;

#define LED_PIN 13 // (Arduino is 13, Teensy is 11, Teensy++ is 6)
bool blinkState = false;

// MPU control/status vars
bool dmpReady = false;  // set true if DMP init was successful
uint8_t mpuIntStatus;   // holds actual interrupt status byte from MPU
uint8_t devStatus;      // return status after each device operation (0 = success, !0 = error)
uint16_t packetSize;    // expected DMP packet size (default is 42 bytes)
uint16_t fifoCount;     // count of all bytes currently in FIFO
uint8_t fifoBuffer[64]; // FIFO storage buffer

// orientation/motion vars
Quaternion q;           // [w, x, y, z]         quaternion container
VectorInt16 aa;         // [x, y, z]            accel sensor measurements
VectorInt16 aaReal;     // [x, y, z]            gravity-free accel sensor measurements
VectorInt16 aaWorld;    // [x, y, z]            world-frame accel sensor measurements
VectorFloat gravity;    // [x, y, z]            gravity vector
float euler[3];         // [psi, theta, phi]    Euler angle container
float ypr[3];           // [yaw, pitch, roll]   yaw/pitch/roll container and gravity vector

// packet structure for InvenSense teapot demo
uint8_t teapotPacket[14] = { '$', 0x02, 0,0, 0,0, 0,0, 0,0, 0x00, 0x00, '\r', '\n' };



// ================================================================
// ===               INTERRUPT DETECTION ROUTINE                ===
// ================================================================

volatile bool mpuInterrupt = false;     // indicates whether MPU interrupt pin has gone high
void dmpDataReady() {
    mpuInterrupt = true;
}
//ros variables
ros::NodeHandle nh;

void readCarControlTopic(const rc_vip::CarControl& msg_CarControl) {
    carControlTimestamp = millis();
    newCarControlmsg = true;

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


ros::Subscriber<rc_vip::CarControl> *subCarControl;
//rc_vip::CarSensors carSensors_msg;
//ros::Publisher pubCarSensors("rc_vip/CarSensors", &carSensors_msg, 1);

void setup() {

    // configure LED for output
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN,LOW);

    // join I2C bus (I2Cdev library doesn't do this automatically)
    #if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
        Wire.begin();
        TWBR = 24; // 400kHz I2C clock (200kHz if CPU is 8MHz)
    #elif I2CDEV_IMPLEMENTATION == I2CDEV_BUILTIN_FASTWIRE
        Fastwire::setup(400, true);
    #endif
    //Serial.begin(115200);
    //while(!Serial);

    // initialize device
    //Serial.println(F("Initializing I2C devices..."));
    //NOTE: this function would block if MPU6050 is not connected
    mpu.initialize();

    // verify connection
    //Serial.println("Testing device connections...");
    if (mpu.testConnection()) {
        //Serial.println("MPU6050 connection successful");
        ;
    } else {
        //Serial.println("MPU6050 connection failed");
        ;
    }


    // wait for ready
    //Serial.println(F("\nSend any character to begin DMP programming and demo: "));
    //while (Serial.available() && Serial.read()); // empty buffer
    //while (!Serial.available());                 // wait for data
    //while (Serial.available() && Serial.read()); // empty buffer again

    // load and configure the DMP
    //Serial.println("Initializing DMP...");
    devStatus = mpu.dmpInitialize();

    // NOTE supply your own gyro offsets here, scaled for min sensitivity
    mpu.setXGyroOffset(220);
    mpu.setYGyroOffset(76);
    mpu.setZGyroOffset(-85);
    mpu.setZAccelOffset(1788); // 1688 factory default for my test chip

    // make sure it worked (returns 0 if so)
    if (devStatus == 0) {

        // turn on the DMP, now that it's ready
        //Serial.println("Enabling DMP...");
        mpu.setDMPEnabled(true);

        // enable Arduino interrupt detection
        //Serial.println("Enabling interrupt detection (Arduino external interrupt 0...)");
        attachInterrupt(0, dmpDataReady, RISING);
        mpuIntStatus = mpu.getIntStatus();

        // set our DMP Ready flag so the main loop() function knows it's okay to use it
        //Serial.println("DMP ready! Waiting for first interrupt...");
        dmpReady = true;

        // get expected DMP packet size for later comparison
        packetSize = mpu.dmpGetFIFOPacketSize();
    } else {
        // ERROR!
        // 1 = initial memory load failed
        // 2 = DMP configuration updates failed
        // (if it's going to break, usually the code will be 1)
        //Serial.println("DMP Initialization failed (code ");

        //Serial.println(")");
        ;
    }

    //nh.initNode();
    //nh.subscribe(subCarControl);


    //pinMode(pinDrive, OUTPUT);
    //pinMode(pinServo, OUTPUT);
    //throttle.attach(pinDrive);
    //steer.attach(pinServo);

    ////ESC requires a low signal durin poweron to prevent accidental input
    //throttle.writeMicroseconds(minThrottleVal);
    //delay(2000);

   
    // if programming failed, don't try to do anything
    if (!dmpReady) {
        //Serial.println("MPU6050 DMP unavailable");
        ;
    } else {
        digitalWrite(LED_PIN,HIGH);
        //subCarControl = new ros::Subscriber<rc_vip::CarControl>("rc_vip/CarControl", &readCarControlTopic);
        
    }
        
    while(true);
}

void loop() {
    while(true);
    //nh.spinOnce();

    //if (mpuInterrupt || fifoCount>packetSize){
    if (false){

        // reset interrupt flag and get INT_STATUS byte
        mpuInterrupt = false;
        mpuIntStatus = mpu.getIntStatus();

        // get current FIFO count
        fifoCount = mpu.getFIFOCount();

        // check for overflow (this should never happen unless our code is too inefficient)
        if ((mpuIntStatus & 0x10) || fifoCount == 1024) {
            // reset so we can continue cleanly
            mpu.resetFIFO();
            //nh.logerror("FIFO overflow!");

        // otherwise, check for DMP data ready interrupt (this should happen frequently)
        } else if (mpuIntStatus & 0x02) {
            // wait for correct available data length, should be a VERY short wait
            while (fifoCount < packetSize) fifoCount = mpu.getFIFOCount();

            // read a packet from FIFO
            mpu.getFIFOBytes(fifoBuffer, packetSize);
            
            // track FIFO count here in case there is > 1 packet available
            // (this lets us immediately read more without waiting for an interrupt)
            fifoCount -= packetSize;

            // display quaternion values in easy matrix form: w x y z
            mpu.dmpGetQuaternion(&q, fifoBuffer);
            //carSensors_msg.imu.orientation.x=q.x;
            //carSensors_msg.imu.orientation.y=q.y;
            //carSensors_msg.imu.orientation.z=q.z;
            //carSensors_msg.imu.orientation.w=q.w;


                // display Euler angles in degrees
                //mpu.dmpGetQuaternion(&q, fifoBuffer);
                //mpu.dmpGetGravity(&gravity, &q);
                //mpu.dmpGetYawPitchRoll(ypr, &q, &gravity);
                //Serial.print("ypr\t");
                //Serial.print(ypr[0] * 180/M_PI);
                //Serial.print("\t");
                //Serial.print(ypr[1] * 180/M_PI);
                //Serial.print("\t");
                //Serial.println(ypr[2] * 180/M_PI);

            // get real acceleration, adjusted to remove gravity
            //mpu.dmpGetQuaternion(&q, fifoBuffer);
            mpu.dmpGetAccel(&aa, fifoBuffer);
            mpu.dmpGetGravity(&gravity, &q);
            mpu.dmpGetLinearAccel(&aaReal, &aa, &gravity);
            //carSensors_msg.imu.linear_acceleration.x = aaReal.x;
            //carSensors_msg.imu.linear_acceleration.y = aaReal.y;
            //carSensors_msg.imu.linear_acceleration.z = aaReal.z;


            //XXX add support for angular acc and covariance
            // blink LED to indicate activity
            blinkState = !blinkState;
            digitalWrite(LED_PIN, blinkState);
        }
    } 

    //failsafe, if there's no new message for over 500ms, halt the motor
    //if ( millis() - carControlTimestamp > 500 ){
    //    throttle.writeMicroseconds(minThrottleVal);
    //} else if (newCarControlmsg) {  
    //    newCarControlmsg = false;
    //    throttle.writeMicroseconds(throttleServoVal);
    //    steer.writeMicroseconds(steeringServoVal);
    //}

}
