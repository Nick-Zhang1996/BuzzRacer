// Arduino firmware for RC-VIP
// WARNING: NEVER push high to both MOSFETs on one side, this will create a short
// and burn out the MOSFETs instantly.

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
#include <rcvip_msgs/CarSensors.h>
#include <rcvip_msgs/CarControl.h>
#include <Servo.h>

//V2 and V3 board have different layout for bridge control lines, select one here.
//#define RC_VIP_PCBV2
#define RC_VIP_PCBV3

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


// ---------- H-bridge Control ------------
// Timer interrupt based PWM control for H bridge
// Current off-time method: tie both end to GND
// Also RN the car only goes in one direction


// All pins are high enable
#define PORT_POS_UP 9    

#ifdef RC_VIP_PCBV2
  #define PORT_POS_DOWN 6 
#endif

#ifdef RC_VIP_PCBV3
  #define PORT_POS_DOWN 8
#endif

  
#define PORT_NEG_UP 11
#define PORT_NEG_DOWN 10

// Timers usage
//timer0 -> Arduino millis() and delay()
//timer1 -> Servo lib
//timer2 -> synchronized multi-channel PWM
// if additional ISP are needed, 2B compare interrupt is still available, not sure about other timers


void enablePWM(){
    cli();//stop interrupts

    digitalWrite(PORT_POS_UP, LOW);
    digitalWrite(PORT_POS_DOWN,LOW);
    digitalWrite(PORT_NEG_UP, LOW);
    digitalWrite(PORT_NEG_DOWN,LOW);

    //set timer2 interrupt 
    TCCR2A = 0;// set entire TCCR2A register to 0
    TCCR2B = 0;// same for TCCR2B
    TCNT2  = 0;//initialize counter value to 0

    // Set CS21 bit for 32 prescaler
    // duty cycle: (16*10^6) / (32*256) Hz = 1.95kHz
    TCCR2B |= (1 << CS21) | (1 << CS20) ; 

    // set compare target, this controls the on-time of PWM
    // for n% signal:
    //OCR2A = (uint8_t) 256.0*onTime (fraction (0-1) );
    // Note, OCR2A < 20 creates erratic behavior(on oscilloscope) worth investicating, it  does NOT set power to 0    
    // You may comment the following line out if you wish a faster recovery from failsafe
    // In that case, when this function is called, OCR2A should already have been set by setHbridgePower()
    // Otherwise, you need to ensure OCR2A is at a safe value before calling enablePWM.
    //OCR2A = 20;


    // enable timer compare interrupt and overflow interrupt
    TIMSK2 |= (1 << OCIE2A) | ( 1 << TOIE2);

    digitalWrite(PORT_NEG_DOWN,HIGH);
    sei();//allow interrupts
  
}

void disablePWM(){
  
  
  cli();//stop interrupts
  //unset timer2 interrupt 
  TCCR2A = 0;// set entire TCCR2A register to 0
  TCCR2B = 0;// same for TCCR2B
  TCNT2  = 0;//initialize counter value to 0
  TIMSK2 = 0;

  sei();//allow interrupts
  
  digitalWrite(PORT_POS_UP, LOW);
  digitalWrite(PORT_POS_DOWN,LOW);
  digitalWrite(PORT_NEG_UP, LOW);
  digitalWrite(PORT_NEG_DOWN,LOW);
  
}


// Called at the falling edge of on-time, enter off-time configuration here
ISR(TIMER2_COMPA_vect){
// digital write takes ~6us to execute
// inline assembly takes <1us
// use with caution, though

/*
    digitalWrite(PORT_POS_UP, LOW);
    digitalWrite(PORT_NEG_UP, LOW);
    digitalWrite(PORT_POS_DOWN,HIGH);
    digitalWrite(PORT_NEG_DOWN,HIGH);
    digitalWrite(13,LOW);
*/
// 0-> POS_UP    
    asm (
      "cbi %0, %1 \n"
      : : "I" (_SFR_IO_ADDR(PORTB)), "I" (PORTB1)
    );
    // allow for P mosfet transient state
    delayMicroseconds(30);
// 1-> POS_DOWN
#ifdef RC_VIP_PCBV2
    asm (
      "sbi %0, %1 \n"
      : : "I" (_SFR_IO_ADDR(PORTD)), "I" (PORTD6)
    );
#endif

#ifdef RC_VIP_PCBV3
    asm (
      "sbi %0, %1 \n"
      : : "I" (_SFR_IO_ADDR(PORTB)), "I" (PORTB0)
    );
#endif
 
}

// Beginning of each Duty Cycle, enter on-time configuration here
ISR(TIMER2_OVF_vect){
// Do this like a pro: write 1 to PINxN to toggle PORTxN
/*
    digitalWrite(PORT_NEG_UP, LOW);
    digitalWrite(PORT_POS_DOWN,LOW);
    digitalWrite(PORT_POS_UP, HIGH);
    digitalWrite(PORT_NEG_DOWN,HIGH);
    digitalWrite(13,HIGH);
*/
    
// 0-> POS_DOWN
#ifdef RC_VIP_PCBV2
    asm (
      "cbi %0, %1 \n"
      : : "I" (_SFR_IO_ADDR(PORTD)), "I" (PORTD6)
    );
#endif

#ifdef RC_VIP_PCBV3
    asm (
      "cbi %0, %1 \n"
      : : "I" (_SFR_IO_ADDR(PORTB)), "I" (PORTB0)
    );
#endif

// 1-> POS_UP
    asm (
      "sbi %0, %1 \n"
      : : "I" (_SFR_IO_ADDR(PORTB)), "I" (PORTB1)
    );
}

// forward only, range 0-1
#define MAX_H_BRIDGE_POWER 0.5
void setHbridgePower(float power){
    if (power<0.01 || power>1.0){
        disablePWM();
        digitalWrite(LED_PIN, LOW);
    } else{
        cli();
        OCR2A = (uint8_t) 256.0*power*MAX_H_BRIDGE_POWER;
        sei();
    }
    return;
}

// -------------- IMU code -------------

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
//which car? turning diameter =42.5cm, steering angle = actan(2*wheelbase/diameter), 25.64 degree
const int leftBoundrySteeringServo = 1750; 
const float steeringLeftLimit = -20; 

const int rightBoundrySteeringServo = 1250;//td = 65.3cm, 17.34deg
const float steeringRightLimit = 20;

//previously 1380
const int midPointSteeringServo = 1500;

//const int minThrottleVal = 1500;
//
//static int throttleServoVal = 1500;
static int steeringServoVal = 1550;

static unsigned long throttleTimestamp = 0;
static unsigned long steeringTimestamp = 0;

unsigned long carControlTimestamp = 0;
unsigned long voltageUpdateTimestamp = 0;
bool newCarControlMsg = false;

//ros variables
//pub,sub, in_buf, out_buf
ros::NodeHandle_<ArduinoHardware, 2, 2, 128, 300 > nh;

bool failsafe = false;

void readCarControlTopic(const rcvip_msgs::CarControl& msg_CarControl) {
    carControlTimestamp = millis();
    newCarControlMsg = true;
    if (failsafe){ // recover from failsafe
        enablePWM();
        failsafe = false;
        digitalWrite(LED_PIN, HIGH);
    }

    if (msg_CarControl.throttle < 0.01 || msg_CarControl.throttle > 1.0001) {
        //throttleServoVal = minThrottleVal;
        disablePWM();
        failsafe = true;
        digitalWrite(LED_PIN, LOW);
    } else {
        //needs a re-calibration
        //throttleServoVal = (int) map( msg_CarControl.throttle, 0.0, 1.0, 1460, 1450);

        // this works only when PWM is enabled
        // so failsafe can override this function
        setHbridgePower(msg_CarControl.throttle);
    }
    // COMMENT THIS OUT for moving motor
    //throttleServoVal = minThrottleVal;

    float tempSteering = constrain(msg_CarControl.steer_angle, steeringLeftLimit, steeringRightLimit);
    if ( tempSteering > 0.0 ){
        steeringServoVal = (int) map(tempSteering, 0.0, steeringRightLimit, midPointSteeringServo, rightBoundrySteeringServo);
    } else if ( tempSteering < 0.0 ){
        steeringServoVal = (int) map(tempSteering, 0.0, steeringLeftLimit, midPointSteeringServo, leftBoundrySteeringServo);
    } else {
        steeringServoVal = 1550;
    }
    steer.writeMicroseconds(steeringServoVal);

    return;
}


ros::Subscriber<rcvip_msgs::CarControl> subCarControl("rc_vip/CarControl", &readCarControlTopic);
rcvip_msgs::CarSensors carSensors_msg;
ros::Publisher pubCarSensors("rc_vip/CarSensors", &carSensors_msg);

void setup() {
    digitalWrite(PORT_POS_UP, LOW);
    digitalWrite(PORT_POS_DOWN,LOW);
    digitalWrite(PORT_NEG_UP, LOW);
    digitalWrite(PORT_NEG_DOWN,LOW);

    pinMode(PORT_POS_UP,OUTPUT);
    pinMode(PORT_POS_DOWN,OUTPUT);
    pinMode(PORT_NEG_UP,OUTPUT);
    pinMode(PORT_NEG_DOWN,OUTPUT);

    // tie one end to GND
    digitalWrite(PORT_NEG_DOWN,HIGH);

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

    //pinMode(pinDrive, OUTPUT);
    pinMode(pinServo, OUTPUT);
    //throttle.attach(pinDrive);
    steer.attach(pinServo);
   
    //throttle.writeMicroseconds(minThrottleVal);
    delay(500);
    OCR2A = 25;
    enablePWM();
}

void loop() {

    //failsafe, if there's no new message for over 500ms, halt the motor
    if ( millis() - carControlTimestamp > 500 and !failsafe ){
        disablePWM();
        failsafe = true;
        digitalWrite(LED_PIN, LOW);
    }

    // get new voltage every 100ms
    if ( millis()-voltageUpdateTimestamp>100 ){
        float voltage = (float)analogRead(VOLTAGEDIVIDER_PIN);
        // depends on experiment value
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
    unsigned long delayTimestamp = millis();
    while(millis()<delayTimestamp+10){
      ;
    }
}
