#include <PWM.h>
#include <Servo.h>

const int pinDrive = 3;
const int pinServo = 5;
Servo servo;

const int right_limit = 60;
const int left_limit = 150;
const int steer_center = (right_limit + left_limit) / 2;

int throttle_val = 0;
int steering_val = steer_center;

/*
 * getMessage: read new throttle value from serial with
 * format $0.45\n or any other non-digit terminating byte.
 * This forces parseFloat to return instead of waiting for 
 * more digits.
 */
void getMessage() {
  while(Serial.available() > 0) {
    char first = Serial.read();
    if(first == '$') {
      float incoming_throttle_data = Serial.parseFloat();
      int temp_throttle = (int) (incoming_throttle_data * 255.0);
      if(temp_throttle < 0 || temp_throttle > 255) {
        throttle_val = 0;
        Serial.print("invalid throttle speed: ");
        Serial.println(incoming_throttle_data);
      } else {
        throttle_val = temp_throttle;
        //Serial.println(throttle);
      }

      float incoming_steering_data = Serial.parseFloat();
      float slope = (right_limit - left_limit) / 2.0;
      int temp_steer = (int) ((incoming_steering_data * slope) + steer_center);
      if(temp_steer < right_limit || temp_steer > left_limit) {
        steering_val = steer_center;
        Serial.print("invalid steering input: ");
        //Serial.println(incoming_steering_data);
        Serial.println(temp_steer);
      } else {
        steering_val = temp_steer;
      }

      String message = "";
      message.concat(throttle_val);
      message.concat(",");
      message.concat(steering_val);
      Serial.println(message);
    }
  }
}

void setup() {
  InitTimersSafe();

  Serial.begin(9600);

  pinMode(pinDrive, OUTPUT);

  bool success = SetPinFrequencySafe(pinDrive, 2000);
  Serial.println(success ? "successfully set frequency" : "error: could not set frequency");

  servo.attach(pinServo);
}

void loop() {
  getMessage();
  pwmWrite(pinDrive, throttle_val);
  servo.write(steering_val);
  
  delay(10);
}

