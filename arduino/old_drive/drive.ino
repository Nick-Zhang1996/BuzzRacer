#include <Servo.h>

//These pins are used because they are on different timers
const int pinDrive = 3;
const int pinServo = 5;
Servo driveOutput;
Servo steerOutput;

const int right_limit = 60;
const int left_limit = 150;
const int steer_center = (right_limit + left_limit) / 2;

int throttle_val = 0;
int steering_val = steer_center;

/*
 * getMessage: read new throttle and steering values from serial with
 * format $0.45,-0.2\n. The first value is the throttle and the second value is
 * the steering. The last byte can be any non-digit value because it forces
 * parseFloat to return instead of waiting for more digits.
 */
void getMessage() {
  while(Serial.available() > 0) {
    char first = Serial.read();
    if(first == '$') {
      float incoming_throttle_data = Serial.parseFloat();
      // Change the multiplier based on what percent of the total power you want to deliver
      float multipler = 0.3 * 180.0;
      int temp_throttle = (int) (incoming_throttle_data * multipler);
      //Serial.print("   ");
      if(temp_throttle < 0 || temp_throttle > 180) {
        throttle_val = 0;
        //Serial.print("invalid throttle speed: ");
        //Serial.println(incoming_throttle_data);
      } else {
        throttle_val = temp_throttle;
        //Serial.println(throttle);
      }

      float incoming_steering_data = Serial.parseFloat();
      float slope = (right_limit - left_limit) / 2.0;
      int temp_steer = (int) ((incoming_steering_data * slope) + steer_center);
      if(temp_steer < right_limit || temp_steer > left_limit) {
        steering_val = steer_center;
        //Serial.print("invalid steering input: ");
        //Serial.println(incoming_steering_data);
        //Serial.println(temp_steer);
      } else {
        steering_val = temp_steer;
      }
    }
  }
  String message = "";
  message.concat(throttle_val);
  message.concat(",");
  message.concat(steering_val);
  message.concat("\n");
  Serial.println(message);
}

void setup() {
  Serial.begin(9600);

  pinMode(pinDrive, OUTPUT);
  pinMode(pinServo, OUTPUT);

  steerOutput.attach(pinServo);
  driveOutput.attach(pinDrive);

  driveOutput.write(0);
  delay(5000);
  Serial.print("#ready\n");
}

void loop() {
  if (Serial.available() > 0) {
    getMessage();
  }
  driveOutput.write(throttle_val);
  steerOutput.write(steering_val);
  delay(10);
}

