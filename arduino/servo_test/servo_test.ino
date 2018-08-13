#include <Servo.h>

int pinServo = 10;
Servo servo;

int servoVal = 105;
// lower bound 60, right turn
// upper bound 150, left turn

void setup() {
  // put your setup code here, to run once:
  servo.attach(pinServo);
  Serial.begin(9600);
  servo.write(servoVal);
}

void loop() {
//  servoVal = Serial.parseInt();
//  if(servoVal != 0) {
//    servo.write(servoVal);
//  }

  for(int x = 60; x <= 150; x++) {
    servo.write(x);
    delay(30);
  }

  for(int x = 150; x >= 60; x--) {
    servo.write(x);
    delay(30);
  }
}
