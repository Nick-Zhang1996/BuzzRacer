#include <Servo.h>

const int pinDrive = 3;
Servo driveOutput;

void calibrate() {
  
}

void setup() {
  Serial.begin(9600);
  
  pinMode(pinDrive, OUTPUT);
  driveOutput.attach(pinDrive);

  driveOutput.write(0);
  delay(5000);
}

void loop() {
  for (int x = 0; x <= 90; x++) {
    driveOutput.write(x);
    delay(30);
  }

  for (int x = 90; x >= 0; x--) {
    driveOutput.write(x);
    delay(30);
  }
}
