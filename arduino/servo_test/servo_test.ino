#include <Servo.h>

int pinServo = 10;
Servo servo;

void setup() {
  // put your setup code here, to run once:
  servo.attach(pinServo);
  Serial.begin(9600);
}

void loop() {
  delay(1000);
  servo.write(0);

  delay(1000);
  servo.write(90);

  delay(1000);
  servo.write(180);

  delay(1000);
  servo.write(90);

}
