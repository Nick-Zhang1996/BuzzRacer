#include <PWM.h>

const int pinDrive = 9;

int throttle = 0;

/*
 * getMessage: read new throttle value from serial with
 * format $0.45\n or any other non-digit terminating byte.
 * This forces parseFloat to return instead of waiting for 
 * more digits.
 */
void getMessage() {
  char first = Serial.read();
  if(first == '$' && Serial.available() > 0) {
    float incoming_throttle_data = Serial.parseFloat();
    if(incoming_throttle_data < 0 || incoming_throttle_data > 1) {
      throttle = 0;
      Serial.print("invalid throttle speed: ");
      Serial.println(throttle);
    } else {
      incoming_throttle_data *= 255;
      throttle = (int) incoming_throttle_data;
    }
  }
}

void setup() {
  InitTimersSafe();

  Serial.begin(9600);

  pinMode(pinDrive, OUTPUT);

  bool success = SetPinFrequencySafe(pinDrive, 2000);
  Serial.println(success ? "successfully set frequency" : "error: could not set frequency");
}

void loop() {
  getMessage();
  pwmWrite(pinDrive, throttle);
}

