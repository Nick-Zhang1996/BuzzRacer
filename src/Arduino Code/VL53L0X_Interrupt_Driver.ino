//Code for driving four VL53L0X sensors and storing measurements in two types of structures
// The first structure is an array (length 4) of custom structures with extra information about the measurement
// The second is a simple int array with four entries, one for each sensor
// *******To do: Both are provided but one should be chosen!!**********

#include "Adafruit_VL53L0X.h"
#include <Wire.h>

#define SENSOR1_WIRE Wire
#define SENSOR2_WIRE Wire
#define SENSOR3_WIRE Wire
#define SENSOR4_WIRE Wire

// Specify Arduino digital pins used for each sensor's interrupt and xshut pins
const byte InterruptPin1 = 2;
const byte InterruptPin2 = 3;
const byte InterruptPin3 = 9;
const byte InterruptPin4 = 10;
const byte XShutPin1 = 4;
const byte XShutPin2 = 5;
const byte XShutPin3 = 8;
const byte XShutPin4 = 11;

// ************LIMITS HOW LONG THE CODE WILL RUN****************
const float timeout = 60000; // allowed run time for code in ms

Adafruit_VL53L0X sensor1;
Adafruit_VL53L0X sensor2;
Adafruit_VL53L0X sensor3;
Adafruit_VL53L0X sensor4;
typedef struct {
  Adafruit_VL53L0X *psensor; // pointer to object
  TwoWire *pwire;
  int id;            // id for the sensor
  int shutdown_pin;  // which pin for shutdown;
  int interrupt_pin; // which pin to use for interrupts.
  Adafruit_VL53L0X::VL53L0X_Sense_config_t
      sensor_config;     // options for how to use the sensor
  uint16_t range;        // range value used in continuous mode stuff.
  uint8_t sensor_status; // status from last ranging in continuous.
} sensorList_t;

sensorList_t sensors[] = {
    {&sensor1, &SENSOR1_WIRE, 0x2A, XShutPin1, InterruptPin1,
     Adafruit_VL53L0X::VL53L0X_SENSE_DEFAULT, 0, 0},
    {&sensor2, &SENSOR2_WIRE, 0x2B, XShutPin2, InterruptPin2,
     Adafruit_VL53L0X::VL53L0X_SENSE_DEFAULT, 0, 0},
    {&sensor3, &SENSOR3_WIRE, 0x2C, XShutPin3, InterruptPin3,
     Adafruit_VL53L0X::VL53L0X_SENSE_DEFAULT, 0, 0},
    {&sensor4, &SENSOR4_WIRE, 0x2D, XShutPin4, InterruptPin4,
     Adafruit_VL53L0X::VL53L0X_SENSE_DEFAULT, 0, 0}
};

const int COUNT_SENSORS = sizeof(sensors) / sizeof(sensors[0]);

volatile boolean update = false; //volatile in order to work better with interrupts

VL53L0X_RangingMeasurementData_t measureData;
VL53L0X_RangingMeasurementData_t *measureDataP = &measureData;
VL53L0X_RangingMeasurementData_t allMeasuresData[4];
int allMeasures[4];
const int sensorNums[] = {1, 2, 3, 4};

float startTime;

void initializeSensors() {
  bool found_any_sensors = false;
  // Set all shutdown pins low to shutdown sensors
  for (int i = 0; i < COUNT_SENSORS; i++) {
    digitalWrite(sensors[i].shutdown_pin, LOW);
  }
  delay(10);

  for (int i = 0; i < COUNT_SENSORS; i++) {
    // one by one enable sensors and set their ID
    digitalWrite(sensors[i].shutdown_pin, HIGH);
    delay(10); // give time to wake up.
    if (sensors[i].psensor->begin(sensors[i].id, false, sensors[i].pwire,
                                  sensors[i].sensor_config)) {
      found_any_sensors = true;
    } else {
      // could do something with LED here in the future?
//      Serial.println("false");
//      Serial.print(i, DEC);
//      Serial.print(F(": failed to start\n"));
    }
  }
  //*************************NOTE*************************************
  // A infinite while loop preventing code from running if no sensors available is commented out here
  // could be uncommented if this is desired
  // Decide what should be done if no sensors can be started!
  //*************************NOTE*************************************
  if (!found_any_sensors) {
    // could do something with LED here in the future?
//    Serial.println("No valid sensors found");
//    while (1)
//      ;
  }

  // set interrupt pins and corresponding functions
  pinMode(InterruptPin1, INPUT_PULLUP);
  pinMode(InterruptPin2, INPUT_PULLUP);
  pinMode(InterruptPin3, INPUT_PULLUP);
  pinMode(InterruptPin4, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(InterruptPin1), sensor1Interrupt, FALLING); // used to be CHANGE but I believe this would lead to a second interrupt when an interrupt is cleared
  attachInterrupt(digitalPinToInterrupt(InterruptPin2), sensor2Interrupt, FALLING);
  attachInterrupt(digitalPinToInterrupt(InterruptPin3), sensor3Interrupt, FALLING);
  attachInterrupt(digitalPinToInterrupt(InterruptPin4), sensor4Interrupt, FALLING);

}

void setSensorPrefs() {
  //***********************NOTE******************************
  // Interrupts are thrown whenever the measurement is below the low threshold
  // We want interrupts for all measurements, so the low threshold is made higher than all possible measurements
  // The high threshold is essentially unused but I believe it is required for 'setInterruptThresholds()'
  //*********************************************************
  FixPoint1616_t LowThreshold = (2000 * 65536.0);
  FixPoint1616_t HighThreshold = (3000 * 65536.0);
  for (int i = 0; i < COUNT_SENSORS; i++) {
    sensors[i].psensor->setGpioConfig(VL53L0X_DEVICEMODE_CONTINUOUS_RANGING,
                    VL53L0X_GPIOFUNCTIONALITY_THRESHOLD_CROSSED_LOW,
                    VL53L0X_INTERRUPTPOLARITY_LOW); // this VL53L0X_DEVICEMODE_CONTINUOUS_RANGING is just ignored bc it's not valid
    // Set Interrupt Tresholds
    // Low reading set to 2000mm  High Set to 3000mm
    sensors[i].psensor->setInterruptThresholds(LowThreshold, HighThreshold, false);
  }
}

void startSensors() {
  noInterrupts();
  for (int i = 0; i < COUNT_SENSORS; i++) {
    // Enable Continous Measurement Mode
    sensors[i].psensor->setDeviceMode(VL53L0X_DEVICEMODE_CONTINUOUS_RANGING, false);
    // Start continuous measurement
    sensors[i].psensor->startMeasurement(); 
    // wait for measurement to start
    delay(10);
  }
  interrupts();
}

void sensor1Interrupt() {
  updateMeasurement(sensor1, sensorNums[0], measureDataP);
}

void sensor2Interrupt() {
  updateMeasurement(sensor2, sensorNums[1], measureDataP);
}

void sensor3Interrupt() {
    updateMeasurement(sensor3, sensorNums[2], measureDataP);
}

void sensor4Interrupt() {
  updateMeasurement(sensor4, sensorNums[3], measureDataP);
}

void updateMeasurement(Adafruit_VL53L0X sensor, int senseNum, VL53L0X_RangingMeasurementData_t *measureDataP) {
  if (measureDataP->RangeStatus != 4) {
    sensor.getRangingMeasurement(
          measureDataP, false); // pass in 'true' to get debug data printout!
    allMeasuresData[senseNum-1] = *measureDataP;
    allMeasures[senseNum-1] = measureDataP->RangeMilliMeter; // == allMeasuresData[senseNum-1].RangeMilliMeter;
    sensor.clearInterruptMask(false);
    update = true;
  }
}

void sensorSetup() {
  // initialize all pins
  for (int i = 0; i < COUNT_SENSORS; i++) {
    pinMode(sensors[i].shutdown_pin, OUTPUT);
    digitalWrite(sensors[i].shutdown_pin, LOW);

    // this if statement (as well as the for loop code above) is copied from an example in the library.
    // not exactly sure if it is needed
    if (sensors[i].interrupt_pin >= 0)
      pinMode(sensors[i].interrupt_pin, INPUT_PULLUP);
  }
  initializeSensors();
  setSensorPrefs();
  startSensors();

  startTime = millis();
}

void setup() {
  sensorSetup();
}

void loop() {
  // nothing required here
}
