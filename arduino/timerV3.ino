#include <LapTimer.h>

double car1start;
double car2start;
double tmpTime;
int car2_first_time = 1;

int state = 0;

int reading;

LapTimer car1(1);
LapTimer car2(2);

void setup() {
  Serial.begin(115200);
  pinMode(5, INPUT); //this is the lower sensor
  pinMode(3, INPUT); //this is the upper sensor that only car 2 can trigger
}

void loop() {

  switch (state) {
    
    case 0:   //only runs once to initialize car1 start time
    if (digitalRead(5) == LOW) {
      tmpTime = millis();
      while(millis() - tmpTime < 500) {  //check upper sensor for .5 sec
        if (digitalRead(3) == LOW) {
          car2start = tmpTime;
          state = 2;
        }
      }
      state = 1;
    }
    break;

    case 1:   //only runs once to initialize car2 start time
    car1start = tmpTime;
    state = 2;
    break;

    case 2:   //always looking for lower sensor to be triggered
    if (digitalRead(5) == LOW) {
      tmpTime = millis();
      state = 3;
      while(millis() - tmpTime < 250) {   //check upper sensor for .25 sec
        if (digitalRead(3) == LOW) {
          if (car2_first_time) {    //check if to initialize car2 for its first real lap
            state = 1;
            car2_first_time = 0;
            car2start = tmpTime;
          } else {
            car2.newLap(tmpTime - car2start);
            car2start = tmpTime;
            state = 2;
          }
          delay(1500);    //wait 1.5 sec before checkking for next car
        }
      }
    }
    break;

    case 3:   //upper sensor wasn't triggered so car1 triggered lower sensor
    car1.newLap(tmpTime - car1start);
    car1start = tmpTime;
    state = 2;
    delay(1500);    //wait 1.5 sec before checking for next car
    break;
  }
}
