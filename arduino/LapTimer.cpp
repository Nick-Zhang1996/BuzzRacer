//Tony Wineman
//3.11.19
//v1.0

#include "LapTimer.h"

void LapTimer::newLap(const double& time) {
    //Update avgLap and sum
    lapsSum -= laps[NUM_LAPS - 1];
    lapsSum += time;
    avgLap = lapsSum / NUM_LAPS;

    //Update laps Array
    for (int i = NUM_LAPS - 1; i > 0; i--) {
        laps[i] = laps[i - 1];
    }
    laps[0] = time;

    //Update Best Lap
    bestLap = bestLap == 0 ? time : bestLap > time ? time : bestLap;

    //Update stdDev
    stdDev = this->stdDevCalc();

    //Print updated info
    this->printData();
}

void LapTimer::printData() const {

    //print which car it is
    Serial.print("Car ");
    Serial.println(carNum);

    //print out lap times
    for (int i = 1; i <= NUM_LAPS; i++) {
        Serial.print("Lap ");
        Serial.print(i);
        Serial.print(": ");
        Serial.print(laps[i-1] / 1000);
        Serial.println(" seconds");
    }

    //print Best Lap
    Serial.print("Best Lap: ");
    Serial.print(bestLap / 1000);
    Serial.println(" seconds");

    //print Avg Lap
    Serial.print("Average Lap: ");
    Serial.print(avgLap / 1000);
    Serial.println(" seconds");

    //print Std Dev
    Serial.print("Standard Deviation: ");
    Serial.print(stdDev / 1000);
    Serial.println(" seconds");
    Serial.println();
}

double LapTimer::stdDevCalc() {
    double aSum = 0;    //sum of (val - avg)^2
    for (int i = 0; i < NUM_LAPS; i++) {
        aSum += (laps[i] - avgLap) * (laps[i] - avgLap);
    }
    return sqrt(aSum / NUM_LAPS);
}
