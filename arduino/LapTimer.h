//Tony Wineman
//3.11.19
//v1.0

#pragma once

#include <Arduino.h>

#define NUM_LAPS 5      //stats based on previous 5 laps

class LapTimer {
    public:
        LapTimer() : bestLap(0), avgLap(0), stdDev(0), lapsSum(0), carNum(1) {}
        LapTimer(int n) : bestLap(0), avgLap(0), stdDev(0), lapsSum(0), carNum(n) {}
        void newLap(const double& time);    //Adds a new Lap Time
        void printData() const;             //Displays Data
    private:
        double stdDevCalc();                //helper function to calc std Dev
        double laps[NUM_LAPS];              //Array of all the NUM_LAPS
        double bestLap;                     //Best Lap Time of all the NUM_LAPS
        double avgLap;                      //Average Lap Time of all the NUM_LAPS
        double stdDev;                      //Standard Deviation of all the NUM_LAPS
        double lapsSum;                     //Sum of all the NUM_LAPS
        int carNum;                         //Which car for printing
};
