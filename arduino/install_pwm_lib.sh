#!/bin/bash

mkdir ~/Arduino/libraries/PWM
unzip -o PWM.zip
cp -r PWM/* ~/Arduino/libraries/PWM/
rm -r PWM

