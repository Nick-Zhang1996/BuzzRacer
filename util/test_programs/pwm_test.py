#! /usr/bin/python

import wiringpi2 as wpi
import time

pin = 6

wpi.wiringPiSetup()
wpi.pinMode(pin, 1)       # Set pin 6 to 1 ( OUTPUT )
wpi.digitalWrite(pin, 1)  # Write 1 ( HIGH ) to pin 6
print wpi.digitalRead(pin)
