//Test software pwm using http://wiringpi.com/reference/software-pwm-library/

#include <wiringPi.h>
#include <softPwm.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int main (void)
{
  printf ("Raspberry Pi wiringPi Software PWM test program\n") ;

  if (wiringPiSetup () == -1)
    exit (1) ;

  int pin = 6;
  softPwmCreate(pin, 0, 100);
  softPwmWrite(pin, 100);
  return 0;
}
