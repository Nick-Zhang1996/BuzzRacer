# test phase lag in control
# and bandwidth for control
# procedure:
# record 1..20Hz, 1 Hz interval, record lag
# with a moderate frequency, keep reducing sleep time till jitter appears
from math import sin
from time import time,sleep,pi

freq = 5
with serial.Serial(CommPort,115200, timeout=0.001,writeTimeout=0) as arduino:
    while(True):
        # formulate a 5Hz sine wave
        steering = radians(24)*sin(2*pi*freq)
        arduino.write((str(mapdata(steering, radians(24),-radians(24),1150,1850))+","+str(mapdata(throttle,-1.0,1.0,1900,1100))+'\n').encode('ascii'))
        sleep(0.05)
