import sys
import os
thisdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(thisdir))

from scipy.signal import savgol_filter
import pickle
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
from common import *
from math import pi,degrees,radians,sin,cos,tan,atan
def loadLog(filename=None):
    if (len(sys.argv) != 2):
        if (filename is None):
            print_error("Specify a log to load")
        print_info("using %s"%(filename))
    else:
        filename = sys.argv[1]
    with open(filename, 'rb') as f:
        log = pickle.load(f)
    log = np.array(log)
    log = log.squeeze(1)
    return log

def prepLog(log,skip=1):
    #time(),x,y,theta,v_forward,v_sideway,omega, car.steering,car.throttle
    t = log[skip:,0]
    t = t-t[0]
    x = log[skip:,1]
    y = log[skip:,2]
    heading = log[skip:,3]
    v_forward = log[skip:,4]
    v_sideway = log[skip:,5]
    omega = log[skip:,6]
    steering = log[skip:,7]
    throttle = log[skip:,8]

    Log = namedtuple('Log', 't x y heading v_forward v_sideway omega steering throttle')
    mylog = Log(t,x,y,heading,v_forward,v_sideway,omega,steering,throttle)
    return mylog

# ax = f( Vx, throttle, steering)
def plotAcc(filename):
    rawlog = loadLog(filename)
    log = prepLog(rawlog,skip=1)
    dt = 0.01
    throttle = log.throttle
    ax = np.hstack([0,np.diff(log.v_forward)])/dt
    ax = savgol_filter(ax, 51,2)


    mask0 = np.abs(log.steering)<radians(30)
    mask1 = np.abs(log.steering)>radians(0)
    mask2 = log.v_forward<2.5
    mask3 = log.v_forward>0.5

    mask = np.bitwise_and(mask0,mask1)
    mask = np.bitwise_and(mask,mask2)
    mask = np.bitwise_and(mask,mask3)

    throttle = throttle[mask]
    ax = ax[mask]
    plt.plot(throttle,ax,'*')

def fitMotor():
    pass

if __name__=="__main__":
    filename = "../../log/jan12/full_state1.p"
    #plotAcc(filename)
    #plt.show()
    fitMotor()
