import sys
import os
thisdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(thisdir))
from common import *
import pickle
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple

def wrapContinuous(val):
    # wrap to -pi,pi
    wrap = lambda x: np.mod(x + np.pi, 2*np.pi) - np.pi
    dval = np.diff(val)
    dval = wrap(dval)
    retval = np.hstack([0,np.cumsum(dval)])+val[0]
    return retval

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
    # NOTE
    #omega = log[skip:,6]
    omega = np.hstack([0,np.diff(wrapContinuous(heading))])/0.01
    steering = log[skip:,7]
    throttle = log[skip:,8]

    Log = namedtuple('Log', 't x y heading v_forward v_sideway omega steering throttle')
    mylog = Log(t,x,y,heading,v_forward,v_sideway,omega,steering,throttle)
    return mylog

filename = '../../log/2022_3_1_sim/full_state2.p'
rawlog = loadLog(filename)
log_old = prepLog(rawlog,skip=1)

filename = '../../log/2022_3_1_sim/full_state1.p'
rawlog = loadLog(filename)
log_new = prepLog(rawlog,skip=1)
dt = 0.01
plt.plot(log_old.steering,label='old steering')
plt.plot(log_new.steering,label='new steering')
plt.legend()
plt.show()
