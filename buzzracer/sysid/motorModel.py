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
from scipy.optimize import minimize
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

def plotAcc2(filename):
    rawlog = loadLog(filename)
    log = prepLog(rawlog,skip=1)
    dt = 0.01
    throttle = log.throttle
    v = log.v_forward
    ax = np.hstack([0,np.diff(log.v_forward)])/dt
    ax = savgol_filter(ax, 51,2)

    start = 100
    end = 3000
    t = log.t[start:end]
    ax = ax[start:end]
    throttle = throttle[start:end]
    v = v[start:end]
    plt.plot(t,ax,'--',label='ax')
    plt.plot(t,throttle,label='throttle')

    # original model
    '''
    u1 = throttle
    ax_guess1 = guess_old(u1,v)
    ax_guess1 -= np.mean(ax_guess1)-np.mean(ax)
    print("guess error = %.2f"%(diff(ax_guess1[7:],ax[7:])))
    plt.plot(t,ax_guess1,label='guess')
    '''

    # candidate model
    ax_guess2 = guess2(throttle, v)
    plt.plot(t,ax_guess2,label='new guess')
    print("guess2 error = %.2f"%(diff(ax_guess2[7:],ax[7:])))

    # fit
    '''
    res = minimize(fun,(6.0,0.322),args=(throttle,v,ax))
    print(res)
    ax_guess_fitted = guess2(throttle, v,res.x[0],res.x[1])
    plt.plot(t,ax_guess_fitted,label='fitted')
    print("guess fitted error = %.2f"%(diff(ax_guess_fitted[7:],ax[7:])))
    '''

def guess_old(u,vx):
    Cm1 = 6.03154
    Cm2 = 0.96769
    Cr = 0.20375
    Cd = 0.00000
    ax = ( Cm1 - Cm2 * vx) * u - Cr - Cd * vx * vx
    return ax

def guess(u,v,c2=0.425):
    ax = np.zeros_like(u)
    # apply a 0.07s delay to control signal
    delay = 7
    #ax[delay:] = c2 * (15.2*u[:-delay] - v[delay:] - 3.157)
    ax[delay:] = 6*(u[:-delay] - v[delay:]/15.2 -0.322)
    return ax

def guess2(u,v,c1=6.17,c2=0.333):
    ax = np.zeros_like(u)
    # apply a 0.07s delay to control signal
    delay = 7
    ax[delay:] = c1*(u[:-delay] - v[delay:]/15.2 -c2)
    return ax

def diff(a,b):
    return np.linalg.norm((a-b)**2)

def fun(x,*args):
    c1 = x[0]
    c2 = x[1]
    u = args[0]
    v = args[1]
    ax = args[2]

    delay = 7
    ax_guess = np.zeros_like(ax)
    ax_guess[delay:] = c1*(u[:-delay] - v[delay:]/15.2 -c2)
    return diff(ax,ax_guess)


def ode(u,c1,c2):
    u = u
    x0 = np.array([0,0])
    A = np.array([[-1.0/c1,0],[1.0/c2,-1.0/c2]])
    B = np.array([15.2/c1,0])
    x = np.zeros((u.shape[0],2))
    dt = 0.01
    for i in range(1,x.shape[0]):
        x[i] += (A @ x[i-1,:] + B * u[i-1])*dt
    return x[:,1]*20000


if __name__=="__main__":
    #filename = "../../log/jan12/full_state1.p"
    #filename = '../../log/2022_2_9_exp/full_state4.p'
    filename = '../../log/2022_3_2_exp/full_state2.p'
    plotAcc2(filename)
    plt.legend()
    plt.show()
