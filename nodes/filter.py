from scipy import signal 
import matplotlib.pyplot as plt
import numpy as np
import pickle
from math import cos,sin,pi

# prepare lowpass filter
# argument: order, omega(-3db)
# TODO consider use a 0 phase shift filter
b, a = signal.butter(1,6,'low',analog=False,fs=100)
#z = signal.lfilter_zi(b,a)
#z = [0]

# plot acceleration
filename = "test1/exp_state.p"
infile = open(filename,'rb')
data = pickle.load(infile)
# how many datapoints to skip from the beginning
skip = 50
dt = 0.01

data = np.array(data)
x = data[:,0]
y = data[:,1]
theta = data[:,2]
vf = data[:,3]
vs = data[:,4]
omega = data[:,5]

def lf(vec):
    result = []
    z = [0]
    previous = vec[0]
    for i in range(len(vec)):
        pending = vec[i]
        if (i>1 and abs(pending-previous)>0.5):
            pending = previous
        val, z = signal.lfilter(b,a,[pending],zi=z)
        result.append(val)
        previous = pending
    return result

def lflf(vec,cutoff=0.1):
    count = 0
    for i in range(len(vec)):
        if (i>1 and abs(vec[i]-vec[i-1])>cutoff):
            vec[i] = vec[i-1]
    return signal.filtfilt(b,a,vec)

vf = lflf(vf[skip:])
vs = lflf(vs[skip:])
omega = lflf(omega[skip:])
theta = theta[skip:]

vx = np.zeros_like(vf)
vy = np.zeros_like(vf)
acc = np.zeros_like(vf)
for i in range(len(vf)):
    vx[i] = vf[i]*cos(theta[i])+vs[i]*cos(theta[i]-pi/2)
    vy[i] = vf[i]*sin(theta[i])+vs[i]*sin(theta[i]-pi/2)
for i in range(1,len(vf)):
    dv = ((vx[i]-vx[i-1])**2+(vy[i]-vy[i-1])**2)**0.5
    acc[i] = dv/dt


#plt.subplot(211)
#plt.plot(data[skip:,3])
#plt.subplot(212)
plt.plot(lflf(acc,1))
plt.show()

