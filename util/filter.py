# testing performance of filters
from scipy import signal 
import matplotlib.pyplot as plt
import numpy as np
import pickle
from math import cos,sin,pi,degrees,radians

# prepare lowpass filter
# argument: order, omega(-3db)
# TODO consider use a 0 phase shift filter
b, a = signal.butter(1,6,'low',analog=False,fs=100)
#z = signal.lfilter_zi(b,a)
#z = [0]

# plot acceleration
filename = "../datadump/test1/exp_state.p"
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


mask = np.array(range(380,690))
vs = vs[:-1]
plt.plot(vs[mask],'b',label='Lateral Vel (m/s)')
plt.plot(np.diff(vs[mask])*100,'r',label='Lateral Accel (m/s2)')
plt.title("Lateral Vel and Acc")
plt.xlabel('Time (0.01s)')
plt.legend()
plt.show()

# Plot: angle between velocity and heading

# path tangent
dx = np.diff(x).astype(np.float64)
dy = np.diff(y).astype(np.float64)
path_heading = np.arctan2(dy,dx)
path_heading = path_heading[skip:]
diff_heading = theta[:-1] - path_heading
lateral_acc = np.diff(vs)*100
plt.plot(theta[:-1])
plt.plot(path_heading)

# good data at 380-690
lateral_acc = lateral_acc[mask-13]
diff_heading = diff_heading[mask]
plt.plot(lateral_acc,label='Lateral Accel 130ms delayed (m/s2)')
plt.plot(diff_heading,label='Vehicle Slide Angle (rad)')
plt.hlines(y=0,xmin=0,xmax=len(mask))
plt.xlabel('Time (0.01s)')
plt.legend()
plt.show()

plt.plot(lateral_acc/diff_heading)
plt.title("Accel/Slide")
plt.show()
print(np.mean(lateral_acc/diff_heading))
print(np.std(lateral_acc/diff_heading))

# vehicle position
#plt.plot(x[mask],y[mask])
#plt.title("Vehicle Position")
#plt.show()

#plt.subplot(211)
#plt.plot(data[skip:,3])
#plt.subplot(212)
#plt.plot(lflf(acc,1))
#plt.show()

