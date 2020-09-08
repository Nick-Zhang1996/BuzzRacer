# analyze log
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('../../src/'))
from common import *
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from mpl_toolkits.mplot3d import Axes3D
from math import degrees,radians


t_vec = []
x_vec = []
heading_vec = []
steering_vec = []
throttle_vec = []
acc_vec = []
v_vec = []

filenames = ["full_state1.p","full_state2.p","full_state3.p","full_state4.p"] 
for filename in filenames:
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    data = np.array(data)

    t = data[:,0]
    t = t-t[0]
    x = data[:,1]
    y = data[:,2]
    heading = data[:,3]
    steering = data[:,4]
    throttle = data[:,5]

    dt = 0.01
    dx = np.diff(x)/dt
    dy = np.diff(y)/dt
    v = (dx**2+dy**2)**0.5
    smooth_v = savgol_filter(v,51,2)
    acc = np.diff(smooth_v)/dt
    throttle_vec.append(throttle[2:])
    steering_vec.append(steering[2:])
    acc_vec.append(acc)
    v_vec.append(smooth_v[1:])

acc = [m for p in acc_vec for m in p]
throttle = [m for p in throttle_vec for m in p]
steering = [m for p in steering_vec for m in p]
v = [m for p in v_vec for m in p]

acc = np.array(acc)
throttle = np.array(throttle)
steering = np.array(steering)
v = np.array(v)


# filter out throttle<0 samples
mask = throttle>-1.1
#mask = throttle>0.01
#mask = np.bitwise_and(mask,np.abs(steering)<radians(1))
#mask = np.bitwise_and(mask,np.abs(v)<2)
mask = np.bitwise_and(mask,np.abs(v)>0.1)
print(acc.shape)
acc = acc[mask]
print(acc.shape)
throttle = throttle[mask]
v = v[mask]

p = np.polyfit(throttle,acc,1)
print(p)
#xx = np.linspace(np.min(throttle),np.max(throttle))
xx = np.linspace(-1,np.max(throttle))
fit_acc = np.polyval(p,xx)

# sample data points measured in step response
sample_throttle = [0, 0.316, 0.276, 0.346, 0.516, 0.238, -1, -0.6]
sample_acc = [-1.368, 0.721, 0.544, 1.08, 4.316, 0.2483, -5.244, -1.6866]

fig = plt.figure()
ax = fig.gca()
ax.plot(throttle,acc,'*',label="acc vs throttle")
ax.plot(xx,fit_acc,label="fit acc")
ax.plot(sample_throttle,sample_acc,'*r',label="fit acc")
ax.legend()
plt.show()

'''
ax = fig.add_subplot(111,projection='3d')
ax.scatter(throttle,steering,acc)
plt.show()
'''
