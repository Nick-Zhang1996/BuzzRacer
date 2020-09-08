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

acc = [m for p in acc_vec for m in p]
throttle = [m for p in throttle_vec for m in p]
steering = [m for p in steering_vec for m in p]

fig = plt.figure()
'''
ax = fig.gca()
ax.plot(throttle,acc,'*',label="acc vs throttle")
ax.legend()
'''

ax = fig.add_subplot(111,projection='3d')
ax.scatter(throttle,steering,acc)
plt.show()
