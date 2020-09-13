# analyze log
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('../../src/'))
from common import *
from kalmanFilter import KalmanFilter
from math import pi
from scipy.signal import savgol_filter

if (len(sys.argv) != 2):
    print_error("Specify a log to load")

filename = sys.argv[1]
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
exp_kf_x = data[:,6]
exp_kf_y = data[:,7]
exp_kf_v = data[:,8]
exp_kf_theta = data[:,9]
exp_kf_omega = data[:,10]

# calculate speed from pos, to use as no-bias but noise reference
dt = 0.01
dx = np.diff(x)
dy = np.diff(y)
# displacement
dr = np.vstack([dx,dy])
d_dir = np.vstack([np.cos(heading),np.sin(heading)])

dot_product = np.empty(dr.shape[1])
for i in range(dr.shape[1]):
    dot_product[i] = np.dot(dr[:,i],d_dir[:,i])+0.001

# add forward/backward signage to speed
v_is_forward = ((dot_product > 0) - 0.5)* (2)
ds = np.sqrt(dx**2+dy**2)
v = ds/dt
v = v*v_is_forward
smooth_v = savgol_filter(v,51,2)
acc = np.diff(smooth_v)/dt

def getLongitudinalAcc(state,throttle,steering):
    vf = state[3]
    acc = throttle * 4.95445214  - 1.01294228 - abs(steering)
    if (vf<0.01 and throttle<0.245):
        acc = 0
    return acc

# test kalman filter
kf_state_vec = []
action_vec = []
kf = KalmanFilter(1.02e-3)
kf.init(x=x[0],y=y[0],theta=heading[0],timestamp=0)
for i in range(len(x)-1):
    # action: steering(rad,left pos),forward acc
    # steering sign convention in log is right positive
    # FIXME this sign discrepancy
    #   state: (x,y,heading,v_forward,v_sideway,omega)
    state = (x[i],y[i],heading[i],v[i],0,0)

    predicted_acc = getLongitudinalAcc(state,throttle[i],steering[i])
    action = (-steering[i],predicted_acc)
    action_vec.append(action)
    # FIXME
    action = (0,0)
    kf.predict(action,timestamp = dt*i)
    z = (x[i],y[i],heading[i])
    z = np.matrix(z).reshape(3,1)
    kf.update(z,timestamp = dt*i)
    kf_state = kf.getState()
    kf_state_vec.append(kf_state)



kf_state = np.array(kf_state_vec)
action_vec = np.array(action_vec)
#kf_x, kf_y, kf_v, kf_theta, kf_omega = kf_state

fig = plt.figure()
ax = fig.gca()
ax.plot(x,y, label="trajectory")
ax.legend()
plt.show()

# plot velocity
fig = plt.figure()
ax = fig.gca()
ax.plot(t[1:],v, label="v")
ax.plot(t[1:],kf_state[:,2], label="kf_v")
#ax.plot(t,exp_kf_v, label="ori_kf_v")
ax.plot(t,throttle, label="throttle")
ax.legend()
plt.show()

# plot acc
'''
fig = plt.figure()
ax = fig.gca()
ax.plot(t[:-2],acc, label="acc measured")
ax.plot(t[:-1],action_vec[:,1], label="kf_a")
ax.plot(t,throttle, label="throttle")
ax.plot(t,np.abs(steering), label="steering")
ax.legend()
plt.show()
'''

fig = plt.figure()
ax = fig.gca()
ax.plot(t,heading/pi*180,label="raw heading")
ax.plot(t[:-1],kf_state[:,3]/pi*180+0.01, label="kf")
ax.legend()
plt.show()
