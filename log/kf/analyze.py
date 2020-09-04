# analyze log
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('../../src/'))
from common import *

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
kf_x = data[:,6]
kf_y = data[:,7]
kf_v = data[:,8]
kf_theta = data[:,9]
kf_omega = data[:,10]

# calculate speed
dt = 0.01
dx = np.diff(x)
dy = np.diff(y)
# displacement
dr = np.vstack([dx,dy])
d_dir = np.vstack([np.cos(heading),np.sin(heading)])

dot_product = np.empty(dr.shape[1])
for i in range(dr.shape[1]):
    dot_product[i] = np.dot(dr[:,i],d_dir[:,i])+0.001

v_is_forward = ((dot_product > 0) - 0.5)* (2)

ds = np.sqrt(dx**2+dy**2)
v = ds/dt
v = v*v_is_forward

# establish cost function

# predict future state based on initial state and control signal
# x0: v0
# u: throttle command, starting from the one aligned with v0, 1*n, n being number of steps to calculate
# return: x: 1*n, predicted state
def predict(x0,u,param=None):
    # acc = K*(Dead(u)-c*v)
    dt = 0.01
    K = 7.569
    c = 0
    x = [x0]
    for val in u:
        if (val > 0):
            K = 7.569
        else:
            K = 3
        a = K*(val - c*x[-1])-1.685
        x.append(x[-1]+a*dt)
    # ignore the very last one, which will be substituted with ground truth in next cycle
    return x[:-1]

# calculate state difference
def stateDiff(x1,x2):
    return (np.sum(x1-x2)**2)


# prepare sample data frames
# v > 0.1

predict_v = []

# visualize model performance
horizon = 50
for i in range(0,len(v),horizon):
    new_state = predict(v[i],throttle[i:i+horizon])
    predict_v.append(new_state)

predict_v = [a for b in predict_v for a in b]

fig = plt.figure()
ax = fig.gca()
#ax.plot(t,heading/np.pi*180.0, label="heading")
#ax.plot(t,x,label="raw")
#ax.plot(t,kf_x, label="kf")
ax.plot(t[1:],v, label="v")
ax.plot(t,predict_v, label="predict v")
ax.plot(t,throttle, label="throttle")
ax.plot(t,steering, label="steering")
ax.legend()
plt.show()
