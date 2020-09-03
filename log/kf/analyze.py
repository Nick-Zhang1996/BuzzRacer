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

fig = plt.figure()
ax = fig.gca()
#ax.plot(t,heading/np.pi*180.0, label="heading")
ax.plot(x,y)
ax.legend()
plt.show()
