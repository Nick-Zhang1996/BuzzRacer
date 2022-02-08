import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
from math import radians,degrees

filename = '../../log/2022_2_7_exp/debug_dict2.p'
with open(filename, 'rb') as f:
    data = pickle.load(f)
measured_steering = np.array(data[0]['measured_steering'])

filename = '../../log/2022_2_7_exp/full_state2.p'
with open(filename, 'rb') as f:
    data = pickle.load(f)
data = np.array(data)
commanded_steering = np.array(data[:,0,7])
t = data[:,0,0] - data[0,0,0]

measured_steering = (measured_steering+0.5*np.pi)%(np.pi)-0.5*np.pi

plt.plot(t,commanded_steering)
plt.plot(t,measured_steering)
plt.show()
