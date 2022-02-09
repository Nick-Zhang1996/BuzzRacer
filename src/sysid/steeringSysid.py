import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
from math import radians,degrees
from scipy.signal import savgol_filter

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
measured_steering_smooth = savgol_filter(measured_steering, 19,2)

# construct estimated steering with first order sys
estimated_steering = np.zeros_like(t)
K = 0.2
for i in range(t.shape[0]-1):
    estimated_steering[i+1] = estimated_steering[i] * (1-K) + commanded_steering[i] * K

#plt.plot(t,commanded_steering,'--',label='command')
plt.plot(t,measured_steering,label='measured')
plt.plot(t,measured_steering_smooth,label='measured_smooth')
#plt.plot(t,estimated_steering,label='estimated')
plt.legend()
plt.show()
