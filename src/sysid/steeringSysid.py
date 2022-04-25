import math
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
from math import radians,degrees
from scipy.signal import savgol_filter
from scipy.optimize import minimize

#filename = '../../log/2022_2_7_exp/debug_dict2.p'
filename = '/home/caleb/RC-VIP/log/steeringSysid/debug_dict4.p'
with open(filename, 'rb') as f:
    data = pickle.load(f)
measured_steering = np.array(data[0]['measured_steering'])

#filename = '../../log/2022_2_7_exp/full_state2.p'
filename = '/home/caleb/RC-VIP/log/steeringSysid/full_state4.p'
with open(filename, 'rb') as f:
    data = pickle.load(f)
data = np.array(data)
commanded_steering = np.array(data[:,0,7])
t = data[:,0,0] - data[0,0,0]

measured_steering = (measured_steering+0.5*np.pi)%(np.pi)-0.5*np.pi
measured_steering_smooth = savgol_filter(measured_steering, 19,2)

# add 7ms delay
# t = t[7:]
# commanded_steering = commanded_steering[:-7]
# measured_steering_smooth = measured_steering_smooth[7:]

def first_order(tau, src, dst):
    # construct estimated steering with first order sys
    guess = np.zeros_like(src)
    for i in range(src.shape[0]-1):
        # guess[i+1] = guess[i] * (1-K) + src[i] * K
        dt = t[i+1] - t[i]
        guess[i+1] = guess[i] + (src[i] - guess[i]) / tau * dt
    err = np.linalg.norm(guess - dst)
    return err


if __name__ == "__main__":

    res = minimize(first_order, [0.2], args=(commanded_steering, measured_steering_smooth))
    print(res)

    # construct estimated steering with first order sys
    estimated_steering = np.zeros_like(t)
    tau = res.x
    # K = 0.2
    # K = 0.0867
    for i in range(t.shape[0] - 1):
        # estimated_omega[i + 1] = estimated_omega[i] * (1 - K) + commanded_steering[i] * K
        dt = t[i + 1] - t[i]
        estimated_steering[i + 1] = estimated_steering[i] + (commanded_steering[i] - estimated_steering[i]) / tau * dt
    err = np.linalg.norm(measured_steering_smooth - estimated_steering) / math.sqrt(t.shape[0])
    mean_err = np.mean(np.abs(measured_steering_smooth - estimated_steering))
    print("estimated err raw= %.3f(norm), %.3f(mean)" % (err, mean_err))
    norm_err = np.linalg.norm(measured_steering_smooth - commanded_steering) / math.sqrt(t.shape[0])
    mean_err = np.mean(np.abs(measured_steering_smooth - commanded_steering))
    print("raw err raw= %.3f(norm), %.3f(mean)" % (norm_err, mean_err))

    # tau = 0.1633
    # estimated_steering2 = np.zeros_like(t)
    # for i in range(t.shape[0] - 1):
    #     # estimated_steering2[i + 1] = estimated_steering2[i] * (1 - K) + commanded_steering[i] * K
    #     dt = t[i + 1] - t[i]
    #     estimated_steering2[i + 1] = estimated_steering2[i] + (commanded_steering[i] - estimated_steering2[i]) / tau * dt
    plt.plot(t, commanded_steering, label='command')
    # plt.plot(t,measured_steering,label='measured')
    plt.plot(t, measured_steering_smooth, label='measured_smooth')
    plt.plot(t, estimated_steering, label='estimated')
    plt.plot(t, estimated_steering - measured_steering_smooth, label="error")
    # plt.plot(t, estimated_steering2, label='nick')
    plt.legend()
    plt.show()
