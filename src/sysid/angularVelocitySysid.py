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

xActual = data[:,0,1]
yActual = data[:,0, 2]
headingActual = data[:,0, 3]
vxActual = data[:,0,4]
vyActual = data[:,0, 5]
omegaActual = data[:,0, 6]
steering = data[:,0, 7]
throttle = data[:,0, 8]

L = 0.09

measured_steering = (measured_steering+0.5*np.pi)%(np.pi)-0.5*np.pi
measured_steering_smooth = savgol_filter(measured_steering, 19, 2)
vxActual = savgol_filter(vxActual, 19, 2)
omegaActual = savgol_filter(omegaActual, 19, 2)

# add 7ms delay
# t = t[7:]
# commanded_steering = commanded_steering[:-7]
# measured_steering_smooth = measured_steering_smooth[7:]

def first_order(paramValues):
    # construct estimated steering with first order sys
    tau_omega, K_us = paramValues
    guess = np.zeros_like(omegaActual)
    for i in range(omegaActual.shape[0]-1):
        # guess[i+1] = guess[i] * (1-K) + src[i] * K
        dt = t[i+1] - t[i]
        o_ss = vxActual[i] / L * (measured_steering_smooth[i] - K_us)
        Omegadot = 1 / tau_omega * (o_ss - guess[i])
        guess[i+1] = guess[i] + Omegadot * dt
    err = np.linalg.norm(guess - omegaActual)
    return err


if __name__ == "__main__":

    res = minimize(first_order, np.array([0.1, -0.1]), method='Nelder-Mead')
    print(res)

    # construct estimated steering with first order sys
    estimated_omega = np.zeros_like(t)
    omega_ss = np.zeros_like(t)
    tau_omega, K_us = res.x
    # K = 0.2
    # K = 0.0867
    for i in range(t.shape[0] - 1):
        # estimated_omega[i + 1] = estimated_omega[i] * (1 - K) + commanded_steering[i] * K
        dt = t[i + 1] - t[i]
        o_ss = vxActual[i] / L * (measured_steering_smooth[i] - K_us)
        omega_ss[i + 1] = o_ss
        Omegadot = 1 / tau_omega * (o_ss - estimated_omega[i])
        estimated_omega[i + 1] = estimated_omega[i] + Omegadot * dt
    err = np.linalg.norm(omegaActual - estimated_omega) / math.sqrt(t.shape[0])

    rsme = 0
    for i in range(t.shape[0]-1):
        rsme += (omegaActual[i] - estimated_omega[i])**2
    rsme = math.sqrt(rsme / t.shape[0])
    print("rsme is:", rsme)

    mean_err = np.mean(abs(omegaActual - estimated_omega))
    print("estimated err raw= %.3f(norm), %.3f(mean)" % (err, mean_err))
    norm_err = np.linalg.norm(omegaActual - omega_ss) / math.sqrt(t.shape[0])
    mean_err = np.mean(np.abs(omega_ss - omegaActual))
    print("raw err raw= %.3f(norm), %.3f(mean)" % (norm_err, mean_err))

    plt.plot(t, omegaActual, label='actual')
    plt.plot(t, omega_ss,label='steady state')
    # plt.plot(t, measured_steering_smooth, label='measured_smooth')
    plt.plot(t, estimated_omega, label='estimated')
    plt.plot(t, estimated_omega - omegaActual, label="error")
    plt.legend()
    plt.show()
