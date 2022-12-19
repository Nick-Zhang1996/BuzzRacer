import math
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
from math import radians,degrees
from scipy.signal import savgol_filter
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

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

skip = 50
commanded_steering = np.array(data[skip:,0,7])
t = data[skip:,0,0] - data[skip,0,0]

xActual = data[skip:,0,1]
yActual = data[skip:,0, 2]
headingActual = data[skip:,0, 3]
vxActual = data[skip:,0,4]
vyActual = data[skip:,0, 5]
omegaActual = data[skip:,0, 6]
steering = data[skip:,0, 7]
throttle = data[skip:,0, 8]

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
    m = 0.1667
    # motor_A = 6.17
    # motor_B = 15.2
    # motor_C = 0.333
    # k_D, c_r, tau_a = paramValues
    # motor_A, motor_B, motor_C, k_D, c_r, tau_a = paramValues
    motor_A, k_D, c_r, tau_a = paramValues
    # motor_A, motor_B, motor_C, tau_a = paramValues
    # k_D = c_r = 0
    # motor_A, motor_B, motor_C = paramValues
    guessV = np.zeros_like(vxActual)
    guessA = np.zeros_like(vxActual)
    for i in range(vxActual.shape[0]-1):
        dt = t[i+1] - t[i]
        # a_x0 = motor_A * (throttle[i] - guessV[i] / motor_B - motor_C)
        a_x0 = motor_A * throttle[i]
        v_xdot = guessA[i] - k_D / m * guessV[i] ** 2 - c_r * guessV[i]
        # print("guessV:", guessV[i])
        # print("guessA:", guessA[i])
        # v_xdot = motor_A * (throttle[i] - guessV[i] / motor_B - motor_C)
        a_xdot = 1 / tau_a * (a_x0 - guessA[i])
        guessV[i+1] = guessV[i] + v_xdot * dt
        guessA[i+1] = guessA[i] + a_xdot * dt
    # plt.plot(t, vxActual, label='actual')
    # plt.plot(t, guessV, label='estimated')
    # plt.plot(t, guessV - vxActual, label="error")
    # plt.show()
    err = np.linalg.norm(guessV - vxActual)
    print(err)
    return err


if __name__ == "__main__":
    m = 0.1667
    motor_A = 6.17
    motor_B = 15.2
    motor_C = 0.333
    posBound = (0, None)
    res = minimize(first_order, np.array([6, 0.2, 0.1, 0.05]), method='Nelder-Mead',
                   bounds=(posBound, posBound, posBound, posBound),
                   options={"maxiter": 10000})

    # bounds = ((0,0.4), (0,0.1), (0,0.2))
    # res = differential_evolution(first_order, bounds)
    print(res)
    # construct estimated steering with first order sys
    # motor_A, motor_B, motor_C, k_D, c_r, tau_a = res.x
    motor_A, k_D, c_r, tau_a = res.x
    # k_D = c_r = 0
    guessV = np.zeros_like(vxActual)
    guessA = np.zeros_like(vxActual)
    for i in range(vxActual.shape[0] - 1):
        dt = t[i + 1] - t[i]
        # a_x0 = motor_A * (throttle[i] - guessV[i] / motor_B - motor_C)
        a_x0 = motor_A * throttle[i]
        v_xdot = guessA[i] - k_D / m * guessV[i] ** 2 - c_r * guessV[i]
        # v_xdot = motor_A * (throttle[i] - vxActual[i] / motor_B - motor_C)
        a_xdot = 1 / tau_a * (a_x0 - guessA[i])
        guessV[i + 1] = guessV[i] + v_xdot * dt
        guessA[i + 1] = guessA[i] + a_xdot * dt
    err = np.linalg.norm(guessV - vxActual) / math.sqrt(t.shape[0])


    rsme = 0
    for i in range(t.shape[0]-1):
        rsme += (vxActual[i] - guessV[i])**2
    rsme = math.sqrt(rsme / t.shape[0])
    print("rsme is:", rsme)

    mean_err = np.mean(abs(vxActual - guessV))
    print("estimated err raw= %.3f(norm), %.3f(mean)" % (err, mean_err))
    # norm_err = np.linalg.norm(omegaActual - omega_ss) / math.sqrt(t.shape[0])
    # mean_err = np.mean(np.abs(omega_ss - omegaActual))
    # print("raw err raw= %.3f(norm), %.3f(mean)" % (norm_err, mean_err))

    plt.plot(t, vxActual, label='actual')
    # plt.plot(t, omega_ss,label='steady state')
    # plt.plot(t, measured_steering_smooth, label='measured_smooth')
    plt.plot(t, guessV, label='estimated')
    plt.plot(t, guessV - vxActual, label="error")
    plt.legend()
    plt.show()

