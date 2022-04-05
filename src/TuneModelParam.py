"""
Identify error-minimizing values for model parameters
"""

# 3/12/22: 15:32 - 16:45, 17:47 - 18:30, 20:30 - 24:51

import numpy as np
from scipy.optimize import minimize, differential_evolution

from src import validateModelKinetoDynamic
# from src import steeringSysid
from src.common import print_error
import xlsxwriter
import pynput.keyboard as kb
from pynput.keyboard import Key, Listener

######################################################################
# Set up for run() from validate model
import math

import matplotlib
import matplotlib.pyplot as plt

# import keyboard as kb
# import PyQt5
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d, splev

matplotlib.use('GTK3Agg')
import cv2
from PIL import Image
import pickle
import sys
import os

sys.path.append(os.path.abspath('../../src/'))
from common import *
from kalmanFilter import KalmanFilter
from math import pi, degrees, radians, sin, cos, tan, atan
from scipy.signal import savgol_filter

from RCPTrack import RCPtrack

from time import sleep

from tire import tireCurve, newTireCurve, oldTireCurve

saveGif = True
gifs = []

if (len(sys.argv) != 2):
    filename = "/home/caleb/RC-VIP/log/feb25/full_state1.p"  # "../log/feb25/full_state1.p"
    print_info("using %s" % (filename))
    # print_error("Specify a log to load")
# else:
#     filename = sys.argv[1]
with open(filename, 'rb') as f:
    data = pickle.load(f)
data = np.array(data)
data = data.squeeze(1)

skip = 1
t = data[skip:, 0]
t = t - t[0]
xActual = data[skip:, 1]  # changed from just 'x'
yActual = data[skip:, 2]  # changed from just 'y'
headingActual = data[skip:, 3]  # changed from just 'heading'
steering = data[skip:, 4]  # this name aligns with my convention
throttle = data[skip:, 5]  # this name aligns with my convention

dt = 0.01
vxGlobal = np.hstack([0, np.diff(xActual)]) / dt  # changed from just 'vx'
vyGlobal = np.hstack([0, np.diff(yActual)]) / dt  # changed from just 'vy'
# omegaActual = np.hstack([0,np.diff(headingActual)])/dt  # changed from just 'omega', gets overwritten below by ekf

# local speed
# forward
vxActual = vxGlobal * np.cos(headingActual) + vyGlobal * np.sin(headingActual)  # changed from vx_car
# lateral, left +
vyActual = -vxGlobal * np.sin(headingActual) + vyGlobal * np.cos(headingActual)  # changed from vy_car

exp_kf_x = data[skip:, 6]
exp_kf_y = data[skip:, 7]
exp_kf_v = data[skip:, 8]
exp_kf_vx = exp_kf_v * np.cos(exp_kf_v)
exp_kf_vy = exp_kf_v * np.sin(exp_kf_v)
exp_kf_theta = data[skip:, 9]
exp_kf_omega = data[skip:, 10]

'''
# use kalman filter results
x = exp_kf_x
y = exp_kf_y
vx = exp_kf_vx
vy = exp_kf_vy
heading = exp_kf_theta
'''
# NOTE using filtered omega
omegaActual = exp_kf_omega  # changed from just 'omega'

data_len = t.shape[0]

history_steps = 5
forward_steps = 3

# start of my set up code
curvature = 0
lf = 0.09 - 0.036
lr = 0.036
L = lr + lf

full_state_vec = []

track = RCPtrack()
track.load()

n_steps = 1000
uu = np.linspace(0, track.track_length_grid, n_steps + 1)
(x_i, y_i) = splev(uu, track.raceline, der=0)
maxDistance = track.uToS(uu[-1])


def show(img):
    plt.imshow(img)
    plt.show()
    return


# end run() from validate model start up code
########################################################################

# global variables specific to tuning model parameters script
count = 0
errorList = []


def on_press(key):
    return


def on_release(key):
    """On pressing 'esc', exports error table to excel"""
    global errorList
    global count
    if key == Key.esc:
        with xlsxwriter.Workbook("default" + str(count) + '.xlsx') as workbook:
            worksheet = workbook.add_worksheet()
            for row_num, row_data in enumerate(errorList):
                for col_num, col_data in enumerate(row_data):
                    if np.isnan(col_data):
                        col_data = 999999
                    worksheet.write(row_num, col_num, col_data)
    # if key == Key.esc:
    #     # Stop listener
    #     return False


listener = kb.Listener(
    on_press=on_press,
    on_release=on_release)
listener.start()


def run(modelMethod=validateModelKinetoDynamic.step_NonlinearKinetoDynamic,
        lookahead_steps=200, run_steps=1040, paramNames=None, paramValues=None):

    step_fun = modelMethod

    debug_dict_hist = {"Torque Accel": [[]], "Steer": [[]], "Omega": [[]], "OmegaDot": [[]], "zeta": [[]], "n": [[]],
                       "xi": [[]], "omega": [[]], "v_x": [[]]}
    for i in range(1, data_len - lookahead_steps - 1):

        # calculate predicted tractory -- KinetoDynamic model
        # SLIGHT PERTURBATION TO HEADING (0.0005 RAD) TO SEE HOW ERROR PROPAGATES
        # noise = np.random.normal(0, 0.15, None)
        # state = {a_x, delta, v_x, Omega, zeta, n, xi}
        # assume that for initial state, actual steer angle and throttle match requests
        oldStateVersion = (xActual[i], yActual[i], headingActual[i], vxActual[i], vyActual[i], omegaActual[i])
        refPoint, n, refHeading, curvature, _, u = track.localTrajectory(oldStateVersion, L, True)
        xi = headingActual[i] - refHeading
        zeta = float(track.uToS(u))
        state = (throttle[i], steering[i], vxActual[i], omegaActual[i], zeta, n, xi)
        control = (throttle[i], steering[i])

        for key in debug_dict_hist:
            debug_dict_hist[key].append([])
        # make prediction from current state
        for j in range(i + 1, i + lookahead_steps):
            # print(state)
            # if fullsim:
            #
            # else:
            dt = t[j + 1] - t[j]
            state, debug_dict = step_fun(state, control, curvature, dt=dt, paramNames=paramNames,
                                         paramValues=paramValues)

            for key in debug_dict:
                value = debug_dict[key]
                debug_dict_hist[key][i - 1].append(value)

            zeta = state[4] % maxDistance  # need to wrap zeta for interpolation function
            u = track.sToU(zeta)
            # print(zeta)
            # print(u)
            xRef, yRef = splev(u, track.raceline, der=0)
            der = np.array(splev(u, track.raceline, der=1))
            headingRef = math.atan2(der[1], der[0])
            n = state[5]
            x = xRef - n * np.sin(headingRef)
            y = yRef + n * np.cos(headingRef)

            heading = headingRef + state[6]  # state[6] is xi

            control = (throttle[j], steering[j])

        # periodic debugging plots
        if i % run_steps == 0:
            return debug_dict_hist, [t[0:run_steps], xActual[0:run_steps], yActual[0:run_steps], vxActual[0:run_steps],
                                     vyActual[0:run_steps], headingActual[0:run_steps], omegaActual[0:run_steps]]


def wrapper(paramValues, *args):
    global count
    filename, model, lookahead_steps, run_steps, movingAverageWindow, paramNames = args
    print("Starting iteration", count)
    print("Running simulation.")
    module = __import__(filename)
    # runMethod = getattr(module, "run")
    # debug_dict_hist, testRunData = runMethod(model, lookahead_steps, run_steps, paramNames, paramValues)
    modelMethod = getattr(module, model)
    debug_dict_hist, testRunData = run(modelMethod, lookahead_steps, run_steps, paramNames, paramValues)

    paramName = paramNames[0]
    if paramName == "accelTimeConstant":
        m = 0.1667
        k_D = 0
        c_r = 0
        for i in range(0, len(paramNames)):
            name = paramNames[i]
            if name == "dragCoeff":
                k_D = paramValues[i]
            elif name == "rollResistCoeff":
                c_r = paramValues[i]
        actualData = testRunData[3]  # vx data (need ax data)
        for i in range(0, len(actualData)):
            left = movingAverageWindow // 2
            right = (movingAverageWindow - 1) // 2
            if i - left < 0:
                left = i
            if i + right >= len(actualData):
                right = len(actualData) - i - 1
            actualData[i] = sum(actualData[i - left:i + right + 1]) / (left + right + 1)  # smoothed velocity

        adjust = k_D / m * actualData ** 2 + c_r * actualData  # adjust for accel from torque, use smoothed velocity
        actualData = np.diff(actualData) / np.diff(testRunData[:][0])  # get accel, length decreased by 1

        actualData += adjust[:-1]
        predictData = debug_dict_hist["Torque Accel"]
    # elif paramName = 'steerTimeConstant':
    #     actualData = testRunData[] todo: get data with steer angle
    #     predictData = debug_dist_hist["Steer"]
    elif paramName == "angVelTimeConstant" or paramName == "Understeer Gradient":
        actualData = testRunData[6]
        predictData = debug_dict_hist["Omega"]
    else:
        print_error("Don't know how you got here but you're wrong. (Invalid parameter name)")
    print("Finished simulation. Starting error analysis.")

    error = np.zeros_like(actualData)
    for i in range(0, len(actualData) - lookahead_steps):
        horizonEnd = i + lookahead_steps
        error[i] = (np.sum((np.transpose(predictData[i]) - actualData[i:horizonEnd - 1]) ** 2) / (
                lookahead_steps + 1)) ** 0.5
    count += 1
    error = [np.sum(error) / run_steps]
    print("Error for", paramValues, "is", error)
    print()

    errorList.append([*paramValues, *error])

    return np.sum(error) / run_steps


if __name__ == "__main__":
    model = 'step_NonlinearKinetoDynamic'
    filename = 'validateModelKinetoDynamic'
    lookahead_steps = 50  # 200
    run_steps = 1110

    # SMOOTHING vxActual!!!!
    movingAverageWindow = 40
    for i in range(0, len(vxActual)):
        left = movingAverageWindow // 2
        right = (movingAverageWindow - 1) // 2
        if i - left < 0:
            left = i
        if i + right >= len(vxActual):
            right = len(vxActual) - i - 1
        vxActual[i] = sum(vxActual[i - left:i + right + 1]) / (left + right + 1)
    paramNames = ("angVelTimeConstant", "Understeer Gradient")
    bounds = ((0, 3), (-0.5, 0.5))

    # paramNames = ("accelTimeConstant", "dragCoeff", "rollResistCoeff")
    # bounds = ((0, 1), (0, 0.1), (0, 0.3))
    # paramNames = ("accelTimeConstant", "rollResistCoeff")
    # bounds = ((0, 0.2), (0, 0.2))
    # paramNames = ("accelTimeConstant",)
    # bounds = ((0, 1),)

    result = differential_evolution(wrapper, bounds,
                                    (filename, model, lookahead_steps, run_steps, movingAverageWindow, paramNames))
    print(result)

    # CANNOT APPEND TO EXISTING WORKBOOK SO BE CAREFUL!! (will overwrite everything)
    wname = paramNames[0][0:5]
    for i in range(1, len(paramNames)):
        wname += " & " + paramNames[i][0:5]
    with xlsxwriter.Workbook(wname + '.xlsx') as workbook:
        worksheet = workbook.add_worksheet(wname)

        for row_num, row_data in enumerate(errorList):
            for col_num, col_data in enumerate(row_data):
                if np.isnan(col_data):
                    col_data = 999999
                worksheet.write(row_num, col_num, col_data)
