# visualize model prediction against actual trajectories
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
import numpy as np
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
    filename = "/home/caleb/RC-VIP/log/steeringSysid/full_state4.p"  # "/home/caleb/RC-VIP/log/feb25/full_state1.p"  # "../log/feb25/full_state1.p"
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

# plt.plot(t, data[skip:, 1], label="x")
# plt.plot(t, data[skip:, 2], label="y")
# plt.plot(t, data[skip:, 3], label="theta")
# plt.plot(t, data[skip:, 4], label="vx")
# plt.plot(t, data[skip:, 5], label="vy")
# plt.plot(t, data[skip:, 6], label="omega")
# plt.plot(t, data[skip:, 7], label="steer")
# plt.plot(t, data[skip:, 8], label="throt")
# plt.legend()
# plt.show()

xActual = data[skip:, 1]
yActual = data[skip:, 2]
headingActual = data[skip:, 3]
vxActual = data[skip:, 4]
vyActual = data[skip:, 5]
omegaActual = data[skip:, 6]
steering = data[skip:, 7]
throttle = data[skip:, 8]

# xActual = data[skip:, 1]  # changed from just 'x'
# yActual = data[skip:, 2]  # changed from just 'y'
# headingActual = data[skip:, 3]  # changed from just 'heading'
# steering = data[skip:, 4]  # this name aligns with my convention
# throttle = data[skip:, 5]  # this name aligns with my convention
#
# dt = 0.01
# vxGlobal = np.hstack([0, np.diff(xActual)]) / dt  # changed from just 'vx'
# vyGlobal = np.hstack([0, np.diff(yActual)]) / dt  # changed from just 'vy'
# # omegaActual = np.hstack([0,np.diff(headingActual)])/dt  # changed from just 'omega', gets overwritten below by ekf
#
# # local speed
# # forward
# vxActual = vxGlobal * np.cos(headingActual) + vyGlobal * np.sin(headingActual)  # changed from vx_car
# # lateral, left +
# vyActual = -vxGlobal * np.sin(headingActual) + vyGlobal * np.cos(headingActual)  # changed from vy_car
#
# exp_kf_x = data[skip:, 6]
# exp_kf_y = data[skip:, 7]
# exp_kf_v = data[skip:, 8]
# exp_kf_vx = exp_kf_v * np.cos(exp_kf_v)
# exp_kf_vy = exp_kf_v * np.sin(exp_kf_v)
# # exp_kf_theta = data[skip:, 9]
# # exp_kf_omega = data[skip:, 10]
#
# '''
# # use kalman filter results
# x = exp_kf_x
# y = exp_kf_y
# vx = exp_kf_vx
# vy = exp_kf_vy
# heading = exp_kf_theta
# '''
# # NOTE using filtered omega
# omegaActual = exp_kf_omega  # changed from just 'omega'

data_len = t.shape[0]

history_steps = 5
forward_steps = 3

# start of my set up code
visuals = True
fullsim = True
curvature = 0
lf = 0.09 - 0.036
lr = 0.036
L = lr + lf

if not fullsim:
    xInit = 0.5
    yInit = 2
    headInit = -pi / 2
    vInit = 1
    throttle = 0
    steering = 0.2
    state = (xInit, yInit, headInit, vInit, 0.08, vInit / L * np.tan(steering))
    data_len = 1000

    curvature = np.tan(steering) / L

full_state_vec = []

track = RCPtrack()
track.load()

if fullsim:
    # code to facilitate curvilinear coordinates:
    n_steps = 1000
    uu = np.linspace(0, track.track_length_grid, n_steps + 1)
    (x_i, y_i) = splev(uu, track.raceline, der=0)
    maxDistance = track.uToS(uu[-1])


def show(img):
    plt.imshow(img)
    plt.show()
    return


def step_kinematic(state, control, dt=0.01):
    ## This is the old step_kinematic code. Replaced with duplicate of KinetaticSimulator.py
    # # constants
    # L = 0.102
    # lr = 0.036
    # # convert to local frame
    # x, y, heading, vxg, vyg, omega = state
    # steering, throttle = tuple(control)
    # vx = vxg * cos(heading) + vyg * sin(heading)
    # vy = -vxg * sin(heading) + vyg * cos(heading)
    #
    # # some convenience variables
    # R = L / tan(steering)
    # beta = atan(lr / R)
    # norm = lambda a, b: (a ** 2 + b ** 2) ** 0.5
    #
    # # advance model
    # vx = max(0, vx + (throttle - 0.24) * 7.0 * dt)
    # # vx = vx + (throttle)*7.0*dt
    # vy = norm(vx, vy) * sin(beta)
    # assert vy * steering > 0
    #
    # # NOTE where to put this
    # omega = vx / R
    #
    # # back to global frame
    # vxg = vx * cos(heading) - vy * sin(heading)
    # vyg = vx * sin(heading) + vy * cos(heading)
    #
    # # apply updates
    # x += vxg * dt
    # y += vyg * dt
    # heading += omega * dt

    # print('x = {0:.3f}    y = {1:.3f}    head = {2:.3f}    vx = {3:.3f}    vy = {4:.3f}    omega = {4:.3f}'.format(
    #     x, y, heading, vxg, vyg, omega))

    # rc = (round(control[0], 2), round(control[1], 2))
    # print(rc, end='    ')

    lf = 0.09 - 0.036
    lr = 0.036
    max_v = 3.0

    '''
    throttle = np.clip(throttle, -1.0, 1.0)
    steering = np.clip(throttle, -radians(27), radians(27))
    '''
    x, y, heading, v_forward, v_sideways, omega = state
    # slow down if car is in collision
    '''
    if (car.in_collision):
        v *= 0.9
    '''
    throttle = control[0]
    steering = control[1]

    beta = np.arctan(np.tan(steering) * lr / (lf + lr))
    dXdt = v_forward * np.cos(heading + beta)
    dYdt = v_forward * np.sin(heading + beta)
    if (v_forward > max_v):
        dvdt = -0.01
    else:
        dvdt = throttle
    omega = v_forward / lr * np.sin(beta)

    x += dt * dXdt
    y += dt * dYdt
    v_forward += dt * dvdt
    heading += dt * omega

    # print('Nonlin omega = {0:.3f}'.format(omega), end='    ')
    # print("v_f_i = {0:.3f}".format(state[3]), end="    ")

    return (x, y, heading, v_forward, v_sideways, omega), {}


# dt might be very variable? average of 0.0076315888991722695 s
def step_NonlinearKinetoDynamic(state, control, curvature,j, dt=0.0076, paramNames=None, paramValues=None):
    """
    step_NonlinearKinetoDynamic implements the paper 'Real-time optimal control of an autonomous RC car with
    minimum-time maneuvers and a novel kineto-dynamical model'

    The variable names used for this function will follow that of the paper.
    The state variable is {a_x, delta, v_x, Omega, zeta, n, xi}, where:
        a_x is longitudinal acceleration due to torque input (naturally in non-inertial vehicle frame)
            Note that a_x is accel due to torque input, not actual accel. i.e. d(v_x)/dt != a_x
        delta is the actual (as opposed to commanded) steering angle
        v_x is the longitudinal velocity (same frame as a_x)
        Omega is the yaw rate
        zeta is the curvilinear distance along the raceline
        n is the lateral deviation from the raceline
        xi is the relative yaw angle; xi = psi - theta, where:
            psi is the yaw of the vehicle
            theta is the yaw of the line tangent to the raceline at the point defined by zeta

    This state vector is a significant departure from that of the other models and this model is therefor better
    implemented in its own script.

    Equations describing motion are:
    tau_omega * omegaDot + omega = omegaSteadyState = v_x
    """
    # Directly measurable parameters
    lr = 0.09 - 0.036
    lf = 0.036
    L = lr + lf
    m = 0.1667  # * 1.5

    # Hard to measure parameters
    # Assume constant K_us, + means understeer, - means oversteer
    # understeer gradient (see saved paper for analytic estimation?)
    K_us = 0.04559  # 0.01079 from less extensive test  # -3 * pi / 180 initial guess
    tau_a = 0.05  # 0.5447  # 0.00643796  # 0.05 initial guess
    tau_delta = 0.05
    tau_omega = 0.228  # 0.1339  # 0.1445 from less extensive test
    k_D = 0.00650  # 0.01167  # 0.28067972  # 0.3  # drag coefficient
    c_r = 0.2016  # 0.08387754  # 0.1 initial guess # frictional resistance
    # paper includes road gradient but for RC-Car it is zero

    # enable parameter tuning process
    if paramNames:
        for ind in range(0, len(paramNames)):
            paramName = paramNames[ind]
            if paramName == "accelTimeConstant":
                tau_a = paramValues[ind]
            elif paramName == "steerTimeConstant":
                tau_delta = paramValues[ind]
            elif paramName == "angVelTimeConstant":
                tau_omega = paramValues[ind]
            elif paramName == "Understeer Gradient":
                K_us = paramValues[ind]
            elif paramName == "dragCoeff":
                k_D = paramValues[ind]
            elif paramName == "rollResistCoeff":
                c_r = paramValues[ind]
            else:
                print_error("Invalid Parameter name for step_NonlinearKinetoDynamic. Check parameter name spelling.")

    a_x, delta, v_x, Omega, zeta, n, xi = state
    a_x0, delta_0 = control

    oldStateVersion = (xActual[j], yActual[j], headingActual[j], vxActual[j], vyActual[j], omegaActual[j])
    refPoint, _, refHeading, curvature, _, u = track.localTrajectory(oldStateVersion, L, True)
    xi = headingActual[j] - refHeading
    zeta = float(track.uToS(u))

    factor = 1
    a_x *= factor
    a_x0 *= factor

    Omegadot = 1 / tau_omega * (v_x / L * (delta - K_us) - Omega)
    v_xdot = a_x - k_D / m * v_x ** 2 - c_r * v_x
    a_xdot = 1 / tau_a * (a_x0 - a_x)
    deltadot = 1 / tau_delta * (delta_0 - delta)

    zetadot = - (v_x * np.cos(xi)) / (n * curvature - 1)
    # print("zeta dot: ", zetadot)
    ndot = v_x * np.sin(xi)
    xidot = Omega + (v_x * np.cos(xi) * curvature) / (n * curvature - 1)

    # Left Riemann Sum integrate
    Omega += Omegadot * dt
    v_x += v_xdot * dt
    a_x += a_xdot * dt
    delta += deltadot * dt
    zeta += zetadot * dt
    n += ndot * dt
    xi += xidot * dt

    # (a_x, delta, v_x, Omega, zeta, n, xi)
    return (a_x, delta_0, v_x, Omega, zeta, n, xi), {"Torque Accel": a_x, "Steer": delta, "Omega": Omega,
                                                   "OmegaDot": Omegadot, "zeta": zeta, "n": n,
                                                   "xi": xi, "omega": Omega, "v_x": v_x}


def test():
    img_track = track.drawTrack()
    img_track = track.drawRaceline(img=img_track)
    # img_track = track.drawRaceline(img=img_track)
    cv2.imshow('validate', img_track)
    cv2.waitKey(10)

    sim_steps = 3173
    x = 1.5
    y = 1.6
    vxg = 1.0
    vyg = 0.5
    heading = radians(30)
    omega = 0.0

    controlArr = np.load("controlValues.npy")
    # steering = radians(25)
    # throttle = 0.1
    throttle = controlArr[1:-1, 0]
    steering = controlArr[1:-1, 1]

    state = ((x, vxg, y, vyg, heading, omega), 0)
    predicted_states = []

    start = 0
    for i in range(start, start + sim_steps):
        control = (throttle[i], steering[i])
        steerTemp = steering[i]
        state = step_kinematic(state, control)[0]
        predicted_states.append(state[0])

        car_state = (state[0][0], state[0][2], state[0][4], 0, 0, 0)
        img = track.drawCar(img_track.copy(), car_state, steerTemp)

        cv2.imshow('validate', img)
        k = cv2.waitKey(10) & 0xFF
        if k == ord('q'):
            print("halt")
            break
        sleep(0.05)

    # plt.plot(predicted_states[1:10][0], predicted_states[1:10][2])
    # plt.show()

    predicted_states = np.array(predicted_states)
    plt.plot(predicted_states[:, 0], predicted_states[:, 2])
    plt.show()


def run(model="step_NonlinearKinetoDynamic", lookahead_steps=200, run_steps=400, paramNames=None, paramValues=None):
    global state
    # step_fun = step_ukf_linear
    # step_fun2 = step_ukf
    # step_fun = step_kinematic_heuristic
    step_fun = step_kinematic
    step_fun2 = globals()[model]

    # plt.plot(xActual, yActual)
    # plt.show()
    if visuals:
        img_track = track.drawTrack()
        img_track = track.drawRaceline(img=img_track)
        cv2.imshow('validate', img_track)
        cv2.waitKey(10)

    debug_dict_hist = {"Torque Accel": [[]], "Steer": [[]], "Omega": [[]], "OmegaDot": [[]], "zeta": [[]], "n": [[]],
                       "xi": [[]], "omega": [[]], "v_x": [[]]}
    if not fullsim:
        nextState = state  # just for the first pass through
    for i in range(1, data_len - lookahead_steps - 1):

        # sleep(0.1)

        if fullsim:
            # prepare states
            # draw car current pos
            if visuals:
                car_state = (xActual[i], yActual[i], headingActual[i], 0, 0, 0)
                img = track.drawCar(img_track.copy(), car_state, steering[i])

            # plot actual future trajectory
            if visuals:
                actual_future_traj = np.vstack([xActual[i:i + lookahead_steps], yActual[i:i + lookahead_steps]]).T
                img = track.drawPolyline(actual_future_traj, lineColor=(255, 0, 0), img=img.copy())  # BLUE
            # show(img)
        else:
            state = nextState
            car_state = (state[0], state[1], state[2], 0, 0, 0)
            if visuals:
                img = track.drawCar(img_track.copy(), car_state, steering)
            initState = state

        # # calculate predicted trajectory -- baseline
        # if fullsim:
        #     # SLIGHT PERTURBATION TO HEADING (0.0005 RAD) TO SEE HOW ERROR PROPAGATES
        #     state = (xActual[i], yActual[i], headingActual[i], vxActual[i], vyActual[i],
        #              omegaActual[i])
        #     control = (throttle[i], steering[i])
        # else:
        #     control = (throttle, steering)
        #
        # nonlinear_states = [state]
        # for j in range(i + 1, i + lookahead_steps):
        #     state, debug_dict = step_fun(state, control)
        #     nonlinear_states.append(state)
        #     if fullsim:
        #         control = (throttle[j], steering[j])
        #
        # nonlinear_states = np.array(nonlinear_states)
        #
        # if not fullsim:
        #     nextState = nonlinear_states[1, :]
        #
        # nonlinear_future_traj = np.vstack([nonlinear_states[:, 0], nonlinear_states[:, 1]]).T
        # # GREEN
        # # img = track.drawPolyline(nonlinear_future_traj, lineColor=(0, 255, 0), img=img)

        # debug_dict_hist is 2 level nested list
        # first dim is time step
        # second is prediction in timestep
        # for key in debug_dict_hist:
        #     debug_dict_hist[key].append([])

        # calculate predicted tractory -- KinetoDynamic model
        if fullsim:
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
        else:
            # SLIGHT PERTURBATION TO HEADING (0.0005 RAD) TO SEE HOW ERROR PROPAGATES
            state = (initState[0], initState[1], initState[2], initState[3], initState[4], initState[5])
            control = (throttle, steering)
        # print("step = %d" % i)
        # print(state)
        # print(control)
        # print("")
        # print("")

        if visuals:
            predicted_states = [(xActual[i], yActual[i], headingActual[i], throttle[i], steering[i], vxActual[i],
                                 omegaActual[i], zeta, n, xi)]
        for key in debug_dict_hist:
            debug_dict_hist[key].append([])
        # make prediction from current state
        for j in range(i + 1, i + lookahead_steps):
            # print(state)
            # if fullsim:
            #
            # else:
            dt = t[j + 1] - t[j]
            state, debug_dict = step_fun2(state, control, curvature, j, dt=dt, paramNames=paramNames,
                                          paramValues=paramValues)
            # state, debug_dict = step_fun2(state, control)

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

            if visuals:
                predicted_states.append([x, y, heading, state[0], state[1], state[2], state[3], state[4], state[5],
                                         state[6]])
            if fullsim:
                control = (throttle[j], steering[j])

        if visuals:
            predicted_states = np.array(predicted_states)
            predicted_future_traj = np.vstack([predicted_states[:, 0], predicted_states[:, 1]]).T
        # RED
        # for k in range(0, lookahead_steps, 10):
        #     img = track.drawCar(img_track.copy(), predicted_states[k], steering[k])
        if visuals:
            img = track.drawPolyline(predicted_future_traj, lineColor=(255, 0, 255), img=img)

            # cv.addWeighted(src1, alpha, src2, beta, 0.0)
            # show(img) # show each individual frame

            # add text
            # font
            font = cv2.FONT_HERSHEY_SIMPLEX
            # org
            org = (50, 50)
            # fontScale
            fontScale = 1
            # Blue color in BGR
            color = (255, 0, 0)
            # Line thickness of 2 px
            thickness = 2
            # Using cv2.putText() method
            img = cv2.putText(img, step_fun2.__name__[5:], org, font,
                              fontScale, color, thickness, cv2.LINE_AA)
            cv2.imshow('validate', img)
            k = cv2.waitKey(10) & 0xFF
            if saveGif:
                gifs.append(Image.fromarray(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)))
            if k == ord('q'):
                print("stopping")
                break

        # periodic debugging plots
        if i % 1075 == 0:
            ax0 = plt.subplot(311)
            ax0.plot(debug_dict_hist["zeta"][i-1], label="curve dist")
            plt.legend()
            ax1 = plt.subplot(312)
            ax1.plot(debug_dict_hist["n"][i-1], label="n")
            plt.legend()
            ax2 = plt.subplot(313)
            ax2.plot(debug_dict_hist["Steer"][i-1], label="calc steer")
            ax2.plot(steering[i:i+lookahead_steps], label="steer req")
            plt.legend()
            plt.show()

            plt.figure(1)
            plt.plot(debug_dict_hist["xi"][i-1], label="xi")
            plt.plot(debug_dict_hist["omega"][i - 1], label="omega")
            plt.plot(omegaActual[i:i + lookahead_steps], label="omega actual")
            integ = np.cumsum((np.array(debug_dict_hist["omega"][i - 1]) - omegaActual[i:i + lookahead_steps-1])*np.diff(t[i:i+lookahead_steps]))
            plt.plot(integ, label="integral")
            plt.legend()

            plt.figure(2)
            plt.plot(debug_dict_hist["Torque Accel"][i-1], label="ax")
            plt.plot(debug_dict_hist["v_x"][i - 1], label="v_x")
            plt.plot(vxActual[i:i + lookahead_steps], label="vx actual")
            plt.plot(throttle[i:i + lookahead_steps], label="ax_0")
            plt.legend()
            plt.show()

        if i % run_steps == 0:
            return debug_dict_hist, [t[0:run_steps], xActual[0:run_steps], yActual[0:run_steps], vxActual[0:run_steps],
                                     vyActual[0:run_steps], headingActual[0:run_steps], omegaActual[0:run_steps]]
            # if kb.is_pressed('p'):
            # plt.plot(debug_dict_hist["slip_f_force"][i - 1], label="slip_f_force")
            # plt.plot(debug_dict_hist["slip_r_force"][i - 1], label="slip_r_force")
            # # plt.show()
            # print("showing heading")
            # print("showing velocity (total)")
            # print("showing local velocity in car frame")
            #
            # # wrap = lambda x: np.mod(x + np.pi, 2 * np.pi) - np.pi
            # # print('1')
            # # ax0 = plt.subplot(211)
            # # print('2')
            # # ax0.plot(wrap(predicted_heading_hist) / np.pi * 180, label="heading predicted")
            # # ax0.plot(headingActual[i:i + lookahead_steps] / np.pi * 180, label="actual")
            # # ax0.legend()
            # # print('hello')
            # # ax1 = plt.subplot(212)
            # # ax1.plot(v_predicted_hist, label="v predicted")
            # # ax1.plot(v_actual_hist, label="actual")
            # # ax1.plot(steering[i:i + lookahead_steps], label="steering")
            # #
            # # ax1.legend()
            #
            # # ax2 = plt.subplot(413)
            # # (xG, yG, heading, vf, vs, omega)
            # plt.plot(predicted_states[:, 3], label="vf")
            # # ax1.plot(predicted_states[:, 4], label="vs")
            # # ax1.plot(predicted_states[:, 2], label="heading")
            # plt.plot(predicted_states[:, 4], label="vs")
            # plt.plot(predicted_states[:, 2], label="heading")
            # plt.plot(debug_dict_hist["vs*omega"][i - 1], label="vs*omega")
            # plt.plot(debug_dict_hist["-vf*omega"][i - 1], label="-vf*omega")
            # plt.plot(predicted_states[:, 5], label="omega")
            # plt.legend()
            #
            # if fullsim:
            #     plt.figure(2)
            #     plt.plot(vxActual[i:i + lookahead_steps], label="vx actual")
            #     plt.plot(vyActual[i:i + lookahead_steps], label="vy actual")
            #     plt.plot(headingActual[i:i + lookahead_steps], label="head actual")
            #     plt.legend()
            #
            #     plt.figure(3)
            #     plt.plot(predicted_states[:, 0], predicted_states[:, 1], label="pred")
            #     plt.plot(xActual[i:i + lookahead_steps], yActual[i:i + lookahead_steps], label="actual")
            #     plt.legend()
            #
            # plt.figure(4)
            # plt.plot(debug_dict_hist["xddot"][i - 1], label="xddot")
            # plt.plot(debug_dict_hist["yddot"][i - 1], label="yddot")
            # plt.plot(debug_dict_hist["phiddot"][i - 1], label="phiddot")
            # plt.legend()
            #
            # plt.figure(1).show()
            # if fullsim:
            #     plt.figure(2).show()
            #     plt.figure(3).show()
            # plt.figure(4).show()
            # plt.show()

            # ax2.plot(vx_car_predicted_hist, label="car vx predicted")
            # ax2.plot(vx_car[i:i + lookahead_steps], label="car vx actual")
            # ax2.plot(vy_car_predicted_hist,'--',label="car vy predicted")
            # ax2.plot(vy_car[i:i+lookahead_steps],'--',label="car vy actual")
            #
            # ax2.plot(throttle[i:i + lookahead_steps], label="throttle")
            # ax2.plot(debug_dict_hist['ax'][i], '--', label="predicted ax")
            # ax2.legend()
            #
            # ax3 = plt.subplot(414)
            # ax3.plot(debug_dict_hist['slip_f'][i], label="predicted slip front")
            # ax3.plot(actual_slip_f[i:i + lookahead_steps], label="actual slip front")
            # ax3.legend()

            # plt.show(img)
            # print("breakpoint")

        '''
        print("showing x")
        plt.plot(x[i:i+lookahead_steps],'b--')
        plt.plot(predicted_full_state_vec[:,0],'*')
        plt.show()

        print("showing y")
        plt.plot(y[i:i+lookahead_steps],'b--')
        plt.plot(predicted_full_state_vec[:,2],'*')
        plt.show()

        print("showing vx")
        plt.plot(vx[i:i+lookahead_steps],'b--')
        plt.plot(predicted_full_state_vec[:,1],'*')
        plt.show()

        print("showing vy")
        plt.plot(vy[i:i+lookahead_steps],'b--')
        plt.plot(predicted_full_state_vec[:,3],'*')
        plt.show()

        print("showing heading")
        plt.plot(heading[i:i+lookahead_steps],'b--')
        plt.plot(predicted_full_state_vec[:,4],'*')
        plt.show()

        print("showing omega")
        plt.plot(omega[i:i+lookahead_steps],'b--')
        plt.plot(predicted_full_state_vec[:,5],'*')
        plt.show()
        '''


if __name__ == "__main__":
    # test()
    # movingAverageWindow = 20
    # vxActualSmooth = np.zeros_like(vxActual)
    # for i in range(0, len(vxActual)):
    #     left = movingAverageWindow // 2
    #     right = (movingAverageWindow - 1) // 2
    #     if i - left < 0:
    #         left = i
    #     if i + right >= len(vxActual):
    #         right = len(vxActual) - i - 1
    #     vxActualSmooth[i] = sum(vxActual[i - left:i + right + 1]) / (left + right + 1)
    # plt.plot(t, vxActual)
    # plt.plot(t[:-1], np.diff(vxActual))
    # plt.plot(t, vxActualSmooth)
    # plt.plot(t[:-1], np.diff(vxActualSmooth))
    # plt.show()
    run()
    if saveGif:
        print("saving gif... be patient")
        gif_filename = "validate_model.gif"
        gifs[0].save(fp=gif_filename, format='GIF', append_images=gifs, save_all=True, duration=20, loop=0)
