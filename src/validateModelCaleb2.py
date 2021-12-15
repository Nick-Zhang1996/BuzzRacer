# visualize model prediction against actual trajectories

import matplotlib
import matplotlib.pyplot as plt

# import keyboard as kb
# import PyQt5
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

from tire import tireCurve

saveGif = True
gifs = []

# if (len(sys.argv) != 2):
#     filename = "../log/jan3/full_state1.p"
#     print_info("using %s"%(filename))
#     #print_error("Specify a log to load")
# else:
#     filename = sys.argv[1]
# with open(filename, 'rb') as f:
#     data = pickle.load(f)
# data = np.array(data)
# data = data.squeeze(1)
#
# skip = 1
# t = data[skip:,0]
# t = t-t[0]
# x = data[skip:,1]
# y = data[skip:,2]
# heading = data[skip:,3]
# steering = data[skip:,4]
# throttle = data[skip:,5]
#
# dt = 0.01
# vx = np.hstack([0,np.diff(x)])/dt
# vy = np.hstack([0,np.diff(y)])/dt
# omega = np.hstack([0,np.diff(heading)])/dt
#
# # local speed
# # forward
# vx_car = vx*np.cos(heading) + vy*np.sin(heading)
# # lateral, left +
# vy_car = -vx*np.sin(heading) + vy*np.cos(heading)
#
# exp_kf_x = data[skip:,6]
# exp_kf_y = data[skip:,7]
# exp_kf_v = data[skip:,8]
# exp_kf_vx = exp_kf_v *np.cos(exp_kf_v)
# exp_kf_vy = exp_kf_v *np.sin(exp_kf_v)
# exp_kf_theta = data[skip:,9]
# exp_kf_omega = data[skip:,10]
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
# omega = exp_kf_omega
#
# data_len = t.shape[0]

history_steps = 5
forward_steps = 3

errorBased = True
fullsim = True
if fullsim:
    # stateArr = np.load("stateValues.npy")
    stateArr = np.load("/home/caleb/Documents/GitHub/RC-VIP/src/stateValues.npy")
    controlArr = np.load("/home/caleb/Documents/GitHub/RC-VIP/src/controlValues.npy")

    xActual = stateArr[:-1, 0]
    yActual = stateArr[:-1, 1]
    headingActual = stateArr[:-1, 2]
    vxActual = stateArr[:-1, 3]
    vyActual = stateArr[:-1, 4]
    omegaActual = stateArr[:-1, 5]

    # data_len = xActual.size # reinstate
    data_len = 1000  # todo get rid of this

    throttle = controlArr[:-1, 0]
    steering = controlArr[:-1, 1]

    plt.plot(xActual, yActual)
    plt.show()
else:

    lf = 0.09 - 0.036
    lr = 0.036
    L = lr + lf

    xInit = 0.5
    yInit = 2
    headInit = -pi / 2
    vInit = 2
    throttle = 0
    steering = 0.2
    state = (xInit, yInit, headInit, vInit, 0, vInit / L * np.tan(steering))
    data_len = 1000

full_state_vec = []

track = RCPtrack()
track.load()


def show(img):
    plt.imshow(img)
    plt.show()
    return


# state: x,vx(global),y,vy,heading,omega
# control: steering(rad),throttle(raw unit -1 ~ 1)
def step_kinematic(state, control, dt=0.03):
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


def step_LTIKinematic(state, control, currentState, currentControl, nextRefTrajState, dt=0.03):
    # car parameters
    lf = 0.09 - 0.036
    lr = 0.036
    L = lr + lf
    m = 1.0

    # currentState is the state at the present time, currentControl is the control at the present time i (as in run())
    # state is a propagated prediction for i + j (as in run())
    # "delta*" variables are in the standard form used in linearization
    x0, y0, heading0, v_forward0, v_sideways0, omega0 = currentState
    x, y, heading, v_forward, v_sideways, omega = state
    deltax = x - x0
    deltay = y - y0
    deltahead = heading - heading0
    deltav = v_forward - v_forward0

    throttle0, steer0 = currentControl
    deltaControl = np.array(control) - np.array(currentControl)
    deltaControl = (0, 0)

    # rs = (round(state[0], 2), round(state[1], 2), round(state[2], 2), round(state[3], 2), round(state[4], 2),
    #       round(state[5], 2))
    # rs = (round(currentState[0], 2), round(currentState[1], 2), round(currentState[2], 2), round(currentState[3], 2),
    #       round(currentState[4], 2), round(currentState[5], 2))
    # rc = (round(deltaControl[0], 2), round(deltaControl[1], 2))
    # rc = (round(currentControl[0], 2), round(currentControl[1], 2))
    # print(rs, rc)
    # print(rc)

    # Varying parameters
    v0 = v_forward0
    psi0 = heading0
    # psidot0 = omega0
    steer0 = currentControl[1]

    A = np.zeros((4, 4))
    A[0][2] = np.cos(psi0)
    A[0][3] = -v0 * np.sin(psi0)
    A[1][2] = np.sin(psi0)
    A[1][3] = v0 * np.cos(psi0)
    # A[2][2] = -np.tan(steer0) * (psidot0 + v0 / L * np.tan(steer0))
    A[3][2] = 1 / L * np.tan(steer0)

    deltaState = np.array([deltax, deltay, deltav, deltahead])

    B = np.zeros((4, 2))
    B[2][0] = 1 / m
    # B[2][1] = -v0 * (1 / (np.cos(steer0)) ** 2) * (psidot0 + v0 / L * np.tan(steer0))
    # print("B[2][1]:  " + str(B[2][1]))
    B[3][1] = (v0 / L) / (np.cos(steer0) ** 2)
    # print("B[3][1]:  " + str(B[3][1])

    deltaStateDot = np.matmul(A, deltaState) + np.matmul(B, deltaControl)  # xdot, ydot, vdot, psidot

    # print('v0 = {0:.3f}    B[3][1] = {1:.3f}    head = {2:.3f}    steer = {3:.3f}    throttle = {4:.3f}'.format(v0, B[3][1], state[3], steer0, control[0]))

    deltaState += deltaStateDot * dt

    # print('LTI omega = {0:.3f}'.format(omega))

    # Add back in linearization point (current state)
    nextxRef, nextyRef, nextheadingRef, nextvfRef, nextvsRef, nextomegaRef = nextRefTrajState
    x = deltaState[0] + nextxRef #x0  # + nextxRef  # (v0 * np.cos(psi0)) * dt +
    y = deltaState[1] + nextyRef #y0  # nextyRef  # (v0 * np.sin(psi0)) * dt +
    heading = deltaState[3] + nextheadingRef #heading0  # + nextheadingRef  # + (v0 / L * np.tan(steer0)) * dt
    # print(heading)
    v_forward = deltaState[2] + nextvfRef #v0  # + nextvfRef  # + (1/m * throttle0) * dt
    # v_forward += control[0] * dt  # todo get rid of this
    # v_forward = max(0, v_forward + (control[0] - 0.24) * 7.0 * dt)  # todo get rid of this
    # v_sideways passes through
    omega = deltaStateDot[3] + nextomegaRef #omega0  # + nextomegaRef  # + v0 / L * np.tan(steer0)

    print('x = {0:.3f}    y = {1:.3f}    head = {2:.3f}    vx = {3:.3f}    vy = {4:.3f}    omega = {4:.3f}'.format(
        deltaState[0], deltaState[1], deltaState[3], deltaState[2], v_sideways, omega))

    if deltaState[3] > 0.15:
        print('heading deviation too large for accurate results')

    print(x0, y0, heading0, v_forward0, v_sideways0, omega0)

    # print("omega = {0:.3f}".format(omega))

    return (x, y, heading, v_forward, v_sideways, omega), {}


# old, kinematic model with correction
# def step_kinematic_heuristic(state,control,dt=0.01):
#     # constants
#     L = 0.102
#     lr = 0.036
#     # convert to local frame
#     x,vxg,y,vyg,heading,omega = tuple(state)
#     steering,throttle = tuple(control)
#     vx = vxg*cos(heading) + vyg*sin(heading)
#     vy = -vxg*sin(heading) + vyg*cos(heading)
#
#     # some convenience variables
#     R = L/tan(steering)
#     beta = atan(lr/R)
#     norm = lambda a,b:(a**2+b**2)**0.5
#
#     #advance model
#     vx = max(0.0,vx + (throttle - 0.24)*7.0*dt)
#     #vx = vx + (throttle)*7.0*dt
#     vy = norm(vx,vy)*sin(beta)
#     assert vy*steering>0
#
#     # NOTE heuristics
#     vy -= 0.68*vx*steering
#
#
#     # NOTE where to put this
#     omega = vx/R
#
#     # back to global frame
#     vxg = vx*cos(heading)-vy*sin(heading)
#     vyg = vx*sin(heading)+vy*cos(heading)
#
#     # apply updates
#     x += vxg*dt
#     y += vyg*dt
#     heading += omega*dt
#
#     return (x,vxg,y,vyg,heading,omega ),{}

# dynamic model with heuristically selected parameters
# def step_dynamics(state,control,dt=0.01):
#     # constants
#     lf = 0.09-0.036
#     lr = 0.036
#     # convert to local frame
#     # x,vxg,y,vyg,heading,omega = tuple(state)
#     x, vxg, y, vyg, heading, omega = (state[0][0], state[0][1], state[0][2], state[0][3], state[0][4], state[0][5])
#     steering,throttle = tuple(control)
#     # forward
#     vx = vxg*cos(heading) + vyg*sin(heading)
#     # lateral, left +
#     vy = -vxg*sin(heading) + vyg*cos(heading)
#
#     # TODO handle vx->0
#     # for small velocity, use kinematic model
#     slip_f = -np.arctan((omega*lf + vy)/vx) + steering
#     slip_r = np.arctan((omega*lr - vy)/vx)
#     # we call these acc but they are forces normalized by mass
#     # TODO consider longitudinal load transfer
#     lateral_acc_f = tireCurve(slip_f) * 9.8 * lr / (lr + lf)
#     lateral_acc_r = tireCurve(slip_r) * 9.8 * lf / (lr + lf)
#     # TODO use more comprehensive model
#     forward_acc_r = (throttle - 0.24)*7.0
#
#     ax = forward_acc_r - lateral_acc_f * sin(steering) + vy*omega
#     ay = lateral_acc_r + lateral_acc_f * cos(steering) - vx*omega
#
#     vx += ax * dt
#     vy += ay * dt
#
#     # leading coeff = m/Iz
#     d_omega = 12.0/(0.1**2+0.1**2)*(lateral_acc_f * lf * cos(steering) - lateral_acc_r * lr )
#     omega += d_omega * dt
#
#     # back to global frame
#     vxg = vx*cos(heading)-vy*sin(heading)
#     vyg = vx*sin(heading)+vy*cos(heading)
#
#     # apply updates
#     # TODO add 1/2 a t2
#     x += vxg*dt
#     y += vyg*dt
#     heading += omega*dt + 0.5* d_omega * dt * dt
#
#     retval = (x,vxg,y,vyg,heading,omega )
#     debug_dict = {"slip_f":slip_f, "slip_r":slip_r, "lateral_acc_f":lateral_acc_f, "lateral_acc_r":lateral_acc_r, 'ax':ax}
#     return retval, debug_dict

# model with parameter from ukf
# def step_ukf(state, control, dt=0.01):
#     # constants
#     lf = 0.09 - 0.036
#     lr = 0.036
#     L = 0.09
#
#     Df = 3.93731
#     Dr = 6.23597
#     C = 2.80646
#     B = 0.51943
#     Cm1 = 6.03154
#     Cm2 = 0.96769
#     Cr = -0.20375
#     Cd = 0.00000
#     Iz = 0.00278
#     m = 0.1667
#
#     # convert to local frame
#     x, vxg, y, vyg, heading, omega = tuple(state)
#     steering, throttle = tuple(control)
#     # forward
#     vx = vxg * cos(heading) + vyg * sin(heading)
#     # lateral, left +
#     vy = -vxg * sin(heading) + vyg * cos(heading)
#
#     # for small velocity, use kinematic model
#     if vx < 0.05:
#         beta = atan(lr / L * tan(steering))
#         norm = lambda a, b: (a ** 2 + b ** 2) ** 0.5
#         # motor model
#         d_vx = ((Cm1 - Cm2 * vx) * throttle - Cr - Cd * vx * vx)
#         vx = vx + d_vx * dt
#         vy = norm(vx, vy) * sin(beta)
#         d_omega = 0.0
#         omega = vx / L * tan(steering)
#
#         slip_f = 0
#         slip_r = 0
#         Ffy = 0
#         Fry = 0
#
#     else:
#         slip_f = -np.arctan((omega * lf + vy) / vx) + steering
#         slip_r = np.arctan((omega * lr - vy) / vx)
#
#         Ffy = Df * np.sin(C * np.arctan(B * slip_f)) * 9.8 * lr / (lr + lf) * m
#         Fry = Dr * np.sin(C * np.arctan(B * slip_r)) * 9.8 * lf / (lr + lf) * m
#
#         # motor model
#         Frx = ((Cm1 - Cm2 * vx) * throttle - Cr - Cd * vx * vx) * m
#
#         # Dynamics
#         d_vx = 1.0 / m * (Frx - Ffy * np.sin(steering) + m * vy * omega)
#         d_vy = 1.0 / m * (Fry + Ffy * np.cos(steering) - m * vx * omega)
#         d_omega = 1.0 / Iz * (Ffy * lf * np.cos(steering) - Fry * lr)
#
#         # discretization
#         vx = vx + d_vx * dt
#         vy = vy + d_vy * dt
#         omega = omega + d_omega * dt
#
#         # back to global frame
#     vxg = vx * cos(heading) - vy * sin(heading)
#     vyg = vx * sin(heading) + vy * cos(heading)
#
#     # apply updates
#     # TODO add 1/2 a t2
#     x += vxg * dt
#     y += vyg * dt
#     heading += omega * dt + 0.5 * d_omega * dt * dt
#
#     retval = (x, vxg, y, vyg, heading, omega)
#     debug_dict = {"slip_f": slip_f, "slip_r": slip_r, "lateral_acc_f": Ffy / m, "lateral_acc_r": Fry / m, 'ax': d_vx}
#     return retval, debug_dict
#
#
# # model with parameter from ukf
# def step_ukf_linear(state, control, dt=0.01):
#     # constants
#     lf = 0.09 - 0.036
#     lr = 0.036
#     L = 0.09
#
#     '''
#     Df = 3.93731
#     Dr = 6.23597
#     C = 2.80646
#     B = 0.51943
#     '''
#     # Cm1 = 6.03154
#     Cm2 = 0.96769
#     # Cr = -0.20375
#     Cm1 = 9.23154
#     Cr = 0.0
#     Cd = 0.00000
#     # Iz = 0.00278
#     m = 0.1667
#     Iz = m * (0.1 ** 2 + 0.1 ** 2) / 12.0 * 6.0
#     K = 5.0
#
#     # convert to local frame
#     x, vxg, y, vyg, heading, omega = tuple(state)
#     steering, throttle = tuple(control)
#     # forward
#     vx = vxg * cos(heading) + vyg * sin(heading)
#     # lateral, left +
#     vy = -vxg * sin(heading) + vyg * cos(heading)
#
#     # for small velocity, use kinematic model
#     if vx < 0.05:
#         beta = atan(lr / L * tan(steering))
#         norm = lambda a, b: (a ** 2 + b ** 2) ** 0.5
#         # motor model
#         d_vx = ((Cm1 - Cm2 * vx) * throttle - Cr - Cd * vx * vx)
#         vx = vx + d_vx * dt
#         vy = norm(vx, vy) * sin(beta)
#         d_omega = 0.0
#         omega = vx / L * tan(steering)
#
#         slip_f = 0
#         slip_r = 0
#         Ffy = 0
#         Fry = 0
#
#     else:
#         slip_f = -np.arctan((omega * lf + vy) / vx) + steering
#         slip_r = np.arctan((omega * lr - vy) / vx)
#
#         # tire model -- pacejka model
#         # Ffy = Df * np.sin( C * np.arctan(B *slip_f)) * 9.8 * lr / (lr + lf) * m
#         # Fry = Dr * np.sin( C * np.arctan(B *slip_r)) * 9.8 * lf / (lr + lf) * m
#
#         Ffy = K * slip_f * 9.8 * lr / (lr + lf) * m
#         Fry = K * slip_r * 9.8 * lf / (lr + lf) * m
#
#         # motor model
#         Frx = ((Cm1 - Cm2 * vx) * throttle - Cr - Cd * vx * vx) * m
#
#         # Dynamics
#         d_vx = 1.0 / m * (Frx - Ffy * np.sin(steering) + m * vy * omega)
#         d_vy = 1.0 / m * (Fry + Ffy * np.cos(steering) - m * vx * omega)
#         d_omega = 1.0 / Iz * (Ffy * lf * np.cos(steering) - Fry * lr)
#
#         # discretization
#         vx = vx + d_vx * dt
#         vy = vy + d_vy * dt
#         omega = omega + d_omega * dt
#
#         # back to global frame
#     vxg = vx * cos(heading) - vy * sin(heading)
#     vyg = vx * sin(heading) + vy * cos(heading)
#
#     # apply updates
#     # TODO add 1/2 a t2
#     x += vxg * dt
#     y += vyg * dt
#     heading += omega * dt + 0.5 * d_omega * dt * dt
#
#     retval = (x, vxg, y, vyg, heading, omega)
#     debug_dict = {"slip_f": slip_f, "slip_r": slip_r, "lateral_acc_f": Ffy / m, "lateral_acc_r": Fry / m, 'ax': d_vx}
#     return retval, debug_dict


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


def run():
    global state

    # step_fun = step_ukf_linear
    # step_fun2 = step_ukf
    # step_fun = step_kinematic_heuristic
    step_fun = step_kinematic
    step_fun2 = step_LTIKinematic

    '''
    plt.plot(x,y)
    plt.show()

    plt.plot(vx)
    plt.plot(vy)
    plt.show()

    plt.plot(heading)
    plt.show()

    plt.plot(omega)
    plt.show()
    '''

    img_track = track.drawTrack()
    img_track = track.drawRaceline(img=img_track)
    cv2.imshow('validate', img_track)
    cv2.waitKey(10)

    lookahead_steps = 33
    debug_dict_hist = {"slip_f": [[]], "slip_r": [[]], "lateral_acc_f": [[]], "lateral_acc_r": [[]], 'ax': [[]]}
    if not fullsim:
        nextState = state  # just for the first pass through
    for i in range(1, data_len - lookahead_steps - 1):

        sleep(0.05)

        if fullsim:
            # prepare states
            # draw car current pos
            car_state = (xActual[i], yActual[i], headingActual[i], 0, 0, 0)
            img = track.drawCar(img_track.copy(), car_state, steering[i])

            # plot actual future trajectory
            actual_future_traj = np.vstack([xActual[i:i + lookahead_steps], yActual[i:i + lookahead_steps]]).T
            img = track.drawPolyline(actual_future_traj, lineColor=(255, 0, 0), img=img.copy())  # BLUE

            # distance travelled in actual future trajectory
            cum_distance_actual = 0.0
            cum_distance_actual_list = []
            for j in range(i, i + lookahead_steps - 1):
                dist = ((xActual[j + 1] - xActual[j]) ** 2 + (yActual[j + 1] - yActual[j]) ** 2) ** 0.5
                cum_distance_actual += dist
                cum_distance_actual_list.append(dist)

            # velocity in horizon
            v_actual_hist = (vxActual[i:i + lookahead_steps] ** 2 + vyActual[i:i + lookahead_steps] ** 2) ** 0.5

            # show(img)

        else:
            state = nextState
            car_state = (state[0], state[1], state[2], 0, 0, 0)
            img = track.drawCar(img_track.copy(), car_state, steering)
            initState = state

        # calculate predicted trajectory -- baseline

        if fullsim:
            state = (xActual[i], yActual[i], headingActual[i], vxActual[i], vyActual[i], omegaActual[i])
            control = (throttle[i], steering[i])
        else:
            control = (throttle, steering)

        nonlinear_states = [state]
        for j in range(i + 1, i + lookahead_steps):
            state, debug_dict = step_fun(state, control)
            nonlinear_states.append(state)
            if fullsim:
                control = (throttle[j], steering[j])

        nonlinear_states = np.array(nonlinear_states)

        if not fullsim:
            nextState = nonlinear_states[1, :]

        predicted_future_traj = np.vstack([nonlinear_states[:, 0], nonlinear_states[:, 1]]).T
        # GREEN
        # img = track.drawPolyline(predicted_future_traj, lineColor=(0, 255, 0), img=img)

        if fullsim:
            # SLIGHT PERTURBATION TO HEADING (0.0005 RAD) TO SEE HOW ERROR PROPAGATES
            state = (xActual[i], yActual[i], headingActual[i] + 0.005, vxActual[i], vyActual[i], omegaActual[i])
            control = (throttle[i], steering[i])
        else:
            # SLIGHT PERTURBATION TO HEADING (0.0005 RAD) TO SEE HOW ERROR PROPAGATES
            state = (initState[0], initState[1], initState[2], initState[3], initState[4], initState[5])
            control = (throttle, steering)

        predicted_states = [state]
        print("step = %d" % i)

        # debug_dict_hist is 2 level nested list
        # first dim is time step
        # second is prediction in timestep
        for key in debug_dict_hist:
            debug_dict_hist[key].append([])

        if fullsim:
            # SLIGHT PERTURBATION TO HEADING (0.0005 RAD) TO SEE HOW ERROR PROPAGATES
            state = (xActual[i]+0.01, yActual[i], headingActual[i] + 0.05, vxActual[i]+0.1, vyActual[i], omegaActual[i])
            control = (throttle[i], steering[i])
        else:
            # SLIGHT PERTURBATION TO HEADING (0.0005 RAD) TO SEE HOW ERROR PROPAGATES
            state = (initState[0], initState[1], initState[2] + 0.005, initState[3], initState[4], initState[5])
            control = (throttle, steering)
        print("step = %d" % i)

        # make prediction from current state
        for j in range(i + 1, i + lookahead_steps):
            # print(state)

            if fullsim:
                nextRefTrajState = (xActual[j], yActual[j], headingActual[j], vxActual[j], vyActual[j],
                                    omegaActual[j])
            else:
                nextRefTrajState = nonlinear_states[j - i]

            if errorBased:
                if fullsim:
                    currentState = (xActual[i], yActual[i], headingActual[i], vxActual[i], vyActual[i], omegaActual[i])
                    currentControl = (throttle[i], steering[i])

                    state, debug_dict = step_fun2(state, control, currentState, currentControl, nextRefTrajState)
                else:
                    state, debug_dict = step_fun2(state, control, initState, control, nextRefTrajState)
            else:
                state, debug_dict = step_fun2(state, control)

            '''
            # NOTE use ground truth in velocity
            # calculate actual velocity in world frame
            # using ground truth in longitudinal vel, estimated value in lateral vel
            vx_car_truth = vx_car[j]
            vy_car_predicted = -state[1]*np.sin(state[4]) + state[3]*np.cos(state[4])

            _vxg = vx_car_truth*cos(state[4])-vy_car_predicted*sin(state[4])
            _vyg = vx_car_truth*sin(state[4])+vy_car_predicted*cos(state[4])

            state = (state[0], _vxg, state[2], _vyg, state[4], state[5])
            '''

            for key in debug_dict:
                value = debug_dict[key]
                debug_dict_hist[key][i].append(value)

            predicted_states.append(state)
            print(state)

            if fullsim:
                control = (throttle[j], steering[j])
            '''
            if (i % 100 ==0 and j<i+3):
                print(debug_dict['slip_f'])
                print(actual_slip_f[i])
            '''

        predicted_states = np.array(predicted_states)
        predicted_future_traj = np.vstack([predicted_states[:, 0], predicted_states[:, 1]]).T
        # MAGENTA
        img = track.drawPolyline(predicted_future_traj, lineColor=(255, 0, 255), img=img)

        # distance travelled in predicted future trajectory
        cum_distance_predicted = 0.0
        cum_distance_predicted_list = []

        for j in range(lookahead_steps - 1):
            dist = ((predicted_future_traj[j + 1, 0] - predicted_future_traj[j, 0]) ** 2 + (
                    predicted_future_traj[j + 1, 1] - predicted_future_traj[j, 1]) ** 2) ** 0.5
            cum_distance_predicted += dist
            cum_distance_predicted_list.append(dist)

        # velocity forward
        vx_predicted_hist = predicted_states[:, 3]
        # velocity predicted
        v_predicted_hist = (predicted_states[:, 3] ** 2 + predicted_states[:, 4] ** 2) ** 0.5
        # vx_car_predicted_hist = predicted_states[:, 1] this is no longer vel in x dir

        # forward
        # vx_car_predicted_hist = predicted_states[:, 1] * np.cos(predicted_states[:, 4]) + predicted_states[:,
        #                                                                                   3] * np.sin(
        #     predicted_states[:, 4])

        # lateral, left +
        # vy_car_predicted_hist = -predicted_states[:, 1] * np.sin(predicted_states[:, 4]) + predicted_states[:,
        #                                                                                    3] * np.cos(
        #     predicted_states[:, 4])

        # heading
        predicted_heading_hist = predicted_states[:, 4]

        # position error predicted vs actual
        # pos_err = ((predicted_future_traj[:, 0] - actual_future_traj[:, 0]) ** 2 + (
        #         predicted_future_traj[:, 1] - actual_future_traj[:, 1]) ** 2) ** 0.5

        # actual slip at front tire
        # NOTE subject to delay etc
        # lf = 0.09 - 0.036
        # actual_slip_f = -np.arctan((omega * lf + vy_car) / vx_car) + steering

        '''
        # calculate predicted trajectory -- longer time step
        state = (x[i],vx[i],y[i],vy[i],heading[i],omega[i])
        control = (steering[i],throttle[i])
        predicted_states = [state]
        speedup = 4
        for j in range(i+1,i+lookahead_steps,speedup):
            state = step_new(state,control,dt=0.01*speedup)
            predicted_states.append(state)
            control = (steering[j],throttle[j])

        predicted_states = np.array(predicted_states)
        predicted_future_traj = np.vstack([predicted_states[:,0],predicted_states[:,2]]).T
        # GREEN
        img = track.drawPolyline(predicted_future_traj,lineColor=(0,255,0),img=img)
        '''

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
        img = cv2.putText(img, step_fun.__name__[5:], org, font,
                          fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow('validate', img)
        k = cv2.waitKey(10) & 0xFF
        if saveGif:
            gifs.append(Image.fromarray(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)))
        if k == ord('q'):
            print("stopping")
            break

        # periodic debugging plots
        if i % 300 == 0:
            # if kb.is_pressed('p'):
            # plt.plot(xActual, yActual)
            # plt.show()
            print("showing heading")
            print("showing velocity (total)")
            print("showing local velocity in car frame")

            # wrap = lambda x: np.mod(x + np.pi, 2 * np.pi) - np.pi
            # print('1')
            # ax0 = plt.subplot(211)
            # print('2')
            # ax0.plot(wrap(predicted_heading_hist) / np.pi * 180, label="heading predicted")
            # ax0.plot(headingActual[i:i + lookahead_steps] / np.pi * 180, label="actual")
            # ax0.legend()
            # print('hello')
            # ax1 = plt.subplot(212)
            # ax1.plot(v_predicted_hist, label="v predicted")
            # ax1.plot(v_actual_hist, label="actual")
            # ax1.plot(steering[i:i + lookahead_steps], label="steering")
            #
            # ax1.legend()

            # ax2 = plt.subplot(413)
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

    # if saveGif:
    #     print("saving gif... be patient")
    #     gif_filename = "validate_model.gif"
    #     gifs[0].save(fp=gif_filename, format='GIF', append_images=gifs, save_all=True, duration=20, loop=0)


if __name__ == "__main__":
    # test()
    run()
    if saveGif:
        print("saving gif... be patient")
        gif_filename = "validate_model.gif"
        gifs[0].save(fp=gif_filename, format='GIF', append_images=gifs, save_all=True, duration=20, loop=0)
