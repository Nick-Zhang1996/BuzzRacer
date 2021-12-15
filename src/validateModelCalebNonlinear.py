# visualize model prediction against actual trajectories
import math

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

from tire import tireCurve, newTireCurve, oldTireCurve

saveGif = True
gifs = []

if (len(sys.argv) != 2):
    filename = "/home/caleb/Documents/GitHub/RC-VIP/log/feb25/full_state1.p"  # "../log/feb25/full_state3.p"
    # filename = "/home/caleb/Documents/GitHub/RC-VIP/log/sep28full_state1.p"
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
fullsim = True
# if fullsim:
#     # stateArr = np.load("stateValues.npy")
#     stateArr = np.load("/home/caleb/Documents/GitHub/RC-VIP/src/stateValues.npy")
#     controlArr = np.load("/home/caleb/Documents/GitHub/RC-VIP/src/controlValues.npy")
#
#     xActual = stateArr[:-1, 0]
#     yActual = stateArr[:-1, 1]
#     headingActual = stateArr[:-1, 2]
#     vxActual = stateArr[:-1, 3]
#     vyActual = stateArr[:-1, 4]
#     omegaActual = stateArr[:-1, 5]
#
#     # data_len = xActual.size # reinstate
#     data_len = 1000  # todo get rid of this
#
#     throttle = controlArr[:-1, 0]
#     steering = controlArr[:-1, 1]
#
#     plt.plot(xActual, yActual)
#     plt.show()
if not fullsim:

    lf = 0.09 - 0.036
    lr = 0.036
    L = lr + lf

    data_len = 1000

    xInit = 0.5
    yInit = 5
    headInit = -pi / 2
    vxInit = 1
    controlArr = np.zeros((data_len, 2))
    controlArr[range(200,data_len), 1] = 0.2
    vyInit = 0.5 * controlArr[0,1] / vxInit # heuristic formula
    state = (xInit, yInit, headInit, vxInit, vyInit, 0)#vxInit / L * np.tan(steering))
    print(state)

full_state_vec = []

track = RCPtrack()
track.load()


def show(img):
    plt.imshow(img)
    plt.show()
    return


# state: x,vx(global),y,vy,heading,omega
# control: steering(rad),throttle(raw unit -1 ~ 1)
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


def step_Nonlinear(state, control, dt=0.01):
    lf = 0.09 - 0.036
    lr = 0.036
    # lf = 0.045
    # lr = 0.045
    L = lr + lf
    h = 0.01
    m = 0.1667 * 1.5
    I = 0.00278 * 0.1
    g = 9.81

    xG, yG, heading, vf, vs, omega = state
    print(state)
    print(control)
    throt, steer = control
    scaleToForce = 0.8  # Scale throttle command to force
    throt = throt * scaleToForce

    frontslip = np.arctan2(vs + lf * omega, vf) - steer
    rearslip = np.arctan2(vs - lr * omega, vf)

    frontFriction = saturationTireModel(frontslip)
    rearFriction = saturationTireModel(rearslip)

    Wf = 0.5 * (scaleToForce * throt * h + m * g * lr) / L  # 0.25 * m * g #
    Wr = 0.5 * (-scaleToForce * throt * h + m * g * lf) / L  # 0.25 * m * g #

    Ffx = -Wf * frontFriction * np.sin(steer)   # Ffx = Ffl * cos(steer) - Ffc * sin(steer); Ffl = 0, Ffc = mu * N
    Ffy = Wf * frontFriction * np.cos(steer)    # Ffy = Ffl * sin(steer) + Ffc * cos(steer); Ffl = 0, Ffc = mu * N
    Frx = throt / 2                          # Frx = Frl * cos(0) + Frc * sin(0); Frl = throttle, sin(0) = 0
    Fry = Wr * rearFriction                     # Fry = Frl * sin(0) + Frc * cos(0); sin(0) = 0, Frc = mu * N

    xddot = vs * omega + 2 / m * (Ffx + Frx)

    yddot = -vf * omega + 2 / m * (Ffy + Fry)

    phiddot = 2 / I * (lf * Ffy - lr * Fry)

    # convert back to global
    vxG = vf * np.cos(heading) - vs * np.sin(heading)
    vyG = vf * np.sin(heading) + vs * np.cos(heading)

    xG += vxG * dt
    yG += vyG * dt
    heading += omega * dt

    vf += xddot * dt
    vs += yddot * dt
    omega += phiddot * dt

    # heading += omega * dt  # where should this be

    print("")

    return (xG, yG, heading, vf, vs, omega), {"slip_f": frontslip, "slip_r": rearslip, "-vf*omega": -vf*omega,
                                              "vs*omega": vs*omega, "xddot": xddot, "yddot": yddot, "phiddot": phiddot,
                                              "slip_f_force": Wf * frontFriction,
                                              "slip_r_force": Wr * rearFriction}


def saturationTireModel(slip):

    slope = -1
    frictionCoef = slope * slip
    frictionCoef = np.clip(frictionCoef, -1, 1)
    print("Friction Coef: {0:.3f}".format(frictionCoef))
    return frictionCoef


def run():
    global state
    global stateLine
    global stateCircle

    # step_fun = step_ukf_linear
    # step_fun2 = step_ukf
    # step_fun = step_kinematic_heuristic
    step_fun = step_kinematic
    step_fun2 = step_Nonlinear

    plt.plot(xActual, yActual)
    plt.show()
    '''
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

    lookahead_steps = 100
    debug_dict_hist = {"slip_f": [[]], "slip_r": [[]], "vs*omega": [[]], "-vf*omega": [[]], "xddot": [[]],
                       "yddot": [[]], "phiddot": [[]], "slip_f_force": [[]], "slip_r_force": [[]]}
    if not fullsim:
        nextState = state  # just for the first pass through
    for i in range(1, data_len - lookahead_steps - 1):

        # sleep(0.1)

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
            img = track.drawCar(img_track.copy(), car_state, controlArr[i, 1])
            initState = state

        # calculate predicted trajectory -- baseline

        if fullsim:
            # state = (xActual[i], yActual[i], headingActual[i], vxActual[i], vyActual[i], omegaActual[i])
            # control = (throttle[i], steering[i])
            # SLIGHT PERTURBATION TO HEADING (0.0005 RAD) TO SEE HOW ERROR PROPAGATES
            state = (xActual[i], yActual[i], headingActual[i], vxActual[i], vyActual[i],
                     omegaActual[i])
            control = (throttle[i], steering[i])
        else:
            control = (controlArr[i, 0], controlArr[i, 1])

        if not fullsim:
            nonlinear_states = [state]
            for j in range(i + 1, i + lookahead_steps):
                state, debug_dict = step_fun(state, control)
                nonlinear_states.append(state)
                if fullsim:
                    control = (throttle[j], steering[j])
                else:
                    control = (controlArr[j, 0], controlArr[j, 1])

            nonlinear_states = np.array(nonlinear_states)

            if not fullsim:
                nextState = nonlinear_states[1, :]

            nonlinear_future_traj = np.vstack([nonlinear_states[:, 0], nonlinear_states[:, 1]]).T
            # GREEN
            img = track.drawPolyline(nonlinear_future_traj, lineColor=(0, 255, 0), img=img)

        # debug_dict_hist is 2 level nested list
        # first dim is time step
        # second is prediction in timestep
        # for key in debug_dict_hist:
        #     debug_dict_hist[key].append([])

        if fullsim:
            # SLIGHT PERTURBATION TO HEADING (0.0005 RAD) TO SEE HOW ERROR PROPAGATES
            state = (xActual[i], yActual[i], headingActual[i], vxActual[i], vyActual[i], omegaActual[i])
            control = (throttle[i], steering[i])
        else:
            # SLIGHT PERTURBATION TO HEADING (0.0005 RAD) TO SEE HOW ERROR PROPAGATES
            state = initState #(initState[0], initState[1], initState[2], initState[3], initState[4], initState[5])
            control = (controlArr[i, 0], controlArr[i, 1])
        print("step = %d" % i)
        print(state)
        print(control)
        print("")
        print("")

        predicted_states = [state]
        for key in debug_dict_hist:
            debug_dict_hist[key].append([])
        # make prediction from current state
        for j in range(i + 1, i + lookahead_steps):
            # print(state)
            if fullsim:
                refTrajState = (xActual[j - 1], yActual[j - 1], headingActual[j - 1], vxActual[j - 1], vyActual[j - 1],
                                omegaActual[j - 1])
                nextRefTrajState = (xActual[j], yActual[j], headingActual[j], vxActual[j], vyActual[j],
                                    omegaActual[j])
            # else:
            # refTrajState = nonlinear_states[j - (i + 1)]
            # nextRefTrajState = nonlinear_states[j - i]

            # state, debug_dict = step_fun2(state, control, refTrajState, nextRefTrajState)
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
                debug_dict_hist[key][i - 1].append(value)

            predicted_states.append(state)
            if fullsim:
                control = (throttle[j], steering[j])
            else:
                control = (controlArr[j, 0], controlArr[j, 1])
            '''
            if (i % 100 ==0 and j<i+3):
                print(debug_dict['slip_f'])
                print(actual_slip_f[i])
            '''

        predicted_states = np.array(predicted_states)
        predicted_future_traj = np.vstack([predicted_states[:, 0], predicted_states[:, 1]]).T
        # MAGENTA
        # for k in range(0, lookahead_steps, 10):
        #     img = track.drawCar(img_track.copy(), predicted_states[k], steering[k])
        img = track.drawPolyline(predicted_future_traj, lineColor=(255, 0, 255), img=img)
            # sleep(0.1)

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
        if i % 1000 == 0:
            print("showing plots")

            # old debugging plots
            # plt.plot(debug_dict_hist["slip_f_force"][i - 1], label="slip_f_force")
            # plt.plot(debug_dict_hist["slip_r_force"][i - 1], label="slip_r_force")
            # ax1.plot(predicted_states[:, 3], label="vf")
            # ax1.plot(predicted_states[:, 2], label="heading")
            # plt.plot(predicted_states[:, 4], label="vs")
            # plt.plot(predicted_states[:, 2], label="heading")
            # plt.plot(debug_dict_hist["vs*omega"][i - 1], label="vs*omega")
            # plt.plot(debug_dict_hist["-vf*omega"][i - 1], label="-vf*omega")
            # plt.plot(predicted_states[:, 5], label="omega")
            # plt.legend()

            if fullsim:
                plt.figure(1)
                ax1 = plt.subplot(311)
                ax1.plot(vxActual[i:i + lookahead_steps], label="vx actual")
                ax1.plot(vyActual[i:i + lookahead_steps], label="vy actual")
                plt.plot(predicted_states[:, 3], label="vx predicted")
                plt.plot(predicted_states[:, 4], label="vy predicted")
                ax1.legend()

                ax2 = plt.subplot(312)
                ax2.plot(omegaActual[i:i + lookahead_steps], label="omega actual")
                ax2.plot(predicted_states[:, 5], label="omega predicted")
                ax2.legend()

                ax3 = plt.subplot(313)
                vxError = predicted_states[:, 3] - vxActual[i:i + lookahead_steps]
                vyError = predicted_states[:, 4] - vyActual[i:i + lookahead_steps]
                omegaError = predicted_states[:, 5] - omegaActual[i:i + lookahead_steps]
                ax3.plot(vxError, label="vx Error")
                ax3.plot(vyError, label="vy Error")
                ax3.plot(omegaError, label="omega Error")
                ax3.legend()


                plt.figure(2)
                ax1 = plt.subplot(311)
                ax1.plot(predicted_states[:, 0], predicted_states[:, 1], label="predicted")
                ax1.plot(xActual[i:i + lookahead_steps], yActual[i:i + lookahead_steps], label="actual")
                ax1.legend()

                ax2 = plt.subplot(312)
                XError = predicted_states[:, 0] - xActual[i:i + lookahead_steps]
                YError = predicted_states[:, 1] - yActual[i:i + lookahead_steps]
                ax2.plot(XError, label="X Error")
                ax2.plot(YError, label="Y Error")
                ax2.legend()

                ax3 = plt.subplot(313)
                ax3.plot(headingActual[i:i + lookahead_steps], label="Heading Actual")
                ax3.plot((predicted_states[:, 2] + math.pi) % (2 * math.pi) - math.pi, label="Heading Predicted")
                ax3.legend()

            plt.figure(3)
            plt.plot(debug_dict_hist["xddot"][i - 1], label="xddot")
            plt.plot(debug_dict_hist["yddot"][i - 1], label="yddot")
            plt.plot(debug_dict_hist["phiddot"][i - 1], label="phiddot")
            plt.legend()

            if fullsim:
                plt.figure(1).show()
                plt.figure(2).show()
            plt.figure(3).show()
            plt.show()

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
    run()
    if saveGif:
        print("saving gif... be patient")
        gif_filename = "validate_model.gif"
        gifs[0].save(fp=gif_filename, format='GIF', append_images=gifs, save_all=True, duration=20, loop=0)
