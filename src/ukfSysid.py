# use ukf to fit model parameters
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('../../src/'))
from common import *
from kalmanFilter import KalmanFilter
from math import pi,degrees,radians,sin,cos,tan,atan
from scipy.signal import savgol_filter

from RCPTrack import RCPtrack
import cv2
from time import sleep

from ukf import UKF

if (len(sys.argv) != 2):
    filename = "../log/ethsim/full_state2.p"
    print_info("using %s"%(filename))
    #print_error("Specify a log to load")
else:
    filename = sys.argv[1]
with open(filename, 'rb') as f:
    data = pickle.load(f)
data = np.array(data)
data = data.squeeze(1)
data = data[:-100,:]

skip = 200
t = data[skip:,0]
t = t-t[0]
x = data[skip:,1]
y = data[skip:,2]
heading = data[skip:,3]
steering = data[skip:,4]
throttle = data[skip:,5]

# NOTE add some noise
x = x+np.random.normal(0.0,2e-3,size=x.shape)
y = y+np.random.normal(0.0,2e-3,size=x.shape)
heading = heading+np.random.normal(0.0,radians(0.5),size=x.shape)

dt = 0.01
vx = np.hstack([0,np.diff(data[:,1])])/dt
vy = np.hstack([0,np.diff(data[:,2])])/dt
vx = vx[skip:]
vy = vy[skip:]

omega = np.hstack([0,np.diff(data[:,3])])/dt
omega = omega[skip:]


# local speed
# forward
vx_car = vx*np.cos(heading) + vy*np.sin(heading)
# lateral, left +
vy_car = -vx*np.sin(heading) + vy*np.cos(heading)

'''
exp_kf_x = data[skip:,6]
exp_kf_y = data[skip:,7]
exp_kf_v = data[skip:,8]
exp_kf_vx = exp_kf_v *np.cos(exp_kf_v)
exp_kf_vy = exp_kf_v *np.sin(exp_kf_v)
exp_kf_theta = data[skip:,9]
exp_kf_omega = data[skip:,10]

# use kalman filter results
x = exp_kf_x
y = exp_kf_y
vx = exp_kf_vx
vy = exp_kf_vy
heading = exp_kf_theta
'''
# NOTE using filtered omega
#omega = exp_kf_omega

data_len = t.shape[0]

full_state_vec = []

track = RCPtrack()
track.load()


def show(img):
    plt.imshow(img)
    plt.show()
    return

def run():
    global x,vx,y,vy,heading,omega,t
    ukf = UKF()
    ukf.initState(x[0],vx[0],y[0],vy[0],heading[0],omega[0])
    print("true")
    print("Df, Dr, C, B, Cm1, Cm2, Cr, Cd, Iz")
    print(ukf.state[-ukf.param_n:])
    true_param = ukf.state[-ukf.param_n:].copy()

    print("initial")
    print("Df, Dr, C, B, Cm1, Cm2, Cr, Cd, Iz")
    # NOTE add noise to initial parameter
    # uniform, * 0.5-1.5
    # keep Iz ground truth
    ukf.state[-ukf.param_n:-1] = ukf.state[-ukf.param_n:-1] * (np.random.rand(ukf.param_n-1)+0.5)
    print(ukf.state[-ukf.param_n:])

    init_param = ukf.state[-ukf.param_n:].copy()
    sim_t = t[0]
    log = {'state':[], 'cov':[]}
    for i in range(x.shape[0]-1):
        #print(x[i],vx[i],y[i],vy[i],heading[i],omega[i])
        control = (throttle[i], steering[i])
        ukf.state, ukf.state_cov = ukf.predict(ukf.state, ukf.state_cov, control, t[i+1]-t[i])
        measurement = (x[i+1], y[i+1], heading[i+1])
        ukf.state, ukf.state_cov = ukf.update( ukf.state, ukf.state_cov, measurement )
        log['state'].append(ukf.state)
        log['cov'].append(ukf.state_cov)


    # print final parameters
    print("final")
    print("Df, Dr, C, B, Cm1, Cm2, Cr, Cd, Iz")
    print(ukf.state[-ukf.param_n:])
    final_param = ukf.state[-ukf.param_n:].copy()

    print("initial:")
    print(init_param/true_param)
    print("final:")
    print(final_param/true_param)
    # plot
    state_hist = np.array(log['state'])
    cov_hist = np.array(log['cov'])

    # x, y, basic
    ax0 = plt.subplot(511)
    ax0.plot(x,'--',label="measure x")
    ax0.plot(y,'--',label="measure y")
    ax0.plot(state_hist[:,2],label="ukf y")
    ax0.plot(state_hist[:,0],label="ukf x")
    ax0.legend()

    # vx,vy
    ax1 = plt.subplot(512)
    ax1.plot(vx,'--',label="measure vx")
    ax1.plot(vy,'--',label="measure vy")
    ax1.plot(state_hist[:,1],label="ukf vx")
    ax1.plot(state_hist[:,3],label="ukf vy")
    ax1.set_ylim([-3, 3])
    ax1.legend()

    # heading
    ax2 = plt.subplot(513)
    ax2.plot(heading,'--',label="measure psi")
    ax2.plot(state_hist[:,4],label="ukf psi")
    ax2.legend()

    # omega
    ax2 = plt.subplot(514)
    ax2.plot(omega,'--',label="measure w")
    ax2.plot(state_hist[:,5],label="ukf w")
    ax2.set_ylim([-5,5])
    ax2.legend()

    # params
    ax2 = plt.subplot(515)
    ax2.plot(state_hist[:,6]/true_param[0],label="param")
    ax2.plot(state_hist[:,7]/true_param[1],label="param")
    ax2.plot(state_hist[:,8]/true_param[2],label="param")
    ax2.plot(state_hist[:,9]/true_param[3],label="param")
    ax2.plot(state_hist[:,10]/true_param[4],label="param")
    ax2.plot(state_hist[:,11]/true_param[5],label="param")
    ax2.plot(state_hist[:,12]/true_param[6],label="param")
    ax2.plot(state_hist[:,13]/true_param[7],label="param")
    ax2.plot(state_hist[:,14]/true_param[8],label="param")
    #ax2.legend()
    plt.show()

def testPredict():
    img_track = track.drawTrack()
    #img_track = track.drawRaceline(img=img_track)
    cv2.imshow('validate',img_track)
    cv2.waitKey(10)

    lookahead_steps = 100
    for i in range(1,data_len-lookahead_steps-1):
        # prepare states
        # draw car current pos
        car_state = (x[i],y[i],heading[i],0,0,0)
        img = track.drawCar(img_track.copy(), car_state, steering[i])

        # plot actual future trajectory
        actual_future_traj = np.vstack([x[i:i+lookahead_steps],y[i:i+lookahead_steps]]).T
        img = track.drawPolyline(actual_future_traj,lineColor=(255,0,0),img=img.copy())

        # calculate predicted trajectory -- using predict() in ukf
        state = (x[i],vx[i],y[i],vy[i],heading[i],omega[i])
        control = (throttle[i],steering[i])
        predicted_states = []
        #print("step = %d"%(i))
        ukf = UKF()
        ukf.initState(x[i], vx[i], y[i], vy[i], heading[i], omega[i])
        predicted_states.append(ukf.state)
        for j in range(i+1,i+lookahead_steps):
            #print(ukf.state_cov[0,0])
            #ukf.state, ukf.state_cov = ukf.predict(ukf.state, ukf.state_cov, control, 0.01)
            #print(ukf.state[:ukf.state_n])
            ukf.state, _ = ukf.predict(ukf.state, ukf.state_cov, control, 0.01)
            predicted_states.append(ukf.state)

            control = (throttle[j],steering[j])

        predicted_states = np.array(predicted_states)
        xx = predicted_states[:,0]
        yy = predicted_states[:,2]
        vv = (predicted_states[:,1]**2 + predicted_states[:,3]**2)**0.5
        vv_real = (vx[i:i+lookahead_steps]**2 + vy[i:i+lookahead_steps]**2)**0.5

        debug = False
        if debug:
            plt.plot(vv)
            plt.plot(vv_real,'--')
            plt.show()

        predicted_future_traj = np.vstack([xx,yy]).T
        img = track.drawPolyline(predicted_future_traj,lineColor=(0,0,255),img=img)

        cv2.imshow('validate',img)
        k = cv2.waitKey(10) & 0xFF
        if k == ord('q'):
            print("halt")
            break

if __name__=="__main__":
    run()
    #testPredict()
