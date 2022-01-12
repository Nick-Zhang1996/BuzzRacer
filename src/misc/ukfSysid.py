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
    #filename = "../log/ethsim/full_state4.p"
    filename = "../log/jan3/full_state1.p"
    print_info("using %s"%(filename))
    #print_error("Specify a log to load")
else:
    filename = sys.argv[1]
with open(filename, 'rb') as f:
    data = pickle.load(f)
data = np.array(data)
data = data.squeeze(1)
data = data[:-100,:]

# wrap heading so there's no discontinuity
#plt.plot(data[:,3])
d_heading = np.diff(data[:,3])
d_heading = (d_heading + np.pi) % (2 * np.pi) - np.pi
data[1:,3] = data[0,3] + np.cumsum(d_heading)
#plt.plot(data[:,3])
#plt.show()


skip = 200
end = -500
t = data[skip:end,0]
t = t-t[0]
x = data[skip:end,1]
y = data[skip:end,2]
heading = data[skip:end,3]
steering = data[skip:end,4]
throttle = data[skip:end,5]

# NOTE add some noise
'''
x = x+np.random.normal(0.0,2e-3,size=x.shape)
y = y+np.random.normal(0.0,2e-3,size=x.shape)
heading = heading+np.random.normal(0.0,radians(0.5),size=x.shape)
'''

dt = 0.01
vx = np.hstack([0,np.diff(data[:,1])])/dt
vy = np.hstack([0,np.diff(data[:,2])])/dt
vx = vx[skip:end]
vy = vy[skip:end]

omega = np.hstack([0,np.diff(data[:,3])])/dt
omega = omega[skip:end]


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

def run_ukf():
    global x,vx,y,vy,heading,omega,t
    ukf = UKF()
    ukf.initState(x[0],vx[0],y[0],vy[0],heading[0],omega[0])
    '''
    print("true")
    print("Df, Dr, C, B, Cm1, Cm2, Cr, Cd, Iz (ratio)")
    print(ukf.state[-ukf.param_n:])
    true_param = ukf.state[-ukf.param_n:].copy()

    print("initial")
    print("Df, Dr, C, B, Cm1, Cm2, Cr, Cd, Iz (ratio)")
    '''
    print("initial param")
    initial_param = ukf.state[-ukf.param_n:].copy()
    Df_ratio, Dr_ratio, C_ratio, B_ratio, Cm1_ratio, Cm2_ratio, Cr_ratio, Cd_ratio, Iz_ratio = initial_param

    print("Df, Dr, C, B, Cm1, Cm2, Cr, Cd, Iz (value)")

    print("Df = %.5f"%(ukf.Df*Df_ratio))
    print("Dr = %.5f"%(ukf.Dr*Dr_ratio))
    print("C = %.5f"%(ukf.C*C_ratio))
    print("B = %.5f"%(ukf.B*B_ratio))
    print("Cm1 = %.5f"%(ukf.Cm1*Cm1_ratio))
    print("Cm2 = %.5f"%(ukf.Cm2*Cm2_ratio))
    print("Cr = %.5f"%(ukf.Cr*Cr_ratio))
    print("Cd = %.5f"%(ukf.Cd*Cd_ratio))
    print("Iz = %.5f"%(ukf.Iz*Iz_ratio))


    # NOTE add noise to initial parameter
    # uniform, * 0.5-1.5
    # keep Iz ground truth
    #ukf.state[-ukf.param_n:-1] = ukf.state[-ukf.param_n:-1] * (np.random.rand(ukf.param_n-1)+0.5)
    #print(ukf.state[-ukf.param_n:])

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
    print("Df, Dr, C, B, Cm1, Cm2, Cr, Cd, Iz (ratio)")
    print(ukf.state[-ukf.param_n:])
    final_param = ukf.state[-ukf.param_n:].copy()
    Df_ratio, Dr_ratio, C_ratio, B_ratio, Cm1_ratio, Cm2_ratio, Cr_ratio, Cd_ratio, Iz_ratio = final_param

    print("Df, Dr, C, B, Cm1, Cm2, Cr, Cd, Iz (value)")

    print("Df = %.5f"%(ukf.Df*Df_ratio))
    print("Dr = %.5f"%(ukf.Dr*Dr_ratio))
    print("C = %.5f"%(ukf.C*C_ratio))
    print("B = %.5f"%(ukf.B*B_ratio))
    print("Cm1 = %.5f"%(ukf.Cm1*Cm1_ratio))
    print("Cm2 = %.5f"%(ukf.Cm2*Cm2_ratio))
    print("Cr = %.5f"%(ukf.Cr*Cr_ratio))
    print("Cd = %.5f"%(ukf.Cd*Cd_ratio))
    print("Iz = %.5f"%(ukf.Iz*Iz_ratio))



    state_cov = [ukf.state_cov[i,i] for i in range(ukf.state.shape[0])]
    print("state_cov")
    print(state_cov)

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
    ax2.plot(state_hist[:,6]/initial_param[0],label="param")
    ax2.plot(state_hist[:,7]/initial_param[1],label="param")
    ax2.plot(state_hist[:,8]/initial_param[2],label="param")
    ax2.plot(state_hist[:,9]/initial_param[3],label="param")
    ax2.plot(state_hist[:,10]/initial_param[4],label="param")
    ax2.plot(state_hist[:,11]/initial_param[5],label="param")
    ax2.plot(state_hist[:,12]/initial_param[6],label="param")
    ax2.plot(state_hist[:,13]/initial_param[7],label="param")
    ax2.plot(state_hist[:,14]/initial_param[8],label="param")
    #ax2.legend()
    plt.show()
    return ukf

def testPredict(ukf,show=False):
    # record ukf param
    ukf_param = ukf.state[-ukf.param_n:].copy()

    if (show):
        img_track = track.drawTrack()
        #img_track = track.drawRaceline(img=img_track)
        cv2.imshow('validate',img_track)
        cv2.waitKey(10)

    lookahead_steps = 100
    state_diff_vec = []
    for i in range(100,data_len-lookahead_steps-1-200):
        # prepare states
        # draw car current pos
        car_state = (x[i],y[i],heading[i],0,0,0)
        if (show):
            img = track.drawCar(img_track.copy(), car_state, steering[i])

        # plot actual future trajectory
        actual_future_traj = np.vstack([x[i:i+lookahead_steps],y[i:i+lookahead_steps]]).T
        if (show):
            img = track.drawPolyline(actual_future_traj,lineColor=(255,0,0),img=img.copy())

        # calculate predicted trajectory 
        state = (x[i],vx[i],y[i],vy[i],heading[i],omega[i])
        control = (throttle[i],steering[i])
        predicted_states = []
        #print("step = %d"%(i))
        #ukf.initState(x[i], vx[i], y[i], vy[i], heading[i], omega[i])
        #predicted_states.append(ukf.state)
        # NOTE check dimension, should be col vector
        joint_state = np.hstack([state, ukf_param]).reshape(-1,1)
        for j in range(i+1,i+lookahead_steps):
            #print(ukf.state_cov[0,0])
            #ukf.state, ukf.state_cov = ukf.predict(ukf.state, ukf.state_cov, control, 0.01)
            #print(ukf.state[:ukf.state_n])
            #ukf.state, _ = ukf.predict(ukf.state, ukf.state_cov, control, 0.01)

            joint_state = ukf.advanceModel(joint_state,control,dt=0.01)

            predicted_states.append(joint_state)

            control = (throttle[j],steering[j])

        predicted_states = np.array(predicted_states)

        xx = predicted_states[:,0]
        yy = predicted_states[:,2]
        hh = predicted_states[:,4]
        vv = (predicted_states[:,1]**2 + predicted_states[:,3]**2)**0.5


        xx_real = x[i:i+lookahead_steps]
        yy_real = y[i:i+lookahead_steps]
        hh_real = heading[i:i+lookahead_steps]
        vv_real = (vx[i:i+lookahead_steps]**2 + vy[i:i+lookahead_steps]**2)**0.5

        debug = False
        if debug:
            plt.plot(vv)
            plt.plot(vv_real,'--')
            plt.show()

        predicted_future_traj = np.hstack([xx,yy])
        if (show):
            img = track.drawPolyline(predicted_future_traj,lineColor=(0,0,255),img=img)

            cv2.imshow('validate',img)
            k = cv2.waitKey(10) & 0xFF
            if k == ord('q'):
                print("halt")
                break

        # calculate difference between actual and predicted traj
        pos_diff = ((xx_real - xx)**2 + (yy_real - yy)**2)**0.5
        vel_diff = vv - vv_real
        heading_diff = hh_real - hh

        pos_diff = np.mean(np.abs(pos_diff))
        vel_diff = np.mean(np.abs(vel_diff))
        heading_diff = np.mean(np.abs(heading_diff))

        state_diff = (pos_diff,  vel_diff, heading_diff)
        state_diff_vec.append(state_diff)
    # sim finish
    state_diff_vec = np.array(state_diff_vec)
    print("state diff, avg")
    state_diff_mean = np.mean(np.abs(state_diff_vec),axis=0)
    print("pos %.2f"%(state_diff_mean[0]))
    print("vel %.2f"%(state_diff_mean[1]))
    print("heading %.2f (deg)"%(degrees(state_diff_mean[2])))


if __name__=="__main__":
    ukf = run_ukf()
    testPredict(ukf,show=True)
