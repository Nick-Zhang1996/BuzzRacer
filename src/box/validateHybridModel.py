# validata long term prediction accuracy of model

import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('../../src/'))
from common import *
from kalmanFilter import KalmanFilter
from math import pi
from scipy.signal import savgol_filter

from hybridSim import hybridSim
from RCPTrack import RCPtrack
import cv2
import torch

if (len(sys.argv) != 2):
    filename = "../log/nov10/full_state1.p"
    print_info("using %s"%(filename))
    #print_error("Specify a log to load")
else:
    filename = sys.argv[1]
with open(filename, 'rb') as f:
    data = pickle.load(f)
data = np.array(data)

skip = 200
t = data[skip:,0]
t = t-t[0]
x = data[skip:,1]
y = data[skip:,2]
heading = data[skip:,3]
steering = data[skip:,4]
throttle = data[skip:,5]

dt = 0.01
vx = np.hstack([0,np.diff(x)])/dt
vy = np.hstack([0,np.diff(y)])/dt
omega = np.hstack([0,np.diff(heading)])/dt

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
omega = exp_kf_omega

data_len = t.shape[0]

history_steps = 5
forward_steps = 3

full_state_vec = []

dtype = torch.double
device = torch.device('cpu') # cpu or cuda
sim = hybridSim(dtype,device,history_steps,forward_steps,dt)

track = RCPtrack()
track.load()

img_track = track.drawTrack()
#img_track = track.drawRaceline(img=img_track)
cv2.imshow('validate',img_track)
cv2.waitKey(10)

def show(img):
    plt.imshow(img)
    plt.show()
    return

def run():
    lookahead_steps = 10
    for i in range(1,data_len-lookahead_steps-1):
        # prepare states
        with torch.no_grad():
            long_acc = sim.getLongitudinalAcc(throttle[i]).detach().item()
        full_state = [x[i],vx[i],y[i],vy[i],heading[i],omega[i],long_acc, steering[i]]
        full_state_vec.append(full_state)

        if (len(full_state_vec)>history_steps):
            full_state_vec.pop(0)
        else:
            continue

        long_acc_vec = [sim.getLongitudinalAcc(throttle[i]).detach().item() for i in range(i+1,i+1+forward_steps)]
        steering_vec = steering[i+1:i+1+forward_steps]
        actions = np.vstack([long_acc_vec,steering_vec]).T
        actions = actions[np.newaxis,...]


        # draw car current pos
        car_state = (x[i],y[i],heading[i],0,0,0)
        img = track.drawCar(img_track.copy(), car_state, steering[i])

        # plot actual future trajectory
        actual_future_traj = np.vstack([x[i:i+lookahead_steps],y[i:i+lookahead_steps]]).T
        img = track.drawPolyline(actual_future_traj,lineColor=(0,0,255),img=img)
        #show(img)


        # plot predicted trajectory
        temp_full_history = full_state_vec.copy()
        predicted_future_traj = []
        predicted_full_state_vec = []
        #print("initial state : %.2f %.2f %.2f %.2f %.2f %.2f"%full_state_vec[-1][0],)
        #print(full_state_vec)
        #print("------")
        with torch.no_grad():
            for j in range(1,lookahead_steps+1):
                # make prediction
                converted_full_state_vec = np.array(temp_full_history)[np.newaxis,...]
                converted_full_state_vec = torch.tensor(converted_full_state_vec,dtype=dtype,device=device,requires_grad=False)

                long_acc_vec = [sim.getLongitudinalAcc(throttle[k]).detach().item() for k in range(i+j,i+j+forward_steps)]
                steering_vec = steering[i+j:i+j+forward_steps]
                actions = np.vstack([long_acc_vec,steering_vec]).T
                actions = actions[np.newaxis,...]
                actions = torch.tensor(actions,dtype=dtype,device=device,requires_grad=False)
                predicted_state = sim(converted_full_state_vec,actions, False).detach().numpy()
                # only use the first prediction
                predicted_state = predicted_state[0,0,:]
                predicted_full_state_vec.append(predicted_state)
                predicted_future_traj.append((predicted_state[0],predicted_state[2]))
                '''
                print("history")
                print(np.array(temp_full_history)[:,:6])
                print("predicted state: ")
                print(np.array(predicted_state)[:6])
                '''
                temp_full_history.pop(0)
                temp_full_history.append(predicted_state)

        predicted_full_state_vec = np.array(predicted_full_state_vec)
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

        img = track.drawPolyline(predicted_future_traj,lineColor=(0,255,0),img=img)
        #show(img)

        cv2.imshow('validate',img)
        k = cv2.waitKey(10) & 0xFF
        if k == ord('q'):
            print("halt")
            break

if __name__=="__main__":
    run()
