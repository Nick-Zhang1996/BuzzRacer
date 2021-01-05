# validate kinematic moel

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

if (len(sys.argv) != 2):
    filename = "../log/jan3/full_state1.p"
    print_info("using %s"%(filename))
    #print_error("Specify a log to load")
else:
    filename = sys.argv[1]
with open(filename, 'rb') as f:
    data = pickle.load(f)
data = np.array(data)
data = data.squeeze(1)

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

'''
# use kalman filter results
x = exp_kf_x
y = exp_kf_y
vx = exp_kf_vx
vy = exp_kf_vy
heading = exp_kf_theta
'''
omega = exp_kf_omega

data_len = t.shape[0]

history_steps = 5
forward_steps = 3

full_state_vec = []

track = RCPtrack()
track.load()


def show(img):
    plt.imshow(img)
    plt.show()
    return

#state: x,vx(global),y,vy,heading,omega
#control: steering(rad),throttle(raw unit -1 ~ 1)
def step(state,control,dt=0.01):
    # constants
    L = 0.102
    lr = 0.036
    # convert to local frame
    x,vxg,y,vyg,heading,omega = tuple(state)
    steering,throttle = tuple(control)
    vx = vxg*cos(heading) + vyg*sin(heading)
    vy = -vxg*sin(heading) + vyg*cos(heading)

    # some convenience variables
    R = L/tan(steering)
    beta = atan(lr/R)
    norm = lambda a,b:(a**2+b**2)**0.5

    #advance model
    vx = max(0,vx + (throttle - 0.24)*7.0*dt)
    #vx = vx + (throttle)*7.0*dt
    vy = norm(vx,vy)*sin(beta)
    assert vy*steering>0


    # NOTE where to put this
    omega = vx/R

    # back to global frame
    vxg = vx*cos(heading)-vy*sin(heading)
    vyg = vx*sin(heading)+vy*cos(heading)

    # apply updates
    x += vxg*dt
    y += vyg*dt
    heading += omega*dt

    return (x,vxg,y,vyg,heading,omega )

def step_new(state,control,dt=0.01):
    # constants
    L = 0.102
    lr = 0.036
    # convert to local frame
    x,vxg,y,vyg,heading,omega = tuple(state)
    steering,throttle = tuple(control)
    vx = vxg*cos(heading) + vyg*sin(heading)
    vy = -vxg*sin(heading) + vyg*cos(heading)

    # some convenience variables
    R = L/tan(steering)
    beta = atan(lr/R)
    norm = lambda a,b:(a**2+b**2)**0.5

    #advance model
    vx = max(0.0,vx + (throttle - 0.24)*7.0*dt)
    #vx = vx + (throttle)*7.0*dt
    vy = norm(vx,vy)*sin(beta)
    assert vy*steering>0

    # NOTE heuristics
    vy -= 0.68*vx*steering


    # NOTE where to put this
    omega = vx/R

    # back to global frame
    vxg = vx*cos(heading)-vy*sin(heading)
    vyg = vx*sin(heading)+vy*cos(heading)

    # apply updates
    x += vxg*dt
    y += vyg*dt
    heading += omega*dt

    return (x,vxg,y,vyg,heading,omega )

max_slip = 0.0
#get lateral acceleration from slip angle (rad
def tireCurve(slip,Cf=0.1):
    global max_slip
    if np.abs(slip) > max_slip:
        max_slip = np.abs(slip)
        print(max_slip)

    # satuation slip angle
    Bf = 10.0
    # peel-away sharpness, lower Cf = more gental peel away(street tire)
    Cf = 0.1
    # no slip tire stiffness
    # use this to control 1deg = 0.1g
    Df = 1.0*(180.0/np.pi)/ Bf / Cf
    retval = Df * np.sin( Cf * np.arctan( Bf * slip ) )
    return retval

def step_dynamics(state,control,dt=0.01):
    # constants
    lf = 0.09-0.036
    lr = 0.036

    # convert to local frame
    x,vxg,y,vyg,heading,omega = tuple(state)
    steering,throttle = tuple(control)
    # forward
    vx = vxg*cos(heading) + vyg*sin(heading)
    # lateral, left +
    vy = -vxg*sin(heading) + vyg*cos(heading)

    # TODO handle vx->0
    # for small velocity, use kinematic model 
    slip_f = -np.arctan((omega*lf + vy)/vx) + steering
    slip_r = np.arctan((omega*lr - vy)/vx)
    # we call these acc but they are forces normalized by mass
    lateral_acc_f = tireCurve(slip_f)
    lateral_acc_r = tireCurve(slip_r)
    # TODO use more comprehensive model
    forward_acc_r = (throttle - 0.24)*7.0

    vx += (forward_acc_r - lateral_acc_f * sin(steering) + vy*omega) * dt
    vy += (lateral_acc_r + lateral_acc_f * cos(steering) - vx*omega) * dt
    # leading coeff = m/Iz
    omega += 0.05*12.0/(0.1**2+0.1**2)*(lateral_acc_f * lf * cos(steering) - lateral_acc_r * lr ) * dt

    # back to global frame
    vxg = vx*cos(heading)-vy*sin(heading)
    vyg = vx*sin(heading)+vy*cos(heading)

    # apply updates
    x += vxg*dt
    y += vyg*dt
    heading += omega*dt

    retval = (x,vxg,y,vyg,heading,omega )
    return retval

def test():
    img_track = track.drawTrack()
    #img_track = track.drawRaceline(img=img_track)
    cv2.imshow('validate',img_track)
    cv2.waitKey(10)

    sim_steps = 1000
    x = 1.5
    y = 1.6
    vxg = 1.0
    vyg = 0.5
    heading = radians(30)
    omega = 0.0

    steering = radians(25)
    throttle = 0.5


    state =  (x,vxg,y,vyg,heading,omega )
    predicted_states = []

    start = 0
    for i in range(start,start+sim_steps):
        control = (steering,throttle)
        state = step_dynamics(state,control)
        predicted_states.append(state)

        car_state = (state[0],state[2],state[4],0,0,0)
        img = track.drawCar(img_track.copy(), car_state, steering)

        '''
        cv2.imshow('validate',img)
        k = cv2.waitKey(10) & 0xFF
        if k == ord('q'):
            print("halt")
            break
        sleep(0.05)
        '''

    predicted_states = np.array(predicted_states)
    plt.plot(predicted_states[:,0],predicted_states[:,2])
    plt.show()

def run():
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
        #print(actual_future_traj)
        #img_actual_traj = track.drawPolyline(actual_future_traj,lineColor=(255,0,0),img=img.copy())
        img = track.drawPolyline(actual_future_traj,lineColor=(255,0,0),img=img.copy())
        #show(img)


        '''
        # calculate predicted trajectory -- baseline
        state = (x[i],vx[i],y[i],vy[i],heading[i],omega[i])
        control = (steering[i],throttle[i])
        predicted_states = [state]
        for j in range(i+1,i+lookahead_steps):
            state = step(state,control)
            predicted_states.append(state)
            control = (steering[j],throttle[j])

        predicted_states = np.array(predicted_states)
        predicted_future_traj = np.vstack([predicted_states[:,0],predicted_states[:,2]]).T
        # GREEN
        img = track.drawPolyline(predicted_future_traj,lineColor=(0,255,0),img=img)
        '''

        # calculate predicted trajectory -- heuristic
        state = (x[i],vx[i],y[i],vy[i],heading[i],omega[i])
        control = (steering[i],throttle[i])
        predicted_states = [state]
        print("step = %d"%(i))
        for j in range(i+1,i+lookahead_steps):
            #print(state)
            state = step_dynamics(state,control)
            predicted_states.append(state)
            control = (steering[j],throttle[j])

        predicted_states = np.array(predicted_states)
        predicted_future_traj = np.vstack([predicted_states[:,0],predicted_states[:,2]]).T
        # RED
        img = track.drawPolyline(predicted_future_traj,lineColor=(0,0,255),img=img)

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

        #cv.addWeighted(src1, alpha, src2, beta, 0.0)

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

        #show(img)
        cv2.imshow('validate',img)
        k = cv2.waitKey(10) & 0xFF
        if k == ord('q'):
            print("halt")
            break

if __name__=="__main__":
    #test()
    run()
