# visualize model prediction against actual trajectories

import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
thisdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(thisdir))
from common import *
from util.kalmanFilter import KalmanFilter
from math import pi,degrees,radians,sin,cos,tan,atan
from scipy.signal import savgol_filter

from track import RCPTrack
import cv2
from time import sleep

from tire import tireCurve
from PIL import Image

from collections import namedtuple

def loadLog(filename=None):
    if (len(sys.argv) != 2):
        if (filename is None):
            print_error("Specify a log to load")
        print_info("using %s"%(filename))
    else:
        filename = sys.argv[1]
    with open(filename, 'rb') as f:
        log = pickle.load(f)
    log = np.array(log)
    log = log.squeeze(1)
    return log

def prepLog(log,skip=1):
    #time(),x,y,theta,v_forward,v_sideway,omega, car.steering,car.throttle
    t = log[skip:,0]
    t = t-t[0]
    x = log[skip:,1]
    y = log[skip:,2]
    heading = log[skip:,3]
    v_forward = log[skip:,4]
    v_sideway = log[skip:,5]
    # NOTE
    #omega = log[skip:,6]
    omega = np.hstack([0,np.diff(wrapContinuous(heading))])/0.01
    steering = log[skip:,7]
    throttle = log[skip:,8]

    Log = namedtuple('Log', 't x y heading v_forward v_sideway omega steering throttle')
    mylog = Log(t,x,y,heading,v_forward,v_sideway,omega,steering,throttle)
    return mylog

def loadMeasuredSteering(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    measured_steering = np.array(data[0]['measured_steering'])
    measured_steering = (measured_steering+0.5*np.pi)%(np.pi)-0.5*np.pi
    measured_steering_smooth = savgol_filter(measured_steering, 19,2)
    return measured_steering_smooth

def show(img):
    plt.imshow(img)
    plt.show()
    return

# model for rigged vehicle, using raw steering
def step_raw(state,control,dt=0.01,slip_f_override=None):
    # constants
    L = 0.09
    lf = 0.04824
    lr = L - lf

    # measured through torsional pendulum
    Iz = 417757e-9
    m = 0.1667

    # convert to local frame
    x,vxg,y,vyg,heading,omega = tuple(state)
    steering,throttle = tuple(control)
    # forward
    vx = vxg*cos(heading) + vyg*sin(heading)
    # lateral, left +
    vy = -vxg*sin(heading) + vyg*cos(heading)

    # for small velocity, use kinematic model 
    if (vx<0.05):
        beta = atan(lr/L*tan(steering))
        norm = lambda a,b:(a**2+b**2)**0.5
        # motor model
        d_vx = 6.17*(throttle - vx/15.2 -0.333)

        vx = vx + d_vx * dt
        vy = norm(vx,vy)*sin(beta)
        d_omega = 0.0
        omega = vx/L*tan(steering)

        slip_f = 0
        slip_r = 0
        Ffy = 0
        Fry = 0

    else:
        # Dynamics
        # motor model
        # NOTE need 0.07s delay
        d_vx = 6.17*(throttle - vx/15.2 -0.333)

        slip_f = -np.arctan((omega*lf + vy)/vx) + steering
        slip_r = np.arctan((omega*lr - vy)/vx)

        #Ffy = tireCurve(slip_f) * m * ( 9.8 *lr/(lr+lf) - d_vx*h/(lr+lf))
        #Fry = 1.15*tireCurve(slip_r) * m * ( 9.8 *lf/(lr+lf) + d_vx*h/(lr+lf))
        Ffy = 0.9*tireCurve(slip_f) * m * 9.8 *lr/(lr+lf)
        Fry = 0.95*tireCurve(slip_r) * m * 9.8 *lf/(lr+lf)

        d_vy = 1.0/m * (Fry + Ffy * np.cos( steering ) - m * vx * omega)
        d_omega = 1.0/Iz * (Ffy * lf * np.cos( steering ) - Fry * lr)

        # discretization
        vx = vx + d_vx * dt
        vy = vy + d_vy * dt
        omega = omega + d_omega * dt 

    # back to global frame
    vxg = vx*cos(heading)-vy*sin(heading)
    vyg = vx*sin(heading)+vy*cos(heading)

    # apply updates
    # TODO add 1/2 a t2
    x += vxg*dt
    y += vyg*dt
    heading += omega*dt + 0.5* d_omega * dt * dt

    retval = (x,vxg,y,vyg,heading,omega )
    debug_dict = {"slip_f":slip_f, "slip_r":slip_r, "lateral_acc_f":Ffy/m, "lateral_acc_r":Fry/m, 'ax':d_vx,'dw':d_omega,'w':omega}
    return retval, debug_dict

# model for rigged vehicle
def step_rig(state,control,dt=0.01,slip_f_override=None):
    # constants
    lf = 0.09-0.036
    lr = 0.036
    L = 0.09

    Df = 3.93731
    Dr = 6.23597
    C = 2.80646
    B = 0.51943
    Iz = 0.00278*0.5
    m = 0.1667

    # convert to local frame
    x,vxg,y,vyg,heading,omega = tuple(state)
    steering,throttle = tuple(control)
    # forward
    vx = vxg*cos(heading) + vyg*sin(heading)
    # lateral, left +
    vy = -vxg*sin(heading) + vyg*cos(heading)

    # for small velocity, use kinematic model 
    if (vx<0.05):
        beta = atan(lr/L*tan(steering))
        norm = lambda a,b:(a**2+b**2)**0.5
        # motor model
        d_vx = 0.425*(15.2*throttle - vx - 3.157)

        vx = vx + d_vx * dt
        vy = norm(vx,vy)*sin(beta)
        d_omega = 0.0
        omega = vx/L*tan(steering)

        slip_f = 0
        slip_r = 0
        Ffy = 0
        Fry = 0

    else:
        slip_f = -np.arctan((omega*lf + vy)/vx) + steering
        slip_r = np.arctan((omega*lr - vy)/vx)

        Ffy = Df * np.sin( C * np.arctan(B *slip_f)) * 9.8 * lr / (lr + lf) * m
        Fry = Dr * np.sin( C * np.arctan(B *slip_r)) * 9.8 * lf / (lr + lf) * m

        # motor model
        #Frx = (1.8*0.425*(15.2*throttle - vx - 3.157))*m
        # Dynamics
        #d_vx = 1.0/m * (Frx - Ffy * np.sin( steering ) + m * vy * omega)
        d_vx = 1.8*0.425*(15.2*throttle - vx - 3.157)

        d_vy = 1.0/m * (Fry + Ffy * np.cos( steering ) - m * vx * omega)
        d_omega = 1.0/Iz * (Ffy * lf * np.cos( steering ) - Fry * lr)

        # discretization
        vx = vx + d_vx * dt
        vy = vy + d_vy * dt
        omega = omega + d_omega * dt 

    # back to global frame
    vxg = vx*cos(heading)-vy*sin(heading)
    vyg = vx*sin(heading)+vy*cos(heading)

    # apply updates
    # TODO add 1/2 a t2
    x += vxg*dt
    y += vyg*dt
    heading += omega*dt + 0.5* d_omega * dt * dt

    retval = (x,vxg,y,vyg,heading,omega )
    debug_dict = {"slip_f":slip_f, "slip_r":slip_r, "lateral_acc_f":Ffy/m, "lateral_acc_r":Fry/m, 'ax':d_vx,'dw':d_omega,'w':omega}
    return retval, debug_dict

def getDistanceTravelled(log, i, lookahead):
    # distance travelled in actual future trajectory
    cum_distance_actual = 0.0
    cum_distance_actual_list = []
    x = log.x
    y = log.y
    for j in range(i,i+lookahead-1):
        dist = ((x[j+1] - x[j])**2 + (y[j+1] - y[j])**2)**0.5
        cum_distance_actual += dist
        cum_distance_actual_list.append(dist)

    return cum_distance_actual, cum_distance_actual_list

def addAlgorithmName(img,step_fun):
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
    return img

def run():
    # setting
    lookahead_steps = 50
    saveGif = False
    gifs = []

    # set prediction function
    step_fun = step_raw
    step_fun_base = step_rig

    # load log
    #filename = '../../log/2022_2_9_exp/full_state4.p'
    filename = '../../log/2022_3_2_exp/full_state2.p'
    rawlog = loadLog(filename)
    log = prepLog(rawlog,skip=1)
    dt = 0.01
    vx_alt = np.hstack([0,np.diff(log.x)])/dt
    vy_alt = np.hstack([0,np.diff(log.y)])/dt
    heading = log.heading
    vx = log.v_forward*np.cos(heading)-log.v_sideway*np.sin(heading)
    vy = log.v_forward*np.sin(heading)+log.v_sideway*np.cos(heading)

    data_len = log.t.shape[0]

    # use measured steering
    #filename = '../../log/2022_2_7_exp/debug_dict2.p'
    '''
    filename = '../../log/2022_2_9_exp/debug_dict4.p'
    measured_steering = loadMeasuredSteering(filename)[:-1]
    offset = (-np.mean(measured_steering) + np.mean(log.steering))
    measured_steering = measured_steering + offset
    '''

    # prep track image
    track = RCPTrack()
    track.load()
    img_track = track.drawTrack()
    img_track = track.drawRaceline(img=img_track)
    cv2.imshow('validate',img_track)
    cv2.waitKey(10)

    # debug_dict_hist[key][log_timestep][prediction_timestep]
    debug_dict_hist = {"slip_f":[[]], "slip_r":[[]], "lateral_acc_f":[[]], "lateral_acc_r":[[]],'ax':[[]],'dw':[[]],'w':[[]]}

    # iterate through log
    for i in range(1,data_len-lookahead_steps-1):
        # prepare state shorthands
        x = log.x
        y = log.y
        heading = log.heading
        omega = log.omega
        steering = log.steering
        #steering = measured_steering
        throttle = log.throttle
        vx_car = log.v_forward
        vy_car = log.v_sideway

        car_state = (x[i],y[i],heading[i],0,0,0)
        img = track.drawCar(img_track.copy(), car_state, steering[i])

        # plot actual future trajectory
        actual_future_traj = np.vstack([x[i:i+lookahead_steps],y[i:i+lookahead_steps]]).T
        # BLUE
        img = track.drawPolyline(actual_future_traj,lineColor=(255,0,0),img=img.copy())

        state = (x[i],vx[i],y[i],vy[i],heading[i],omega[i])
        control = (steering[i],throttle[i])
        predicted_states = [state]
        print("step = %d"%(i))

        # prepare debug_dict_hist
        for key in debug_dict_hist:
            debug_dict_hist[key].append([])

        # plot predicted trajectory
        for j in range(i+1,i+lookahead_steps):
            # make prediction
            state, debug_dict = step_fun(state,control)

            for key in debug_dict:
                value = debug_dict[key]
                debug_dict_hist[key][i].append(value)
            predicted_states.append(state)
            # delay for throttle
            index = max(j-7,0)
            index = j
            control = (steering[j],throttle[index])

        predicted_states = np.array(predicted_states)
        predicted_future_traj = np.vstack([predicted_states[:,0],predicted_states[:,2]]).T
        # GREEN
        img = track.drawPolyline(predicted_future_traj,lineColor=(0,255,0),img=img)

        '''
        # plot benchmark prediction trajectory
        state = (x[i],vx[i],y[i],vy[i],heading[i],omega[i])
        control = (steering[i],throttle[i])
        predicted_states = [state]
        for j in range(i+1,i+lookahead_steps):
            # make prediction
            state, debug_dict = step_fun_base(state,control)

            predicted_states.append(state)
            control = (steering[j],throttle[j])

        predicted_states = np.array(predicted_states)
        predicted_future_traj = np.vstack([predicted_states[:,0],predicted_states[:,2]]).T
        # RED
        img = track.drawPolyline(predicted_future_traj,lineColor=(100,100,255),img=img)
        '''


        img = addAlgorithmName(img, step_fun)

        cv2.imshow('validate',img)
        k = cv2.waitKey(10) & 0xFF
        if saveGif:
            gifs.append(Image.fromarray(cv2.cvtColor(img.copy(),cv2.COLOR_BGR2RGB)))
        if k == ord('q'):
            print("stopping")
            break


        # distance travelled in predicted future trajectory
        cum_distance_predicted = 0.0
        cum_distance_predicted_list = []
        for j in range(lookahead_steps-1):
            dist = ((predicted_future_traj[j+1,0] - predicted_future_traj[j,0])**2 + (predicted_future_traj[j+1,1] - predicted_future_traj[j,1])**2)**0.5
            cum_distance_predicted += dist
            cum_distance_predicted_list.append(dist)

        # velocity in horizon
        v_actual_hist = (vx[i:i+lookahead_steps]**2 + vy[i:i+lookahead_steps]**2)**0.5

        cum_distance_actual, cum_distance_actual_list = getDistanceTravelled(log, i, lookahead_steps)

        # velocity predicted
        v_predicted_hist = (predicted_states[:,1]**2 + predicted_states[:,3]**2)**0.5
        vx_car_predicted_hist = predicted_states[:,1]

        # forward
        vx_car_predicted_hist = predicted_states[:,1]*np.cos(predicted_states[:,4]) + predicted_states[:,3]*np.sin(predicted_states[:,4])
        # lateral, left +
        vy_car_predicted_hist = -predicted_states[:,1]*np.sin(predicted_states[:,4]) + predicted_states[:,3]*np.cos(predicted_states[:,4])

        # heading
        predicted_heading_hist = predicted_states[:,4]

        # position error predicted vs actual
        pos_err = ((predicted_future_traj[:,0] - actual_future_traj[:,0])**2 + (predicted_future_traj[:,1] - actual_future_traj[:,1])**2)**0.5

        # actual slip at front tire
        lf = 0.09-0.036
        actual_slip_f = -np.arctan((omega*lf + vy_car)/vx_car) + steering
        # periodic debugging plots
        if (False and i % 100 == 0):
            print("showing heading")
            print("showing velocity (total)")
            print("showing local velocity in car frame")

            # heading
            wrap = lambda x: np.mod(x + np.pi, 2*np.pi) - np.pi
            heading_predicted = wrapContinuous(wrap(predicted_heading_hist))/np.pi*180
            heading_actual = wrapContinuous(heading[i:i+lookahead_steps])/np.pi*180

            ax0 = plt.subplot(311)
            ax0.plot(heading_predicted,label="heading predicted")
            ax0.plot(heading_actual,label="actual")
            ax0.legend()

            # omega
            d_heading_predicted = np.diff(heading_predicted)/dt
            d_heading_actual = np.diff(heading_actual)/dt
            ax1 = plt.subplot(312)
            ax1.plot(np.array(debug_dict_hist['w'][i])/np.pi*180,label="omega predicted")
            #ax1.plot(omega[i:i+lookahead_steps]/np.pi*180,label="actual")
            ax1.plot(d_heading_actual, label='d_heading_actual')
            #ax1.plot(d_heading_predicted, label='d_heading_predicted')
            ax1.legend()

            # d omega
            '''
            domega = np.diff(omega)/dt
            ax2 = plt.subplot(313)
            ax2.plot(np.array(debug_dict_hist['dw'][i])/np.pi*180,label="d_omega predicted")
            ax2.plot(domega[i:i+lookahead_steps]/np.pi*180,label="actual")
            ax2.legend()
            '''

            # total velocity
            '''
            ax1 = plt.subplot(412)
            ax1.plot(v_predicted_hist,label="v predicted")
            ax1.plot(v_actual_hist,label="actual")
            #ax1.plot(steering[i:i+lookahead_steps],label="steering")
            ax1.legend()
            '''

            # forward velocity
            ax2 = plt.subplot(313)
            ax2.plot(vx_car_predicted_hist,label="car vx predicted")
            ax2.plot(vx_car[i:i+lookahead_steps],label="car vx actual")
            #ax2.plot(vy_car_predicted_hist,'--',label="car vy predicted")
            #ax2.plot(vy_car[i:i+lookahead_steps],'--',label="car vy actual")
            #ax2.plot(throttle[i:i+lookahead_steps],label="throttle")
            #ax2.plot(debug_dict_hist['ax'][i],'--',label="predicted ax")
            ax2.legend()

            # fron slip
            '''
            ax3 = plt.subplot(414)
            ax3.plot(debug_dict_hist['slip_f'][i],label="predicted slip front")
            ax3.plot(actual_slip_f[i:i+lookahead_steps],label="actual slip front")
            ax3.plot(steering[i:i+lookahead_steps],label="steering")
            # heading
            wrap = lambda x: np.mod(x + np.pi, 2*np.pi) - np.pi
            ax3 = plt.subplot(414)
            ax3.plot(wrap(predicted_heading_hist)/np.pi*180,label="heading predicted")
            ax3.plot(heading[i:i+lookahead_steps]/np.pi*180,label="actual")
            ax3.legend()
            '''

            plt.show()


# find a numerical value for error
def err():
    lookahead_steps = 50
    step_fun = step_raw
    #filename = '../../log/2022_2_9_exp/full_state4.p'
    filename = '../../log/2022_3_2_exp/full_state2.p'
    rawlog = loadLog(filename)
    log = prepLog(rawlog,skip=200)
    dt = 0.01
    vx = np.hstack([0,np.diff(log.x)])/dt
    vy = np.hstack([0,np.diff(log.y)])/dt
    data_len = log.t.shape[0]

    # use measured steering
    #filename = '../../log/2022_2_7_exp/debug_dict2.p'
    #filename = '../../log/2022_2_9_exp/debug_dict4.p'
    '''
    measured_steering = loadMeasuredSteering(filename)[:-1]
    offset = (-np.mean(measured_steering) + np.mean(log.steering))
    measured_steering = measured_steering + offset
    '''

    cum_error = 0.0
    # iterate through log
    for i in range(500,min(1900,data_len-lookahead_steps-1)):
        # prepare state shorthands
        x = log.x
        y = log.y
        heading = log.heading
        omega = log.omega
        steering = log.steering
        #steering = measured_steering
        throttle = log.throttle
        vx_car = log.v_forward
        vy_car = log.v_sideway

        car_state = (x[i],y[i],heading[i],0,0,0)
        # actual trajectory
        actual_future_traj = np.vstack([x[i:i+lookahead_steps],y[i:i+lookahead_steps]]).T

        state = (x[i],vx[i],y[i],vy[i],heading[i],omega[i])
        control = (steering[i],throttle[i])
        predicted_states = [state]

        # predicted trajectory
        for j in range(i+1,i+lookahead_steps):
            # make prediction
            state, debug_dict = step_fun(state,control)

            predicted_states.append(state)
            # delay for throttle
            index = max(j-7,0)
            index = j
            control = (steering[j],throttle[index])

        predicted_states = np.array(predicted_states)
        predicted_future_traj = np.vstack([predicted_states[:,0],predicted_states[:,2]]).T
        this_err = np.linalg.norm(actual_future_traj - predicted_future_traj)
        cum_error += this_err
        continue
    print(cum_error)

def wrapContinuous(val):
    # wrap to -pi,pi
    wrap = lambda x: np.mod(x + np.pi, 2*np.pi) - np.pi
    dval = np.diff(val)
    dval = wrap(dval)
    retval = np.hstack([0,np.cumsum(dval)])+val[0]
    return retval


if __name__=="__main__":
    run()
    err()
    exit(0)
    if saveGif:
        print("saving gif... be patient")
        gif_filename = "validate_model.gif"
        gifs[0].save(fp=gif_filename,format='GIF',append_images=gifs,save_all=True,duration = 20,loop=0)
