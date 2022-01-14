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
    omega = log[skip:,6]
    steering = log[skip:,7]
    throttle = log[skip:,8]

    Log = namedtuple('Log', 't x y heading v_forward v_sideway omega steering throttle')
    mylog = Log(t,x,y,heading,v_forward,v_sideway,omega,steering,throttle)
    return mylog



def show(img):
    plt.imshow(img)
    plt.show()
    return

#state: x,vx(global),y,vy,heading,omega
#control: steering(rad),throttle(raw unit -1 ~ 1)
def step_kinematic(state,control,dt=0.01):
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

    return (x,vxg,y,vyg,heading,omega ),{}

# old, kinematic model with correction
def step_kinematic_heuristic(state,control,dt=0.01):
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

    return (x,vxg,y,vyg,heading,omega ),{}

# dynamic model with heuristically selected parameters
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
    # TODO consider longitudinal load transfer
    lateral_acc_f = tireCurve(slip_f) * 9.8 * lr / (lr + lf)
    lateral_acc_r = tireCurve(slip_r) * 9.8 * lf / (lr + lf)
    # TODO use more comprehensive model
    forward_acc_r = (throttle - 0.24)*7.0

    ax = forward_acc_r - lateral_acc_f * sin(steering) + vy*omega
    ay = lateral_acc_r + lateral_acc_f * cos(steering) - vx*omega

    vx += ax * dt
    vy += ay * dt

    # leading coeff = m/Iz
    d_omega = 12.0/(0.1**2+0.1**2)*(lateral_acc_f * lf * cos(steering) - lateral_acc_r * lr )
    omega += d_omega * dt

    # back to global frame
    vxg = vx*cos(heading)-vy*sin(heading)
    vyg = vx*sin(heading)+vy*cos(heading)

    # apply updates
    # TODO add 1/2 a t2
    x += vxg*dt
    y += vyg*dt
    heading += omega*dt + 0.5* d_omega * dt * dt

    retval = (x,vxg,y,vyg,heading,omega )
    debug_dict = {"slip_f":slip_f, "slip_r":slip_r, "lateral_acc_f":lateral_acc_f, "lateral_acc_r":lateral_acc_r, 'ax':ax}
    return retval, debug_dict

# model with parameter from ukf
def step_ukf(state,control,dt=0.01):
    # constants
    lf = 0.09-0.036
    lr = 0.036
    L = 0.09

    Df = 3.93731
    Dr = 6.23597
    C = 2.80646
    B = 0.51943
    Cm1 = 6.03154
    Cm2 = 0.96769
    Cr = -0.20375
    Cd = 0.00000
    Iz = 0.00278
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
        d_vx = (( Cm1 - Cm2 * vx) * throttle - Cr - Cd * vx * vx)
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
        Frx = (( Cm1 - Cm2 * vx) * throttle - Cr - Cd * vx * vx)*m

        # Dynamics
        d_vx = 1.0/m * (Frx - Ffy * np.sin( steering ) + m * vy * omega)
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
    debug_dict = {"slip_f":slip_f, "slip_r":slip_r, "lateral_acc_f":Ffy/m, "lateral_acc_r":Fry/m, 'ax':d_vx}
    return retval, debug_dict

# model with parameter from ukf
# add in weight transfer
def step_ukf_linear_orig(state,control,dt=0.01,slip_f_override=None):
    # constants
    # CG to rear axle
    lr = 0.036
    # CG to front axle
    lf = 0.09-lr
    # CG height from ground
    h = 0.02
    # wheelbase
    L = 0.09

    '''
    Df = 3.93731
    Dr = 6.23597
    C = 2.80646
    B = 0.51943
    '''
    #Cm1 = 6.03154
    Cm2 = 0.96769
    #Cr = -0.20375
    Cm1 = 9.23154
    Cr = 0.0
    Cd = 0.00000
    #Iz = 0.00278
    m = 0.1667
    Iz = m*(0.1**2+0.1**2)/12.0 * 6.0
    K = 5.0

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
        d_vx = (( Cm1 - Cm2 * vx) * throttle - Cr - Cd * vx * vx)
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
        if not(slip_f_override is None):
            slip_f = slip_f_override
        slip_r = np.arctan((omega*lr - vy)/vx)

        # tire model -- pacejka model
        #Ffy = Df * np.sin( C * np.arctan(B *slip_f)) * 9.8 * lr / (lr + lf) * m
        #Fry = Dr * np.sin( C * np.arctan(B *slip_r)) * 9.8 * lf / (lr + lf) * m


        # Longitudinal Dynamics
        Frx = (( Cm1 - Cm2 * vx) * throttle - Cr - Cd * vx * vx)*m

        # Lateral Dynamics
        # we would have:
        #d_vx = 1.0/m * (Frx - Ffy * np.sin( steering ) + m * vy * omega)
        # but this needs Ffy, which appears later in
        #Ffy = K * slip_f * ( 9.8 * lr - h * d_vx)/L * m
        # solving these two eq gives
        d_vx = ( Frx - K*slip_f/L*9.8*lr*m*np.sin(steering) + m*vy*omega ) / (m - K*slip_f*h/L*m*np.sin(steering))

        Ffy = K * slip_f * ( 9.8 * lr - h * d_vx)/L * m
        Fry = K * slip_r * ( 9.8 * lf  + h * d_vx)/L * m

        # verify 
        d_vx_test = 1.0/m * (Frx - Ffy * np.sin( steering ) + m * vy * omega)
        assert( np.abs(d_vx_test - d_vx) < 0.00001)



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
    debug_dict = {"slip_f":slip_f, "slip_r":slip_r, "lateral_acc_f":Ffy/m, "lateral_acc_r":Fry/m, 'ax':d_vx}
    return retval, debug_dict

# model with parameter from ukf
# add in weight transfer
def step_ukf_linear(state,control,dt=0.01,slip_f_override=None):
    # constants
    # CG to rear axle
    lr = 0.036
    # CG to front axle
    lf = 0.09-lr
    # CG height from ground
    h = 0.02
    # wheelbase
    L = 0.09

    Cm1 = 9.23154
    Cm2 = 0.96769
    Cr = 0.0
    Cd = 0.00000
    m = 0.1667
    Iz = m*(0.1**2+0.1**2)/12.0 * 6.0
    K = 5.0

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
        d_vx = (( Cm1 - Cm2 * vx) * throttle - Cr - Cd * vx * vx)
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
        if not(slip_f_override is None):
            slip_f = slip_f_override
        slip_r = np.arctan((omega*lr - vy)/vx)

        # tire model -- pacejka model
        #Ffy = Df * np.sin( C * np.arctan(B *slip_f)) * 9.8 * lr / (lr + lf) * m
        #Fry = Dr * np.sin( C * np.arctan(B *slip_r)) * 9.8 * lf / (lr + lf) * m


        # Longitudinal Dynamics
        Frx = (( Cm1 - Cm2 * vx) * throttle - Cr - Cd * vx * vx)*m

        # Lateral Dynamics
        # we would have:
        #d_vx = 1.0/m * (Frx - Ffy * np.sin( steering ) + m * vy * omega)
        # but this needs Ffy, which appears later in
        #Ffy = K * slip_f * ( 9.8 * lr - h * d_vx)/L * m
        # solving these two eq gives
        d_vx = ( Frx - K*slip_f/L*9.8*lr*m*np.sin(steering) + m*vy*omega ) / (m - K*slip_f*h/L*m*np.sin(steering))

        Ffy = K * slip_f * ( 9.8 * lr - h * d_vx)/L * m
        Fry = K * slip_r * ( 9.8 * lf  + h * d_vx)/L * m

        # verify 
        d_vx_test = 1.0/m * (Frx - Ffy * np.sin( steering ) + m * vy * omega)
        assert( np.abs(d_vx_test - d_vx) < 0.00001)



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
    debug_dict = {"slip_f":slip_f, "slip_r":slip_r, "lateral_acc_f":Ffy/m, "lateral_acc_r":Fry/m, 'ax':d_vx}
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
    lookahead_steps = 100
    saveGif = False
    gifs = []

    # set prediction function
    step_fun = step_ukf_linear
    step_fun_base = step_ukf_linear_orig

    # load log
    filename = "../../log/jan12/full_state1.p"
    rawlog = loadLog(filename)
    log = prepLog(rawlog,skip=1)
    dt = 0.01
    vx = np.hstack([0,np.diff(log.x)])/dt
    vy = np.hstack([0,np.diff(log.y)])/dt
    data_len = log.t.shape[0]

    # prep track image
    track = RCPTrack()
    track.load()
    img_track = track.drawTrack()
    img_track = track.drawRaceline(img=img_track)
    cv2.imshow('validate',img_track)
    cv2.waitKey(10)

    # debug_dict_hist[key][log_timestep][prediction_timestep]
    debug_dict_hist = {"slip_f":[[]], "slip_r":[[]], "lateral_acc_f":[[]], "lateral_acc_r":[[]],'ax':[[]]}

    # iterate through log
    for i in range(1,data_len-lookahead_steps-1):
        # prepare state shorthands
        x = log.x
        y = log.y
        heading = log.heading
        omega = log.omega
        steering = log.steering
        throttle = log.throttle
        vx_car = log.v_forward
        vy_car = log.v_sideway

        car_state = (x[i],y[i],heading[i],0,0,0)
        img = track.drawCar(img_track.copy(), car_state, steering[i])

        # plot actual future trajectory
        actual_future_traj = np.vstack([x[i:i+lookahead_steps],y[i:i+lookahead_steps]]).T
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
            control = (steering[j],throttle[j])

        predicted_states = np.array(predicted_states)
        predicted_future_traj = np.vstack([predicted_states[:,0],predicted_states[:,2]]).T
        # RED
        img = track.drawPolyline(predicted_future_traj,lineColor=(0,0,255),img=img)

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
        # GREEN
        img = track.drawPolyline(predicted_future_traj,lineColor=(0,255,0),img=img)


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
        if (i % 100 == 0):
            print("showing heading")
            print("showing velocity (total)")
            print("showing local velocity in car frame")

            wrap = lambda x: np.mod(x + np.pi, 2*np.pi) - np.pi
            ax0 = plt.subplot(411)
            ax0.plot(wrap(predicted_heading_hist)/np.pi*180,label="heading predicted")
            ax0.plot(heading[i:i+lookahead_steps]/np.pi*180,label="actual")
            ax0.legend()

            ax1 = plt.subplot(412)
            ax1.plot(v_predicted_hist,label="v predicted")
            ax1.plot(v_actual_hist,label="actual")
            #ax1.plot(steering[i:i+lookahead_steps],label="steering")

            ax1.legend()

            ax2 = plt.subplot(413)
            ax2.plot(vx_car_predicted_hist,label="car vx predicted")
            ax2.plot(vx_car[i:i+lookahead_steps],label="car vx actual")
            #ax2.plot(vy_car_predicted_hist,'--',label="car vy predicted")
            #ax2.plot(vy_car[i:i+lookahead_steps],'--',label="car vy actual")

            ax2.plot(throttle[i:i+lookahead_steps],label="throttle")
            #ax2.plot(debug_dict_hist['ax'][i],'--',label="predicted ax")
            ax2.legend()

            ax3 = plt.subplot(414)
            ax3.plot(debug_dict_hist['slip_f'][i],label="predicted slip front")
            ax3.plot(actual_slip_f[i:i+lookahead_steps],label="actual slip front")
            ax3.plot(steering[i:i+lookahead_steps],label="steering")

            ax3.legend()

            plt.show()


if __name__=="__main__":
    run()
    exit(0)
    if saveGif:
        print("saving gif... be patient")
        gif_filename = "validate_model.gif"
        gifs[0].save(fp=gif_filename,format='GIF',append_images=gifs,save_all=True,duration = 20,loop=0)
