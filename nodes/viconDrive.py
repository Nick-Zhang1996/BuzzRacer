#!/usr/bin/python

# utilize track.py to operate a vehicle
# this node directly publish to RCchannel

import sys
import serial
import platform
import cv2
import os.path
sys.path.insert(0,'../src')
from threading import Lock
from signal import SIGINT
from signal import signal as syssignal
#import rospy
import numpy as np
from scipy import signal 
from time import sleep,time
from math import radians,degrees,isnan,sin,cos,atan2,asin
import matplotlib.pyplot as plt
#from std_msgs.msg import Header
#from sensor_msgs.msg import Joy
#from rcvip_msgs.msg import RCchannel
#from rcvip_msgs.msg import Vicon as Vicon_msg
from track import RCPtrack
from tf import TF
from vicon import Vicon
import pickle
from skidpad import Skidpad
from car import Car
from PIL import Image

# settings
twoCars = True
saveLog = True
saveGif = True

# static variables, for share in this file
s = RCPtrack()
sp = Skidpad()
img_track = None
showobj = None
visualization_ts = 0.0
tf = TF()
vi = Vicon()

#debug
state_vec = []
offset_vec = []
vf_vec = []

# state vector
lock_state = Lock()
# (x,y,heading,v_longitudinal, v_lateral, angular rate)
state_car = None
vicon_dt = 0.01
# lowpass filter
# argument: order, omega(-3db)
b, a = signal.butter(1,6,'low',analog=False,fs=100)
# filter state
z_vf = [0]
z_vs = [0]
z_omega = [0] # omega, angular rate in rad/s
z_steering = [0]
last_vf = 0
last_vs = 0
last_omega = 0
last_steering = 0


# visualization
lock_visual = Lock()
shared_visualization_img = None
flag_new_visualization_img = False

# track pose
# for normal space origin
#q_t = tf.euler2q(radians(180),0,radians(90))
#T = np.hstack([q_t,np.array([0,1,0])])

# for upright origin
q_t = tf.euler2q(0,0,0)
T = np.hstack([q_t,np.array([-0.03,-0.03,0])])

exitSeverity = 0
def exitHandler(signal_received, frame):
    global exitSeverity
    exitSeverity += 1
    if exitSeverity > 1:
        print("second sigterm, force quit")
        exit(1)

    cv2.destroyAllWindows()
    vi.stopUpdateDaemon()
    if saveGif:
        gifimages[0].save(fp="./gifs/mk103exp"+str(no)+".gif",format='GIF',append_images=gifimages,save_all=True,duration = 50,loop=0)

    if saveLog:
        output = open(logFolder+'exp_state'+str(no)+'.p','wb')
        pickle.dump(state_vec,output)
        output.close()

        output = open(logFolder+'exp_offset'+str(no)+'.p','wb')
        pickle.dump(offset_vec,output)
        output.close()

        #print("saved to No." + str(no))
        #print("showing offset_vec")

        plt.plot(offset_vec)
        plt.show()

        plt.plot(vf_vec)
        plt.show()
    print("Program finished")
    exit(0)
    return

# register signal handler to save everything when ctrl-C is pressed
syssignal(SIGINT,exitHandler)

def mapdata(x,a,b,c,d):
    y=(x-a)/(b-a)*(d-c)+c
    return int(y)
# read data from vicon feed
# convert from vicon world frame to track frame
# update local copy of state
def ctrlloop(car,car2,track,cooldown=False):
    global visualization_ts
    global flag_new_visualization_img
    global shared_visualization_img
    global last_vf,last_vs,last_omega,last_steering
    global z_vf,z_vs,z_omega,z_steering
    global state_car


    # control for car 1
    # state update
    #(x,y,z,rx,ry,rz) = vi.getState(car.vicon_id)
    (x,y,heading) = vi.getState2d(car.vicon_id)
    kf_x,vx,ax,kf_y,vy,ay,kf_heading,omega = vi.getKFstate(car.vicon_id)

    #state_car = (x,y,heading, vf, vs, omega)
    # assume no lateral velocity
    vf = (vx**2+vy**2)**0.5
    state_car = (x,y,heading,vf,0,omega)
    
    if (not cooldown):
        throttle,steering,valid,debug_dic = car.ctrlCar(state_car,track,reverse=False)
    else:
        throttle,steering,valid,debug_dic= car.ctrlCar(state_car,track,v_override=0,reverse=False)
        throttle = 0
    #print(degrees(steering),throttle)
    
    # only log debugging information for car No.1
    offset_vec.append(debug_dic['offset'])
    vf_vec.append(vf)

    

    car.steering = steering
    car.throttle = throttle
    car.actuate(steering,throttle)
    #print(offset, degrees(steering), throttle)

    # control for car 2
    if not (car2 is None):
        # state update
        (x,y,heading) = vi.getState2d(car2.vicon_id)
        kf_x,vx,ax,kf_y,vy,ay,kf_heading,omega = vi.getKFstate(car2.vicon_id)

        state_car2 = (x,y,heading,vf,0,omega)

        
        if (not cooldown):
            throttle,steering,valid,debug_dic = car2.ctrlCar(state_car2,track,reverse=False)
        else:
            throttle,steering,valid,debug_dic= car2.ctrlCar(state_car2,track,v_override=0,reverse=False)
            throttle = 0

        car2.steering = steering
        car2.throttle = throttle
        car2.actuate(steering,throttle)
    
    if (car2 is None):
        print("%.2f, %.2f"% (car.throttle,degrees(car.steering)))
    else:
        print("%.2f, %.2f,%.2f, %.2f"% (car.throttle,degrees(car.steering),car2.throttle,degrees(car2.steering)))

    # visualization
    # restrict update rate to 0.1s/frame
    if (time()-visualization_ts>0.1):
        # plt doesn't allow updating from a different thread
        lock_visual.acquire()
        #shared_visualization_img = track.drawCar((x,y),heading,steering,img_track.copy())
        shared_visualization_img = track.drawCar(img_track.copy(), state_car, car.steering)
        if not (car2 is None):
            shared_visualization_img = track.drawCar(shared_visualization_img, state_car2, car2.steering)

        lock_visual.release()
        visualization_ts = time()
        flag_new_visualization_img = True

no = 1

if __name__ == '__main__':
# ROS init
    #rospy.init_node('viconDrive', anonymous=False)
    #pub = rospy.Publisher("rc_vip/CarControl", carControl_msg, queue_size=1)
    #pub = rospy.Publisher("vip_rc/channel", RCchannel, queue_size=1)


    # setup log file
    # log file will record state of the vehicle for later analysis
    #   state: (x,y,heading,v_forward,v_sideway,omega)
    logFolder = "./log/"
    logPrefix = "exp_state"
    logSuffix = ".p"
    # global
    #no = 1
    while os.path.isfile(logFolder+logPrefix+str(no)+logSuffix):
        no += 1
    logFilename = logFolder+logPrefix+str(no)+logSuffix

    # define tracks
    # skid pad
    #sp.initSkidpad(radius=0.5,velocity=1)

    # current track setup in mk103, L shaped
    # width 0.563, length 0.6
    mk103 = RCPtrack()
    mk103.initTrack('uuruurddddll',(5,3),scale=0.57)
    # add manual offset for each control points
    adjustment = [0,0,0,0,0,0,0,0,0,0,0,0]
    adjustment[4] = -0.5
    adjustment[8] = -0.5
    adjustment[9] = 0
    adjustment[10] = -0.5
    mk103.initRaceline((2,2),'d',4,offset=adjustment)


    # select track
    track = mk103

    porsche_setting = {'wheelbase':90e-3,
                     'max_steer_angle_left':radians(27.1),
                     'max_steer_pwm_left':1150,
                     'max_steer_angle_right':radians(27.1),
                     'max_steer_pwm_right':1850,
                     'serial_port' : '/dev/ttyUSB0',
                     'max_throttle' : 0.5}

    lambo_setting = {'wheelbase':98e-3,
                     'max_steer_angle_left':asin(2*98e-3/0.52),
                     'max_steer_pwm_left':1100,
                     'max_steer_angle_right':asin(2*98e-3/0.47),
                     'max_steer_pwm_right':1850,
                     'serial_port' : '/dev/ttyUSB1',
                     'max_throttle' : 0.5}

    # porsche 911
    car = Car(porsche_setting)
    # ensure vicon state has been updated
    sleep(0.05)
    car.vicon_id = vi.getItemID('nick_mr03_porsche')
    if car.vicon_id is None:
        print("error, can't find car in vicon")
        exit(1)

    # lambo TODO: calibrate lambo chassis
    if (twoCars):
        car2 = Car(lambo_setting)
        car2.vicon_id = vi.getItemID('nick_mr03_lambo')
    else:
        car2 = None

    img_track = track.drawTrack()
    img_track = track.drawRaceline(img=img_track)
    cv2.imshow('car',img_track)
    cv2.waitKey(1)

    # prepare save gif, this provides an easy to use visualization for presentation
    saveGif = True
    gifimages = []
    if saveGif:
        gifimages.append(Image.fromarray(cv2.cvtColor(img_track.copy(),cv2.COLOR_BGR2RGB)))


    # main loop
    while True:
        # control
        ctrlloop(car,car2,track)

        # logging (only for car 1)
        state_vec.append(state_car)

        # update visualization
        if flag_new_visualization_img:
            lock_visual.acquire()
            #showobj.set_data(shared_visualization_img)
            cv2.imshow('car',shared_visualization_img)
            if saveGif:
                gifimages.append(Image.fromarray(cv2.cvtColor(shared_visualization_img.copy(),cv2.COLOR_BGR2RGB)))
            lock_visual.release()
            #plt.draw()
            flag_new_visualization_img = False
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break

    # NOTE currently not used :cooldown
    # this is forl slowing down the vehicle while maintining steering control
    # so when the experiment is over the vehicle is not left going high speed on the track
    for i in range(200):
        ctrlloop(p,cooldown=True)
        state_vec.append(state_car)

        if flag_new_visualization_img:
            lock_visual.acquire()
            #showobj.set_data(shared_visualization_img)
            cv2.imshow('car',shared_visualization_img)
            lock_visual.release()
            #plt.draw()
            flag_new_visualization_img = False
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
    exitHandler(None,None)

