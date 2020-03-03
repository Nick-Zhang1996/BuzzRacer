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
from math import radians,degrees,isnan,sin,cos,atan2
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
saveLog = False

# static variables, for share in this file
s = RCPtrack()
sp = Skidpad()
img_track = None
showobj = None
visualization_ts = 0.0
tf = TF()
vi = Vicon()

#debug
offset_vec = []
omega_offset_vec = []
state_vec = []

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

def exitHandler(signal_received, frame):
    cv2.destroyAllWindows()
    vi.stopUpdateDaemon()
    if saveGif:
        gifimages[0].save(fp="./mk103exp"+str(no)+".gif",format='GIF',append_images=gifimages,save_all=True,duration = 50,loop=0)

    if saveLog:
        output = open(logFolder+'exp_state'+str(no)+'.p','wb')
        pickle.dump(state_vec,output)
        output.close()

        output = open(logFolder+'exp_offset'+str(no)+'.p','wb')
        pickle.dump(offset_vec,output)
        output.close()

        output = open(logFolder+'exp_dw'+str(no)+'.p','wb')
        pickle.dump(omega_offset_vec,output)
        output.close()

        print("saved to No." + str(no))
        print(" showing offset_vec")
        plt.plot(offset_vec)
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
    (x,y,z,rx,ry,rz) = vi.getState(car.vicon_id)
    # get body pose in track frame
    (x,y,heading) = tf.reframeR(T,x,y,z,tf.euler2Rxyz(rx,ry,rz))

    #state_car = (x,y,heading, vf_lf[0], vs_lf[0], omega_lf[0])
    state_car = (x,y,heading,0,0,0)
    
    if (not cooldown):
        throttle,steering,valid,other = car.ctrlCar(state_car,track,reverse=False)
    else:
        throttle,steering,valid,other= car.ctrlCar(state_car,track,v_override=0,reverse=False)
        throttle = 0
    #print(degrees(steering),throttle)
    
    # only log debugging information for car No.1
    if len(other)==2:
        offset = other[0]
        omega_offset = other[1]
        offset_vec.append(offset)
        omega_offset_vec.append(omega_offset)

    car.steering = steering
    car.throttle = throttle
    car.actuate(steering,throttle)
    #print(offset, degrees(steering), throttle)

    # control for car 2
    if not (car2 is None):
        # state update
        (x,y,z,rx,ry,rz) = vi.getState(car2.vicon_id)
        # get body pose in track frame
        (x,y,heading) = tf.reframeR(T,x,y,z,tf.euler2Rxyz(rx,ry,rz))
        state_car2 = (x,y,heading,0,0,0)

        
        if (not cooldown):
            throttle,steering,valid,other = car2.ctrlCar(state_car2,track,reverse=False)
        else:
            throttle,steering,valid,other= car2.ctrlCar(state_car2,track,v_override=0,reverse=False)
            throttle = 0

        car2.steering = steering
        car2.throttle = throttle
        car2.actuate(steering,throttle)
    
    if (car2 is None):
        print(car.throttle,car.steering)
    else:
        print(car.steering,car.throttle,car2.steering,car2.throttle)

    # visualization
    # restrict update rate to 0.1s/frame
    if (time()-visualization_ts>0.1):
        # plt doesn't allow updating from a different thread
        lock_visual.acquire()
        #shared_visualization_img = track.drawCar((x,y),heading,steering,img_track.copy())
        shared_visualization_img = track.drawCar(img_track.copy(), state_car, steering)
        if not (car2 is None):
            shared_visualization_img = track.drawCar(shared_visualization_img, state_car2, steering)

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
    mk103.initTrack('uuruurddddll',(5,3),scale=0.60)
    mk103.initRaceline((2,2),'d',4)

    # select track
    track = mk103


    

    # porsche 911
    car = Car(serial_port='/dev/ttyUSB0')
    # ensure vicon state has been updated
    sleep(0.05)
    car.vicon_id = vi.getItemID('nick_mr03_porsche')
    if car.vicon_id is None:
        print("error, can't find car in vicon")
        exit(1)

    # lambo TODO: calibrate lambo chassis
    if (twoCars):
        car2 = Car(wheelbase=90e-3,max_steering=radians(27.1),serial_port='/dev/ttyUSB1',max_throttle=0.3)
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

