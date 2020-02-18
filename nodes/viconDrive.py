#!/usr/bin/python

# utilize track.py to operate a vehicle
# this node directly publish to RCchannel

import sys
import serial
import platform
import cv2
sys.path.insert(0,'../src')
from threading import Lock
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
from track import RCPtrack,TF
from vicon import Vicon
import pickle
from skidpad import Skidpad
from car import Car

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
local_state = None
previous_state = (0,0,0,0,0,0)
vicon_dt = 0.01
# lowpass filter
# argument: order, omega(-3db)
b, a = signal.butter(1,6,'low',analog=False,fs=100)
# filter state
z_vf = [0]
z_vs = [0]
z_omega = [0] # omega, angular rate in rad/s
last_vf = 0
last_vs = 0
last_omega = 0


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
T = np.hstack([q_t,np.array([0.5,0.3,0])])

def mapdata(x,a,b,c,d):
    y=(x-a)/(b-a)*(d-c)+c
    return int(y)
# read data from vicon feed
# convert from vicon world frame to track frame
# update local copy of state
def ctrlloop(track,cooldown=False):
    global visualization_ts
    global flag_new_visualization_img
    global shared_visualization_img
    global previous_state
    global last_vf,last_vs,last_omega
    global z_vf,z_vs,z_omega
    global local_state

    # state update
    (x,y,z,rx,ry,rz) = vi.getViconUpdate()
    # get body pose in track frame
    (x,y,heading) = tf.reframeR(T,x,y,z,tf.euler2Rxyz(rx,ry,rz))
    vx = (x - previous_state[0])/vicon_dt
    vy = (y - previous_state[1])/vicon_dt
    omega = (heading - previous_state[2])/vicon_dt
    vf = vx*cos(heading) + vy*sin(heading)
    vs = vx*sin(heading) - vy*cos(heading)

    # low pass filter
    if (abs(vf-last_vf)>0.5):
        vf_lf, z_vf = signal.lfilter(b,a,[last_vf],zi=z_vf)
    else:
        vf_lf, z_vf = signal.lfilter(b,a,[vf],zi=z_vf)

    if (abs(vs-last_vs)>0.5):
        vs_lf, z_vs = signal.lfilter(b,a,[last_vs],zi=z_vs)
    else:
        vs_lf, z_vs = signal.lfilter(b,a,[vs],zi=z_vs)

    if (abs(omega-last_omega)>0.5):
        omega_lf, z_omega = signal.lfilter(b,a,[last_omega],zi=z_omega)
    else:
        omega_lf, z_omega = signal.lfilter(b,a,[omega],zi=z_omega)

    last_vf = vf
    last_vs = vs
    last_omega = omega

    lock_state.acquire()
    local_state = (x,y,heading, vf_lf[0], vs_lf[0], omega_lf[0])
    previous_state = local_state
    lock_state.release()
    
    if (not cooldown):
        throttle,steering,valid,other = car.ctrlCar(local_state,sp,reverse=False)
    else:
        throttle,steering,valid,other= car.ctrlCar(local_state,sp,v_override=0,reverse=False)
    
    print(steering, throttle)

    offset = other[0]
    omega_offset = other[1]
    offset_vec.append(offset)
    omega_offset_vec.append(omega_offset)
    #rospy.loginfo(str((x,y,heading,throttle,steering,valid)))
    #print(str((x,y,degrees(heading),throttle,steering,valid)))
    #print(valid)

    # for using carControl
    #msg = carControl_msg()
    #msg.throttle = throttle
    #msg.steer_angle = degrees(steering)

    # for using channel directly
    '''
    msg = RCchannel()
    msg.header = Header()
    msg.header.stamp = rospy.Time.now()
    msg.ch[0] = mapdata(steering, radians(24),-radians(24),1150,1850)
    msg.ch[1] = mapdata(throttle,-1.0,1.0,1900,1100)

    pub.publish(msg)
    '''
# need to supply 4 valurs for two cars even if only one interface is being used
    #arduino.write((str(mapdata(steering, radians(24),-radians(24),1150,1850))+","+str(mapdata(throttle,-1.0,1.0,1900,1100))+",1500,1500"+'\n').encode('ascii'))
    arduino.write((str(mapdata(steering, radians(24),-radians(24),1150,1850))+","+str(mapdata(throttle,-1.0,1.0,1900,1100))+'\n').encode('ascii'))
    #print((str(mapdata(steering, radians(24),-radians(24),1150,1850))+","+str(mapdata(throttle,-1.0,1.0,1900,1100))+'\n').encode('ascii'))

    # visualization
    # add throttling
    if (time()-visualization_ts>0.1):
        # plt doesn't allow updating from a different thread
        lock_visual.acquire()
        #shared_visualization_img = track.drawCar((x,y),heading,steering,img_track.copy())
        shared_visualization_img = track.drawCar(img_track.copy(), local_state, steering)
        lock_visual.release()
        visualization_ts = time()
        flag_new_visualization_img = True

if __name__ == '__main__':
    host_system = platform.system()
    if host_system == "Linux":
        CommPort = '/dev/ttyUSB0'
    elif host_system == "Darwin":
        CommPort = '/dev/tty.wchusbserial1420'

# ROS init
    #rospy.init_node('viconDrive', anonymous=False)
    #pub = rospy.Publisher("rc_vip/CarControl", carControl_msg, queue_size=1)
    #pub = rospy.Publisher("vip_rc/channel", RCchannel, queue_size=1)

    if (False):
        # MK111 track
        # row, col
        track_size = (6,4)
        s.initTrack('uuurrullurrrdddddluulddl',track_size, scale=0.565)
        # add manual offset for each control points
        adjustment = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

        adjustment[0] = -0.2
        adjustment[1] = -0.2
        #bottom right turn
        adjustment[2] = -0.2
        adjustment[3] = 0.5
        adjustment[4] = -0.2

        #bottom middle turn
        adjustment[6] = -0.2

        #bottom left turn
        adjustment[9] = -0.2

        # left L turn
        adjustment[12] = 0.5
        adjustment[13] = 0.5

        adjustment[15] = -0.5
        adjustment[16] = 0.5
        adjustment[18] = 0.5

        adjustment[21] = 0.35
        adjustment[22] = 0.35

        # start coord, direction, sequence number of origin(which u gives the exit point for origin)
        s.initRaceline((3,3),'d',10,offset=adjustment)
    else:
        # example: simple track
        s.initTrack('uurrddll',(3,3),scale=1.0)
        s.initRaceline((0,0),'l',0)
        

    # sample track
    #setup visualization of current car location, comment out if running the code on car computer
    #img_track = s.drawTrack()
    #img_track = s.drawRaceline(img=img_track)

    # skid pad
    sp.initSkidpad(radius=1,velocity=1)
    car = Car()
    img_track = sp.drawTrack()
    cv2.imshow('car',img_track)
    cv2.waitKey(1)

    # visualization update loop
    with serial.Serial(CommPort,115200, timeout=0.001,writeTimeout=0) as arduino:
        #for i in range(1000):
        while True:
            ctrlloop(sp)
            state_vec.append(local_state)

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

        # cooldown
        for i in range(200):
            ctrlloop(sp,cooldown=True)
            state_vec.append(local_state)

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

    cv2.destroyAllWindows()
    output = open('exp_state.p','wb')
    pickle.dump(state_vec,output)
    output.close()

    output = open('exp_offset.p','wb')
    pickle.dump(offset_vec,output)
    output.close()

    output = open('exp_dw.p','wb')
    pickle.dump(omega_offset_vec,output)
    output.close()

    
    plt.plot(omega_offset_vec)
    plt.show()

    plt.plot(offset_vec)
    plt.show()

