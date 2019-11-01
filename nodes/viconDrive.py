#!/usr/bin/python

# utilize track.py to operate a vehicle
# this node directly publish to RCchannel

import sys
sys.path.insert(0,'../src')
from threading import Lock
#import rospy
import numpy as np
from time import sleep,time
from math import radians,degrees
import matplotlib.pyplot as plt
#from std_msgs.msg import Header
#from sensor_msgs.msg import Joy
#from rcvip_msgs.msg import RCchannel
#from rcvip_msgs.msg import Vicon as Vicon_msg
from track import RCPtrack,TF
from vicon import Vicon

# static variables, for share in this file
s = RCPtrack()
img_track = None
showobj = None
visualization_ts = 0.0
tf = TF()
vi = Vicon()

lock_state = Lock()
local_state = None

lock_visual = Lock()
shared_visualization_img = None
flag_new_visualization_img = False

# track pose
q_t = tf.euler2q(radians(180),0,radians(90))
T = np.hstack([q_t,np.array([0,1,0])])


def mapdata(x,a,b,c,d):
    y=(x-a)/(b-a)*(d-c)+c
    return y
# read data from vicon feed
# convert from vicon world frame to track frame
# update local copy of state
def update_state():
    global visualization_ts
    global flag_new_visualization_img
    global shared_visualization_img
    (x,y,z,rx,ry,rz) = vi.getViconUpdate()
    # get body pose in track frame
    (x,y,heading) = tf.reframeR(T,x,y,z,tf.euler2Rxyz(rx,ry,rz))
    lock_state.acquire()
    local_state = (x,y,heading)
    lock_state.release()
    
    throttle,steering,valid = s.ctrlCar((x,y),heading,reverse=False)
    #rospy.loginfo(str((x,y,heading,throttle,steering,valid)))
    #print(str((x,y,degrees(heading),throttle,steering,valid)))
    print(valid)

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

    # visualization
    # add throttling
    if (time()-visualization_ts>0.1):
        # plt doesn't allow updating from a different thread
        lock_visual.acquire()
        shared_visualization_img = s.drawCar((x,y),heading,steering,img_track.copy())
        lock_visual.release()
        visualization_ts = time()
        flag_new_visualization_img = True

if __name__ == '__main__':
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
        

# ROS init
    #rospy.init_node('viconDrive', anonymous=False)
    #pub = rospy.Publisher("rc_vip/CarControl", carControl_msg, queue_size=1)
    #pub = rospy.Publisher("vip_rc/channel", RCchannel, queue_size=1)

    #setup visualization of current car location, comment out if running the code on car computer
    img_track = s.drawTrack()
    img_track = s.drawRaceline(img=img_track)
    showobj = plt.imshow(img_track)
    showobj.set_data(img_track)
    plt.draw()
    plt.pause(0.01)
    #rospy.Subscriber("/vicon_tf", Vicon_msg, vicon_callback)

    # visualization update loop
    #while not rospy.is_shutdown():
    while True:
        update_state()

        if flag_new_visualization_img:
            print("new img")
            lock_visual.acquire()
            showobj.set_data(shared_visualization_img)
            lock_visual.release()
            plt.draw()
            flag_new_visualization_img = False
            plt.pause(0.005)

    # may not be necesary
    #rospy.spin()
