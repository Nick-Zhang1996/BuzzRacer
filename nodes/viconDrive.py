#!/usr/bin/python

# utilize track.py to operate a vehicle
# this node directly publish to RCchannel

from ../src/track import RCPtrack
from math import radians,degrees
import rospy
import std_msgs.msg
from sensor_msgs.msg import Joy
from rcvip_msgs.msg import RCchannel
#from rcvip_msgs.msg import CarControl as carControl_msg
from rcvip_msgs.msg import Vicon as Vicon_msg

def mapdata(x,a,b,c,d):
    y=(x-a)/(b-a)*(d-c)+c
    return y

def vicon_callback(data):
    x = data.data[0]
    y = data.data[1]
    heading = data.data[5]
    throttle,steering,valid = s.ctrlCar((x,y),heading)
    rospy.loginfo(throttle,steering,valid)

    # for using carControl
    #msg = carControl_msg()
    #msg.header = data.header
    #msg.throttle = throttle
    #msg.steer_angle = degrees(steering)

    # for using channel directly
    msg.header = data.header
    msg.ch[0] = mapdata(steering, radians(24),-radians(24),1150,1850)
    msg.ch[1] = mapdata(throttle,-1.0,1.0,1900,1100)

    pub.publish(msg)

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
    rospy.init_node('viconDrive', anonymous=False)
    rospy.Subscriber("/vicon_tf", Vicon, vicon_callback)
    #pub = rospy.Publisher("rc_vip/CarControl", carControl_msg, queue_size=1)
    pub = rospy.Publisher("vip_rc/channel", RCchannel, queue_size=1)

    rospy.spin()
