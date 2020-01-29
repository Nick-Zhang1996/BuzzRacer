#!/usr/bin/env python

#drive the car(equipped with onboard computer) with a joystick
import rospy
import std_msgs.msg
from sensor_msgs.msg import Joy
from rcvip_msgs.msg import CarControl as carControl_msg

def mapdata(x,a,b,c,d):
    y=(x-a)/(b-a)*(d-c)+c
    return y

def joystick_callback(data):
    #rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
    print(data.axes)
    msg = carControl_msg()
    msg.header = data.header
    if (data.axes[2]<0.9): #brake priority
        # Brake, L trigger 1.0 undepressed, -1 fully depressed
        msg.throttle = mapdata(data.axes[2],1.0,-1.0,0.0,-1.0)
    else:
        # gas R trigger 1.0 undepressed, -1 fully depressed
        msg.throttle = mapdata(data.axes[5],1.0,-1.0,0.0,1.0)

    # left joystick? 1.0 left, -1 right
    msg.steer_angle = mapdata(data.axes[0],1.0,-1.0,-20.0,20.0)
    pub.publish(msg)


if __name__ == '__main__':

    rospy.init_node('Joystick2CarControl', anonymous=False)
    rospy.Subscriber("joy", Joy, joystick_callback)
    pub = rospy.Publisher("rc_vip/CarControl", carControl_msg, queue_size=1)

    rospy.spin()
