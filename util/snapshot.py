#!/usr/bin/env python

# take a snapshot from the image_raw topic


import argparse
import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from calibration import imageutil

def snap(data):
    print('called')
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)

    #frame = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
    frame = cv_image

    retval = cv2.imwrite(filename,frame)
    print('picture taken')
    print(retval)

    reason='job done'
    rospy.signal_shutdown(reason)
    return

#reading command line arguments
parser = argparse.ArgumentParser(description = 'capture a snapshot from the image_raw topic')
parser.add_argument('filename',help='output filename without extension')
args = parser.parse_args()

filename = './'+args.filename + '.jpeg'

bridge = CvBridge()

# Define the codec and create VideoWriter object

# start recording
rospy.init_node('recording_node', anonymous=False)
#create subscriber for camera input
rospy.Subscriber("/image_raw", Image, snap,queue_size=1)
rospy.spin()

