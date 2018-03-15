#!/usr/bin/env python

# record a video from the image_raw topic


import argparse
import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
#from skvideo.io import VideoWriter
from cv2 import VideoWriter

def record(data):
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)

    #frame = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
    frame = cv_image

    out.write(frame)

    # uncomment to show live video
    #cv2.imshow('frame',frame)
    #if cv2.waitKey(10) & 0xFF == ord('q'):
    #    pass
    return

#reading command line arguments
parser = argparse.ArgumentParser(description = 'record a video from the image_raw topic')
parser.add_argument('filename',help='output filename without extension')
args = parser.parse_args()

filename = './'+args.filename + '.avi'

bridge = CvBridge()

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
print(fourcc)
out = VideoWriter(filename,fourcc, 30.0, (1280,960))

# start recording
rospy.init_node('recording_node', anonymous=False)
#create subscriber for camera input
rospy.Subscriber("camera/image_raw", Image, record,queue_size=1)


# stop the recording, for interactive using
raw_input('press Enter to stop')

# OR:
#rospy.spin()


# Release everything if job is finished
out.release()
#cv2.destroyAllWindows()
print('Program finished')
