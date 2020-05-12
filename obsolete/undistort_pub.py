import numpy as np
import cv2
import pickle
import warnings
import rospy

from sensor_msgs.msg import Image
from calibration import imageutil
from cv_bridge import CvBridge, CvBridgeError

from timeUtil import execution_timer

x_size = 640
y_size = 480
crop_y_size = 240
cam = imageutil('../calibrated/')

def callback(data)
    try:
        image = bridge.imgmsg_to_cv2(data, "rgb8")
    except CvBridgeError as e:
        print(e)

    image = cam.undistort(image)
    image_message = bridge.cv2_to_imgmsg(image, encoding="rgb8")
    undis_pub.publish(image_message)

bridge = CvBridge()
rospy.init_node('undistort_node',log_level=rospy.DEBUG, anonymous=False)

vidin = rospy.Subscriber("image_raw", Image,callback,queue_size=1,buff_size = 2**24)
undis_pub = rospy.Publisher('img_undis',Image, queue_size=1)
testimg = None
rospy.spin()

    


