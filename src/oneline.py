# test the integrated system
# this node allows the vehicle to track a white line in a dark background

import rospy
import cv2
import numpy as np
import tiscamera
import threading
import matplotlib.pyplot as plt
import matplotlib

from sensor_msgs.msg import Image
from std_msgs.msg import Float64 as float_msg

# for debug only
def show(img):
    plt.imshow(img, cmap='gray')
    plt.show()
    return

class oneline:
    

    @staticmethod
    def init():

        oneline.bridge = CvBridge()
        rospy.init_node('oneline_node',log_level=rospy.DEBUG, anonymous=False)

        #shutdown routine
        #rospy.on_shutdown(oneline.cleanup)
        oneline.throttle = 0
        oneline.steering = 0
        oneline.throttle_pub = rospy.Publisher("/throttle",float_msg, queue_size=1)
        oneline.steering_pub = rospy.Publisher("/steering",float_msg, queue_size=1)
        oneline.test_pub = rospy.Publisher('img_test',Image, queue_size=1)
        oneline.sizex=1080
        oneline.sizey=768

        return

    @staticmethod
    def publish():
        oneline.throttle_pub.publish(oneline.throttle)
        oneline.steering_pub.publish(oneline.steering)
        return

    @staticmethod
    def imagecallback(data):
        try:
            cv_image = vision.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        gray = cv2.cvtColor(cv_image,cv2.COLOR_BGR2GRAY)
        # add cropping here

        # binary conversion
        avg = np.average(gray)
        gray[gray>avg]=255
        gray[gray<avg]=0
        gray[gray==255]=1

        image_message = oneline.bridge.cv2_to_imgmsg(gray, encoding="passthrough")
        oneline.test_pub.publish(image_message)

        # unwarp
        # calculate curvature
        return



