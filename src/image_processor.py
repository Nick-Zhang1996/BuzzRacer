#!/usr/bin/env python
#Based on code from http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class image_converter:
  def __init__(self):
    self.bridge = CvBridge()
    #Listen for images from cameras
    self.image_sub = rospy.Subscriber("/image_raw", Image, self.callback)

  #When an image from the camera comes in, convert it into cv2 and then process
  def callback(self, data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    #Do image processing on cv_image here
    

ic = image_converter()
rospy.init_node('image_processor', anonymous=True)
try:
  rospy.spin()
except KeyboardInterrupt:
  print("Shutting down")