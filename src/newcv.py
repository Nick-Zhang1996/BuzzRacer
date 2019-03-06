#This file contains algorithm developed for lane extraction and subsequent processing for use on RCP tracks

import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#import pickle
#import warnings
#import rospy
#import threading
#from os import listdir, mkdir, remove
#from os.path import isfile, join, isdir
#from getpass import getuser
#
#from sensor_msgs.msg import Image, CompressedImage 
#from rc_vip.msg import CarSensors as carSensors_msg
#from rc_vip.msg import CarControl as carControl_msg
#
#from calibration import imageutil
#from cv_bridge import CvBridge, CvBridgeError
#
from timeUtil import execution_timer
from time import time

from math import sin, cos, radians, degrees, atan2

from os import listdir
from os.path import isfile, join

def showpic(img):
    #showimg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img,cmap='gray')
    plt.show()
    return

# normalize an image with (0,255)
def normalize(data):
    data = data.astype(np.float32)
    data = data/255
    mean = np.mean(data)
    stddev = np.std(data)
    data = (data-mean)/stddev
    return data

path_to_file = "../img/trackpics/"
filenames = [join(path_to_file,f) for f in listdir(path_to_file) if isfile(join(path_to_file, f))]
#filepath = "../img/trackpics/pic1.jpg"


# pipeline

# clip unwanted portion
for filename in filenames:
    img = cv2.imread(filename)
    img = cv2.resize(img, (640,480))
    img = img[160:,:,:]
    #kernel = np.ones((9,9),np.float32)/81
    #img = cv2.filter2D(img,-1,kernel)

# straight edge extraction
# normalize
    img_l = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_l = normalize(img_l)
    img_l = np.array(img_l>0.7,dtype="uint8")

# red curve extraction
    img_c = img.astype("float")
    img_c = img_c[:,:,2]-img_c[:,:,0]-img_c[:,:,1]
    img_c = np.array(img_c>20,dtype="uint8")

# reconstruction

    img_l[img_c>0] = 5
    showpic(edges)

