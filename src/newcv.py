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
    # can have an upper limit of 300, nothing interesting beyond that usually, but this messes up avg calculation in the presence of chessboard
    img = img[160:,:,:]
    # original image
    plt.imshow(img[:,:,::-1])
    plt.show()
    #kernel = np.ones((9,9),np.float32)/81
    #img = cv2.filter2D(img,-1,kernel)

# straight edge extraction
# normalize
    img_l = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_l = normalize(img_l)
    img_l = np.array(img_l>0.7,dtype="uint8")

# red curve extraction, this is pretty good
    img_c = img.astype("float")
    # BGR
    img_c = img_c[:,:,2]-img_c[:,:,0]-img_c[:,:,1]
    img_c = np.array(img_c>20,dtype="uint8")
    kernel = np.ones((13,13),np.uint8)
    img_c = cv2.dilate(img_c,kernel, iterations=1)

    # remove chessboard pattern at finishing line
    #mask = img_l.copy()
    #kernel = np.ones((3,3),np.uint8)
    #mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)

    #mask[img_c==1]=1
    #kernel = np.ones((4,4),np.uint8)
    #mask = cv2.erode(mask,kernel,iterations=1)


    # remove red corner from  image
    img_l[img_c==1] = 0
    kernel = np.ones((7,7),np.uint8)
    img_l = cv2.morphologyEx(img_l,cv2.MORPH_OPEN,kernel)

    #der_x = cv2.Sobel(img_l,cv2.CV_16S,1,0).astype(np.int8)
    der_y = cv2.Sobel(img_l,cv2.CV_16S,0,1).astype(np.int8)
    #der_mag = np.sqrt(der_x**2+der_y**2)
    edges = np.array(der_y<0,dtype="uint8")
    kernel = np.ones((5,5),np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    connectivity = 8
    output = cv2.connectedComponentsWithStats(edges, connectivity, cv2.CV_16U)
    # The first cell is the number of labels
    num_labels = output[0]
    # The second cell is the label matrix
    labels = output[1]
    # The third cell is the stat matrix
    stats = output[2]
    # The fourth cell is the centroid matrix
    centroids = output[3]

    # select edges closest to upper , left and right boundary (to exclude chessboard pattern noise)
    # first remove small components
    selected = []
    min_left = 1000
    min_right = 1000
    min_top = 1000

    label_left = -1
    label_right = -1
    label_top = -1

    print(stats)
    # loop is baaaad, do min with index or sth
    for i in range(1,num_labels):
        if stats[i,cv2.CC_STAT_AREA]<140 or stats[i,cv2.CC_STAT_TOP]>70:
            continue
        x = centroids[i,0]
        y = centroids[i,1]
        left = x
        right = 640-x
        if left < min_left:
            min_left = left
            label_left = i
        if right < min_right:
            min_right = right
            label_right = i
        if y < min_top:
            min_top = left
            label_top = i

# baaad coding
    labels[labels==label_left] = 255
    labels[labels==label_right] = 254
    labels[labels==label_top] = 253
    labels[labels<253] = 0
    labels[labels==255] = label_left
    labels[labels==254] = label_right
    labels[labels==253] = label_top

    showpic(labels*5)
    



    #lines = cv2.HoughLines(edges,8,5*np.pi/180,200)
    ##lines = cv2.HoughLinesP(edges,5,np.pi/180,200,maxLineGap)
    #for rho,theta in lines.reshape(-1,2):
    #    a = np.cos(theta)
    #    b = np.sin(theta)
    #    x0 = a*rho
    #    y0 = b*rho
    #    x1 = int(x0 + 1000*(-b))
    #    y1 = int(y0 + 1000*(a))
    #    x2 = int(x0 - 1000*(-b))
    #    y2 = int(y0 - 1000*(a))

    #    cv2.line(img,(x1,y1),(x2,y2),(255,0,0),2)

    #plt.imshow(img)
    #plt.show()

