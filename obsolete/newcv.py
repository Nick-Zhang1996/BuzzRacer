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
#from rcvip_msgs.msg import CarSensors as carSensors_msg
#from rcvip_msgs.msg import CarControl as carControl_msg
#
#from calibration import imageutil
#from cv_bridge import CvBridge, CvBridgeError
#
from timeUtil import execution_timer
from time import time

from math import sin, cos, radians, degrees, atan2

from os import listdir
from os.path import isfile, join

g_transformMatrix = np.array(
        [[ -2.83091069e-02,   5.06336559e-05,   8.95601891e+00],
         [ -1.16359091e-03,  -4.99665389e-02,  -9.03024252e+00],
         [ -3.69582054e-05,  -6.06021304e-03,   1.00000000e+00]]

        )
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
    print(filename)
    img = cv2.imread(filename)
    ori_img = cv2.resize(img, (640,480))
    img = ori_img.copy()
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

    if not label_left in selected:
        selected.append(label_left)
    if not label_right in selected:
        selected.append(label_right)
    if not label_top in selected:
        selected.append(label_top)

    # extract points, convert to carframe, then show
    lines = []
    line_range = []
    for i in selected:
        pts = labels==i
        pts = pts.nonzero()
        pts = np.array(pts,dtype='float32')
        # 0->row,y, 1->col,x
        pts[0,:] += 160
        pts = pts[::-1,:]

        pts = pts.T.reshape(1,-1,2)
        pts = cv2.perspectiveTransform(pts, g_transformMatrix)

        # 0->x 1->y
        fit = np.polyfit(pts[0,:,0],pts[0,:,1],1)
        lines.append(fit)
        line_range.append([min(pts[0,:,0]),max(pts[0,:,0])])
        print(line_range)

    warp_img = cv2.warpPerspective(ori_img,g_transformMatrix, (1000,1000))
    plt.imshow(warp_img)
    plt.show()
    # next, TODO visualize everything, original image + fit line, 
    # recycle visualization from onelane


