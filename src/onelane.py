# this is one of the main files for Nick's summer work on his personal RC car
# The focus here is on control algorithms, new sensors, etc. NOT on determing the correct pathline to follow
# Therefore, the pathline will be a clearly visiable dark tape on a pale background. The line is about 0.5cm wide
# This file contains code that deals with the track setup at Nick's house, and may not be suitable for other uses

import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import warnings
import rospy
import threading
from os import listdir, mkdir, remove
from os.path import isfile, join, isdir
from getpass import getuser

from sensor_msgs.msg import Image
from std_msgs.msg import Float64 as float_msg
from calibration import imageutil
from cv_bridge import CvBridge, CvBridgeError

from timeUtil import execution_timer
from time import time

from math import sin, cos, radians, degrees, atan2

if (getuser()=='odroid'):
    DEBUG = False
    calibratedFilepath = "/home/odroid/catkin_ws/src/rc-vip/calibrated/"
else:
    DEBUG = True
    calibratedFilepath = "/home/nickzhang/catkin_ws/src/rc-vip/calibrated/"

x_size = 640
y_size = 480
crop_y_size = 240
cam = imageutil(calibratedFilepath)

g_wheelbase = 25.8
g_track = 16.0
g_lookahead = 70
g_max_steer_angle = 30.0
g_fileIndex = 1
g_slip_compensator = 1.4

g_transformMatrix = np.array(
       [[ -7.28065913e-02,   7.37353326e-04,   2.42476984e+01],
        [ -5.37538652e-03,  -1.42401754e-01,  -9.81881827e+00],
        [ -1.00912583e-04,  -5.56329041e-03,   1.00000000e+00],]
       )

class driveSys:

    debugImageIndex = 0
    lastDebugImageTimestamp = 0

    @staticmethod
    def init():

        driveSys.bridge = CvBridge()
        rospy.init_node('driveSys_node',log_level=rospy.DEBUG, anonymous=False)

        #shutdown routine
        #rospy.on_shutdown(driveSys.cleanup)
        driveSys.throttle = 0
        driveSys.steering = 0
        driveSys.vidin = rospy.Subscriber("image_raw", Image,driveSys.callback,queue_size=1,buff_size = 2**24)
        driveSys.throttle_pub = rospy.Publisher("/throttle",float_msg, queue_size=1)
        driveSys.steering_pub = rospy.Publisher("/steer_angle",float_msg, queue_size=1)
        driveSys.test_pub = rospy.Publisher('img_test',Image, queue_size=1)
        driveSys.testimg = None
        driveSys.sizex=x_size
        driveSys.sizey=y_size
        driveSys.scaler = 25
        # unit: cm
        driveSys.lanewidth=15
        driveSys.lock = threading.Lock()

        driveSys.data = None
        while not rospy.is_shutdown():
            driveSys.lock.acquire()
            # XXX does this create only a reference?
            driveSys.localcopy = driveSys.data
            driveSys.lock.release()

            if driveSys.localcopy is not None:
                driveSys.drive(driveSys.localcopy)

        rospy.spin()

        return

    # TODO handle basic cropping and color converting at this level, or better yet before it is published
    # update current version of data, thread safe
    @staticmethod
    def callback(data):
        driveSys.lock.acquire()
        driveSys.data = data
        driveSys.lock.release()
        return

    @staticmethod
    def publish():
        driveSys.throttle_pub.publish(driveSys.throttle)
        driveSys.steering_pub.publish(driveSys.steering)
        rospy.loginfo("throttle = %f steering = %f",driveSys.throttle,driveSys.steering)
        if (driveSys.testimg is not None):
            image_message = driveSys.bridge.cv2_to_imgmsg(driveSys.testimg, encoding="passthrough")
            driveSys.test_pub.publish(image_message)
        return
    
    # handles frame pre-processing and post status update
    @staticmethod
    def drive(data, noBridge = False):
        try:
            frame = driveSys.bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError as e:
            print(e)

        # handle this later so we don't lose information here
        #frame = cam.undistort(ori_frame)
        #crop
        #DEBUG: save every frame
        driveSys.saveImg()

        frame = frame[240:,:]

        frame = frame.astype(np.float32)
        frame = frame[:,:,2]-frame[:,:,0]+frame[:,:,1]-frame[:,:,0]

        retval = driveSys.findCenterline(frame)
        if (retval is not None):
            fit = retval
            steer_angle = driveSys.purePursuit(fit)
            if (steer_angle is None):
                throttle = 0.0
                steer_angle = 0.0
                # we have a lane but no steer angle? worth digging
                rospy.logdebug("can't find steering angle for current path")
                driveSys.saveImg()

            elif (steer_angle > g_max_steer_angle):
                steer_angle = g_max_steer_angle
                throttle = 0.3
                rospy.loginfo("insufficient steering - R")
            elif (steer_angle < - g_max_steer_angle):
                steer_angle = -g_max_steer_angle
                throttle = 0.3
                rospy.loginfo("insufficient steering - L")
            else:
                steer_angle = steer_angle * calcSteer(steer_angle)
                throttle = 0.5

        else:
            throttle = 0
            steer_angle = 0
            rospy.loginfo("can't find centerline")
            # not saving debug image here since this state is too general and contains the car not being placed on track at all, just pure garbage
            driveSys.saveImg()


        driveSys.throttle = throttle
        driveSys.steering = steer_angle
        driveSys.publish()
        return


    # calculate the actual steering angle, compensate for slip angle
    @staticmethod
    def calcSteer(angle):
        # values obtained from testing

        return np.clip(angle*g_slip_compensator,-g_max_steer_angle,g_max_steer_angle)

    # given a gray image, spit out:
    #   a centerline curve x=f(y), 2nd polynomial. with car's rear axle  as (0,0)
    @staticmethod
    def findCenterline(gray, returnBinary = False):

        t.s('copy original')
        ori = gray.copy()
        t.e('copy original')


        t.s('blur')
        #alpha = 20
        #gauss = cv2.GaussianBlur(gray, (0, 0), sigmaX=alpha, sigmaY=alpha)
        blurred = cv2.blur(gray, (30, 30))
        t.e('blur')

        t.s('Norm')
        gray = gray - blurred
        binary = normalize(gray)>1
        binary = binary.astype(np.uint8)
        t.e('Norm')
        #showg(ori)
        #showg(binary)


        # TODO a erosion here may help separate bad labels later

        t.s('connected comp')
        #label connected components
        connectivity = 8 
        #XXX will doing one w/o stats for fast removal quicker?
        output = cv2.connectedComponentsWithStats(binary, connectivity, cv2.CV_32S)
        # The first cell is the number of labels
        num_labels = output[0]
        # The second cell is the label matrix
        labels = output[1]
        # The third cell is the stat matrix
        stats = output[2]
        # The fourth cell is the centroid matrix
        centroids = output[3]


        # apply known rejection standards here
        goodLabels = []
        # label 0 is background, start at 1
        for i in range(1,num_labels):
            if (stats[i,cv2.CC_STAT_AREA]>1000 and stats[i,cv2.CC_STAT_TOP]+stats[i,cv2.CC_STAT_HEIGHT] > 190 and stats[i,cv2.CC_STAT_HEIGHT]>80):
                goodLabels.append(i)

                # DEBUG
                #binary[labels==i]=0
        pass

        if (len(goodLabels)==1):
            finalGoodLabel = goodLabels[0]
            binary = labels==finalGoodLabel

        elif (len(goodLabels)==0):
            print('no good feature')
            # find no lane, let caller deal with it
            if (returnBinary):
                return None, binary
            else:
                return None

        elif (len(goodLabels)>1):
            # if there are more than 1 good label, pick the big one
            finalGoodLabel = goodLabels[np.argmax(stats[goodLabels,cv2.CC_STAT_AREA])]
            rospy.logdebug("multiple good labels exist, no = "+str(len(goodLabels)))

            binary = labels==finalGoodLabel
            # note: frequently this is 2
            # driveSys.saveImg()
        else:
            pass

        t.e('connected comp')
        
# BEGIN: DEBUG -----------------
        '''
        cv2.namedWindow('binary')
        cv2.imshow('binary',binary)
        cv2.createTrackbar('label','binary',0,len(goodLabels)-1,nothing)
        last_selected = 1

        # visualize the remaining labels
        while(1):

            selected = goodLabels[cv2.getTrackbarPos('label','binary')]

            binaryGB = binary.copy()
            binaryGB[labels==selected] = 0
            testimg = 255*np.dstack([binary,binaryGB,binaryGB])
            cv2.imshow('binary',testimg)

            #list info here

            if (selected != last_selected):
                print('label --'+str(selected))
                print('Area --\t'+str(stats[selected,cv2.CC_STAT_AREA]))
                print('Bottom --\t'+str(stats[selected,cv2.CC_STAT_TOP]+stats[selected,cv2.CC_STAT_HEIGHT]))
                print('Height --\t'+str(stats[selected,cv2.CC_STAT_HEIGHT]))
                print('WIDTH --\t'+str(stats[selected,cv2.CC_STAT_WIDTH]))
                print('---------------------------------\n\n')
                last_selected = selected

            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                print('next')
                break
        cv2.destroyAllWindows()
        return
        '''

# END: DEBUG -------------------------

        t.s('fitPoly')
        with warnings.catch_warnings(record=True) as w:
            # XXX do we need the astype here?
            centerPoly = fitPoly((labels == finalGoodLabel).astype(np.uint8), yOffset = 240)
            if ( centerPoly is None):
                rospy.loginfo("fail to fit poly - None")
                driveSys.saveImg()
                if (returnBinary):
                    return None, binary
                else:
                    return None

            if len(w)>0:
                #raise Exception('fail to fit poly')
                #print('fail to fit poly')
                rospy.loginfo('fail to fit poly')
                driveSys.saveImg()
                if (returnBinary):
                    return None, binary
                else:
                    return None
        t.e('fitPoly')

        # debug the curve fitting in original image 
        #testimg = 100* np.dstack([binary, binary, binary])
        #testimg = drawPoly(testimg.astype(np.uint8), centerPoly, 240,480, yOffset = -240)
        #show(testimg)

        '''
        t.s('generate testimg')
        # Generate x and y values for plotting
        ploty = np.linspace(0, gray.shape[0]-1, gray.shape[0] )
        left_fitx = left_poly[0]*ploty**2 + left_poly[1]*ploty + left_poly[2]
        right_fitx = right_poly[0]*ploty**2 + right_poly[1]*ploty + right_poly[2] 
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the blank image
        binary_output =  np.zeros_like(gray,dtype=np.uint8)
        cv2.fillPoly(binary_output, np.int_([pts]), 1)

        # Draw centerline onto the image
        centerlinex = center_poly[0]*ploty**2 + center_poly[1]*ploty + center_poly[2]
        pts_center = np.array(np.transpose(np.vstack([centerlinex, ploty])))
        cv2.polylines(binary_output,np.int_([pts_center]), False, 5,10)

        #driveSys.testimg = np.dstack(40*[binary_output,binary_output,binary_output])
        # END-DEBUG
        t.e('generate testimg')
        '''
        # get centerline in top-down view

        t.s('change centerline perspective')


        
        # prepare sample points
        # TODO do this symbolically
        # XXX don't use constant here
        # the centerPoly is in original, distorted space
        ploty = np.linspace(240, 480-1, 240 )
        plotx = np.polyval(centerPoly, ploty)
        mask = np.logical_and(plotx>0,plotx<640)
        mask = np.logical_and(mask, ploty<480)
        mask = np.logical_and(mask, ploty>0)
        plotx = plotx[mask]
        ploty = ploty[mask]


        ptsCenter = np.array(np.transpose(np.vstack([plotx, ploty])))
        ptsCenter = cam.undistortPts(np.reshape(ptsCenter,(1,-1,2)))


        # undistortPts() maps points to locations way beyond reasonable range
        # thus it is necessary to discard datapoints beyond a certain range
        # from experiments this threshold size is 1.3*originalSize
        # for a (480,640) image, the center is (240,320), and the new half side length 
        # is (240*1.3, 320*1.3) Note this space also extends to negative coordinates
        # the four bounding coordinate is (TopLeft, TopRight, BottomL, BotR) (x,y)
        # (-96,-72), (736,-72), (-96,552), (736,552)
        # XXX this is not conclusive
        x = ptsCenter[0,:,0] 
        y = ptsCenter[0,:,1]
        mask = np.logical_and(x>-96,x<736)
        mask = np.logical_and(mask, y<552)
        mask = np.logical_and(mask, y>-72)
        ptsCenter = ptsCenter[:,mask,:]
        if (len(ptsCenter[0])<50):
            print('all datapoints out side of valid range')
            if (returnBinary):
                return None, binary
            else:
                return None
        




        # if for some reason the new dimension is not added in undistortPts, add it here
        #ptsCenter = ptsCenter[np.newaxis,...]

        # unwarp and change of units
        # TODO use a map to spped it up
        ptsCenter = cv2.perspectiveTransform(ptsCenter, g_transformMatrix)
            
        # now ptsCenter should contain points in vehicle coordinate with x axis being rear axle,unit in cm
        fit = np.polyfit(ptsCenter[0,:,1],ptsCenter[0,:,0],2)

        # DEBUG: draw the ptscenter and the fitline

        # DEBUG: plot persp transformed points
        #canvasSize = (800,800)
        #debugimg = np.zeros((800,800),dtype=np.uint8)
        #x = ptsCenter[0,:,0] 
        #y = ptsCenter[0,:,1] 
        #x *= 10
        #y *= 10
        #x += 400

        #x[x>=canvasSize[1]] = canvasSize[1]-1
        #x[x<0] = 0
        #y[y>=canvasSize[0]] = canvasSize[0]-1
        #y[y<0] = 0
        #debugimg[[y.astype(np.int).tolist(),x.astype(np.int).tolist()]]=1

        #py = np.linspace(0,10,801)
        #px = np.polyval(fit,py)
        #px *= 10
        #py *= 10
        #px += 400
        #px[px>=canvasSize[1]] = canvasSize[1]-1
        #px[px<0] = 0
        #py[py>=canvasSize[0]] = canvasSize[0]-1
        #py[py<0] = 0

        #debugimg[[py.astype(np.int).tolist(),px.astype(np.int).tolist()]]=2

        #showg(debugimg)

        t.e('change centerline perspective')

        if (returnBinary):
            return fit, binary
        else:
            return fit

# fit-> x = f(y)
    @staticmethod
    def purePursuit(fit,lookahead=g_lookahead, returnDebugInfo = False):
        #pic = debugimg(fit)
        # anchor point coincide with rear axle
        # calculate target point
        a = fit[0]
        b = fit[1]
        c = fit[2]
        p = []
        p.append(a**2)
        p.append(2*a*b)
        p.append(b**2+2*a*c+1)
        p.append(2*b*c)
        p.append(c**2-lookahead**2)
        p = np.array(p)
        
        # roots are y coord of pursuit points
        roots = np.roots(p)
        roots = roots[np.abs(roots.imag)<0.00001]
        roots = roots.real
        roots = roots[(roots<lookahead) & (roots>0)]
        if ((roots is None) or (len(roots)==0)):
            if (returnDebugInfo):
                return None, (None,None), None
            else:
                calcopy, "bgr8")
            name = g_saveDir + str(driveSys.debugImageIndex) + ".png"
            driveSys.debugImageIndex += 1
            rospy.loginfo("debug img %s saved", str(driveSys.debugImageIndex) + ".png")
            cv2.imwrite(name, cv2_image)
            return

# universal functions



# draw a polynomial x = f(y) onto an IMG, with y =  [start,finish), when plotting, shift the curve in y direction by yOffset pixels. This is useful when the polynomial is represented in an uncropped version of the current img
# if painting on a grayscale image, user must specify color as an int
def drawPoly(img, poly, start, finish, color =  (255,255,255), thickness = 3, yOffset = 0):
    ploty = np.linspace(start, finish-1, finish-start)
    plotx = np.polyval(poly, ploty)
    pts = np.array(np.transpose(np.vstack([plotx, ploty+yOffset])))
    if (img.shape[-1] ==3):
        cv2.polylines(img, np.int_([pts]), False, color = color,thickness = thickness)
    elif (len(img.shape) == 2):
        cv2.polylines(img, np.int_([pts]), False, color = color,thickness = thickness)
    else:
        print('channel != 1 or 3, expect a normal picture')

    return img


# generate a top-down view for transformed fitted curve
def debugimg(poly):
    # Generate x and y values for plotting
    ploty = np.linspace(0,40,41)

    binary_output =  np.zeros([41,20],dtype=np.uint8)

    # Draw centerline onto the image
    x = poly[0]*ploty**2 + poly[1]*ploty + poly[2]
    x = x+10
    ploty = 40-ploty
    pts = np.array(np.transpose(np.vstack([x, ploty])))
    cv2.polylines(binary_output,np.int_([pts]), False, 1,1)

    return  binary_output

def showg(img):
    plt.imshow(img,cmap='gray',interpolation='nearest')
    plt.show()
    return

def show(img):
    plt.imshow(img,interpolation='nearest')
    plt.show()
    return

def showmg(img1,img2=None,img3=None,img4=None):
    plt.subplot(221)
    plt.imshow(img1,cmap='gray',interpolation='nearest')
    if (img2 is not None):
        plt.subplot(222)
        plt.imshow(img2,cmap='gray',interpolation='nearest')
    if (img3 is not None):
        plt.subplot(223)
        plt.imshow(img3,cmap='gray',interpolation='nearest')
    if (img4 is not None):
        plt.subplot(224)
        plt.imshow(img4,cmap='gray',interpolation='nearest')

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

# do a local normalization on grayscale image with [0,1] space
# alpha and beta are the sigma values for the two blur
def localNormalize(float_gray, alpha=2, beta=20):

    blur = cv2.GaussianBlur(float_gray, (0, 0), sigmaX=alpha, sigmaY=alpha)
    num = float_gray - blur

    blur = cv2.GaussianBlur(num*num, (0, 0), sigmaX=beta, sigmaY=beta)
    den = cv2.pow(blur, 0.5)

    gray = num / den

    cv2.normalize(gray, dst=gray, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
    return gray

def warp(image):

    warped = cv2.warpPerspective(image, M, (image.shape[1],image.shape[0]), flags=cv2.INTER_LINEAR)
    return warped


# given a binary image BINARY, where 1 means data and 0 means nothing
# return the best fit polynomial x = f(y)
# yOffset : offset to be added to the y coordinates of nonzero pixels
# NOTE: the polynomial is in the image coordinate system, which may need additional work if
# the image given is cropped or shifted. This can be done with yOffset. 
def fitPoly(binary, yOffset = 0):
    # TODO would nonzero on a dense arrray consume lots of time?
    data = binary.nonzero()
    
    if (len(data)!=2):
        rospy.logfatal("Data structure for fitPoly() is wrong, expecting 2 channel for x and y")
    #    print(data)
    # raise some error here
        return None
    
    # TODO properly remove a portion of the datapoints
    #reject = np.floor(np.random.rand(30)*len(data[0])).astype(np.int16)
    #data = np.delete(data,tuple(reject),axis=1)

    nonzeroy = np.array(data[0])
    if (len(nonzeroy) == 0):
        rospy.loginfo("fitPoly -- error: No nonzero points to work with")
        return None

    nonzerox = np.array(data[1])
    x = nonzerox
    y = nonzeroy + yOffset

    fit = np.polyfit(y, x, 2)
    return fit


# find the centerline of two polynomials
def findCenterFromSide(left,right):
    return (left+right)/2

    
def testvid(filename):
    cap = cv2.VideoCapture(filename)
    if (cap is None):
        print('No such file : '+filename)
        return

    cv2.namedWindow('original')
    cv2.namedWindow('kinematics')
    cv2.moveWindow('original', 40,30)
    cv2.moveWindow('kinematics', 700,30)

    debugImg = None

    while(cap.isOpened()):
        ret, image = cap.read()

        # we hold undistortion after lane finding because this operation discards data
        image = cam.undistort(image)
        undistorted = image.copy()
        image = image.astype(np.float32)
        image = image[:,:,0]-image[:,:,2]+image[:,:,1]-image[:,:,2]

        #crop
        # XXX is this messing up our perspective transformation?
        #image = image[240:,:]

        driveSys.lanewidth=15

        driveSys.scaler = 25

        t.s()
        fit, debugImg = driveSys.findCenterline(image, returnBinary=True)
        if (fit is None):
            print("Oops, can't find lane in this frame")

        else:
            t.s('pure pursuit')
            steer_angle, (x,y) = driveSys.purePursuit(fit, returnDebugInfo = True)
            t.e('pure pursuit')
            if (steer_angle is None):
                print("err: curve found, but can't find steering angle")
            else:
                print(steer_angle)

            #genKinematicsImg(fit, steer_angle, (x,y))

        cv2.imshow('originalistorted',undistorted)
        if (debugImg is not None):
        
            cv2.imshow('kinematics',255*np.dstack([debugImg,debugImg,debugImg]).astype(np.uint8))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return


# run the pipeline on a test img    
def testimg(filename):
    print('----------')
    image = cv2.imread(filename)
    original = image.copy()

    winname = filename
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
    cv2.imshow(winname,original)
    if (cv2.waitKey(10000) == ord('d')):
        remove(filename)
        print(filename+' removed')
        cv2.destroyAllWindows()
        return
    else:
        cv2.destroyAllWindows()


    # special handle for images saved wrong
    #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    if (image is None):
        print('No such file : '+filename)
        return

    # handle undistortion later because this process loses important data
    # image = cam.undistort(image)
    #crop
    image = image[240:,:]
    image = image.astype(np.float32)
    image = image[:,:,0]-image[:,:,2]+image[:,:,1]-image[:,:,2]

    driveSys.lanewidth=15

    driveSys.scaler = 25

    t.s()
    fit, binary = driveSys.findCenterline(image, returnBinary = True)
    # NOTE binary here is distorted 
    #showg(binary)
    unwarpped = genWarpedBinary(binary)
    unwarpped = 100 * np.dstack([unwarpped, unwarpped, unwarpped])

    if (fit is None):
        print("Oops, can't find lane in this frame")
        winname = filename
        cv2.namedWindow(winname)        # Create a named window
        cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
        cv2.imshow(winname,original)
        if (cv2.waitKey(10000) == ord('d')):
            remove(filename)
            print(filename+' removed')
        else:
            pass

        cv2.destroyAllWindows()

    else:
        t.s('pure pursuit')
        steer_angle, (x,y) , curvature = driveSys.purePursuit(fit, returnDebugInfo = True)
        t.e('pure pursuit')
        if (steer_angle is None):
            print("err: curve found, but can't find steering angle")
        else:
            print(steer_angle)
            kinematics = genKinematicsImg(fit, steer_angle, (x,y), curvature)
            show(cv2.addWeighted(unwarpped, 0.8, kinematics, 1.0, 0.0))

    t.e()
    return

# draw a line of LENGTH at CENTER, angled at ANGLE (right positive, in deg)
def drawLine(img, center, angle, length = 60, weight = 4, color = (255,255,255)):
    dsin = lambda x : sin(radians(x))
    dcos = lambda x : cos(radians(x))
    bottomLeft = (int(center[0] - 0.5*length*dsin(angle)), int(center[1] + 0.5*length*dcos(angle)))
    upperRight = (int(center[0] + 0.5*length*dsin(angle)), int(center[1] - 0.5*length*dcos(angle)))
    cv2.line(img, bottomLeft, upperRight, color, weight)
    return img

# draw an image for debugging kinematics.
# Given fit, the function returns an image including the following components
# vehicle, front wheels, projected pathline, lookahead circle, target pursuit point
# FIT : x = f(y) in car coord
def genKinematicsImg(fit, steer_angle, targetCoord, curvature):
    canvasSize = ( 800, 800)
    if (steer_angle is None):
        fitOnly = True
    else:
        fitOnly = False

    img = np.zeros((canvasSize[0],canvasSize[1],3),dtype = np.uint8)
    pixelPerCM = 10

    # shift transforms car coord to canvas coord
    shift = lambda x,y : (int(x*pixelPerCM+canvasSize[0]/2), int(canvasSize[1] - y*pixelPerCM))
    shiftx = lambda x : int(x*pixelPerCM+canvasSize[0]/2)
    shifty = lambda y :  int(canvasSize[1] - y*pixelPerCM)

    toCanvas = lambda x : int(x*pixelPerCM)

    # draw the car
    cv2.rectangle(img, shift(-g_track/2,g_wheelbase), shift(g_track/2, 0), (255,255,255),3)

    # draw the pathline
    ploty = np.linspace(0, 140, 141 )
    plotx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
    cv2.polylines(img, np.array([map(shift,plotx,ploty)]), False, (255,255,255), 3)

    # Plot the camera view window, two sets are possible
    # set 1: if working with undistorted image
    cv2.polylines(img, np.array([map(shift,[13,-15,-72,72],[47,47,131,131])]),True, (100,0,0),3)

    # set 2: if working with distorted image (original frame)
    # since the boundary would be a curve and I don't want to calculate that, you'll have to deal 
    # with undistorted view

    if (not fitOnly):
        x,y = targetCoord

        # draw front wheels with correct steer_angle
        drawLine(img, shift(-g_track/2, g_wheelbase), steer_angle, length = toCanvas(6))
        drawLine(img, shift(g_track/2, g_wheelbase), steer_angle, length = toCanvas(6))

        # draw lookahead circle
        cv2.circle(img, shift(0,0), toCanvas(g_lookahead), color = (0,0,100), thickness = 3)
        # target point
        cv2.circle(img, shift(*targetCoord), 10, color = (255,0,0), thickness = 2)

        # draw the expected pathline leading to the target coordinate
        start = 0 if curvature>0 else degrees(atan2(y,x))
        end = degrees(atan2(y,x)) if curvature>0 else 180
        radius = toCanvas(1/np.abs(curvature))
        start = 0
        end = 360
        cv2.ellipse(img, 
                    center = shift(int(1/curvature),0), 
                    axes = (radius,radius), 
                    angle = 0,
                    startAngle = start, 
                    endAngle = end , 
                    color = (0,255,0), 
                    thickness = 3
                    )
    else:
        pass

    # debug texts
    #show(img)
    return img

def nothing(x):
    pass
def constrain(x,lower,upper):
    x = lower if x<lower else x
    x = upper if x>upper else x
    return x

# WARNING: extremely inefficient, DEBUG only
def testPerspectiveTransform():
    img = cv2.imread("../img/perspectiveCalibration23cm.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #showg(gray)
    unwarppedSize = (800,800)
    unwarpped = np.zeros( unwarppedSize, dtype= np.uint8)

    for i in range(0,gray.shape[0]):
        for j in range(gray.shape[1]):
            ret = cv2.perspectiveTransform(np.array([[[float(j),float(240+i)]]]),g_transformMatrix)
            x = ret[0,0,0]
            y = ret[0,0,1]
            x = x*10 + 400
            y = unwarppedSize[1] - (y*10)
            x = constrain(int(x), 0, unwarppedSize[1]-1)
            y = constrain(int(y), 0, unwarppedSize[0]-1)
            unwarpped[y,x] = gray[i,j]
    showg(unwarpped)
    return

# WARNING: extremely inefficient, DEBUG only
def testUndistort(gray):
    #img = cv2.imread("../debug/run1/19.png")
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #showg(gray)
    unwarppedSize = (800,800)
    unwarpped = np.zeros( unwarppedSize, dtype= np.uint8)
    count = 0

    for i in range(0,gray.shape[0]):
        for j in range(gray.shape[1]):
            ret = cam.undistortPts(np.array([j,i], dtype = np.float).reshape(-1,1,2))
            x = ret[0,0,0]
            y = ret[0,0,1]
            x = constrain(int(x), 0, unwarppedSize[1]-1)
            y = constrain(int(y), 0, unwarppedSize[0]-1)
            unwarpped[y,x] = gray[i,j]
            count += 1 if gray[i,j]!=0 else 0
    print(count)
    showg(unwarpped)
    
    return
# this takes a CROPPED binary (240 lower rows only)
# and generate an perspective transformed image that matches genKinematicsImg()
# by default this takes an distorted img, if BINARY is undistorted, set UNDISTORTIMG to False
def genWarpedBinary(binary, undistortImg = True):

    canvasSize = ( 800, 800)
    pixelPerCM = 10
    unwarpped = np.zeros( canvasSize, dtype= np.uint8)
    nonzeroPts = binary.nonzero()
    nonzeroPts = np.vstack([[nonzeroPts[1],nonzeroPts[0]+240]]).T.reshape(-1,1,2)
    nonzeroPts = nonzeroPts.astype(np.float)
    if (undistortImg):
        nonzeroPts = cam.undistortPts(nonzeroPts)
    else: 
        pass

    ret = cv2.perspectiveTransform( nonzeroPts, g_transformMatrix)
    # 0->col->x
    ret[:,0,0] = ret[:,0,0]*10 + canvasSize[0]/2
    ret[:,0,1] = canvasSize[1] - (ret[:,0,1]*10)
    x = ret[:,0,0]
    y = ret[:,0,1]
    x[x>=canvasSize[1]] = canvasSize[1]-1
    x[x<0] = 0
    y[y>=canvasSize[0]] = canvasSize[0]-1
    y[y<0] = 0

    unwarpped[[y.astype(np.int).tolist(),x.astype(np.int).tolist()]]=1
    #showg(unwarpped)

    return unwarpped


t = execution_timer(False)
if __name__ == '__main__':

    print('begin')

    # ---------------- for pictures ------------------
    '''

        path_to_file = '../debug/run2/'
        testpics = [join(path_to_file,f) for f in listdir(path_to_file) if isfile(join(path_to_file, f))]
        
        #testpics =[ '../debug/run1/0.png']
        for i in range(len(testpics)):
            testimg(testpics[i])
    '''

    # ---------------- for vids --------------
    if (DEBUG):

        #testvid('../img/run1.avi')

        path_to_file = '../debug/run5/'
        testpics = [join(path_to_file,f) for f in listdir(path_to_file) if isfile(join(path_to_file, f))]
        if len(testpics)==0 :
            print('empty folder')

        else:

            #testpics =[ '../debug/run1/0.png']
            for i in range(len(testpics)):
                testimg(testpics[i])

            t.summary()

    else:
        g_saveDir = "/home/odroid/catkin_ws/src/rc-vip/debug/run%d" % (g_fileIndex)
        while (isdir(g_saveDir)):
            g_fileIndex += 1
            g_saveDir = "/home/odroid/catkin_ws/src/rc-vip/debug/run%d" % (g_fileIndex)

        g_saveDir = "/home/odroid/catkin_ws/src/rc-vip/debug/run%d" % (g_fileIndex)
        mkdir(g_saveDir)
        g_saveDir += "/"

        driveSys.init()
