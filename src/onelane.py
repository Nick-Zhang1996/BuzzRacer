# this is one of the main files for Nick's summer work on his personal RC car
# The focus here is on control algorithms, new sensors, etc. NOT on determing the correct pathline to follow
# Therefore, the pathline will be a clearly visiable dark tape on a pale background. The line is about 0.5cm wide
# This file contains code that deals with the track setup at Nick's house, and may not be suitable for other uses

# TODO - deal with non-smooth pathlines

import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import warnings
import rospy
import threading

from sensor_msgs.msg import Image
from std_msgs.msg import Float64 as float_msg
from calibration import imageutil
from cv_bridge import CvBridge, CvBridgeError

from timeUtil import execution_timer

x_size = 640
y_size = 480
crop_y_size = 240
cam = imageutil('../calibrated/')

class driveSys:

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
            localcopy = driveSys.data
            driveSys.lock.release()

            if localcopy is not None:
                driveSys.drive(localcopy)

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
    def drive(data,noBridge = False):
        try:
            ori_frame = driveSys.bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError as e:
            print(e)

        #crop
        frame = ori_frame[240:,:]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        retval = driveSys.findCenterline(frame)
        if (retval is not None):
            throttle = 0.247
            fit = retval
            steer_angle = driveSys.purePursuit(fit)
            if (steer_angle is not None):
                steer = driveSys.calcSteer(steer_angle)
            else:
                saveImg(ori_frame)
        else:
            throttle = 0
            steer = 0


        driveSys.throttle = throttle
        driveSys.steering = steer
        driveSys.publish()
        return


    # given a steering angle, provide a -1.0-1.0 value for rostopic /steer_angle
    # XXX this is a temporary measure, this should be handled by arduino
    @staticmethod
    def calcSteer(angle):
        # values obtained from testing
        val = 0.0479*angle+0.2734
        print('steer=',val)
        if (val>1 or val<-1):
            print('insufficient steering')

        return np.clip(val,-1,1)

    # given a gray image, spit out:
    #   a centerline curve x=f(y), 2nd polynomial. with car's rear axle  as (0,0)
    @staticmethod
    def findCenterline(gray):

        ori = gray.copy()

        alpha = 20
        gauss = cv2.GaussianBlur(gray, (0, 0), sigmaX=alpha, sigmaY=alpha)

        gray = gray - gauss
        binary = normalize(gray)>1
        binary = binary.astype(np.uint8)


        # TODO a erosion here may help separate bad labels later

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
            if (stats[i,cv2.CC_STAT_AREA]>1000 and stats[i,cv2.CC_STAT_TOP]+stats[i,cv2.CC_STAT_HEIGHT] > 220 and stats[i,cv2.CC_STAT_HEIGHT]>80):
                goodLabels.append(i)

        if (len(goodLabels)==1):
            finalGoodLabel = goodLabels[0]
        elif (len(goodLabels)==0):
            print('no good feature')
            # find no lane, let caller deal with it
            return None
        elif (len(goodLabels)>1):
            # if there are more than 1 good label, pick the big one
            finalGoodLabel = np.amax(stats[:,cv2.CC_STAT_AREA][1:])
            rospy.logdebug("multiple good labels exist, no = "+str(len(goodLabels)))
            # TODO we probably want to record that frame
        
        with warnings.catch_warnings(record=True) as w:
            centerPoly = fitPoly((labels == finalGoodLabel).astype(np.uint8))
            if len(w)>0:
                #raise Exception('fail to fit poly')
                print('fail to fit poly')
                rospy.logdebug('fail to fit poly')
                return None

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

            #XXX don't forget to calibrate new camera

            # prepare sample points
            ploty = np.linspace(0, gray.shape[0]-1, gray.shape[0] )
            centerlinex = centerPoly[0]*ploty**2 + centerPoly[1]*ploty + centerPoly[2]

            # convert back to uncropped space
            ploty += y_size/2
            ptsCenter = np.array(np.transpose(np.vstack([centerlinex, ploty])))
            ptsCenter = cam.undistortPts(np.reshape(ptsCenter,(1,-1,2)))

            # unwarp and change of units
            for i in range(len(ptsCenter[0])):
                ptsCenter[0,i,0],ptsCenter[0,i,1] = transform(ptsCenter[0,i,0],ptsCenter[0,i,1])
                
            # now ptsCenter should contain points in vehicle coordinate with x axis being rear axle,unit in cm
            fit = np.polyfit(ptsCenter[0,:,1],ptsCenter[0,:,0],2)
	    t.e('change centerline perspective')

            return fit

    @staticmethod
    def purePursuit(fit,lookahead=27):
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
        roots = np.roots(p)
        roots = roots[np.abs(roots.imag)<0.00001]
        roots = roots.real
        roots = roots[(roots<lookahead) & (roots>0)]
        if ((roots is None) or (len(roots)==0)):
            return None
        roots.sort()
        y = roots[-1]
        x = fit[0]*(y**2) + fit[1]*y + fit[2]

        # find curvature to that point
        curvature = (2*x)/(lookahead**2)

        # find steering angle for this curvature
        # not sure about this XXX
        wheelbase = 11
        steer_angle = math.atan(wheelbase*curvature)/math.pi*180
        return steer_angle
            


# universal functions

# save frame as an image for debug
def saveImg(frame, steering=0, throttle=0):
    #text = "Steering: %f, Throttle: %f" % (steering, throttle)
    #cv2.putText(frame, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    image_message = driveSys.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
    nameTime = str(round(time.time()))
    name = './pics/' + nameTime + ".png"
    cv2.imwrite(name, frame)
    rospy.loginfo("debug img %s saved", nameTime+'.png')
    return



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

# matrix obtained from matlab linear fit, mse=1.79 on 17 data points
def transform(x,y):
    return 0.035*x-11.5713, -0.1111*y+74.1771

# XXX this is not accurate
def obs_transform(x, y):
    # alpha = 20 degrees, verticalFOV(vFov) = 15 degrees, horizontalFOV(hFov) = 15 degrees, h = 5.4 cm
    alpha = 3
    vFov = 27.0
    hFov = 40.0
    h = 5.4

    ob = h / math.cos(math.radians(90 - alpha - vFov))
    op = math.cos(math.radians(vFov)) * ob
    bp = math.sin(math.radians(vFov)) * ob

    if y > 0 and y <= 240:
        angle = math.degrees(math.atan((240-y)/240.0*bp/op)) + 90.0 - alpha
        actualY = math.tan(math.radians(angle))*h
    else:
        angle = 90 - alpha - math.degrees(math.atan((y-240)/240*bp/op))
        actualY = math.tan(math.radians(angle))*h

    om = actualY * math.tan(math.radians(hFov))
    
    if x > 0 and x <= 320:
        actualX = -(320-x)/320.0*om
    else:
        actualX = (x-320)/320.0*om
        
    actualY = actualY + 14
    
    return actualX, actualY
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
# return the best fit polynomial
def fitPoly(binary):
    # TODO would nonzero on a dense arrray consume lots of time?
    data = binary.nonzero()
    
    if (len(data)!=2):
        print(data)
    # raise some error here
        return
    
    reject = np.floor(np.random.rand(30)*len(data[0])).astype(np.int16)
    data = np.delete(data,tuple(reject),axis=1)

    nonzeroy = np.array(data[0])
    nonzerox = np.array(data[1])
    x = nonzerox
    y = nonzeroy

    fit = np.polyfit(y, x, 2)
    return fit


# find the centerline of two polynomials
def findCenterFromSide(left,right):
    return (left+right)/2

    
# run the pipeline on a test img    
def testimg(filename):
    image = cv2.imread(filename)
    if (image is None):
        print('No such file'+filename)
        return

    # we hold undistortion after lane finding because this operation discards data
    #image = cam.undistort(image)
    image = image.astype(np.float32)
    image = image[:,:,0]-image[:,:,2]+image[:,:,1]-image[:,:,2]

    #crop
    image = image[240:,:]
    driveSys.lanewidth=15
    driveSys.scaler = 25

    t.s()
    fit = driveSys.findCenterline(image)
    if (fit is None):
        print("Oops, can't find lane in this frame")
    else:
        steer_angle = driveSys.purePursuit(fit)
        steer = driveSys.calcSteer(steer_angle)
    t.e()
    return

def nothing(x):
    pass


# test perspective changing algorithm against measured value
def testperspective():
    src = np.array([[207,370], [220,387],[238,411],[430,368],[461,376],[486,379],[497,386],[554,385],[580,384],[612,384],[432,423],[330,333],[394,411],[390,398],[369,338],[600,394],[613,405]])
    dest = np.array([[-5,33],  [-4,31],  [-3,29],[4,33],[5,32],[6,32],[6,31],[8,31],[9,31],[10,31],[3,28],[0,38],[2,29],[2,30],[2,37],[9,30],[9,29]])
    mse = np.array([0,0],dtype=np.float64)
    for (a,b) in zip(src,dest):
        guess = transform(a[0],a[1])
        diff = guess-b
        mse += diff**2

    print(mse**0.5)
    return

    

t = execution_timer(True)
if __name__ == '__main__':

    print('begin')
    #testpics =['../perspectiveCali/mid.png','../perspectiveCali/left.png','../img/0.png','../img/1.png','../img/2.png','../img/3.png','../img/4.png','../img/5.png','../img/6.png','../img/7.png'] 
    testpics =[ '../img/pic1.jpeg',
                '../img/pic2.jpeg',
                '../img/pic3.jpeg',
                '../img/pic4.jpeg',
                '../img/pic5.jpeg',
                '../img/pic6.jpeg',
                '../img/pic7.jpeg',
                '../img/pic8.jpeg',
                '../img/pic9.jpeg',
                '../img/pic10.jpeg',
                '../img/pic11.jpeg',
                '../img/pic12.jpeg',
                '../img/pic13.jpeg',
                '../img/pic14.jpeg',
                '../img/pic15.jpeg',
                '../img/pic16.jpeg',
                '../img/pic17.jpeg',
                '../img/pic18.jpeg',
                '../img/pic19.jpeg',
                '../img/pic20.jpeg',
                '../img/pic21.jpeg',
                '../img/pic22.jpeg',
                '../img/pic23.jpeg',
                '../img/pic24.jpeg',
                '../img/pic25.jpeg',
                '../img/pic26.jpeg',
                '../img/pic27.jpeg']
    
    #driveSys.init()
    for i in range(27):
        testimg(testpics[i])
    t.summary()
