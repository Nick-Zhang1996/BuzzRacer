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
            frame = driveSys.bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError as e:
            print(e)

        #crop
        frame = frame[240:,:]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        retval = driveSys.findCenterline(frame)
        if (retval is not None):
            (curvature,offset)=retval
            rospy.loginfo("curvature = %f offset = %f",curvature,offset)
            throttle = 0.247
            steer = driveSys.purePursuit(fit)
        else:
            throttle = 0
            steer = 0


        driveSys.throttle = throttle
        driveSys.steering = steer
        driveSys.publish()
        return

    # given curvature and offset, calculate appropriate steering value for the car
    @staticmethod
    def calcSteer(curvature,offset):
        ref = -offset*0.3
        print('steer=',ref)
        return np.clip(ref,-1,1)

    # given a gray image, spit out:
    #   a centerline curve x=f(y), 2nd polynomial. with car as (0,0)
    #   a predicted lane width
    @staticmethod
    def findCenterline(gray):
        #showg(gray)
        sobel_kernel=7
        thresh=(0.6, 1.3)

        # normalize
        t.s('normalize')
        gray = normalize(gray)
        t.e('normalize')

        # Calculate the x and y gradients
        t.s('sobel\t')
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=sobel_kernel)
        t.e('sobel\t')

        #graddir = np.arctan2(sobely, sobelx)

        # find the norm (magnitute) of gradient
        norm = np.sqrt(np.square(sobelx)+np.square(sobely))
        norm = normalize(norm)
        #norm > 1 to get good edges

        
        # find left edges of while lanes
        t.s('find left edges')
        binary_output =  np.zeros_like(gray,dtype=np.uint8)
        # XXX gray>1.5 is a sketchy solution that cut data size in half
        binary_output[(gray>1.5)&(sobelx>0) & (norm>1)] = 1
        #showg(binary_output)
        
        #label connected components
        connectivity = 8 
        output = cv2.connectedComponentsWithStats(binary_output, connectivity, cv2.CV_32S)
        # The first cell is the number of labels
        num_labels = output[0]
        # The second cell is the label matrix
        labels = output[1]
        # The third cell is the stat matrix
        stats = output[2]
        # The fourth cell is the centroid matrix
        centroids = output[3]

        '''
        # for DEBUG
        
        # Map component labels to hue val
        label_hue = np.uint8(179*labels/np.max(labels))
        blank_ch = 255*np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

        # cvt to BGR for display
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

        # set bg label to black
        labeled_img[label_hue==0] = 0

        showg(labeled_img)
        '''

        # find the two longest left edges
        line_labels = np.argsort(stats[:,cv2.CC_STAT_AREA][1:])[-2:]+1


        # list of centroids with corresponding left/right edge (of a white line)
        long_edge_centroids = []
        long_edge_lr = ""
        long_edge_label = []

        if (stats[line_labels[0],cv2.CC_STAT_AREA]>300):
            long_edge_centroids.append(centroids[line_labels[0],0])
            long_edge_lr += 'L'
            long_edge_label.append(labels==line_labels[0])
        if (stats[line_labels[1],cv2.CC_STAT_AREA]>300):
            long_edge_centroids.append(centroids[line_labels[1],0])
            long_edge_lr += 'L'
            long_edge_label.append(labels==line_labels[1])

        t.e('find left edges')

        # find right edge of lanes
        # XXX gray>1.5 is a sketchy solution that cut data size in half
        t.s('find right edg')
        binary_output =  np.zeros_like(gray,dtype=np.uint8)
        binary_output[(gray>1.5)&(sobelx<0) & (norm>1)] = 1
        
        #label connected components
        connectivity = 8 
        output = cv2.connectedComponentsWithStats(binary_output, connectivity, cv2.CV_32S)
        # The first cell is the number of labels
        num_labels = output[0]
        # The second cell is the label matrix
        labels = output[1]
        # The third cell is the stat matrix
        stats = output[2]
        # The fourth cell is the centroid matrix
        centroids = output[3]


        line_labels = np.argsort(stats[:,cv2.CC_STAT_AREA][1:])[-2:]+1

        if ( stats[line_labels[0],cv2.CC_STAT_AREA]>300):
            long_edge_centroids.append(centroids[line_labels[0],0])
            long_edge_lr += 'R'
            long_edge_label.append(labels==line_labels[0])

        if ( stats[line_labels[1],cv2.CC_STAT_AREA]>300):

            long_edge_centroids.append(centroids[line_labels[1],0])
            long_edge_lr += 'R'
            long_edge_label.append(labels==line_labels[1])


        # rank the edges based on centroid
        order = np.argsort(long_edge_centroids)
        long_edge_centroids = np.array(long_edge_centroids)[order]
        temp_lr = ""
        for i in order:
            temp_lr += long_edge_lr[i]
        long_edge_lr = temp_lr
        long_edge_label = np.array(long_edge_label)[order]

        t.e('find right edg')
        # now we analyze the long edges we have
        # case notation: e.g.(LR) -> left edge, right edge, from left to right

        # this logical is based on the assumption that the edges we find are lane edges
        # now we distinguish between several situations
        t.s('find centerline - lr analysis')
        flag_fail_to_find = False
        flag_good_road = False
        flag_one_lane = False
        centerPoly = None

        # case 1: if we find one and only one pattern (?RL?), we got a match
        if (long_edge_lr.count('RL')==1):
            index = long_edge_lr.find('RL')
            with warnings.catch_warnings(record=True) as w:
                left_poly = fitPoly(long_edge_label[index])
                index += 1
                right_poly = fitPoly(long_edge_label[index])
                if len(w)>0:
                    raise Exception('fail to fit poly')

                else:
                    flag_good_road = True
                    center_poly = findCenterFromSide(left_poly,right_poly)
        
        # case 2: we only see one edge of any sort
        if (len(long_edge_lr)==1):
            with warnings.catch_warnings(record=True) as w:
                side_poly = fitPoly(long_edge_label[0])
                if len(w)>0:
                    raise Exception('fail to fit poly')
                else:
                    flag_one_lane = True

        # case 3: if we get  (LR), then we are stepping on a lane, but don't know which that lane is (LR)
        # in this case drive on this lane until we see the other lane 
        elif (long_edge_lr == 'LR'):
            index = 0
            with warnings.catch_warnings(record=True) as w:
                left_poly = fitPoly(long_edge_label[index])
                index += 1
                right_poly = fitPoly(long_edge_label[index])
                if len(w)>0:
                    raise Exception('fail to fit poly')

                else:
                    flag_one_lane = True
                    side_poly = findCenterFromSide(left_poly,right_poly)

        # otherwise we are completely lost
        else:
            flag_fail_to_find = True
            pass

        # based on whether the line inclines to the left or right, guess which side it is
        if (flag_one_lane == True):
            x0 = side_poly[0]*1**2 + side_poly[1]*1 + side_poly[2] - x_size/2
            x1 = side_poly[0]*crop_y_size**2 + side_poly[1]*crop_y_size + side_poly[2] - x_size/2
            if (x1-x0>0):
                side = 'right'
            else:
                side = 'left'
        t.e('find centerline - lr analysis')

        
        binary_output=None
        if (flag_good_road == True):
            # DEBUG - for producing anice testimg
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

            driveSys.testimg = np.dstack(40*[binary_output,binary_output,binary_output])
            # END-DEBUG
            t.e('generate testimg')
            '''
            pass

            # get centerline in top-down view

	    t.s('change centerline perspective')

            # prepare sample points
            ploty = np.linspace(0, gray.shape[0]-1, gray.shape[0] )
            centerlinex = center_poly[0]*ploty**2 + center_poly[1]*ploty + center_poly[2]

            # convert back to uncropped space
            ploty += y_size/2
            pts_center = np.array(np.transpose(np.vstack([centerlinex, ploty])))
            pts_center = cam.undistortPts(np.reshape(pts_center,(1,-1,2)))

            # unwarp and change of units
            for i in range(len(pts_center)):
                pts_center[0,i,0],pts_center[0,i,1] = transform(pts_center[0,i,0],pts_center[0,i,1])
                
            # now pts_center should contain points in vehicle coordinate with x axis being rear axle,unit in cm
            #fit(y,x)
            fit = np.polyfit(pts_center[0,:,1],pts_center[0,:,0],2)
	    t.e('change centerline perspective')


            return fit


        if (flag_one_lane == True):

	    '''
            # DEBUG - for producing anice testimg

	    t.s('generate testimg')
            # Generate x and y values for plotting
            ploty = np.linspace(0, gray.shape[0]-1, gray.shape[0] )

            binary_output =  np.zeros_like(gray,dtype=np.uint8)

            # Draw centerline onto the image
            sidelinex = side_poly[0]*ploty**2 + side_poly[1]*ploty + side_poly[2]
            pts_side = np.array(np.transpose(np.vstack([sidelinex, ploty])))
            cv2.polylines(binary_output,np.int_([pts_side]), False, 1,1)

            driveSys.testimg = np.dstack(250*[binary_output,binary_output,binary_output])
	    t.e('generate testimg')
            # END-DEBUG
	    '''

            # get centerline in top-down view

	    t.s('change centerline perspective')

            # prepare sample points
            ploty = np.linspace(0, gray.shape[0]-1, gray.shape[0] )
            sidelinex = side_poly[0]*ploty**2 + side_poly[1]*ploty + side_poly[2]

            # convert back to uncropped space
            ploty += y_size/2
            pts_side = np.array(np.transpose(np.vstack([sidelinex, ploty])))
            pts_side = cam.undistortPts(np.reshape(pts_side,(1,-1,2)))

            # unwarp and change of units
            for i in range(len(pts_side)):
                pts_side[0,i,0],pts_side[0,i,1] = transform(pts_side[0,i,0],pts_side[0,i,1])
                
                # now pts_side should contain points in vehicle coordinate with x axis being rear axle,unit in cm
                #XXX this is really stupid and inefficient
                if (side == 'left'):
                    pts_side[0,i,0] = pts_side[0,i,0]+0.5*driveSys.lanewidth
                else:
                    pts_side[0,i,0] = pts_side[0,i,0]-0.5*driveSys.lanewidth

            # now pts_side should contain points in vehicle coordinate with x axis being rear axle,unit in cm
            #fit(y,x)
            fit = np.polyfit(pts_side[0,:,1],pts_side[0,:,0],2)

	    t.e('change centerline perspective')


            return fit

        return None

    @staticmethod
    def purePursuit(fit,lookahead=27):
        # anchor point coincide with rear axle
        # calculate target point
        a = fit[0]
        b = fit[1]
        c = fit[2]
        p = []
        p.append(a**2)
        p.append(a*b)
        p.append(b**2+a*c+1)
        p.append(b*c)
        p.append(c**2-lookahead**2)
        p = np.array(p)
        roots = np.roots(p)
        roots = roots[(roots<lookahead) & (roots>0)]
        if (roots is None):
            return None
        roots.sort()
        y = roots[-1]
        x = fit[0]*(y**2) + fit[1]*y + fit[2]

        # find curvature to that point
        curvature = (2*x)/(lookahead**2)

        # find steering angle for this curvature
        # not sure about this XXX
        wheelbase = 11
        steer_angle = math.atan(wheelbase, 1/curvature)/math.pi*180
        steer_output = steer_angle/30.0
        return steer_output
            


# universal functions


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


def transform(x, y):
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

def warp(image):

    warped = cv2.warpPerspective(image, M, (image.shape[1],image.shape[0]), flags=cv2.INTER_LINEAR)
    return warped


# given a binary image BINARY, where 1 means data and 0 means nothing
# return the best fit polynomial
def fitPoly(binary):
    nonzero = binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
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
    # we hold undistortion after lane finding because this operation discards data
    #image = cam.undistort(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #crop
    image = image[240:,:]
    driveSys.lanewidth=15
    driveSys.scaler = 25

    t.s()
    fit = driveSys.findCenterline(image)
    steer = driveSys.purePursuit(fit)
    t.e()
    print('steer = ',steer)
    return
    

t = execution_timer(True)
if __name__ == '__main__':
    print('begin')
    #testpics =['../perspectiveCali/mid.png','../perspectiveCali/left.png','../img/0.png','../img/1.png','../img/2.png','../img/3.png','../img/4.png','../img/5.png','../img/6.png','../img/7.png'] 
    testpics =['../img/0.png','../img/1.png','../img/2.png','../img/3.png','../img/4.png','../img/5.png','../img/6.png','../img/7.png'] 
    
    #driveSys.init()
    #total 8 pics
    for i in range(8):
        testimg(testpics[i])
    t.summary()
