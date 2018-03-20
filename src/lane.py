import numpy as np
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

x_size = 640
y_size = 480
cam = imageutil ('../calibrated/')

src_points = np.array([[68,344],[153,295],[496,303],[591,353]])
dst_points = np.array([[0.25*x_size,y_size-0.25*y_size],[0.25*x_size,y_size-0.567*y_size],[0.75*x_size,y_size-0.567*y_size],[0.75*x_size,y_size-0.25*y_size]])

src_points = src_points.astype(np.float32)
dst_points = dst_points.astype(np.float32)


class driveSys:

    @staticmethod
    def init():

        driveSys.bridge = CvBridge()
        rospy.init_node('driveSys_node',log_level=rospy.DEBUG, anonymous=False)

        #shutdown routine
        #rospy.on_shutdown(driveSys.cleanup)
        driveSys.throttle = 0
        driveSys.steering = 0
        driveSys.throttle_pub = rospy.Publisher("/throttle",float_msg, queue_size=1)
        driveSys.steering_pub = rospy.Publisher("/steering",float_msg, queue_size=1)
        driveSys.test_pub = rospy.Publisher('img_test',Image, queue_size=1)
        driveSys.sizex=x_size
        driveSys.sizey=y_size

        return

    @staticmethod
    def publish():
        driveSys.throttle_pub.publish(driveSys.throttle)
        driveSys.steering_pub.publish(driveSys.steering)
        rospy.loginfo("throttle = %f steering = %f",driveSys.throttle,driveSys.steering)
        image_message = driveSys.bridge.cv2_to_imgmsg(driveSys.testimg, encoding="passthrough")
        driveSys.test_pub.publish(image_message)
        return

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

def imagecallback(data):
    try:
        cv_image = vision.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)

    pipeline(cv_image)


    return
# takes a grayscale frame, return a centerline
# centerline is based on un-undistorted, un-unwarped image
def pipeline(frame):

    #crop
    frame = frame[240:,:]

    # normalize
    frame = normalize(frame)
    center_poly = findCenterline(frame)
    #ploty = np.linspace(0, gray.shape[0]-1, gray.shape[0] )
    #centerlinex = center_poly[0]*ploty**2 + center_poly[1]*ploty + center_poly[2]
    (steering,throttle) = drive(center_poly)
    driveSys.throttle = throttle
    driveSys.steering = steering
    driveSys.publish()
    return

# given the centerline poly, drive the car
# NEED WORK
def drive(center_poly):
    if (center_poly is None)
        throttle = 0.1
    else
        throttle = 0.2
        
    ploty = 0.25*y_size
    goalx = center_poly[0]*ploty**2 + center_poly[1]*ploty + center_poly[2]
    centerx = x_size/2
    diff = goalx-centerx
    steering = np.float32(diff)/150

    return (steering, throttle)


# given a binary image BINARY, where 1 means data and 0 means nothing
# return the best fit polynomial
def fitPoly(binary):
    nonzero = binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx = nonzerox
    lefty = nonzeroy

    left_fit = np.polyfit(lefty, leftx, 2)
    return left_fit


# find the centerline of two polynomials
def findCenterFromSide(left,right):
    return (left+right)/2

#direction
def findCenterline(gray, sobel_kernel=7, thresh=(0.6, 1.3)):
    #showg(gray)

    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=sobel_kernel)

    #graddir = np.arctan2(sobely, sobelx)

    norm = np.sqrt(np.square(sobelx)+np.square(sobely))
    norm = normalize(norm)
    #norm > 1 to get good edges

    
    # find left edges of while lanes
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


    # find right edge of lanes
    # XXX gray>1.5 is a sketchy solution that cut data size in half
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

    # now we analyze the long edges we have
    # case notation: e.g.(LR) -> left edge, right edge, from left to right

    # this logical is based on the assumption that the edges we find are lane edges
    # now we distinguish between several situations
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
    
    # case 2: we only see one edge of any sort, go to it till we see more (L,R)
    if (len(long_edge_lr)==1):
        with warnings.catch_warnings(record=True) as w:
            center_poly = fitPoly(long_edge_label[0])
            if len(w)>0:
                raise Exception('fail to fit poly')

    # case 3: if we get  (LR), then we are stepping on a lane, but don't know which that lane is (LR)
    # in this case drive on this lane until we see the other lane (TODO- find what lane it is)
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
                center_poly = findCenterFromSide(left_poly,right_poly)
        # if it's turning right, it's probably a left lane
        # vice versa
        # NOT IMPLEMENTED
    # otherwise we are completely lost
    else:
        flag_fail_to_find = True
        pass
    

    binary_output=None
    if (flag_good_road == True or flag_one_lane == True):
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

        fillspace = np.zeros([240,640])
        temp = np.vstack([fillspace,binary_output])
        temp = np.dstack([temp,temp,temp])
        temp = cam.undistort(temp)
        warped = warp(temp)

        driveSys.testimg = np.dstack([binary_output,binary_output,binary_output])
        #showmg(gray,sobelx,norm,binary_output)
        #show(warped)

    return center_poly
    
    
def testimg(filename):
    image = cv2.imread(filename)
    # we hold undistortion after lane finding because this operation discards data
    #image = cam.undistort(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pipeline(image)
    return
    

if __name__ == '__main__':
    print('begin')
    testpics =['../perspectiveCali/mid.png','../perspectiveCali/left.png','../img/0.png','../img/1.png','../img/2.png','../img/3.png','../img/4.png','../img/5.png','../img/6.png','../img/7.png'] 
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    Minv = cv2.getPerspectiveTransform(dst_points,src_points)
    driveSys.init()
    #total 8 pics
    for i in range(8):
        testimg(testpics[i])

