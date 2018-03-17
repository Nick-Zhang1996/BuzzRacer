import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import warnings
from calibration import imageutil

x_size = 640
y_size = 480
cam = imageutil ('../calibrated/')

src_points = np.array([[68,344],[153,295],[496,303],[591,353]])
dst_points = np.array([[0.25*x_size,0.25*y_size],[0.25*x_size,0.567*y_size],[0.75*x_size,0.567*y_size],[0.75*x_size,0.25*y_size]])

src_points = src_points.astype(np.float32)
dst_points = dst_points.astype(np.float32)

def showg(img):
    plt.imshow(img,cmap='gray',interpolation='nearest')
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

# takes a grayscale frame, return a centerline
def pipeline(frame):

    #crop
    frame = frame[240:,:]

    # normalize
    frame = normalize(frame)
    showg(frame)
    sobel_threshold(frame)

    #frame[frame>1]=1
    #frame[frame<1]=0

#direction
def sobel_threshold(gray, sobel_kernel=7, thresh=(0.6, 1.3)):

    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=sobel_kernel)

    #graddir = np.arctan2(sobely, sobelx)

    norm = np.sqrt(np.square(sobelx)+np.square(sobely))
    norm = normalize(norm)
    showg(norm)
    #norm > 1 to get good edges

    
    # find RIGHT line
    # find left edge of lanes
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

    line_labels = np.argsort(stats[:,cv2.CC_STAT_AREA][1:])[-2:]+1
    # find the right line
    if (centroids[line_labels[0]][0]>centroids[line_labels[1]][0]):
        right_label = line_labels[0]
    else:
        right_label = line_labels[1]

    binary_output =  np.zeros_like(gray,dtype=np.uint8)
    binary_output[labels==right_label]=1
    showg(binary_output)

    return binary_output
    
    
def testimg(filename):
    image = cv2.imread(filename)
    image = cam.undistort(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pipeline(image)
    return
    

if __name__ == '__main__':
    print('begin')
    testpics =['../img/0.png','../img/1.png','../img/2.png','../img/3.png','../img/4.png','../img/5.png','../img/6.png','../img/7.png'] 
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    Minv = cv2.getPerspectiveTransform(dst_points,src_points)
    #total 8 pics
    for i in range(8):
        testimg(testpics[i])

