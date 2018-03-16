import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import warnings

x_size = 1280
y_size = 720

src_points = np.array([[304,742],[596,534],[706,533],[1034,742]])-np.array([[8,78],[8,78],[8,78],[8,78]])
dst_points = np.array([[0.25*x_size,y_size],[0.25*x_size,0.25*y_size],[0.75*x_size,0.25*y_size],[0.75*x_size,y_size]])

src_points = src_points.astype(np.float32)
dst_points = dst_points.astype(np.float32)

class Line():
    radius_smooth_factor = 0.1
    offset_smooth_factor = 0.1
    poly_smooth_factor = 0.7

    def __init__(self):
        #if the last frame yielded a valid result
        self.leftIsValid = False
        self.rightIsValid = False
        #left 
        self.leftx = None
        self.polyLeft = None
        #right 
        self.rightx = None
        self.polyRight = None

        #average radius
        self.radius = None
        #average offsets
        self.offset = None
        #average coefficients for best fits
    def updateRadius(self, newradius):
        radius_smooth_factor=self.radius_smooth_factor
        if (self.radius is None):
            self.radius = newradius
        else:
            self.radius = self.radius * (1-radius_smooth_factor)+ newradius * radius_smooth_factor

        return self.radius

    def updateOffset(self, newoffset):
        offset_smooth_factor = self.offset_smooth_factor
        if (self.offset is None):
            self.offset = newoffset
        else:
            self.offset = self.offset * (1-offset_smooth_factor)+ newoffset * offset_smooth_factor

        return self.offset
        
        
    def updateLeft(self, newleft):
        poly_smooth_factor = self.poly_smooth_factor
        if (self.polyLeft is None):
            self.polyLeft = newleft
        else:
            self.polyLeft = self.polyLeft * (1-poly_smooth_factor)+ newleft * poly_smooth_factor

        return self.polyLeft

    def updateRight(self, newright):
        poly_smooth_factor = self.poly_smooth_factor
        if (self.polyRight is None):
            self.polyRight = newright
        else:
            self.polyRight = self.polyRight * (1-poly_smooth_factor)+ newright * poly_smooth_factor

        return self.polyRight



def calibration():
    objpoints_set=[]
    imgpoints_set=[]
    objpoints = np.zeros([9*6,3],np.float32)
    objpoints[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    
    for i in range(1,21,1):
        filename = "reference/camera_cal/calibration"+str(i)+".jpg"
        image = cv2.imread(filename)
        ret, corners = cv2.findChessboardCorners(image, (9,6))
        if ret:
            #imgpoints_set.append(np.squeeze(corners))
            imgpoints_set.append(corners)
            objpoints_set.append(objpoints)

    #ret, mtx, dist, rvecs,tvecs = cv2.calibrateCamera(objpoints_set,imgpoints_set,image.shape[-2:-4:-1],None, None)
    ret, mtx, dist, rvecs,tvecs = cv2.calibrateCamera(objpoints_set,imgpoints_set,image.shape[0:2],None, None)
            
    return mtx, dist

def show_undistorted():
    mtx, dist = calibration()
    img = cv2.imread("reference/camera_cal/calibration1.jpg")
    dst = cv2.undistort(img,mtx,dist)
    plt.imshow(dst)
    plt.show()
    return None


#Below are functions I wrote during development, many are not used,
#but I keep them here as a record of the different methods I tried

#these functions return a probability mask, each pixel is a float
# that describe the likeliness that this pixel is a part of a lane marking

#direction
def dir_threshold(img, sobel_kernel=15, thresh=(0.6, 1.3)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output


def _sobel(img, dir, sobel_kernel=15):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if dir=='x':
        # Calculate the x gradients
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        # Calculate the y gradients
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    sobel = np.absolute(sobel)
    sobel = sobel/np.max(sobel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    return sobel

def sobelx_thres(img, sobel_kernel=15, thresh = (0.15,1.0)):

    binary_output =  np.zeros(img.shape[0:2])
    sobel = _sobel(img, 'x')
    binary_output[(sobel >= thresh[0]) & (sobel <= thresh[1])] = 1
    return binary_output

def dir_float(img, sobel_kernel=15, thresh=(0.6, 1.3)):
    mid = np.mean(thresh)
    diff = mid - thresh[0]
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    float_output =  np.zeros_like(absgraddir)
    float_output = 1- np.abs(absgraddir-mid)/diff
    float_output = float_output**3
    float_output[(absgraddir <= thresh[0]) | (absgraddir >= thresh[1])] = 0
    return float_output

#magnitude
def mag_float(img, sobel_kernel=13):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    mag = np.sqrt(sobelx**2+sobely**2)
    mag = mag/np.max(mag)
    return mag

def mag_threshold(img, thresh, sobel_kernel=13):
    image_mag = mag_float(img,sobel_kernel=sobel_kernel)
    
    binary_output =  np.zeros([img.shape[0],img.shape[1]])
    binary_output[(image_mag >= thresh[0]) & (image_mag <= thresh[1])] = 1
    return binary_output

#reduce color samples
def dec_color(img, level):
    level = int(256/level)
    return img//level*level


def visual_subsample():
    image = mpimg.imread(r'reference/test_images/test3.jpg')
    #400,660
    image = image[400:660,:,:]
    f, axarr = plt.subplots(2,2)
    axarr[0,0].set_title("5")
    axarr[0,0].imshow(dec_color(image,5))

    axarr[1,0].set_title("10")
    axarr[1,0].imshow(dec_color(image,10))

    axarr[0,1].set_title("30")
    axarr[0,1].imshow(dec_color(image,30))

    axarr[1,1].set_title("original")
    axarr[1,1].imshow(image)
    plt.show()
    return None


def visual_hls(img):
    original=img
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    f, axarr = plt.subplots(2,2)
    
    axarr[0,0].set_title("H")
    axarr[0,0].imshow(img[:,:,0], cmap='gray')

    axarr[1,0].set_title("L")
    axarr[1,0].imshow(img[:,:,1], cmap='gray')

    axarr[0,1].set_title("S")
    axarr[0,1].imshow(img[:,:,2], cmap='gray')

    axarr[1,1].set_title("original")
    axarr[1,1].imshow(original)
    plt.show()
    return None

def avg_h(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    img[:,:,0]=np.mean(img[:,:,0])
    
    img = cv2.cvtColor(img, cv2.COLOR_HLS2RGB)
    return img

def visual_mis(image):

    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    image_mag = mag_float(image)

    f, axarr = plt.subplots(1,3)
    axarr[0].set_title("S")
    axarr[0].imshow(image_hls[:,:,2],cmap='gray')

    axarr[1].set_title("direction")
    image_dir = dec_color(image, 10)
    image_dir = dir_threshold(image_dir)
    axarr[1].imshow(image_dir,cmap='gray')

    axarr[2].set_title("magnitute")
    axarr[2].imshow(image_mag,cmap='gray')

    result = np.add(image_dir, image_mag)
    plt.imshow(image_mag,cmap='gray')
    plt.show()

    return None

def s_thres(img,thresh=[100,255]):
    s_channel = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = s_channel[:,:,2]

    binary_output =  np.zeros_like(s_channel)
    binary_output[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

def warp(image):

    warped = cv2.warpPerspective(image, M, (image.shape[1],image.shape[0]), flags=cv2.INTER_LINEAR)
    return warped

def show(img1,img2=None,img3=None,img4=None,title1=None,title2=None,title3=None,title4=None,):

    f, axarr = plt.subplots(2,2)
    
    if (title1 is not None):
        axarr[0,0].set_title(title1)
    axarr[0,0].imshow(img1, cmap='gray')

    if (img2 is not None):
        if(title2 is not None):
            axarr[1,0].set_title(title2)
        axarr[1,0].imshow(img2, cmap='gray')

    if (img3 is not None):
        if(title3 is not None):
            axarr[0,1].set_title(title3)
        axarr[0,1].imshow(img3, cmap='gray')

    if (img4 is not None):
        if(title4 is not None):
            axarr[1,1].set_title(title4)
        axarr[1,1].imshow(img4, cmap='gray')

    plt.show()
    return None
# MAIN PROGRSM
input_filename = 'reference/project_video.mp4'

mtx, dist = calibration()
M = cv2.getPerspectiveTransform(src_points, dst_points)
Minv = cv2.getPerspectiveTransform(dst_points,src_points)

cap = cv2.VideoCapture(input_filename)
size = (1280,720)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out_vid = cv2.VideoWriter('output_mag.avi',fourcc,20.0,size)

line = Line()

while(cap.isOpened()):
    ret, frame = cap.read()
    if (ret==True):

        original_image=frame

        original_image = cv2.undistort(original_image, mtx, dist, None, mtx)
        undist = original_image

        #L channel
        img = cv2.cvtColor(original_image, cv2.COLOR_BGR2HLS)
        l = img[:,:,1]

        #B channel
        b_channel = cv2.cvtColor(original_image, cv2.COLOR_BGR2LAB)
        b_channel = b_channel[:,:,2]
        b_max = np.max(b_channel)
        b_min = np.min(b_channel)
        b_channel = 255*((b_channel-b_min)/b_max)
        b_channel = b_channel.astype(np.uint8)

        #combine
        lnb = np.zeros_like(b_channel)
        lnb[(b_channel>70) | (l>200)]=255
        binary_warped = warp(lnb)

        #reuse previous x locations for line base
        if (line.leftIsValid & line.rightIsValid):
            leftx_base = line.leftx
            rightx_base = line.rightx
        elif (line.leftIsValid and (not line.rightIsValid)):
            #only do left
            midpoint = size[0]//2
            histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,midpoint:], axis=0)
            rightx_base = np.argmax(histogram) + midpoint
            leftx_base = line.leftx

        elif (line.rightIsValid and (not line.leftIsValid)):
            #only do right
            midpoint = size[0]//2
            histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:midpoint], axis=0)
            leftx_base = np.argmax(histogram)
            rightx_base = line.rightx

        else:
            # Take a histogram of the bottom half of the image
            histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
            # Find the peak of the left and right halves of the histogram
            # These will be the starting point for the left and right lines
            midpoint = np.int(histogram.shape[0]/2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            #cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            #cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        if left_lane_inds.size>0:
            # Extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds] 
            with warnings.catch_warnings(record=True) as w:
                left_fit = np.polyfit(lefty, leftx, 2)
                if len(w)>0:
                    left_fit = line.polyLeft
                    line.leftIsValid = False
                else:
                    left_fit = line.updateLeft(left_fit)
                    line.leftIsValid = True
                    line.leftx=leftx[0]
        else:
            left_fit = line.polyLeft
            line.leftIsValid = False
        
        if right_lane_inds.size>0:
            # Extract left and right line pixel positions
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds] 
            with warnings.catch_warnings(record=True) as w:
                # Fit a second order polynomial to each
                right_fit = np.polyfit(righty, rightx, 2)
                if len(w)>0:
                    right_fit = line.polyRight
                    line.rightIsValid = False
                else:
                    right_fit = line.updateRight(right_fit)
                    line.rightIsValid = True
                    line.rightx = rightx[0]
        else:
            right_fit = line.polyRight
            line.rightIsValid = False

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        #out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        #out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        warped=binary_warped

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, size) 
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

        # Define y-value where we want radius of curvature
        y_eval = np.max(ploty)
        left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
        right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
        avg = np.mean([left_curverad,right_curverad])
        avg = avg/2
        avg = line.updateRadius(avg)
        avg = int(avg)
        if (avg>5000):
            text = "Straight"
        else:
            text = 'radius= '+str(avg)+'m'
            

        offset = (rightx[0]+leftx[0])/2-640
        offset = offset/80
        offset = line.updateOffset(offset)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text2 = 'offset= '+"{:.2f}".format(offset)+'m'
        cv2.putText(result,text,(300,100), font, 2.5,(0,0,0),3)
        cv2.putText(result,text2,(300,200), font, 2.5,(0,0,0),3)

        out_vid.write(result)
    else:
        break

cap.release()
out_vid.release()
