# for a single image, perform camera calibration

import glob
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# prepare object points
dim = (9,7)
objp = np.zeros((dim[1]*dim[0],3), np.float32)
objp[:,:2] = np.mgrid[0:dim[0],0:dim[1]].T.reshape(-1,2)
#objp = objp*0.021
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#image_filename = './calibration/IMG_7027.JPG'
#img = cv.imread(image_filename)
image_filename_vec = glob.glob('./calibration/*.JPG')

print(f'reading images...')
for fname in image_filename_vec:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, dim, None)
    # If found, add object points, image points (after refining them)
    objpoints = []
    imgpoints = []
    if ret == True:
        objpoints.append(objp)
        # refind corners
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, dim, corners2, ret)

        # show images
        #plt_img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        #plt.imshow(plt_img)
        #plt.show()

# Calibration
print(f'calibrating...')
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(f'residual: {ret}')
print(f'mtx: {mtx}')

# check reprojection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )

src = np.array([[4.5,3.5,0]])
dst, _ = cv.projectPoints(src, rvecs[0], tvecs[0], mtx, dist)
dst = tuple(dst.flatten().astype(int))

img = cv.imread(image_filename_vec[0])
cv.circle(img,dst, 5, (0,0,255), -1)

plt_img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
plt.imshow(plt_img)
plt.show()


