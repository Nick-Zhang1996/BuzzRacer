# This file creates the mapping equation from camera coordinates to car coordinate.
# It accomplishes this by analyzing a special calibration image.And coorelate features
# in image with their known real world coordinates
# Author: Nick Zhang

import cv2
import numpy as np
import matplotlib.pyplot as plt


filename = '../perspective.png'
#filename = '../img/perspectiveCalibration23cm.png'

# all real world coordinates are in cm

# 'vertical' wise offset from center of rear axle(which is the origin of car frame)

# offset from front axle 
zeroOffset = 12.8
wheelbase = 9.8

# size of interal corners, (row,column)
patternSize = (5,5)
# side length of a chessboard unit:cm
gridSize = 2.0

# calculate  coordinate of the right, bottom corner
baseOffset = np.array([(patternSize[0]-1.0)/2.0*gridSize,zeroOffset+wheelbase+gridSize])

image = cv2.imread(filename)
image = cv2.resize(image,(640,480))
if (image is None):
    print('no such file')
    exit()

#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);
blur = cv2.GaussianBlur(image, (0, 0), 3);
image = cv2.addWeighted(image, 3, blur, -2, 0);
avg = np.average(image)


# the corners given are from right to left, then bottom to top
retval, cameraFrameCorners = cv2.findChessboardCorners(image, patternSize, flags = cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE)
if (cameraFrameCorners.shape[0] != patternSize[0]*patternSize[1]):
    print('error: expected ' + str(patternSize[0]*patternSize[1]) + 'corners, ' + str(cameraFrameCorners.shape[0]) + 'found')

    # if you wannt debug comment the following line out
    exit(1)


# DEBUG: label all corners found
for i in range(cameraFrameCorners.shape[0]):
    cv2.circle(image,tuple(cameraFrameCorners[i,0]), 3, (0,0,255), -1)

cv2.imshow('image',image)
cv2.waitKey(0) 
cv2.destroyAllWindows()


carFrameCorners = np.zeros_like(cameraFrameCorners)
index = 0
# construct the corresponding coordinates for real world frame
for j in range(patternSize[1]):
    for i in range(patternSize[0]):
        carFrameCorners[index, 0, 0] = -i*gridSize
        carFrameCorners[index, 0, 1] = j*gridSize
        index += 1

carFrameCorners += baseOffset
xFrom = cameraFrameCorners[:,0,0]
yFrom = cameraFrameCorners[:,0,1]

xTo = carFrameCorners[:,0,0]
yTo = carFrameCorners[:,0,1]


# TODO: find f -> y_f = f(x,y), using linear algebra 

# for now we use a simplified linear fit on only one input
# y_f = f(y) = a*y + b
fitx = np.polyfit( xFrom, xTo, 1)
fity = np.polyfit( yFrom, yTo, 1)


print('------reference: linear fit--------')
print('fitx = '+str(fitx[0])+' *x + '+str(fitx[1]))
print('fity = '+str(fity[0])+' *y + '+str(fity[1]))

xGuess = np.polyval(fitx,cameraFrameCorners[:,0,0])
yGuess = np.polyval(fity,cameraFrameCorners[:,0,1])

# mse = sum( sqrt( dx^2 + dy^2))

dx = xGuess - carFrameCorners[:,0,0]
dy = yGuess - carFrameCorners[:,0,1]

# root of mse
rmse = np.average( (dx**2 + dy**2) ) **0.5
errx = np.average(np.abs(dx))
erry = np.average(np.abs(dy))
print('rmse = %.2f, errx = %.2f, erry = %.2f'%(rmse, errx, erry))


print('\n\n------ Orthogonal Projection - lstsq--------')

# y = ax2+bx+cy2+dy+e, find least square solution to the linear algebra problem

x = cameraFrameCorners[:,0,0]
y = cameraFrameCorners[:,0,1]


# B = A P
A = np.vstack([x**2, x, y**2, y, np.ones_like(y)]).T
B = carFrameCorners.reshape(-1,2)

P, residuals, _, _ = np.linalg.lstsq(A,B)

print(P)
print(' individual average  Euclidean 2-norm, errx = %.2f, erry = %.2f' % ((residuals[0]/len(x))**0.5,(residuals[1]/len(y))**0.5))

print(' average Euclidean 2-norm ' + str((np.sum(residuals)/len(x))**0.5))

print('\n\n------ Homography - Opencv -------')
m, mask = cv2.findHomography(srcPoints = cameraFrameCorners.reshape(1,-1,2), dstPoints = carFrameCorners.reshape(1,-1,2))

print(m)

result = cv2.perspectiveTransform(cameraFrameCorners, m)
xGuess = result[:,0,0]
yGuess = result[:,0,1]

dx = xGuess - carFrameCorners[:,0,0]
dy = yGuess - carFrameCorners[:,0,1]

# root of mse
rmse = np.average( (dx**2 + dy**2) ) **0.5
errx = np.average(np.abs(dx))
erry = np.average(np.abs(dy))
print('rmse = %.2f, errx = %.2f, erry = %.2f'%(rmse, errx, erry))

