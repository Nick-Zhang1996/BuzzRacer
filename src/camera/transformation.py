# for a single image, perform camera calibration
import glob
import pickle
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Load Calibration
with open('camera.p','rb') as f:
    ret, mtx, dist, rvecs, tvecs = pickle.load(f)
print(f'residual: {ret}')
print(f'mtx: {mtx}')

objpoints = [[3,1,0],[7,1,0],[5,7,0],[1,11,0],[7,11,0]]
imgpoints = [[1551,618],[1438,279],[914,366],[267,850],[462,131]]


objpoints = np.array(objpoints, dtype=float)
imgpoints = np.array(imgpoints, dtype=float)

# load track
img = cv.imread('resources/track.png')
objpoints = [objpoints[i] for i in (0,1,3,4)]
imgpoints = [imgpoints[i] for i in (0,1,3,4)]

objpoints = np.array(objpoints, dtype=float)
imgpoints = np.array(imgpoints, dtype=float)

ret, rvec, tvec = cv.solvePnP(objpoints, imgpoints, mtx, dist, flags=cv.SOLVEPNP_P3P)
print(f'PnP residual {ret}')

# test transformation
for objpoint in objpoints:
    src = np.array([objpoint],dtype=float)
    dst, _ = cv.projectPoints(src, rvec, tvec, mtx, dist)
    dst = tuple(dst.flatten().astype(int))
    cv.circle(img,dst, 15, (0,0,255), -1)
    print(dst)

plt_img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
plt.imshow(plt_img)
plt.show()


