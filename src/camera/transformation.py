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

# test transformation
image_filename_vec = glob.glob('./calibration/*.JPG')

src = np.array([[4.5,3.5,0]])
dst, _ = cv.projectPoints(src, rvecs[0], tvecs[0], mtx, dist)
dst = tuple(dst.flatten().astype(int))

img = cv.imread(image_filename_vec[0])
cv.circle(img,dst, 5, (0,0,255), -1)

plt_img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
plt.imshow(plt_img)
plt.show()


