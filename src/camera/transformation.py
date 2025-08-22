# for a single image, perform camera calibration
import glob
import pickle
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

objpoints = [[3, 1], [7, 1], [5, 7], [1, 11], [7, 11]]
imgpoints = [[1551, 618], [1438, 279], [914, 366], [267, 850], [462, 131]]

objpoints = np.array(objpoints, dtype=np.float32)*0.3
imgpoints = np.array(imgpoints, dtype=np.float32)

# load track
img = cv.imread('resources/track.png')
mat = cv.getPerspectiveTransform(objpoints[:4], imgpoints[:4])
with open('resources/transform.p', 'wb') as f:
    pickle.dump(mat, f)

# test transformation
for objpoint in objpoints:
    src = np.array(objpoint.reshape(1, 1, -1), dtype=np.float32)
    dst = cv.perspectiveTransform(src, mat)
    # dst.shape = (1,1,2)
    dst = tuple(dst.flatten().astype(int))
    cv.circle(img, dst, 15, (0, 0, 255), -1)
    print(dst)

plt_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(plt_img)
plt.show()
