import matplotlib.pyplot as plt
from calibration import imageutil
cam = imageutil('../calibrated/')

pic=plt.imread('../perspectiveCali/mid2.png')
img = cam.undistort(pic)
plt.imshow(img)
plt.show()
