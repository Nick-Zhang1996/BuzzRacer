import numpy as np
import cv2
import matplotlib
matplotlib.use('gtk4agg')
import matplotlib.pyplot as plt


img = np.zeros((100,100,3),dtype=np.uint8)
img[:,:,2] = 200
#cv2.imshow('dummy',img)
#cv2.waitKey(1000)
#cv2.destroyAllWindows()

plt.plot([1,2,3])
plt.show()

plt.imshow(img)
plt.show()

cv2.imshow('dummy',img)
cv2.waitKey(1000)
cv2.destroyAllWindows()

plt.imshow(img)
plt.show()
