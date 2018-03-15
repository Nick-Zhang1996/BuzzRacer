import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import warnings

x_size = 640
y_size = 480

def showg(img):
    plt.imshow(img,cmap='gray')
    plt.show()
    return

# takes a grayscale frame, return a centerline
def pipeline(frame):
    frame = frame[220:,:]
    frame = frame.astype(np.float32)
    frame = frame/255
    mean = np.mean(frame)
    stddev = np.std(frame)
    frame = (frame-mean)/stddev
    showg(frame)
    frame[frame>1]=1
    frame[frame<1]=0
    showg(frame)
    
    
def testimg(filename):
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pipeline(image)
    return
    

if __name__ == '__main__':
    print('begin')
    testpics =['../img/0.png','../img/1.png','../img/2.png','../img/3.png','../img/4.png','../img/5.png','../img/6.png','../img/7.png'] 
    for i in range(8):
        testimg(testpics[i])

