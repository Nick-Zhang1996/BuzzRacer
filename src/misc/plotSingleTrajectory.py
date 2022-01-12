import sys
from common import *
import pickle
import matplotlib.pyplot as plt
from TrackFactory import TrackFactory
import cv2

def plotTraj(track, filename, img, color):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    data = np.array(data).squeeze(1)

    x = data[:,1]
    y = data[:,2]
    points = np.vstack([x,y]).T
    track.drawPolyline(points, img, lineColor=color,thickness=2)
    return img


# load blank canvas
with open("track_img.p", 'rb') as f:
    img = pickle.load(f)


track = TrackFactory(name='full')
grayness = 180

filename = "../log/kinematics_results/full_state11.p"
img = plotTraj(track,filename, img.copy(), (255,0,0))

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.show()
