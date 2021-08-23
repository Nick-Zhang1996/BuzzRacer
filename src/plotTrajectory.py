import sys
from common import *
import pickle
import matplotlib.pyplot as plt
from TrackFactory import TrackFactory
import cv2

def plotTraj(filename, img, color):
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

filename = "../log/kinematics_results/full_state8.p"
img = plotTraj(filename, img, (255,0,0))

filename = "../log/kinematics_results/full_state9.p"
img = plotTraj(filename, img, (0,255,0))

filename = "../log/kinematics_results/full_state10.p"
img = plotTraj(filename, img, (0,0,255))

track = TrackFactory(name='full')
grayness = 180
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.show()
