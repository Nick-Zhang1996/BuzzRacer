import cv2
import sys
import os
import pickle
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from common import *
from track.TrackFactory import TrackFactory

def plotTraj(track, filename, img, color, text):
    global offset
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    data = np.array(data).squeeze(1)

    x = data[:,1]
    y = data[:,2]
    points = np.vstack([x,y]).T
    track.drawPolyline(points, img, lineColor=color,thickness=2)
    return img

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org = (20, 50+offset)
    # fontScale
    fontScale = 1
    # Line thickness of 2 px
    thickness = 2
    # Using cv2.putText() method
    img = cv2.putText(img, text, org, font,
                       fontScale, color, thickness, cv2.LINE_AA)
    offset += 30

    return img


# load blank canvas
with open("track_img.p", 'rb') as f:
    img = pickle.load(f)


track = TrackFactory(name='full')

offset = 0

filename = '../log/2022_8_25_sim/full_state1.p'
img1 = plotTraj(track,filename, img.copy(), (0,0,255), "CVaR")

filename = '../log/2022_8_25_sim/full_state2.p'
img2 = plotTraj(track,filename, img.copy(), (0,255,0), "Baseline")


# combine img together
#img = (img1/2 + img2/2)
#img = np.minimum(img1,img2)
#img = np.array(img,dtype=np.uint8)

# side by side image
img = np.hstack([img1,img2])




img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
fig = plt.figure()
plt.imshow(img)
fig.savefig('out.png', bbox_inches='tight',transparent=True, pad_inches=0)
plt.show()
