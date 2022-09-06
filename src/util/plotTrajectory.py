import cv2
import sys
import os
import pickle
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from common import *
from track.TrackFactory import TrackFactory
from xml.dom import minidom
import xml.etree.ElementTree as ET

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
    track_img = pickle.load(f)

with open("track_boundary_img.p", 'rb') as f:
    boundary_img = pickle.load(f)
    boundary_img[:,:,0] = boundary_img[:,:,1]
    boundary_img[:,:,2] = boundary_img[:,:,1]


# load config cvar.xml
config_filename = './configs/cvar.xml'
config = minidom.parse(config_filename)
config_track= config.getElementsByTagName('track')[0]
track = TrackFactory(config_track)

offset = 0

#blank = np.zeros_like(track_img)
filename = '../log/2022_8_30_sim/full_state2.p'
img2 = plotTraj(track,filename, track_img, (0,255,0), "Baseline")

filename = '../log/2022_8_30_sim/full_state1.p'
img1 = plotTraj(track,filename, track_img, (0,0,255), "CVaR")


#img1 = img1 - boundary_img
#img2 = img2 - boundary_img


# combine img together
#img = (img1/2 + img2/2)
#img = np.maximum(img1,img2)

'''
img = img1 + img2
mask = np.sum(img,axis=2) > 0
track_img[mask,:] = img[mask,:]
'''
img = track_img
img = np.array(img,dtype=np.uint8)

# side by side image
#img = np.hstack([img1,img2])

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
fig = plt.figure()
plt.imshow(img)
fig.savefig('out.png', bbox_inches='tight',transparent=True, pad_inches=0)
plt.show()
