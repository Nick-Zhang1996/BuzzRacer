import sys
from common import *
import pickle
import matplotlib.pyplot as plt
from TrackFactory import TrackFactory
import cv2

# load blank canvas
with open("track_img.p", 'rb') as f:
    img = pickle.load(f)

# load log
if (len(sys.argv) != 2):
    filename = "../log/kinematics_results/full_state10.p"
    print_warning("reading default file" + filename)
else:
    filename = sys.argv[1]

with open(filename, 'rb') as f:
    data = pickle.load(f)
data = np.array(data).squeeze(1)

x = data[:,1]
y = data[:,2]
points = np.vstack([x,y]).T

track = TrackFactory(name='full')
grayness = 180
track.drawPolyline(points, img, lineColor=(grayness,grayness,grayness),thickness=2)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.show()
