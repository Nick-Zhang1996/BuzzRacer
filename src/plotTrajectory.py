import sys
from common import *
import pickle
import matplotlib.pyplot as plt
from TrackFactory import TrackFactory

# load blank canvas
with open("track_img.p", 'rb') as f:
    img = pickle.load(f)

# load log
if (len(sys.argv) != 2):
    filename = "../log/plot_traj/full_state1.p"
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

plt.imshow(img)
plt.show()
