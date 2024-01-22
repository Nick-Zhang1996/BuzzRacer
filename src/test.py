# test curvilinear qpSmooth
from track.TrackFactory import TrackFactory
import matplotlib.pyplot as plt

track = TrackFactory.build('nascar')
track.optimizePath(offset=0.1)
track.save()

track = TrackFactory.build('nascar')
track = track.load()

img_track = track.drawTrack()
img_track = track.drawRaceline(img=img_track)
img_track_rgb = cv2.cvtColor(img_track.copy(),cv2.COLOR_BGR2RGB)
plt.imshow(img_track_rgb)
plt.show()
