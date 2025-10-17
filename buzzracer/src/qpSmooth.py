import cv2
import numpy as np

from common import *
import argparse
from track.TrackFactory import TrackFactory
import matplotlib.pyplot as plt


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('track_name', choices = TrackFactory.availableTrackNames())
    args = parser.parse_args()

    # on my desktop computer there's a conflict between plt and cv2, likely due to gtk version conflict
    # force initialize here
    plt.plot([1,2])
    plt.show()
    #cv2.imshow('dummy',np.zeros((10,10,3),dtype=np.uint8))
    #cv2.waitKey(100)
    #cv2.destroyAllWindows()

    # optimize and save
    track = TrackFactory.build(args.track_name)


    track.optimizePath(offset=0.15)
    track.save()

    # verify results: load and show
    #load_track = TrackFactory.build(args.track_name, track=track)
    load_track = TrackFactory.build('saved')
    print("-----------------")
    print_info("testing loading")
    load_track = load_track.load()
    img_track = load_track.drawTrack()
    img_track = load_track.drawRaceline(img=img_track)
    img_track = cv2.cvtColor(img_track,cv2.COLOR_BGR2RGB)
    plt.imshow(img_track)
    plt.show()
