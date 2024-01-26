import cv2
from common import *
import argparse
from track.TrackFactory import TrackFactory
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('track_name', choices = TrackFactory.availableTrackNames())
    args = parser.parse_args()

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
