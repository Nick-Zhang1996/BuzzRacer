''' Extension for saving consecutive multiple snapshots of car as multiple exposure photo'''
import os
from threading import Event

import cv2
import matplotlib.pyplot as plt

from buzzracer.common import BASEDIR
from buzzracer.extension.Extension import Extension

class SnapshotSaver(Extension):
    ''' Extension for saving consecutive multiple snapshots of car as multiple exposure photo'''
    def __init__(self):
        Extension.__init__(self, 'snapshot_saver')
        self.recording = Event()
        ''' Currently taking snapshot'''

        self.img = None
        self.timestep = 0
        ''' Elapsed time steps while taking snapshot, for counting keyframe'''
        self.snapshot_count = 0
        ''' Current number of shutter opening during this snapshot '''
        self.interval = 20
        ''' Number of frames between snapshots '''
        self.background = None
        ''' Background of track. '''

    def post_init(self):
        # need to wait for Visualization to complete
        self.background = self.main.track.draw_track()
        self.background = self.main.track.draw_raceline(img=self.background)


    def toggle_snapshot(self):
        ''' Start/Stop snapshot.
        
        First time called this will start snapshot, second time will stop
        Called in Main when user press 's'
        '''
        if self.recording.is_set():
            self.recording.clear()
            self.print_info('snapshot stopping')
        else:
            self.recording.set()
            self.timestep = 0
            self.print_info('snapshot started')

    def post_update(self):
        # save a multiple exposure photo
        if (self.recording.is_set() and self.timestep % self.interval == 0):
            if self.img is None:
                img = self.background.copy()
            else:
                img = self.img
            for car in self.main.cars:
                img = self.main.visualization.draw_car(img, car)
            self.print_info('snapshot taken frame %d' % (self.timestep))
            self.img = img

        if not self.recording.is_set():
            self.save_snapshot()
        self.timestep += 1

    def save_snapshot(self):
        if not self.img is not None:
            self.snapshot_count += 1
            filename = os.path.join(BASEDIR, 'snapshot{self.snapshot_count}.png')
            cv2.imwrite(filename, self.img)
            self.print_info('saved snapshot at '+filename)
            plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
            plt.show()
            self.img = None

    def final(self):
        self.save_snapshot()
