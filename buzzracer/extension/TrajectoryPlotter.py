# draw trajectory
import numpy as np
from common import *
from extension.Extension import Extension

import cv2
import os.path


class TrajectoryPlotter(Extension):
    def __init__(self):
        Extension.__init__(self, 'trajectory_plotter')
        print_ok(self.prefix()+' in use')

    def init(self):

        self.resolve_logname()

    def prepare_gif(self):
        pass

    def resolve_logname(self,):
        # setup log file
        # log file will record state of the vehicle for later analysis
        logFolder = '../gifs/'
        logPrefix = 'sim'
        logSuffix = '.gif'
        no = 1
        while os.path.isfile(logFolder+logPrefix+str(no)+logSuffix):
            no += 1

        self.log_no = no
        self.logFilename = logFolder+logPrefix+str(no)+logSuffix

    def update(self):
        self.gifimages.append(Image.fromarray(cv2.cvtColor(
            self.main.visualization.visualization_img.copy(), cv2.COLOR_BGR2RGB)))

    def final(self):
        print_ok('[Gifsaver]: saving gif.. This may take a while')
        gif_filename = '../gifs/sim'+str(self.log_no)+'.gif'
        # TODO better way of determining duration
        self.gifimages[0].save(fp=gif_filename, format='GIF',
                               append_images=self.gifimages, save_all=True, duration=30, loop=0)
        print_ok('[Gifsaver]: gif saved at '+gif_filename)
