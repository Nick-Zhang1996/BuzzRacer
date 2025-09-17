''' Extension to save a gif image '''
import os.path

import cv2
from PIL import Image

from buzzracer.common import BASEDIR
from buzzracer.extensions.extension import Extension


class Gifsaver(Extension):
    ''' Extension to save the experiment visualization as a gif image '''

    def __init__(self):
        Extension.__init__(self, 'gif_saver')
        self.count = 0
        ''' Number of frames currently'''
        self.gifimages: list[Image.Image] = []
        ''' Frames of visualizations '''

        self.log_no = None
        self.gif_filename = None
        self.resolve_gif_filename()

    def init(self):
        # prepare save gif, this provides an easy to use visualization for presentation
        self.prepare_gif()

    def prepare_gif(self):
        self.gifimages.append(
            Image.fromarray(
                cv2.cvtColor(self.main.visualization.img_track.copy(),
                             cv2.COLOR_BGR2RGB)))

    def resolve_gif_filename(self, ):
        # setup log file
        # log file will record state of the vehicle for later analysis
        log_folder = os.path.join(BASEDIR, 'gifs')
        log_prefix = 'test'
        log_suffix = '.gif'
        no = 1
        while os.path.isfile(log_folder + log_prefix + str(no) + log_suffix):
            no += 1

        self.log_no = no
        self.gif_filename = log_folder + log_prefix + str(no) + log_suffix

    def update(self):
        self.gifimages.append(
            Image.fromarray(
                cv2.cvtColor(self.main.visualization.visualization_img.copy(),
                             cv2.COLOR_BGR2RGB)))

    def post_update(self):
        self.count += 1
        # save first and second rendering image
        # if (self.count == 1):
        #     img = self.main.visualization.visualization_img.copy()
        #     filename = "./first_frame_" + self.main.algorithm + ".png"
        #     cv2.imwrite(filename,img)
        #     self.print_info(self.prefix()+"saved first frame at "+filename)
        #     plt.imshow(img)
        #     plt.show()
        # if (self.count == 1):
        #     img = self.main.visualization.visualization_img.copy()
        #     filename = "./Qfstudy/ccmppi_Qf_" + str(self.main.params['Qf']) + ".png"
        #     cv2.imwrite(filename,img)
        #     self.print_info(self.prefix()+"saved frame at "+filename)
        #     self.main.exit_request.set()

        # if (self.count == 2):
        #     img = self.main.visualization.visualization_img.copy()
        #     filename = "./second_frame_" + self.main.algorithm + ".png"
        #     cv2.imwrite(filename,img)
        #     self.print_info(self.prefix()+"saved second frame at "+filename)

    def final(self):
        self.print_ok(self.prefix() + 'saving final frame')

        self.print_ok('saving gif.. This may take a while')
        gif_filename = os.path.join(BASEDIR, 'gifs', f'run{self.log_no}.gif')
        self.gifimages[0].save(fp=gif_filename,
                               format='GIF',
                               append_images=self.gifimages,
                               save_all=True,
                               duration=30,
                               loop=0)
        self.print_ok('gif saved at ' + gif_filename)
