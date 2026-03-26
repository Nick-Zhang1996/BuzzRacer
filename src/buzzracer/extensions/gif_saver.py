''' Extension to save a gif image '''
import os.path

import cv2
from PIL import Image

from buzzracer.common import BASEDIR, ExperimentType, get_logger
from buzzracer.extensions.extension import Extension, ExtensionConfig, ExtensionState

logger = get_logger(__name__)


class GifsaverConfig(ExtensionConfig):
    def __init__(self, main_config):
        super().__init__(main_config)
        self.log_no, self.gif_filename = Gifsaver.resolve_gifname(main_config)


@Extension.register('gif_saver', GifsaverConfig, ExtensionState)
class Gifsaver(Extension):
    ''' Extension to save the experiment visualization as a gif image '''

    def __init__(self, config, state):
        super().__init__(config, state)
        self.count = 0
        ''' Number of frames currently'''
        self.gifimages: list[Image.Image] = []
        ''' Frames of visualizations '''

    def init(self):
        # prepare save gif, this provides an easy to use visualization for presentation
        self.main.visualization.save_frames.set()
        self.gifimages.append(
            Image.fromarray(cv2.cvtColor(self.main.visualization.img_track.copy(), cv2.COLOR_BGR2RGB)))

    @staticmethod
    def resolve_gifname(main_config):
        ''' Find log file name. '''
        # create a folder using date and type of experiment
        name_map = {ExperimentType.Simulation: 'sim',
                    ExperimentType.Realworld: 'exp'}
        prefix = name_map[main_config.experiment_type]

        log_folder = os.path.join(BASEDIR, 'outputs', 'gifs', main_config.experiment_name)

        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        log_suffix = '.gif'
        log_no = 1
        gif_filename = os.path.join(log_folder, prefix + str(log_no) + log_suffix)
        while os.path.isfile(gif_filename):
            log_no += 1
            gif_filename = os.path.join(log_folder, prefix + str(log_no) + log_suffix)
        return log_no, gif_filename

    def update(self):
        frame = self.main.visualization.get_current_frame()
        if frame is not None:
            self.gifimages.append(frame)
        else:
            logger.warning("Ignoring empty frame received")

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
        self.gifimages[0].save(fp=self.config.gif_filename,
                               format='GIF',
                               append_images=self.gifimages,
                               save_all=True,
                               duration=30,
                               loop=0)
        self.print_ok('gif saved at ' + self.config.gif_filename)
