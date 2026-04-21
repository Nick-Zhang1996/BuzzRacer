''' Extension to save a gif image '''
import os.path

import cv2
from PIL import Image

from buzzracer.common import BASEDIR, ExperimentType, get_logger, LoggingFilter
from buzzracer.extensions.extension import Extension, ExtensionConfig, ExtensionState

logger = get_logger(__name__)
logger.addFilter(LoggingFilter(interval=5.0))


class GifsaverConfig(ExtensionConfig):
    def __init__(self, main_config):
        super().__init__(main_config)
        self.log_no, self.gif_filename = Gifsaver.resolve_gifname(main_config)
        self.gif_fps: float = 1000.0 / 30.0
        ''' Output GIF playback rate in frames per simulated second. '''


@Extension.register('gif_saver', GifsaverConfig, ExtensionState)
class Gifsaver(Extension):
    ''' Extension to save the experiment visualization as a gif image '''

    def __init__(self, config, state):
        super().__init__(config, state)
        self.count = 0
        ''' Number of frames currently'''
        self.gifimages: list[Image.Image] = []
        ''' Frames of visualizations '''
        self.capture_dt = 1.0 / self.config.gif_fps
        self.next_capture_sim_t = 0.0
        self.pending_capture = False

    def init(self):
        # Fallback first frame in case the visualization backend is not ready yet.
        self.gifimages.append(
            Image.fromarray(cv2.cvtColor(self.main.visualization.img_track.copy(), cv2.COLOR_BGR2RGB)))

    def _request_frame_if_supported(self):
        """Ask async visualization backends for a frame only when needed."""
        visualization = self.main.visualization
        if hasattr(visualization, 'new_frame'):
            visualization.new_frame.set()

    def _try_capture_frame(self) -> bool:
        frame = self.main.visualization.get_current_frame()
        if frame is None:
            return False
        self.gifimages.append(frame)
        return True

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
        return

    def post_update(self):
        self.count += 1
        sim_t = self.main.simulator.sim_t

        if self.pending_capture:
            if self._try_capture_frame():
                self.pending_capture = False
                while self.next_capture_sim_t <= sim_t + 1e-12:
                    self.next_capture_sim_t += self.capture_dt
            return

        if sim_t + 1e-12 < self.next_capture_sim_t:
            return

        self._request_frame_if_supported()
        if self._try_capture_frame():
            while self.next_capture_sim_t <= sim_t + 1e-12:
                self.next_capture_sim_t += self.capture_dt
        else:
            self.pending_capture = True

    def final(self):
        self.print_ok(self.prefix() + 'saving final frame')

        self.print_ok('saving gif.. This may take a while')
        if self.pending_capture:
            self._request_frame_if_supported()
            if self._try_capture_frame():
                self.pending_capture = False

        if len(self.gifimages) == 0:
            logger.warning('No GIF frames captured, nothing saved')
            return

        duration_ms = max(1, int(round(1000.0 / self.config.gif_fps)))
        self.gifimages[0].save(fp=self.config.gif_filename,
                               format='GIF',
                               append_images=self.gifimages[1:],
                               save_all=True,
                               duration=duration_ms,
                               loop=0)
        self.print_ok('gif saved at ' + self.config.gif_filename)
