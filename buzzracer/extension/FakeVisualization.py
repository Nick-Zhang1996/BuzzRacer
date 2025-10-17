from threading import Event
from extension.Extension import Extension

class FakeVisualization(Extension):
    def __init__(self,main):
        super().__init__(main)
        self.update_visualization = Event()
        self.update_freq = 100
        self.frame_dt = 1.0/self.update_freq
        # NOTE
        self.frame_dt = 0.0
        self.count = 0
        # default setting, will be overridden if defined in config
        self.car_graphics = False
        self.track = self.main.track
        self.main.breakpoint = Event()
