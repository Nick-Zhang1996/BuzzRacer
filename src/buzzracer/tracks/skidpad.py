"""Circular skidpad track."""

import numpy as np

from buzzracer.tracks.curvilinear_track import CurvilinearTrack


class Skidpad(CurvilinearTrack):
    """A circular curvilinear track with fixed-width boundaries."""

    def __init__(self, config):
        super().__init__(config)

        self.radius = getattr(config, 'radius', 2.0)
        self.width = getattr(config, 'width', 0.5)
        self.ccw = True

        theta = np.linspace(
            0.0,
            2.0 * np.pi,
            config.discretized_raceline_len,
            endpoint=False,
        )
        centerline = np.column_stack(
            (self.radius * np.cos(theta), self.radius * np.sin(theta))
        )
        half_width = np.full(theta.shape, self.width / 2.0)
        self.data = self.build_track(centerline, half_width, half_width)

        self.raceline_s = self.data.raceline_s
        self.raceline_len_m = self.data.raceline_len_m
        self.start_pos = self.data.start_pos
        self.start_dir = self.data.start_dir
        config.x_limit = self.data.x_max - self.data.x_min
        config.y_limit = self.data.y_max - self.data.y_min
