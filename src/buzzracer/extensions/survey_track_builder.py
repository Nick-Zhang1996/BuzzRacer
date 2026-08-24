from __future__ import annotations

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

from buzzracer.common import BASEDIR, get_logger
from buzzracer.extensions.extension import Extension, ExtensionConfig, ExtensionState

logger = get_logger(__name__)


class SurveyTrackBuilderConfig(ExtensionConfig):
    def __init__(self, main_config):
        super().__init__(main_config)
        self.wand_optitrack_id = 1
        self.sample_spacing_m = 0.01
        self.wand_height_threshold_m = 0.05
        self.x_limits = (-1.0, 3.0)
        self.y_limits = (-1.0, 6.0)
        self.output_filename = os.path.join('outputs', 'mapped_track.p')


class SurveyTrackBuilderState(ExtensionState):
    def __init__(self, config):
        super().__init__(config)
        self.internal_id = None


@Extension.register('survey_track_builder', SurveyTrackBuilderConfig, SurveyTrackBuilderState)
class SurveyTrackBuilder(Extension):
    """ Build a track survey with an Optitrack wand.
    Describe the left boundary, right boundary, and reference path.
    The origin of the wand is where the boundary is"""

    curve_names = ('left_boundary', 'right_boundary', 'reference')
    curve_labels = {
        'left_boundary': 'left boundary',
        'right_boundary': 'right boundary',
        'reference': 'reference',
    }
    curve_colors = {
        'left_boundary': 'tab:orange',
        'right_boundary': 'tab:green',
        'reference': 'tab:purple',
    }

    def __init__(self, config, state):
        super().__init__(config, state)
        self.curves = {
            name: []
            for name in self.curve_names
        }
        self.current_curve_index = 0
        self.mapping_phase = 'waiting_to_start'
        self.current_position_xy = None
        self.current_position_is_low = False
        self.current_height = None
        self.figure = None
        self.axes = None
        self.curve_artists = {}
        self.current_position_artist = None
        self.status_text = None
        self.help_text = None
        self.buttons = {}
        self.ui_closed = False
        self.saved = False

    def post_init(self):
        self._refresh_internal_id()
        self._setup_ui()

    def update(self):
        if self.ui_closed:
            return

        if self.state.internal_id is None:
            self._refresh_internal_id()
            self._update_status_text('Waiting for wand to appear in OptiTrack.')
            self._draw_ui()
            return

        vi = Extension.main.vi.vi
        state = vi.get_state(self.state.internal_id)
        if state is None:
            return

        x, y, z, _, _, _ = state
        x_local, y_local, z_local = vi.rotation_world_to_track.apply([x, y, z])  # 0.02ms
        self.current_position_xy = [float(x_local), float(y_local)]
        self.current_height = float(z_local)
        self.current_position_is_low = self.current_height < self.config.wand_height_threshold_m

        if self.mapping_phase == 'recording':
            self._try_log_current_position()

        self._update_plot()
        self._update_status_text()
        self._draw_ui()

    def final(self):
        if self.figure is not None and not self.ui_closed:
            plt.close(self.figure)
            self.ui_closed = True

    def _refresh_internal_id(self):
        vi = Extension.main.vi.vi
        if hasattr(vi, 'internal_id_lookup'):
            self.state.internal_id = vi.internal_id_lookup.get(self.config.wand_optitrack_id)
        else:
            self.state.internal_id = vi.get_internal_id(self.config.wand_optitrack_id)

    def _setup_ui(self):
        plt.ion()
        self.figure, self.axes = plt.subplots(figsize=(8, 8))
        self.figure.canvas.mpl_connect('close_event', self._on_close)
        self.figure.subplots_adjust(bottom=0.28)
        self.axes.set_title('Survey Track Builder')
        self.axes.set_xlabel('x [m]')
        self.axes.set_ylabel('y [m]')
        self.axes.set_xlim(*self.config.x_limits)
        self.axes.set_ylim(*self.config.y_limits)
        self.axes.set_aspect('equal', adjustable='box')
        self.axes.grid(True, alpha=0.3)

        for name in self.curve_names:
            artist, = self.axes.plot(
                [], [],
                color=self.curve_colors[name],
                linewidth=2,
                label=self.curve_labels[name],
            )
            self.curve_artists[name] = artist

        self.current_position_artist, = self.axes.plot(
            [], [],
            marker='o',
            linestyle='None',
            markersize=8,
            markerfacecolor='white',
            markeredgecolor='white',
            markeredgewidth=1.5,
            label='wand',
        )
        self.axes.legend(loc='upper right')

        self.status_text = self.figure.text(0.05, 0.18, '', ha='left', va='bottom')
        self.help_text = self.figure.text(0.05, 0.10, '', ha='left', va='bottom')

        button_specs = [
            ('start', [0.05, 0.02, 0.12, 0.05], 'Start', self._on_start),
            ('stop', [0.19, 0.02, 0.12, 0.05], 'Stop', self._on_stop),
            ('remap', [0.33, 0.02, 0.12, 0.05], 'Remap', self._on_remap),
            ('next', [0.47, 0.02, 0.12, 0.05], 'Next', self._on_next),
            ('save_exit', [0.61, 0.02, 0.20, 0.05], 'Save & Exit', self._on_save_exit),
        ]
        for name, rect, label, callback in button_specs:
            button = Button(self.figure.add_axes(rect), label)
            button.on_clicked(callback)
            self.buttons[name] = button

        self._update_status_text()
        self._draw_ui()
        plt.show(block=False)

    @property
    def current_curve_name(self):
        if self.current_curve_index >= len(self.curve_names):
            return None
        return self.curve_names[self.current_curve_index]

    def _current_curve_points(self):
        current_name = self.current_curve_name
        if current_name is None:
            return []
        return self.curves[current_name]

    def _try_log_current_position(self):
        if not self.current_position_is_low or self.current_position_xy is None:
            return

        current_points = self._current_curve_points()
        if not current_points:
            current_points.append(self.current_position_xy.copy())
            return

        previous_point = np.asarray(current_points[-1], dtype=float)
        current_point = np.asarray(self.current_position_xy, dtype=float)
        if np.linalg.norm(current_point - previous_point) >= self.config.sample_spacing_m:
            current_points.append(self.current_position_xy.copy())

    def _update_plot(self):
        if self.axes is None or self.ui_closed:
            return

        for name, artist in self.curve_artists.items():
            points = self.curves[name]
            if points:
                data = np.asarray(points, dtype=float)
                artist.set_data(data[:, 0], data[:, 1])
            else:
                artist.set_data([], [])

        if self.current_position_xy is None:
            self.current_position_artist.set_data([], [])
        else:
            self.current_position_artist.set_data(
                [self.current_position_xy[0]],
                [self.current_position_xy[1]],
            )
            if self.current_position_is_low:
                self.current_position_artist.set_markerfacecolor('tab:red')
                self.current_position_artist.set_markeredgecolor('tab:red')
            else:
                self.current_position_artist.set_markerfacecolor('none')
                self.current_position_artist.set_markeredgecolor('white')

    def _update_status_text(self, message=None):
        if self.status_text is None or self.help_text is None:
            return

        current_name = self.current_curve_name
        if current_name is None:
            curve_line = 'All curves completed.'
        else:
            curve_label = self.curve_labels[current_name]
            point_count = len(self.curves[current_name])
            curve_line = (
                f'Current curve: {curve_label} | phase: {self.mapping_phase} '
                f'| logged points: {point_count}'
            )

        if self.current_position_xy is None or self.current_height is None:
            wand_line = 'Wand position: unavailable'
        else:
            wand_state = 'LOW' if self.current_position_is_low else 'HIGH'
            wand_line = (
                f'Wand: x={self.current_position_xy[0]:.3f}, '
                f'y={self.current_position_xy[1]:.3f}, '
                f'z={self.current_height:.3f} ({wand_state})'
            )

        self.status_text.set_text(f'{curve_line}\n{wand_line}')

        if message is not None:
            help_text = message
        elif current_name is None:
            help_text = 'Click Save & Exit to write outputs/mapped_track.p and quit.'
        elif self.mapping_phase == 'waiting_to_start':
            help_text = (
                f'Place wand at start of {self.curve_labels[current_name]} and click Start.'
            )
        elif self.mapping_phase == 'recording':
            help_text = (
                f'Recording {self.curve_labels[current_name]}. '
                'Only low wand positions are logged. Click Stop when done.'
            )
        else:
            if current_name == self.curve_names[-1]:
                help_text = (
                    f'Review {self.curve_labels[current_name]}. '
                    'Click Remap to redo, or Save & Exit to finish.'
                )
            else:
                help_text = (
                    f'Review {self.curve_labels[current_name]}. '
                    'Click Remap to redo, or Next to continue.'
                )

        self.help_text.set_text(help_text)

    def _draw_ui(self):
        if self.figure is None or self.ui_closed:
            return
        self.figure.canvas.draw_idle()
        plt.pause(0.001)

    def _on_start(self, _event):
        if self.current_curve_name is None:
            self._update_status_text('All curves are already mapped. Click Save & Exit.')
            return
        if self.mapping_phase != 'waiting_to_start':
            self._update_status_text('Start is only available before recording begins.')
            return

        self.mapping_phase = 'recording'
        logger.info('Started mapping %s', self.current_curve_name)
        self._update_status_text()

    def _on_stop(self, _event):
        if self.mapping_phase != 'recording':
            self._update_status_text('Stop is only available while recording.')
            return

        self.mapping_phase = 'review'
        logger.info(
            'Stopped mapping %s with %d points',
            self.current_curve_name,
            len(self._current_curve_points()),
        )
        self._update_status_text()

    def _on_remap(self, _event):
        if self.current_curve_name is None:
            self._update_status_text('All curves are complete. Click Save & Exit to finish.')
            return
        if self.mapping_phase not in ('waiting_to_start', 'review'):
            self._update_status_text('Remap is only available before start or after stop.')
            return

        self.curves[self.current_curve_name] = []
        self.mapping_phase = 'waiting_to_start'
        logger.info('Cleared mapping for %s', self.current_curve_name)
        self._update_plot()
        self._update_status_text()

    def _on_next(self, _event):
        if self.current_curve_name is None:
            self._update_status_text('All curves are already mapped. Click Save & Exit.')
            return
        if self.mapping_phase != 'review':
            self._update_status_text('Next is only available after you stop recording.')
            return
        if len(self._current_curve_points()) < 2:
            self._update_status_text('Record at least two points before moving to the next curve.')
            return

        logger.info('Completed mapping %s', self.current_curve_name)
        self.current_curve_index += 1
        self.mapping_phase = 'waiting_to_start'
        self._update_status_text()

    def _on_save_exit(self, _event):
        if self.mapping_phase == 'recording':
            self._update_status_text('Stop recording before saving.')
            return

        missing_curves = [
            self.curve_labels[name]
            for name in self.curve_names
            if len(self.curves[name]) < 2
        ]
        if missing_curves:
            joined_names = ', '.join(missing_curves)
            self._update_status_text(f'Complete these curves first: {joined_names}.')
            return

        output_path = self._save_track()
        logger.info('Saved mapped track to %s', output_path)
        self.saved = True
        self.mapping_phase = 'finished'
        self._update_status_text(f'Saved track to {output_path}. Exiting...')
        self._draw_ui()
        Extension.main.exit_request.set()
        plt.close(self.figure)
        self.ui_closed = True

    def _save_track(self):
        output_path = os.path.join(BASEDIR, self.config.output_filename)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        payload = {
            name: [point.copy() for point in points]
            for name, points in self.curves.items()
        }
        with open(output_path, 'wb') as file_obj:
            pickle.dump(payload, file_obj)
        return output_path

    def _on_close(self, _event):
        self.ui_closed = True
        if self.saved:
            return
        logger.warning('Track builder window closed before saving. Exiting without writing a track file.')
        Extension.main.exit_request.set()
