"""Build a survey-based CurvilinearTrack from mapped boundary curves."""
from __future__ import annotations

import argparse
import os
import pickle
from time import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from scipy.interpolate import splprep, splev

from buzzracer.common import BASEDIR, get_logger, print_info, print_ok
from buzzracer.tracks.rcp_track import RCPTrackRaceline
from buzzracer.tracks.survey_track import SurveyTrack
from buzzracer.tracks.track import Track, TrackConfig

try:
    from qp_smooth import QpSmooth
except ImportError:
    from scripts.qp_smooth import QpSmooth


logger = get_logger(__name__, level=10)


def _resolve_repo_path(path_text: str) -> str:
    if os.path.isabs(path_text):
        return path_text
    return os.path.join(BASEDIR, path_text)


def _sanitize_curve(points) -> np.ndarray:
    """Drop duplicate neighbors and trailing closure points from one survey curve."""
    curve = np.asarray(points, dtype=float)
    if curve.ndim != 2 or curve.shape[1] != 2:
        raise ValueError('Each curve must have shape (N, 2)')

    filtered = [curve[0]]
    for point in curve[1:]:
        if np.linalg.norm(point - filtered[-1]) > 1e-6:
            filtered.append(point)

    curve = np.asarray(filtered, dtype=float)
    if len(curve) > 1 and np.linalg.norm(curve[0] - curve[-1]) < 1e-6:
        curve = curve[:-1]

    if len(curve) < 4:
        raise ValueError('Each curve must contain at least 4 distinct points')
    return curve


def _trim_curve(curve: np.ndarray, trim_count: int) -> np.ndarray:
    """Trim points from the end of a curve while keeping at least 3 points."""
    trim_count = int(trim_count)
    max_trim = max(0, len(curve) - 3)
    trim_count = min(max(trim_count, 0), max_trim)
    if trim_count == 0:
        return curve.copy()
    return curve[:-trim_count].copy()


def _align_curve_direction(reference_curve: np.ndarray, curve: np.ndarray) -> np.ndarray:
    """Flip a curve if needed so it follows the same travel direction as the reference."""
    forward_cost = (
        np.linalg.norm(curve[0] - reference_curve[0]) +
        np.linalg.norm(curve[-1] - reference_curve[-1])
    )
    reverse_cost = (
        np.linalg.norm(curve[-1] - reference_curve[0]) +
        np.linalg.norm(curve[0] - reference_curve[-1])
    )
    if reverse_cost < forward_cost:
        return curve[::-1].copy()
    return curve.copy()


def _close_curve(curve: np.ndarray) -> np.ndarray:
    """Append the start point to the end so the curve forms an explicit loop."""
    if np.linalg.norm(curve[0] - curve[-1]) < 1e-9:
        return curve.copy()
    return np.vstack([curve, curve[0]])


def load_survey_curves(filename: str) -> dict[str, np.ndarray]:
    """Load and sanitize the three curves created by SurveyTrackBuilder."""
    t0 = time()
    with open(filename, 'rb') as file_obj:
        raw_curves = pickle.load(file_obj)
    curves = {
        'left_boundary': _sanitize_curve(raw_curves['left_boundary']),
        'right_boundary': _sanitize_curve(raw_curves['right_boundary']),
        'reference': _sanitize_curve(raw_curves['reference']),
    }
    dt = time() - t0
    logger.debug('load_survey_curves took %.3fs', dt)
    return curves


def trim_curves_with_sliders(curves: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Interactive matplotlib UI for trimming overlap off the end of each curve."""
    t0 = time()
    curve_order = ('left_boundary', 'right_boundary', 'reference')
    curve_labels = {
        'left_boundary': 'left boundary',
        'right_boundary': 'right boundary',
        'reference': 'centerline',
    }
    curve_colors = {
        'left_boundary': 'tab:orange',
        'right_boundary': 'tab:green',
        'reference': 'tab:blue',
    }
    trim_values = {name: 0 for name in curve_order}
    trimmed_curves = {name: curves[name].copy() for name in curve_order}

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.28)
    ax.set_title('Trim curve overlap with sliders, then close the window')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.margins(0.05)

    all_points = np.vstack([curves[name] for name in curve_order])
    x_padding = max(0.05, 0.05 * (np.max(all_points[:, 0]) - np.min(all_points[:, 0])))
    y_padding = max(0.05, 0.05 * (np.max(all_points[:, 1]) - np.min(all_points[:, 1])))
    ax.set_xlim(np.min(all_points[:, 0]) - x_padding, np.max(all_points[:, 0]) + x_padding)
    ax.set_ylim(np.min(all_points[:, 1]) - y_padding, np.max(all_points[:, 1]) + y_padding)

    artists = {}
    start_markers = {}
    for name in curve_order:
        artist, = ax.plot([], [], color=curve_colors[name], linewidth=2, label=curve_labels[name])
        start_marker, = ax.plot([], [], marker='o', linestyle='None',
                                color=curve_colors[name], markersize=6)
        artists[name] = artist
        start_markers[name] = start_marker

    ax.legend(loc='best')

    slider_specs = [
        ('left_boundary', [0.18, 0.17, 0.70, 0.03]),
        ('right_boundary', [0.18, 0.11, 0.70, 0.03]),
        ('reference', [0.18, 0.05, 0.70, 0.03]),
    ]
    sliders = {}

    def _update_plot():
        for name in curve_order:
            trimmed = _trim_curve(curves[name], trim_values[name])
            trimmed_curves[name] = trimmed
            artists[name].set_data(trimmed[:, 0], trimmed[:, 1])
            start_markers[name].set_data([trimmed[0, 0]], [trimmed[0, 1]])
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()

    for name, rect in slider_specs:
        slider = Slider(
            fig.add_axes(rect),
            f'{curve_labels[name]} trim',
            valmin=0,
            valmax=max(0, len(curves[name]) - 3),
            valinit=0,
            valstep=1,
        )

        def _on_change(value, *, curve_name=name):
            trim_values[curve_name] = int(value)
            _update_plot()

        slider.on_changed(_on_change)
        sliders[name] = slider

    _update_plot()
    plt.show()
    dt = time() - t0
    logger.debug('trim_curves_with_sliders took %.3fs', dt)
    return trimmed_curves


def build_closed_raceline(reference_curve: np.ndarray) -> RCPTrackRaceline:
    """Fit a periodic spline to the surveyed centerline and re-parameterize it by arc length."""
    t0 = time()
    if np.linalg.norm(reference_curve[0] - reference_curve[-1]) < 1e-9:
        reference_curve = reference_curve[:-1]
    raw_tck, _ = splprep(reference_curve.T, s=0.0, per=1)
    raceline_s, raceline_len_m = Track.reparam_raceline(raw_tck, 1.0)
    start_pos = tuple(np.array(splev(0.0, raceline_s, der=0)).flatten())
    start_tangent = np.array(splev(0.0, raceline_s, der=1)).flatten()
    start_dir = float(np.arctan2(start_tangent[1], start_tangent[0]))
    raceline = RCPTrackRaceline(
        raceline_s=raceline_s,
        raceline_len_m=raceline_len_m,
        start_pos=start_pos,
        start_dir=start_dir,
    )
    dt = time() - t0
    logger.debug('build_closed_raceline took %.3fs', dt)
    return raceline


def build_survey_track(left_boundary: np.ndarray,
                       right_boundary: np.ndarray,
                       reference_curve: np.ndarray) -> SurveyTrack:
    """Build one SurveyTrack using the surveyed boundaries and initial centerline."""
    # TODO: time this function with finer granularity, I want to know exectly which line is slow
    t0 = time()
    boundary_points = np.vstack([left_boundary, right_boundary, reference_curve])
    x_span = float(np.max(boundary_points[:, 0]) - np.min(boundary_points[:, 0]))
    y_span = float(np.max(boundary_points[:, 1]) - np.min(boundary_points[:, 1]))
    config = TrackConfig(
        resolution=200,
        discretized_raceline_len=max(512, len(reference_curve) * 4),
        x_limit=x_span,
        y_limit=y_span,
    )

    track = SurveyTrack(config, left_boundary, right_boundary)
    track.set_raceline(build_closed_raceline(reference_curve))
    dt = time() - t0
    logger.debug('build_survey_track took %.3fs', dt)
    return track


def show_final_track(track: SurveyTrack, initial_reference: np.ndarray):
    """Display the surveyed boundaries, initial centerline, and optimized raceline."""
    t0 = time()
    optimized_reference = track.data.r_vec
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(track.left_boundary[:, 0], track.left_boundary[:, 1],
            color='tab:orange', linewidth=2, label='left boundary')
    ax.plot(track.right_boundary[:, 0], track.right_boundary[:, 1],
            color='tab:green', linewidth=2, label='right boundary')
    ax.plot(initial_reference[:, 0], initial_reference[:, 1],
            color='0.6', linestyle='--', linewidth=1.5, label='initial centerline')
    ax.plot(optimized_reference[:, 0], optimized_reference[:, 1],
            color='tab:red', linewidth=2, label='optimized reference')
    ax.set_title('Survey Track')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.show()
    dt = time() - t0
    logger.debug('show_final_track took %.3fs', dt)


def main():
    t0_total = time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=os.path.join('outputs', 'mapped_track.p'))
    parser.add_argument('--output', default='survey_track.p')
    parser.add_argument('--qp-offset', type=float, default=0.15)
    args = parser.parse_args()

    t0 = time()
    input_path = _resolve_repo_path(args.input)
    print_info(f'loading mapped curves from {input_path}')
    curves = load_survey_curves(input_path)
    dt = time() - t0
    logger.debug('main: load stage took %.3fs', dt)

    t0 = time()
    curves = trim_curves_with_sliders(curves)
    dt = time() - t0
    logger.debug('main: trim UI stage took %.3fs', dt)

    t0 = time()
    reference_curve = curves['reference']
    left_boundary = _align_curve_direction(reference_curve, curves['left_boundary'])
    right_boundary = _align_curve_direction(reference_curve, curves['right_boundary'])

    # Keep all three curves explicitly closed. SurveyTrack will treat the left
    # and right boundaries as two separate closed loops and use even-odd ray
    # casting across both loops to define the track surface.
    reference_curve = _close_curve(reference_curve)
    left_boundary = _close_curve(left_boundary)
    right_boundary = _close_curve(right_boundary)
    dt = time() - t0
    logger.debug('main: align/close stage took %.3fs', dt)

    t0 = time()
    track = build_survey_track(left_boundary, right_boundary, reference_curve)
    initial_reference = track.data.r_vec.copy()
    dt = time() - t0
    logger.debug('main: initial track build took %.3fs', dt)

    t0 = time()
    optimizer = QpSmooth()
    optimized_raceline = optimizer.optimize_raceline(
        track.rcp_raceline,
        track=track,
        offset=args.qp_offset,
        visualize=False,
        visualize_final_result=False,
        save_gif=False,
        save_steps=False,
    )
    track.set_raceline(optimized_raceline)
    dt = time() - t0
    logger.debug('main: QpSmooth + rebuild took %.3fs', dt)

    t0 = time()
    show_final_track(track, initial_reference)
    dt = time() - t0
    logger.debug('main: final display took %.3fs', dt)

    t0 = time()
    track.save(args.output)
    dt = time() - t0
    logger.debug('main: save took %.3fs', dt)
    print_ok(f'saved survey track to assets/{args.output}')
    logger.debug('main: total runtime took %.3fs', time() - t0_total)


if __name__ == '__main__':
    main()
