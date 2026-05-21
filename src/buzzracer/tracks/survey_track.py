"""Curvilinear track built from surveyed left/right boundaries."""
from __future__ import annotations

from math import cos, sin

import cv2
import numpy as np
from scipy.interpolate import splev

from buzzracer.common import get_logger
from buzzracer.tracks.curvilinear_track import CurvilinearTrack
from buzzracer.tracks.track import TrackConfig

logger = get_logger(__name__)


def _close_boundary(boundary: np.ndarray) -> np.ndarray:
    """Ensure one surveyed boundary is stored as an explicit closed loop."""
    boundary = np.asarray(boundary, dtype=float)
    if len(boundary) > 1 and np.linalg.norm(boundary[0] - boundary[-1]) < 1e-9:
        return boundary.copy()
    return np.vstack([boundary, boundary[0]])


class SurveyTrack(CurvilinearTrack):
    """Track backed by fixed surveyed boundaries and a movable reference path."""

    def __init__(self, config: TrackConfig, left_boundary: np.ndarray, right_boundary: np.ndarray):
        super().__init__(config)
        self.left_boundary = _close_boundary(left_boundary)
        self.right_boundary = _close_boundary(right_boundary)
        self.rcp_raceline = None

    def set_raceline(self, raceline):
        """Set the current raceline and rebuild curvilinear track data from it."""
        self.rcp_raceline = raceline
        r_vec, left, right = self.process_raceline(raceline)
        self.data = self.build_track(r_vec, left, right)

    def process_raceline(self, raceline):
        """Sample one raceline and compute left/right widths from the fixed survey polygon."""
        n_points = self.config.discretized_raceline_len
        ss = np.linspace(0.0, raceline.raceline_len_m, n_points, endpoint=False)
        r_vec = np.array(splev(ss, raceline.raceline_s, der=0)).T
        dr_vec = np.array(splev(ss, raceline.raceline_s, der=1))
        heading_vec = np.arctan2(dr_vec[1], dr_vec[0])
        boundary = self.create_boundary(r_vec, heading_vec)
        return r_vec, boundary[:, 0], boundary[:, 1]

    def save(self, filename='survey_track.p'):
        """Save this survey track using the base curvilinear-track format."""
        super().save(filename)

    def draw_track(self, img=None):
        """Render the fixed surveyed track boundaries."""
        if img is None:
            x_pix = int((self.data.x_max - self.data.x_min) * self.config.resolution)
            y_pix = int((self.data.y_max - self.data.y_min) * self.config.resolution)
            img = 255 * np.ones((y_pix, x_pix, 3), dtype=np.uint8)
        img = self.draw_polyline(self.left_boundary, img=img, lineColor=(0, 0, 0), thickness=2)
        img = self.draw_polyline(self.right_boundary, img=img, lineColor=(0, 0, 0), thickness=2)
        return img

    def draw_raceline(self, raceline=None, bound=None, img=None, points=None, speed_profile=None):
        """Draw one raceline using the inherited polyline renderer."""
        del points, speed_profile
        if img is None:
            img = self.draw_track()
        if raceline is None:
            raceline = self.data.raceline_s
        if bound is None:
            bound = self.data.raceline_len_m
        ss = np.linspace(0.0, bound, 1000, endpoint=False)
        points = np.array(splev(ss, raceline, der=0)).T
        return self.draw_polyline(points, img=img, lineColor=(0, 0, 255), thickness=2)

    def draw_point(self, img, coord, color=(0, 0, 0)):
        """Draw one point in track coordinates onto an OpenCV image."""
        src = self.m2canvas(coord)
        return cv2.circle(img, src, 3, color, -1)

    def precise_track_boundary(self, coord, heading):
        """Return signed left/right margins based on ray intersections with both boundary loops."""
        origin = np.asarray(coord, dtype=float)
        left_dir = np.array([cos(heading + np.pi / 2), sin(heading + np.pi / 2)])
        right_dir = np.array([cos(heading - np.pi / 2), sin(heading - np.pi / 2)])

        left_distance = self._raycast_distance(origin, left_dir)
        right_distance = self._raycast_distance(origin, right_dir)
        sign = 1.0 if self._point_in_track(origin) else -1.0
        return sign * left_distance, sign * right_distance

    def _point_in_track(self, point):
        """Return True when a point is inside the track surface between the two loops."""
        if self._point_on_any_boundary(point):
            return True
        total_crossings = (
            self._count_ray_crossings(point, self.left_boundary) +
            self._count_ray_crossings(point, self.right_boundary)
        )
        return total_crossings % 2 == 1

    def _raycast_distance(self, origin, direction):
        """Return the first positive ray intersection distance with either boundary loop."""
        min_distance = None
        for boundary in (self.left_boundary, self.right_boundary):
            for idx in range(len(boundary) - 1):
                segment_start = boundary[idx]
                segment_end = boundary[idx + 1]
                distance = self._ray_segment_intersection(
                    origin, direction, segment_start, segment_end)
                if distance is None:
                    continue
                if min_distance is None or distance < min_distance:
                    min_distance = distance
        if min_distance is None:
            return 0.0
        return float(min_distance)

    def _point_on_any_boundary(self, point):
        for boundary in (self.left_boundary, self.right_boundary):
            for idx in range(len(boundary) - 1):
                if self._point_on_segment(point, boundary[idx], boundary[idx + 1]):
                    return True
        return False

    def _count_ray_crossings(self, point, boundary):
        """Count horizontal ray crossings for one closed boundary loop."""
        x_coord, y_coord = point
        crossing_count = 0

        for idx in range(len(boundary) - 1):
            p0 = boundary[idx]
            p1 = boundary[idx + 1]
            y0 = p0[1]
            y1 = p1[1]

            if (y0 > y_coord) == (y1 > y_coord):
                continue

            x_at_y = p0[0] + (y_coord - y0) * (p1[0] - p0[0]) / (y1 - y0)
            if x_at_y >= x_coord:
                crossing_count += 1

        return crossing_count

    @staticmethod
    def _cross_2d(vec_a, vec_b):
        return vec_a[0] * vec_b[1] - vec_a[1] * vec_b[0]

    @classmethod
    def _ray_segment_intersection(cls, origin, direction, segment_start, segment_end):
        """Return distance along a ray to a segment intersection, if it exists."""
        segment = segment_end - segment_start
        denom = cls._cross_2d(direction, segment)
        if np.isclose(denom, 0.0):
            return None

        offset = segment_start - origin
        ray_distance = cls._cross_2d(offset, segment) / denom
        segment_alpha = cls._cross_2d(offset, direction) / denom

        if ray_distance <= 1e-9:
            return None
        if segment_alpha < -1e-9 or segment_alpha > 1.0 + 1e-9:
            return None
        return ray_distance

    @staticmethod
    def _point_on_segment(point, segment_start, segment_end, tol=1e-9):
        """Return True if a point lies on a line segment."""
        segment = segment_end - segment_start
        offset = point - segment_start
        if abs(segment[0] * offset[1] - segment[1] * offset[0]) > tol:
            return False

        dot = np.dot(offset, segment)
        if dot < -tol:
            return False
        if dot > np.dot(segment, segment) + tol:
            return False
        return True
