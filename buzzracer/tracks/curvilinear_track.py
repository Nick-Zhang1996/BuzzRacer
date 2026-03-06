""" a track defined by a spline """
# pylint: disable=unbalanced-tuple-unpacking
from typing import Any
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import splprep, splev

from buzzracer.tracks.track import Track, LocalTrajOutput
from buzzracer.types import CartesianState


@dataclass
class CurvilinearTrackData:
    r_vec: np.ndarray  # (N,2)
    s_vec: np.ndarray  # (N) Cumularive curve length
    phi_vec: np.ndarray  # (N,) Ref path tangent heading
    curvature_vec: np.ndarray  # (N,)  Signed curvature
    left_width_vec: np.ndarray  # (N,) distance to left boundary
    right_width_vec: np.ndarray  # (N,) distance to right boundary
    left_boundary_vec: np.ndarray  # (N,) Left boundary points
    right_boundary_vec: np.ndarray  # (N,) Right boundary points
    discretized_raceline: np.ndarray  # (N,5), [x,y, heading, left width, right width]
    raceline_len_m: float   # Total curve length
    raceline_s: Any  # splprep result, maps ss -> r
    curvature_s: Any  # splprep result, maps ss -> curvature
    phi_s: Any  # splprep result, maps ss -> tangent angle
    x_min: float  # Minimum x coordinate of track, for visualization boundary
    x_max: float
    y_min: float
    y_max: float


class CurvilinearTrack(Track):
    """ Base class for Track defined by a curve. """

    def __init__(self, config):
        Track.__init__(self, config)
        # default parameters, to be override
        self.data: CurvilinearTrackData

        # Derived class should override the constructor,
        # call self.build_continuous_track(r) to create a curvilinear track
        # see NascarTrack.py for example
        # e.g.
        # self.data = CurvilinearTrack.build_continuous_track(r, left_width, right_width)
        return

    def local_trajectory(self, state: CartesianState):
        """ Given the state of the car, provide geometry information of the raceline.
        Args:
            state: CartesianState
        Return:
            LocalTrajOutput
        """
        data = self.data
        x = state[0]
        y = state[1]

        dxx = data.r_vec[:, 0]-x
        dyy = data.r_vec[:, 1]-y
        index = np.argmin(dxx**2+dyy**2)
        raceline_point = data.r_vec[index]

        dr = data.r_vec[index+1] - data.r_vec[index]
        track_to_car = (x-data.r_vec[index, 0], y-data.r_vec[index, 1])
        # Positive offset means car is to the left of the trajectory(need to turn right)
        offset = np.cross(dr/np.linalg.norm(dr), track_to_car).item()
        left_margin = data.left_width_vec[index] - offset
        right_margin = data.right_width_vec[index] + offset
        return LocalTrajOutput(ref_point=raceline_point,
                               lateral_err=offset,
                               raceline_dir=data.raceline_dir[index],
                               curvature=data.curvature_vec[index],
                               v_target=data.v_target[index],
                               progress=data.s_vec[index],
                               left_margin=left_margin,
                               right_margin=right_margin)

    def is_outside(self, coord):
        """
        Return true if vehicle is unsalvageably outside of the track
        for use by Watchdog to terminate an experiment
        """
        state = CartesianState(coord[0], coord[1])
        retval = self.local_trajectory(state)
        return retval.lateral_err > (self.width/2)*1.5

    @staticmethod
    def build_continuous_track(r_vec, left_width, right_width):
        """ Build a Curvilinear Track.
        Args:
            r_vec: np.ndarray (N, 2) Reference points for curve.
            left_width: np.ndarray (N,) Half width from ref curve to left boundary
            right_width: np.ndarray (N,) Half width from ref curve to right boundary
        Return:
            CurvilinearTrackData Object
        """
        assert len(r_vec.shape) == 2
        assert r_vec.shape[1] == 2
        n = r_vec.shape[0]
        assert left_width.shape == (n,)
        assert right_width.shape == (n,)
        xx = r_vec[:, 0]
        yy = r_vec[:, 1]

        s = 0
        s_vec = [s]
        for i in range(n):
            s += ((xx[(i+1) % n]-xx[i])**2 + (yy[(i+1) % n]-yy[i])**2)**0.5
            s_vec.append(s)
        raceline_len_m = s
        # Dim: 2*n
        r_vec = np.vstack([r_vec, r_vec[-1]])  # splprep requres [0] == [-1]

        tck, _ = splprep(r_vec.T, u=s_vec, s=0, per=1)
        raceline_s = tck

        # let raceline curve be r(u)
        # dr = r'(u), parameterized with xx/u
        dr = np.array(splev(s_vec, raceline_s, der=1))
        # ddr = r''(u)
        ddr = np.array(splev(s_vec, raceline_s, der=2))

        def _norm(x):
            return np.linalg.norm(x, axis=0)
        # radius of curvature can be calculated as R = |y'|^3/sqrt(|y'|^2*|y''|^2-(y'*y'')^2)
        curvature_vec = 1.0/(_norm(dr)**3/(_norm(dr)**2*_norm(ddr)
                                           ** 2 - np.sum(dr*ddr, axis=0)**2)**0.5)
        curvature_s, _ = splprep(curvature_vec.reshape(1, -1), u=s_vec, s=0, per=1)
        ss = np.linspace(0, raceline_len_m, n)
        r_vec = np.array(splev(ss, raceline_s, der=0))
        dr_vec = np.array(splev(ss, raceline_s, der=1))
        phi_vec = np.arctan2(dr_vec[1, :], dr_vec[0, :])
        phi_s, _ = splprep(phi_vec.reshape(1,-1), u=ss, s=0, per=1)
        # Normal direction vector, dim (n,2)
        lateral = np.vstack(
            [np.cos(phi_vec+np.pi/2), np.sin(phi_vec+np.pi/2)]).T
        # boundary, n*2
        upper = r_vec.T + lateral * left_width[:,np.newaxis]/2
        lower = r_vec.T - lateral * right_width[:,np.newaxis]/2

        x_min = np.min(np.hstack([upper[:, 0], lower[:, 0]])) - 0.1
        x_max = np.max(np.hstack([upper[:, 0], lower[:, 0]])) + 0.1
        y_min = np.min(np.hstack([upper[:, 1], lower[:, 1]])) - 0.1
        y_max = np.max(np.hstack([upper[:, 1], lower[:, 1]])) + 0.1

        # shift track to first quadrant so x,y>0
        # x_limit = x_max - x_min
        # y_limit = y_max - y_min
        # upper[:,0] -= x_min
        # upper[:,1] -= y_min
        # lower[:,0] -= x_min
        # lower[:,1] -= y_min
        # r_vec[:,0] -= x_min
        # r_vec[:,1] -= y_min

        # plt.plot(upper[:,0],upper[:,1])
        # plt.plot(lower[:,0],lower[:,1])
        # plt.plot(r_vec[0,:],r_vec[1,:],'o')
        # plt.show()
        discretized_raceline = np.vstack(
            [r_vec, phi_vec, left_width, right_width]).T

        return CurvilinearTrackData(
            r_vec=r_vec.T,
            s_vec=s_vec,
            phi_vec=phi_vec,
            curvature_vec=curvature_vec,
            left_width_vec=left_width,
            right_width_vec=right_width,
            left_boundary_vec=upper,
            right_boundary_vec=lower,
            discretized_raceline=discretized_raceline,
            raceline_len_m=raceline_len_m,
            raceline_s=raceline_s,
            curvature_s=curvature_s,
            phi_s=phi_s,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )

    def m2canvas(self, coord):
        data = self.data
        config = self.config
        x_new = int(
            (np.clip(coord[0], data.x_min, data.x_max)-data.x_min) * config.resolution)
        y_new = int(
            (data.y_max - np.clip(coord[1], data.y_min, data.y_max)) * config.resolution)
        return (x_new, y_new)

    def draw_track(self):
        """ Create an image of the track. """
        data = self.data
        config = self.config
        x_pix = int((data.x_max - data.x_min)*config.resolution)
        y_pix = int((data.y_max - data.y_min)*config.resolution)

        # height, width
        img = 255*np.ones([y_pix, x_pix, 3], dtype=np.uint8)
        img = data.draw_polyline(
            data.left_boundary_vec, img, lineColor=(0, 0, 0), thickness=2)
        img = data.draw_polyline(
            data.right_boundary_vec, img, lineColor=(0, 0, 0), thickness=2)
        return img

    def precise_track_boundary(self, coord, heading):
        del heading
        state = CartesianState(coord[0], coord[1])
        retval = self.local_trajectory(state)
        return (retval.left_margin, retval.right_margin)
