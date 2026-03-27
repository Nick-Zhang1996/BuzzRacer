""" a track defined by a spline """
# pylint: disable=unbalanced-tuple-unpacking
from dataclasses import dataclass
from math import sin, cos

import numpy as np
from deprecated import deprecated
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
from scipy.spatial import KDTree

from buzzracer.common import wrap
from buzzracer.types import CurvilinearState, CartesianState
from buzzracer.tracks.track import Track, LocalTrajOutput, Tck


def map(val, x, y, a, b):
    """ Map val from [x,y] to [a,b]"""
    return (val-x) / (y-x) * (b-a) + a


@dataclass
class CurvilinearTrackData:
    r_vec: np.ndarray
    """ (N,2) """
    s_vec: np.ndarray
    """  (N) Cumularive curve length """
    phi_vec: np.ndarray
    """  (N,) Ref path tangent heading """
    curvature_vec: np.ndarray
    """  (N,)  Signed curvature """
    left_width_vec: np.ndarray
    """ (N,) Distance to left boundary """
    right_width_vec: np.ndarray
    """  (N,) Distance to right boundary """
    speed_vec: np.ndarray
    """  (N,) Target speed """
    discretized_raceline: np.ndarray
    """  (N,5), [x,y, heading, left width, right width] """
    kd_tree: KDTree
    """  KD tree of raceline points, for converting cartesian -> curvilinear coordinate """

    raceline_len_m: float
    """  Total curve length in m """
    raceline_s: Tck
    """  splprep result, maps ss -> r """
    curvature_s: Tck
    """  splprep result, maps ss -> curvature """
    phi_s: Tck
    """  splprep result, maps ss -> tangent angle """
    speed_s: Tck
    """  splprep result, map ss ->> target speed """

    left_boundary_vec: np.ndarray
    """  (N,) Left boundary points """
    right_boundary_vec: np.ndarray
    """  (N,) Right boundary points """
    start_pos: tuple[float, float]
    """  Start position """
    start_dir: float
    """  Start heading """

    x_min: float
    """  Minimum x coordinate of track, for visualization boundary """
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
        # call self.build_track(r) to create a curvilinear track
        # see NascarTrack.py for example
        # e.g.
        # self.data = self.build_track(r, left_width, right_width)
        return

    def local_trajectory(self, state: CartesianState):
        """ Given the state of the car, provide geometry information of the raceline.
        Args:
            state: CartesianState
        Return:
            LocalTrajOutput
        """
        # TODO interpolate between reference points
        data = self.data
        x = state[0]
        y = state[1]

        dxx = data.r_vec[:, 0]-x
        dyy = data.r_vec[:, 1]-y
        index = np.argmin(dxx**2+dyy**2)
        raceline_point = data.r_vec[index]
        N = len(data.r_vec)

        dr = data.r_vec[(index+1) % N] - data.r_vec[index]
        track_to_car = (x-data.r_vec[index, 0], y-data.r_vec[index, 1])
        # Positive offset means car is to the left of the trajectory(need to turn right)
        offset = np.cross(dr/np.linalg.norm(dr), track_to_car).item()
        left_margin = data.left_width_vec[index] - offset
        right_margin = data.right_width_vec[index] + offset
        return LocalTrajOutput(ref_point=raceline_point,
                               lateral_err=offset,
                               raceline_dir=data.phi_vec[index],
                               curvature=data.curvature_vec[index],
                               v_target=data.speed_vec[index],
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
        return retval.left_margin < 0 or retval.right_margin < 0

    def build_track(self, r_vec, left_width, right_width):
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
        for i in range(n-1):
            s += ((xx[(i+1) % n]-xx[i])**2 + (yy[(i+1) % n]-yy[i])**2)**0.5
            s_vec.append(s)
        s_vec = np.array(s_vec)
        raceline_len_m = s
        # Dim: 2*n
        # r_vec = np.vstack([r_vec, r_vec[-1]])  # splprep requres [0] == [-1]
        assert np.all(np.diff(s_vec) > 0)
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
        nominator = _norm(dr)**2*_norm(ddr) ** 2 - np.sum(dr*ddr, axis=0)**2
        nominator = np.clip(nominator, a_min=0, a_max=None)**0.5
        curvature_vec = nominator / _norm(dr)**3
        assert not np.any(np.isnan(curvature_vec))
        curvature_s, _ = splprep(curvature_vec.reshape(1, -1), u=s_vec, s=0, per=1)
        ss = np.linspace(0, raceline_len_m, n)
        r_vec = np.array(splev(ss, raceline_s, der=0))
        dr_vec = np.array(splev(ss, raceline_s, der=1))
        phi_vec = np.arctan2(dr_vec[1, :], dr_vec[0, :])
        phi_s, _ = splprep(phi_vec.reshape(1, -1), u=ss, s=0, per=1)
        # Normal direction vector, dim (n,2)
        lateral = np.vstack(
            [np.cos(phi_vec+np.pi/2), np.sin(phi_vec+np.pi/2)]).T
        # boundary, n*2
        upper = r_vec.T + lateral * left_width[:, np.newaxis]
        lower = r_vec.T - lateral * right_width[:, np.newaxis]

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

        # plt.plot(upper[:, 0], upper[:, 1])
        # plt.plot(lower[:, 0], lower[:, 1])
        # plt.plot(r_vec[0, :], r_vec[1, :], 'o')
        # plt.show()
        discretized_raceline = np.vstack(
            [r_vec, phi_vec, left_width, right_width]).T
        start_pos = tuple(r_vec[:, 0])
        start_dir = phi_vec[0]
        spd_profile = Track.generate_speed_profile(raceline_s,
                                                   raceline_len_m,
                                                   mu=0.8,
                                                   acc_max_fun=lambda x: 5.0,
                                                   dec_max_fun=lambda x: 3.3)

        speed_vec = np.array(splev(ss, spd_profile.speed_tck, der=0)).flatten()

        # KD tree for finding closest point on raceline
        kd_tree = KDTree(r_vec.T)

        return CurvilinearTrackData(
            r_vec=r_vec.T,
            s_vec=np.array(s_vec),
            phi_vec=phi_vec,
            curvature_vec=curvature_vec,
            left_width_vec=left_width,
            right_width_vec=right_width,
            speed_vec=speed_vec,
            discretized_raceline=discretized_raceline,
            kd_tree=kd_tree,

            raceline_len_m=raceline_len_m,
            raceline_s=raceline_s,
            curvature_s=curvature_s,
            phi_s=phi_s,
            speed_s=spd_profile.speed_tck,

            left_boundary_vec=upper,
            right_boundary_vec=lower,
            start_pos=start_pos,
            start_dir=start_dir,

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

    def cart_to_curv(self, cart: CartesianState) -> CurvilinearState:
        """Transform cartesian states to curvilinear states, relies on
        self.raceline_s.

        Args:
            cart: Cartesian state
            guess_s: estimated s (progress along ref curve)
        Returns:
            curv: curvilinear state

        """
        data: CurvilinearTrackData = self.data
        _, idx = data.kd_tree.query([cart.x, cart.y])
        N = len(data.s_vec)
        rs = data.s_vec[idx]
        rphi = data.phi_vec[idx]
        dx = cart.x - data.r_vec[idx, 0]
        dy = cart.y - data.r_vec[idx, 1]

        ds = dx * cos(rphi) + dy * sin(rphi)  # dot(displacement, curve tangent)
        n = cos(rphi) * dy - sin(rphi)*dx
        s_step = (data.s_vec[(idx+1) % N] - rs) % data.raceline_len_m  # handle wrap around
        phi_step = wrap(data.phi_vec[(idx+1) % N] - rphi)

        # Interpolated ref pi
        precise_rphi = map(ds, 0, s_step, rphi, rphi+phi_step)
        phi = wrap(cart.heading - precise_rphi)

        # NOTE cartesian v_sideway is not exactly the same as curvilinear v_sideway
        return CurvilinearState(progress=rs+ds,
                                lateral_err=n,
                                heading_err=phi,
                                v_forward=cart.v_forward,
                                v_sideway=cart.v_sideway,
                                rel_omega=cart.omega)

    @deprecated
    def old_cart_to_curv(self, cart: CartesianState, guess_s: float = None) -> CurvilinearState:
        """Transform cartesian states to curvilinear states, relies on
        self.raceline_s.

        Args:
            cart: Cartesian state
            guess_s: estimated s (progress along ref curve)
        Returns:
            curv: curvilinear state

        """
        data = self.data

        def dist(s):
            val = np.linalg.norm(
                np.array(splev(s % data.raceline_len_m,
                         data.raceline_s)).flatten()
                - np.array([cart.x, cart.y])
            )
            return val

        if guess_s is None:
            # initial guess to avoid local minima
            xx = np.linspace(0.0, data.raceline_len_m, 100)
            yy = [dist(x) for x in xx]
            guess_s = xx[np.argmin(yy)]
            ds = 2*data.raceline_len_m/100
            fit = minimize(dist, x0=guess_s, method='L-BFGS-B',
                           bounds=((guess_s-ds, guess_s+ds),), tol=1e-3)
        else:
            fit = minimize(dist, x0=guess_s, method='L-BFGS-B',
                           bounds=((guess_s-0.2, guess_s+0.2),), tol=1e-3)

        s = fit.x[0]

        r = np.array(splev(s % data.raceline_len_m,
                     data.raceline_s, der=0))
        dr = np.array(splev(s % data.raceline_len_m,
                      data.raceline_s, der=1))
        dr = dr/np.linalg.norm(dr)
        n = np.cross(dr, np.array([cart.x, cart.y]) - r)
        phi = wrap(cart.heading - np.arctan2(dr[1], dr[0]))
        return CurvilinearState(progress=s,
                                lateral_err=n,
                                heading_err=phi,
                                v_forward=cart.v_forward,
                                v_sideway=cart.v_sideway,
                                rel_omega=cart.omega)

    def curv_to_cart(self, curv: CurvilinearState) -> CartesianState:
        """Transform curvilinear state to cartesian state. 
        Need self.data.raceline_s.

        Args:
            curv: Curvilinear state
        Returns:
            cart: Transformed cartesian state

        """
        data = self.data
        r = np.array(splev(curv.progress % data.raceline_len_m, data.raceline_s, der=0))
        dr = np.array(splev(curv.progress % data.raceline_len_m, data.raceline_s, der=1))
        dr = dr/np.linalg.norm(dr)

        # ccw 90 deg
        A = np.array([[0, -1], [1, 0]])
        x, y = r + (A @ dr)*curv.lateral_err
        ref_heading = np.arctan2(dr[1], dr[0])
        heading = wrap(curv.heading_err + ref_heading)
        return CartesianState(x=x,
                              y=y,
                              heading=heading,
                              v_forward=curv.v_forward,
                              v_sideway=curv.v_sideway,
                              omega=curv.rel_omega)
