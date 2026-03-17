"""Base class for RCPTrack and Skidpad This class provides API for interacting
with a Track object A track object provides information on the trajectory and
provide access for drawing the track."""
import os.path
import pickle
import logging
from math import cos, sin
from typing import NamedTuple, TYPE_CHECKING, Callable
from dataclasses import dataclass
from deprecated import deprecated

import cv2
import numpy as np
from scipy.interpolate import splev, splprep, interp1d
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from buzzracer.common import wrap, BASEDIR
from buzzracer.types import CurvilinearState, CartesianState

if TYPE_CHECKING:
    from buzzracer.tracks.curvilinear_track import CurvilinearTrackData

# Splprep result
type Tck = list[np.ndarray, np.ndarray, int]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LocalTrajOutput(NamedTuple):
    ref_point: np.ndarray  # Closest point on raceline
    lateral_err: float  # Lateral error. Left deviation is positive.
    raceline_dir: float  # Orientation of raceline tangent. CCW positive
    curvature: float  # Signed curvature, CCW positive
    v_target: float  # Reference speed at ref_point
    progress: float  # Curve length along raceline in Frenet frame.
    left_margin: float  # Distance to left boundary
    right_margin: float  # Distance to right boundary


class SpeedProfileOutput(NamedTuple):
    speed_tck: Tck
    min_v: float
    max_v: float


@dataclass
class TrackConfig:
    # pixels per meter
    resolution: int = 200
    discretized_raceline_len: int = 1024
    # track dimension, in meters
    x_limit: float = 0.0
    y_limit: float = 0.0


class Track:
    main = None

    def __init__(self, config: TrackConfig):
        self.config = config
        self.data: CurvilinearTrackData

        # obstacles
        self.obstacle = False
        self.obstacle_count = 0
        self.obstacle_filename = None
        self.obstacle_radius = None
        self.obstacles = None
        ''' dim:[n_obstacles, 2], coordinate of obstacles'''

    def init(self):
        self.set_up_obstacles()

    def draw_raceline(self, raceline, bound, img=None, points=None, speed_profile=None):
        del raceline, bound, points, speed_profile
        return img

    def draw_track(self):
        """ Given the state of the car, provide geometry information of the raceline.
        Args:
            state: CartesianState
        Return:
            LocalTrajOutput
        """
        raise NotImplementedError

    def local_trajectory(self, state) -> LocalTrajOutput:
        raise NotImplementedError

    def is_outside(self, coord):
        ''' Determine if an coordinate is outside of track boundary, used in watchdog '''
        config = self.config
        grace = 1.0
        x, y = coord
        return x < -grace or y < -grace or x > config.x_limit+grace or y > config.y_limit+grace

    def is_in_obstacle(self, state):
        ''' check if vehicle is currently in collision with obstacle
         only give index of the first obstacle if multiple obstacle is in collision'''
        if not self.obstacle:
            return (False, -1)
        dist = self.obstacle_radius
        x, y, _, _, _, _ = state
        min_dist = 100.0
        for i in range(self.obstacles.shape[0]):
            obs = self.obstacles[i]
            dist = ((x-obs[0])**2+(y-obs[1])**2)**0.5
            if dist < min_dist:
                min_dist = dist
            if dist < self.obstacle_radius:
                return (True, i)
        return (False, -1)

    def m2canvas(self, coord):
        config = self.config
        x_new = int(np.clip(coord[0], 0, config.x_limit) * config.resolution)
        y_new = int(
            (config.y_limit-np.clip(coord[1], 0, config.y_limit)) * config.resolution)
        return (x_new, y_new)

    def canvas2m(self, coord):
        config = self.config
        x_new = coord[0] / config.resolution
        y_new = config.y_limit = coord[1] / config.resolution
        return (x_new, y_new)

    def draw_circle(self, img, coord, radius_m, color=(0, 0, 0)):
        ''' draw a circle on canvas at coord '''
        src = self.m2canvas(coord)
        radius_pix = int(radius_m * self.config.resolution)
        img = cv2.circle(img, src, radius_pix, color, -1)
        return img

    def plot_obstacles(self, cars, img=None):
        if not self.obstacle:
            return img
        # plot obstacles
        for obs in self.obstacles:
            img = self.draw_circle(img, obs, 0.1, color=(255, 100, 100))
        for car in cars:
            has_collided, obs_id = self.is_in_obstacle(car.state)
            if has_collided:
                # plot obstacle in collision red
                img = self.draw_circle(
                    img, self.obstacles[obs_id], 0.1, color=(100, 100, 255))

        return img

    def draw_polyline(self, points, img=None, lineColor=(0, 0, 255), thickness=3):
        ''' Draw a polynomial line defined in track space
            points: a list of coordinates in format (x,y)
        '''
        config = self.config
        if img is None:
            img = np.zeros([int(config.resolution*config.x_limit),
                           int(config.resolution*config.y_limit), 3], dtype='uint8')

        pts = [self.m2canvas(point) for point in points]
        for i in range(len(points)-1):
            p1 = np.array(pts[i])
            p2 = np.array(pts[i+1])
            if pts[i] is None or pts[i+1] is None:
                continue
            img = cv2.line(img, tuple(p1), tuple(p2), color=lineColor, thickness=thickness)
        return img

    def draw_trajectory(self, traj_points, img=None, lineColor=(0, 0, 255), thickness=3):
        return self.draw_polyline(traj_points[:, 1:3], img, lineColor, thickness)

    # draw ONE arrow, unit: meter, coord sys: dimensioned
    # source: source of arrow, in meter
    # orientation, radians from x axis, ccw positive
    # length: in pixels, though this is only qualitative
    def draw_arrow(self, source, orientation, length, color=(0, 0, 0), thickness=2, img=None):
        config = self.config
        if img is None:
            img = np.zeros([int(config.resolution*config.x_limit),
                           int(config.resolution*config.y_limit), 3], dtype='uint8')

        length = int(length)
        src = self.m2canvas(source)

        # y-axis positive direction in real world and cv plotting is reversed
        dest = (int(src[0] + cos(orientation)*length),
                int(src[1] - sin(orientation)*length))

        img = cv2.circle(img, src, 3, (0, 0, 0), -1)
        img = cv2.line(img, src, dest, color, thickness)

        return img

    # NOTE obstacles
    # obstacle related class variables need to be set prior
    def set_up_obstacles(self):
        config = self.config
        if not self.obstacle:
            self.obstacle_count = 0
            return
        filename = os.path.join(BASEDIR, self.obstacle_filename)

        if os.path.isfile(filename):
            with open(filename, 'rb') as f:
                obstacles = pickle.load(f)
            self.obstacle_count = obstacles.shape[0]
            logger.info(
                f'loading obstacles at {filename}, count = {obstacles.shape[0]}')
            logger.info(
                ' if you wish to create new obstacles,'
                'remove current obstacle file or change parameter obstacle_filename')
        else:
            logger.info(
                f'generating new obstacles, count = {self.obstacle_count}')
            obstacles = np.random.random((self.obstacle_count, 2))
            # save obstacles
            if not filename is None:
                with open(filename, 'wb') as f:
                    pickle.dump(obstacles, f)
                logger.info(f'saved obstacles at {filename}')

        # spread obstacle to entire track
        obstacles[:, 0] *= config.x_limit
        obstacles[:, 1] *= config.y_limit

        self.obstacles = obstacles

    # def prepare_discretized_raceline(self):
    #     """depends on self.raceline_s, self.raceline_len_m."""
    #     ss = np.linspace(0, self.raceline_len_m, self.config.discretized_raceline_len)
    #     rr = splev(ss % self.raceline_len_m, self.raceline_s, der=0)
    #     drr = splev(ss % self.raceline_len_m, self.raceline_s, der=1)
    #     heading_vec = np.arctan2(drr[1], drr[0])
    #     vv = self.sToV(ss)
    #     top_speed = 10
    #     vv[vv > top_speed] = top_speed

    #     # parameter, distance along track
    #     self.ss = ss
    #     self.raceline_points = np.array(rr)
    #     self.raceline_headings = heading_vec
    #     self.raceline_velocity = vv

    #     # describe track boundary as offset from raceline
    #     self.create_boundary()
    #     self.discretized_raceline = np.vstack(
    #         [self.raceline_points,
    #          self.raceline_headings,
    #          vv,
    #          self.raceline_left_boundary,
    #          self.raceline_right_boundary]).T
    #     return

    def precise_track_boundary(self, coord, heading):
        """ Return the distance to left and right boundary (left, right) """
        raise NotImplementedError

    def create_boundary(self, coord_vec: np.ndarray, heading_vec: np.ndarray, show=False):
        '''
         Find margin to left/right boundary along a reference path 
         using self.precise_track_boundary()
         Args:
            coord_vec: (N,2), x,y coordinates.
            heading_vec: (N,), heading in rad. 
                Left/right are referenced from normal direction of the heading
         Return:
            retval: (N,2), left, right margin in meters
        '''
        N = coord_vec.shape[0]
        assert coord_vec.shape == (N, 2)
        assert heading_vec.shape == (N,)

        left_boundary = []
        right_boundary = []

        left_boundary_points = []
        right_boundary_points = []
        for i in range(N):
            # find normal direction
            coord = coord_vec[i]
            heading = heading_vec[i]

            left, right = self.precise_track_boundary(coord, heading)
            left_boundary.append(left)
            right_boundary.append(right)

            # debug boundary points
            left_point = (coord[0] + left * cos(heading+np.pi/2),
                          coord[1] + left * sin(heading+np.pi/2))
            right_point = (coord[0] + right * cos(heading-np.pi/2),
                           coord[1] + right * sin(heading-np.pi/2))

            left_boundary_points.append(left_point)
            right_boundary_points.append(right_point)

        retval = np.vstack([left_boundary, right_boundary]).T
        assert retval.shape == (N, 2)

        if show:
            img = self.draw_track()
            img = self.draw_raceline(self.data.raceline_s, self.data.raceline_len_m, img=img)
            img = self.draw_polyline(
                left_boundary_points, lineColor=(0, 255, 0), img=img)
            img = self.draw_polyline(
                right_boundary_points, lineColor=(0, 0, 255), img=img)
            plt.imshow(img)
            plt.show()
            return img
        return retval

    def cart_to_curv(self, cart: CartesianState, guess_s: float = None) -> CurvilinearState:
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
                           bounds=((guess_s-ds, guess_s+ds),))
        else:
            fit = minimize(dist, x0=guess_s, method='L-BFGS-B',
                           bounds=((guess_s-0.2, guess_s+0.2),))

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
        r = np.array(splev(curv.progress % data.raceline_len_m,
                     data.raceline_s, der=0))
        dr = np.array(splev(curv.progress % data.raceline_len_m,
                      data.raceline_s, der=1))
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

    @staticmethod
    def reparam_raceline(raceline, bound):
        """ Reparameterize a spline to curve length.
        Args:
            raceline: tck object from splprep
            bound: The end range of the original parameter. Assume parameter start from 0
        Return:
            raceline_s: tck object of re-parameterized spline
            raceline_len_m: The curve length of the raceline
        """
        s_vec = [0]
        n_steps = 1000
        uu = np.linspace(0, bound, n_steps+1)

        def dist(a, b):
            return ((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5
        path_len = 0
        for i in range(n_steps):
            (x_i, y_i) = splev(uu[i % n_steps], raceline, der=0)
            (x_i_1, y_i_1) = splev(uu[(i+1) % n_steps], raceline, der=0)
            # distance between two steps
            ds = dist((x_i, y_i), (x_i_1, y_i_1))
            path_len += ds
            s_vec.append(path_len)
        raceline_len_m = path_len

        ss = np.array(s_vec)
        assert np.all(np.diff(ss) > 0)

        rr = splev(uu % bound, raceline)
        raceline_s, _ = splprep(rr, u=ss, s=0, per=1)

        return raceline_s, raceline_len_m

    @staticmethod
    def generate_speed_profile(raceline_s,
                               raceline_len_m,
                               mu: float = 0.7,
                               acc_max_fun=lambda x: 1.5,
                               dec_max_fun=lambda x: 1.5,
                               n_steps=1000,
                               show=False):
        """ Generate speed profile given traction constraints, braking/acceleration limit.

        Args:
            mu: Coefficient of friction for the radius of traction circle. maximum traction = mu*g
            acc_max_fun: Given velocity, provide maximum acceleration available. ~3.3m/s2 for miniz
            dec_max_fun: Given velocity, provide maximum deceleration available. ~4.5m/s2 for miniz
            n_steps: Discretization steps,
            show: If True, plot speed profile
        Output:
            SpeedProfileOutput
        """
        g = 9.81
        raceline = raceline_s
        # u values for control points
        ss = np.linspace(0, raceline_len_m, n_steps+1)

        # let raceline curve be r(u)
        # dr = r'(u), parameterized with uu
        dr = np.array(splev(ss, raceline, der=1))
        # ddr = r''(u)
        ddr = np.array(splev(ss, raceline, der=2))

        def _norm(x):
            return np.linalg.norm(x, axis=0)

        # Radius of curvature can be calculated as R = |y'|^3/sqrt(|y'|^2*|y''|^2-(y'*y'')^2)
        # curvature = 1/R, always positive
        curvature = (_norm(dr)**2*_norm(ddr) ** 2 -
                     np.sum(dr*ddr, axis=0)**2)**0.5 / _norm(dr)**3

        # First pass, based on lateral acceleration
        v1 = (mu*g/curvature)**0.5

        def dist(a, b):
            return ((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5
        # Second pass, based on engine capacity and available longitudinal traction
        # Start from the index with lowest speed
        min_xx = np.argmin(v1)
        v2 = np.zeros_like(v1)
        v2[min_xx] = v1[min_xx]
        for i in range(min_xx, min_xx+n_steps):
            # lateral acc at next step if the car mainains speed
            a_lat = v2[i % n_steps]**2*curvature[(i+1) % n_steps]

            # is there available traction for acceleration?
            if ((mu*g)**2-a_lat**2) > 0:
                a_lon_available_traction = ((mu*g)**2-a_lat**2)**0.5
                # constrain with motor capacity
                a_lon = min(acc_max_fun(
                    v2[i % n_steps]), a_lon_available_traction)

                (x_i, y_i) = splev(ss[i % n_steps], raceline, der=0)
                (x_i_1, y_i_1) = splev(ss[(i+1) %
                                          n_steps], raceline, der=0)
                # distance between two steps
                ds = dist((x_i, y_i), (x_i_1, y_i_1))
                # assume vehicle accelerate uniformly between the two steps
                v2[(i+1) % n_steps] = min((v2[i % n_steps] **
                                           2 + 2*a_lon*ds)**0.5, v1[(i+1) % n_steps])
            else:
                v2[(i+1) % n_steps] = v1[(i+1) % n_steps]

        v2[-1] = v2[0]
        # Third pass, backwards for braking capacity (deceleration)
        min_xx = np.argmin(v2)
        v3 = np.zeros_like(v1)
        v3[min_xx] = v2[min_xx]
        for i in np.linspace(min_xx, min_xx-n_steps, n_steps+2):
            i = int(i)
            a_lat = v3[i % n_steps]**2*curvature[(i-1+n_steps) % n_steps]
            a_lon_available_traction = abs((mu*g)**2-a_lat**2)**0.5
            a_lon = min(dec_max_fun(v3[i % n_steps]), a_lon_available_traction)

            (x_i, y_i) = splev(ss[i % n_steps], raceline, der=0)
            (x_i_1, y_i_1) = splev(ss[(i-1+n_steps) %
                                      n_steps], raceline, der=0)
            # distance between two steps
            ds = dist((x_i, y_i), (x_i_1, y_i_1))
            # print(ds)
            v3[(i-1+n_steps) % n_steps] = min((v3[i % n_steps] **
                                               2 + 2*a_lon*ds)**0.5, v2[(i-1+n_steps) % n_steps])
            # print(v3[(i-1+n_steps)%n_steps],v2[(i-1+n_steps)%n_steps])

        v3[-1] = v3[0]

        # v_fun = interp1d(ss, v3, kind='cubic')
        tck, _ = splprep([v3], u=ss, s=0, k=3)
        max_v = max(v3)
        min_v = min(v3)

        # three pass of velocity profile
        if show:
            # p0, = plt.plot(curvature, label='curvature')
            p1, = plt.plot(v1, label='1st pass')
            p2, = plt.plot(v2, label='2nd pass')
            p3, = plt.plot(v3, label='3rd pass')
            plt.legend(handles=[p1, p2, p3])
            plt.show()

        return SpeedProfileOutput(tck, min_v, max_v)

    """
    @deprecated
    def verify_speed_profile(self, *, speed_profile_fun, mu=0.7, show_traction_circle=False):
        # calculate theoretical lap time
        g = 9.81
        t_total = 0
        path_len = 0
        xx = np.linspace(0, self.track_length_grid, n_steps+1)
        def dist(a, b): return ((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5
        vv = speed_profile_fun(xx)
        for i in range(n_steps):
            (x_i, y_i) = splev(xx[i % n_steps], self.rcp_raceline, der=0)
            (x_i_1, y_i_1) = splev(xx[(i+1) % n_steps], self.rcp_raceline, der=0)
            # distance between two steps
            ds = dist((x_i, y_i), (x_i_1, y_i_1))
            path_len += ds
            t_total += ds/(vv[i % n_steps]+vv[(i+1) % n_steps])*2

        print_info('Theoretical value:')
        print_info('\t min speed = %.2fm/s' % min(vv))
        print_info('\t top speed = %.2fm/s' % max(vv))
        print_info('\t total time = %.2fs' % t_total)
        print_info('\t path len = %.2fm' % path_len)

        # cartesian distance from two u(parameter)
        def distuu(u1, u2): return dist(
            splev(u1, self.rcp_raceline, der=0), splev(u2, self.rcp_raceline, der=0))

        vel_vec = []
        ds_vec = []

        # get velocity at each point
        for i in range(n_steps):
            # tangential direction
            tan_dir = splev(xx[i], self.rcp_raceline, der=1)
            tan_dir = np.array(tan_dir/np.linalg.norm(tan_dir))
            vel_now = vv[i] * tan_dir
            vel_vec.append(vel_now)

        vel_vec = np.array(vel_vec)

        lat_acc_vec = []
        lon_acc_vec = []
        dtheta_vec = []
        theta_vec = []
        v_vec = []
        dt_vec = []

        # get lateral and longitudinal acceleration
        for i in range(n_steps-1):

            theta = np.arctan2(vel_vec[i, 1], vel_vec[i, 0])
            theta_vec.append(theta)

            dtheta = np.arctan2(vel_vec[i+1, 1], vel_vec[i+1, 0]) - theta
            dtheta = (dtheta+np.pi) % (2*np.pi)-np.pi
            dtheta_vec.append(dtheta)

            speed = np.linalg.norm(vel_vec[i])
            next_speed = np.linalg.norm(vel_vec[i+1])
            v_vec.append(speed)

            dt = distuu(xx[i], xx[i+1])/speed
            dt_vec.append(dt)

            lat_acc_vec.append(speed*dtheta/dt)
            lon_acc_vec.append((next_speed-speed)/dt)

        dt_vec = np.array(dt_vec)
        lon_acc_vec = np.array(lon_acc_vec)
        lat_acc_vec = np.array(lat_acc_vec)

        # get acc_vector, track frame
        dt_vec2 = np.vstack([dt_vec, dt_vec]).T
        acc_vec = np.diff(vel_vec, axis=0)
        acc_vec = acc_vec / dt_vec2

        # plot acceleration vector cloud
        # with x,y axis being vehicle frame, x lateral
        if (show_traction_circle):
            p0, = plt.plot(lat_acc_vec, lon_acc_vec, '*', label='data')

            # draw the traction circle
            cc = np.linspace(0, 2*np.pi)
            circle = np.vstack([np.cos(cc), np.sin(cc)])*mu*g
            p1, = plt.plot(circle[0, :], circle[1, :], label='1g')
            plt.gcf().gca().set_aspect('equal', 'box')
            plt.xlim(-12, 12)
            plt.ylim(-12, 12)
            plt.xlabel('Lateral Acceleration')
            plt.ylabel('Longitudinal Acceleration')
            plt.legend(handles=[p0, p1])
            plt.show()

            p0, = plt.plot(theta_vec, label='theta')
            p1, = plt.plot(v_vec, label='v')
            p2, = plt.plot(dtheta_vec, label='dtheta')
            acc_mag_vec = (acc_vec[:, 0]**2+acc_vec[:, 1]**2)**0.5
            p0, = plt.plot(acc_mag_vec, '*', label='acc vec2mag')
            p1, = plt.plot((lon_acc_vec**2+lat_acc_vec**2)
                           ** 0.5, label='acc mag')

            p2, = plt.plot(lon_acc_vec, label='longitudinal')
            p3, = plt.plot(lat_acc_vec, label='lateral')
            plt.legend(handles=[p0, p1])
            plt.show()
        print('theoretical laptime %.2f' % t_total)

        self.reconstruct_raceline()
        return t_total
    """
