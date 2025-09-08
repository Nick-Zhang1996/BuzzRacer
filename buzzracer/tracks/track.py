"""Base class for RCPTrack and Skidpad This class provides API for interacting
with a Track object A track object provides information on the trajectory and
provide access for drawing the track."""
import os.path
import pickle
from math import cos, sin
from typing import Callable

import cv2
import numpy as np
from scipy.interpolate import splev
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from buzzracer.common import ConfigObject, wrap
from buzzracer.types import CurvilinearState, CartesianState


class Track(ConfigObject):
    def __init__(self, main, config):
        self.main = main
        # the following variables need to be overriden in subclass initilization
        # pixels per meter
        self.resolution = None
        self.discretized_raceline_len = 1024
        self.raceline_len_m: float = 0.0
        ''' Total length of raceline in meters'''
        self.raceline_s = None
        ''' Spline to map track progress to raceline points.
        The tck coefficients from splprep, use as track_point = splev(s_m, self.raceline_s)'''
        self.sToV: Callable = lambda s: 0.0
        ''' Function to provide reference velocity given raceline s_m'''
        self.precise_track_boundary: Callable = lambda coord, heading: (0, 0)
        ''' left, right = self.precise_track_boundary(coord, heading) '''
        self.curvature_s = lambda s: 0.0
        ''' Function to map track progress to signed curvature of raceline, 
        The tck coefficients from splprep, use as curvature = splev(s_m, self.raceline_s)'''

        self.ss: np.ndarray = np.array(0)
        ''' np.linspace(0, self.raceline_len_m, self.discretized_raceline_len)'''
        self.raceline_points: np.ndarray = np.array(0)
        ''' dim:(len, 2) splev(ss % self.raceline_len_m, self.raceline_s) '''
        self.raceline_headings: np.ndarray = np.array(0)
        ''' dim:(len,) An array of reference headings '''
        self.raceline_velocity: np.ndarray = np.array(0)
        ''' dim:(len,) An array of reference velocity, from self.sToV(ss)'''
        self.discretized_raceline: np.ndarray = np.array(0)
        ''' dim: (len, 5)
        [raceline_points, raceline_headings, vv, raceline_left_boundary, raceline_right_boundary]
        '''
        self.raceline_left_boundary: np.ndarray = np.array(0)
        ''' dim:(len,) An array of distances from ref raceline to left boundary'''
        self.raceline_right_boundary: np.ndarray = np.array(0)
        ''' dim:(len,) An array of distances from ref raceline to right boundary'''

        # obstacles
        self.obstacle = False
        self.obstacle_count = 0
        self.obstacle_filename = None
        self.obstacle_radius = None
        self.obstacles = None
        ''' dim:[n_obstacles, 2], coordinate of obstacles'''

        # track dimension, in meters
        self.x_limit = None
        self.y_limit = None

        ConfigObject.__init__(self, config)

    def init(self):
        self.set_up_obstacles()

    # NOTE funs that need to move to this file TODO

    # NOTE need to be overridden in each subclass Track

    # draw a raceline
    def draw_raceline(self, img=None):
        raise NotImplementedError

    # draw a picture of the track
    def draw_track(self, img=None, show=False):
        raise NotImplementedError

    def local_trajectory(self, state):
        raise NotImplementedError

    # NOTE universal function for all Track classes
    def set_resolution(self, res):
        self.resolution = res
        return

    # determine if an coordinate is outside of track boundary, used in watchdog
    def is_outside(self, coord):
        grace = 1.0
        x, y = coord
        return x < -grace or y < -grace or x > self.x_limit+grace or y > self.y_limit+grace

    # check if vehicle is currently in collision with obstacle
    # only give index of the first obstacle if multiple obstacle is in collision
    def is_in_obstacle(self, state):
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

    # NOTE plotting related
    def m2canvas(self, coord):
        x_new = int(np.clip(coord[0], 0, self.x_limit) * self.resolution)
        y_new = int(
            (self.y_limit-np.clip(coord[1], 0, self.y_limit)) * self.resolution)
        return (x_new, y_new)

    # draw a circle on canvas at coord
    def draw_circle(self, img, coord, radius_m, color=(0, 0, 0)):
        src = self.m2canvas(coord)
        radius_pix = int(radius_m * self.resolution)
        img = cv2.circle(img, src, radius_pix, color, -1)
        return img

    def plot_obstacles(self, img=None):
        if not self.obstacle:
            return img
        if img is None:
            if not self.main.visualization.update_visualization.is_set():
                return
            img = self.main.visualization.visualization_img

        # plot obstacles
        for obs in self.obstacles:
            img = self.draw_circle(img, obs, 0.1, color=(255, 100, 100))
        for car in self.main.cars:
            has_collided, obs_id = self.is_in_obstacle(car.states)
            if has_collided:
                # plot obstacle in collision red
                img = self.draw_circle(
                    img, self.obstacles[obs_id], 0.1, color=(100, 100, 255))

        if img is None:
            self.main.visualization.visualization_img = img
        else:
            return img

    # draw a polynomial line defined in track space
    # points: a list of coordinates in format (x,y)
    def draw_polyline(self, points, img=None, lineColor=(0, 0, 255), thickness=3):

        if img is None:
            img = np.zeros([int(self.resolution*self.x_limit),
                           int(self.resolution*self.y_limit), 3], dtype='uint8')

        pts = [self.m2canvas(point) for point in points]
        for i in range(len(points)-1):
            p1 = np.array(pts[i])
            p2 = np.array(pts[i+1])
            if pts[i] is None or pts[i+1] is None:
                continue
            img = cv2.line(img, tuple(p1), tuple(
                p2), color=lineColor, thickness=thickness)
        return img

    def draw_trajectory(self, traj_points, img=None, lineColor=(0, 0, 255), thickness=3):
        return self.draw_polyline(traj_points[:, 1:3], img, lineColor, thickness)

    # draw ONE arrow, unit: meter, coord sys: dimensioned
    # source: source of arrow, in meter
    # orientation, radians from x axis, ccw positive
    # length: in pixels, though this is only qualitative
    def draw_arrow(self, source, orientation, length, color=(0, 0, 0), thickness=2, img=None):
        if img is None:
            img = np.zeros([int(self.resolution*self.x_limit),
                           int(self.resolution*self.y_limit), 3], dtype='uint8')

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
        if not self.obstacle:
            self.obstacle_count = 0
            return
        filename = os.path.join(self.main.basedir, self.obstacle_filename)

        if os.path.isfile(filename):
            with open(filename, 'rb') as f:
                obstacles = pickle.load(f)
            self.obstacle_count = obstacles.shape[0]
            self.print_ok(
                f'loading obstacles at {filename}, count = {obstacles.shape[0]}')
            self.print_ok(
                ' if you wish to create new obstacles,'
                'remove current obstacle file or change parameter obstacle_filename')
        else:
            self.print_ok(
                f'generating new obstacles, count = {self.obstacle_count}')
            obstacles = np.random.random((self.obstacle_count, 2))
            # save obstacles
            if not filename is None:
                with open(filename, 'wb') as f:
                    pickle.dump(obstacles, f)
                self.print_ok(f'saved obstacles at {filename}')

        # spread obstacle to entire track
        obstacles[:, 0] *= self.x_limit
        obstacles[:, 1] *= self.y_limit

        self.obstacles = obstacles

    def prepare_discretized_raceline(self):
        """depends on self.raceline_s, self.raceline_len_m."""
        ss = np.linspace(0, self.raceline_len_m, self.discretized_raceline_len)
        rr = splev(ss % self.raceline_len_m, self.raceline_s, der=0)
        drr = splev(ss % self.raceline_len_m, self.raceline_s, der=1)
        heading_vec = np.arctan2(drr[1], drr[0])
        vv = self.sToV(ss)
        top_speed = 10
        vv[vv > top_speed] = top_speed

        # parameter, distance along track
        self.ss = ss
        self.raceline_points = np.array(rr)
        self.raceline_headings = heading_vec
        self.raceline_velocity = vv

        # describe track boundary as offset from raceline
        self.create_boundary()
        self.discretized_raceline = np.vstack(
            [self.raceline_points,
             self.raceline_headings,
             vv,
             self.raceline_left_boundary,
             self.raceline_right_boundary]).T
        return

    def create_boundary(self, show=False):
        '''
         construct a (self.discretized_raceline_len * 2) vector
         to record the left and right track boundary as an offset to the discretized raceline
         depends on self.precise_track_boundary(coord,heading)
        '''
        left_boundary = []
        right_boundary = []

        left_boundary_points = []
        right_boundary_points = []

        for i in range(self.discretized_raceline_len):
            # find normal direction
            coord = self.raceline_points[:, i]
            heading = self.raceline_headings[i]

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

            # DEBUG
            # plot left/right boundary
            # left_point = (coord[0] + left * cos(heading+np.pi/2),coord[1] \
            # + left * sin(heading+np.pi/2))
            # right_point = (coord[0] + right * cos(heading-np.pi/2),coord[1] \
            # + right * sin(heading-np.pi/2))
            # img = self.draw_track()
            # img = self.draw_raceline(img = img)
            # img = self.draw_point(img,coord,color=(0,0,0))
            # img = self.draw_point(img,left_point,color=(0,0,0))
            # img = self.draw_point(img,right_point,color=(0,0,0))
            # plt.imshow(img)
            # plt.show()

        self.raceline_left_boundary = left_boundary
        self.raceline_right_boundary = right_boundary

        if show:
            img = self.draw_track()
            img = self.draw_raceline(img=img)
            img = self.draw_polyline(
                left_boundary_points, lineColor=(0, 255, 0), img=img)
            img = self.draw_polyline(
                right_boundary_points, lineColor=(0, 0, 255), img=img)
            plt.imshow(img)
            plt.show()
            return img
        return

    def cart_to_curv(self, cart: CartesianState, guess_s: float = None) -> CurvilinearState:
        """Transform cartesian states to curvilinear states, relies on
        self.raceline_s.

        Args:
            cart: Cartesian state
            guess_s: estimated s (progress along ref curve)
        Returns:
            curv: curvilinear state

        """

        def dist(s):
            val = np.linalg.norm(
                np.array(splev(s % self.raceline_len_m,
                         self.raceline_s)).flatten()
                - np.array([cart.x, cart.y])
            )
            return val

        if (guess_s is None):
            # initial guess to avoid local minima
            xx = np.linspace(0.0, self.raceline_len_m, 100)
            yy = [dist(x) for x in xx]
            guess_s = xx[np.argmin(yy)]
            ds = 2*self.raceline_len_m/100
            fit = minimize(dist, x0=guess_s, method='L-BFGS-B',
                           bounds=((guess_s-ds, guess_s+ds),))
        else:
            fit = minimize(dist, x0=guess_s, method='L-BFGS-B',
                           bounds=((guess_s-0.2, guess_s+0.2),))

        s = fit.x[0]

        r = np.array(splev(s % self.raceline_len_m,
                     self.raceline_s, der=0))
        dr = np.array(splev(s % self.raceline_len_m,
                      self.raceline_s, der=1))
        dr = dr/np.linalg.norm(dr)
        n = np.cross(dr, np.array([cart.x, cart.y]) - r)
        phi = wrap(cart.heading - np.arctan2(dr[1], dr[0]))
        return CurvilinearState(progress=s,
                                lateral_err=n,
                                rel_heading=phi,
                                v_forward=cart.v_forward,
                                v_sideway=cart.v_sideway,
                                rel_omega=cart.omega)

    def curv_to_cart(self, curv: CurvilinearState) -> CartesianState:
        """Transform curvilinear state to cartesian state. Need
        self.raceline_s.

        Args:
            curv: Curvilinear state
        Returns:
            cart: Transformed cartesian state

        """
        r = np.array(splev(curv.progress % self.raceline_len_m,
                     self.raceline_s, der=0))
        dr = np.array(splev(curv.progress % self.raceline_len_m,
                      self.raceline_s, der=1))
        dr = dr/np.linalg.norm(dr)

        # ccw 90 deg
        A = np.array([[0, -1], [1, 0]])
        x, y = r + (A @ dr)*curv.lateral_err
        ref_heading = np.arctan2(dr[1], dr[0])
        heading = wrap(curv.rel_heading + ref_heading)
        return CartesianState(x=x,
                              y=y,
                              heading=heading,
                              v_forward=curv.v_forward,
                              v_sideway=curv.v_sideway,
                              omega=curv.rel_omega)
