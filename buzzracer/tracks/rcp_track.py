''' Subclass of Track for RCP style modular track.
Desperately need cleanup and refactoring '''

from __future__ import annotations
import os
import pickle
from math import atan2, sin, cos, pi, copysign, isnan
from bisect import bisect
from typing import NamedTuple
from enum import Enum
from collections.abc import Callable

import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import splprep, splev, interp1d
from scipy.optimize import minimize


from buzzracer.common import BASEDIR, get_logger
from buzzracer.tracks.track import Track
from buzzracer.utilities.execution_timer import ExecutionTimer


class Dir(Enum):
    ''' Directions for grid'''
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

    @staticmethod
    def from_char(char: str) -> Dir:
        _char_to_dir = {'u': Dir.UP, 'd': Dir.DOWN,
                        'l': Dir.LEFT, 'r': Dir.RIGHT}
        try:
            return _char_to_dir[char]
        except KeyError:
            logger.error('unexpected value in description')
            raise

    @staticmethod
    def move(initial: tuple[int, int], direction: Dir) -> tuple[int, int]:
        _dir_to_tuple = {Dir.UP: (0, 1), Dir.DOWN: (
            0, -1), Dir.RIGHT: (1, 0), Dir.LEFT: (-1, 0)}
        move = _dir_to_tuple[direction]
        return (initial[0]+move[0], initial[1]+move[1])


class GridSize(NamedTuple):
    rows: int
    cols: int


class Node:
    ''' One tile in RCPTrack '''
    _lut = {'WE': [(Dir.RIGHT, Dir.RIGHT), (Dir.LEFT, Dir.LEFT)],  # Left/Right straight tile
            # Up/Down straight tile
            'NS': [(Dir.UP, Dir.UP), (Dir.DOWN, Dir.DOWN)],
            # Apex at south east
            'SE': [(Dir.UP, Dir.RIGHT), (Dir.LEFT, Dir.DOWN)],
            'SW': [(Dir.UP, Dir.LEFT), (Dir.RIGHT, Dir.DOWN)],
            'NE': [(Dir.DOWN, Dir.RIGHT), (Dir.LEFT, Dir.UP)],
            'NW': [(Dir.RIGHT, Dir.UP), (Dir.DOWN, Dir.LEFT)]
            }

    def __init__(self, previous: Node = None, entrydir: Node = None):
        self.entry: Dir | None = entrydir
        ''' entry direction '''
        self.previous: Node | None = previous
        ''' previous node '''
        self.next: Node | None = None
        ''' next node '''
        self.exit: Dir | None = None
        ''' exit direction '''
        return

    def set_exit(self, val: Dir):
        ''' Set exit direction for this node'''
        self.exit = val

    def set_entry(self, val: Dir):
        ''' Set entry direction for this node'''
        self.entry = val

    def __repr__(self):
        return f'Node(entry={self.entry}, exit={self.exit})'

    def to_name(self) -> str:
        signature = (self.entry, self.exit)
        for name, signature_vec in Node._lut.items():
            if signature in signature_vec:
                return name
        raise RuntimeError(f"Invalid entry/exit tuple,{signature}")


class SpeedProfileOutput(NamedTuple):
    ''' u -> reference_speed, 0 < u < len(self.ctrl_pts)]'''
    speed_profile_fun: Callable
    min_v: float
    max_v: float


class LocalTrajOutput(NamedTuple):
    ref_point: np.ndarray
    lateral_err: float
    heading_err: float
    curvature: float
    v_target: float
    progress: float


logger = get_logger('RCPTrack')


class RCPTrack(Track):
    # TODO: remove main, set main as class variable for Track
    def __init__(self, main=None, config=None):
        Track.__init__(self, main, config)
        self.t = ExecutionTimer(True)
        # TODO set variable directly
        self.resolution = 200
        ''' resolution : pixels per grid side length '''
        self.debug = {}  # TODO: remove
        ''' Dictionary for debugging'''

        self.scale: float = 0.6
        ''' Edge length of one grid in meters (default 0.6m)'''
        self.gridsize: GridSize = GridSize(0, 0)
        ''' (rows, cols), grid size of the track '''
        self.track_length_grid: int = 0
        ''' Total grid length of the track'''
        self.grid_sequence: list[tuple[int, int]] = []
        ''' List of the grid (row, col) that defines the track'''
        self.grid: list[list[Node | str | None]] = []
        self.x_limit: float = 0
        ''' X-direction bound in meters, i.e. width of track'''
        self.y_limit: float = 0
        ''' Y-direction bound in meters, i.e. height of track'''

    def init_track(self,
                   description: str,
                   gridsize: GridSize,
                   start_grid: tuple[int, int] = (0, 0),
                   scale: float = 0.6):
        ''' Build an RCP style track.
        Args:
            description: directions to go to reach next grid.
                includes u(p), r(ight),d(own), l(eft)
                e.g For a basic 3 by 3 square track like this
                        |-|
                        | |
                        |_|
                The trajectory description, starting from the bottom left corner (origin), clockwise
                is 'uurrddll'. The direction does not matter
            gridsize: Gridsize(rows, cols), size of the track
            start_grid: The grid to start the track (row, col)
            scale: Edge length of one grid in meters (default 0.6m)
        '''

        self.scale = scale
        self.gridsize = gridsize
        self.track_length_grid = len(description)
        self.grid_sequence = []

        self.x_limit = self.gridsize.cols*self.scale
        self.y_limit = self.gridsize.rows*self.scale

        grid = [[None for _ in range(gridsize.rows)]
                for _ in range(gridsize.cols)]

        current_index = start_grid
        self.grid_sequence.append(current_index)
        current_node = grid[start_grid[0]][start_grid[1]] = Node()
        for i, dir_char in enumerate(description):
            move_dir = Dir.from_char(dir_char)
            current_node.set_exit(move_dir)
            next_index = Dir.move(current_index, move_dir)
            self.grid_sequence.append(next_index)

            if next_index == start_grid:
                grid[start_grid[0]][start_grid[1]].set_entry(move_dir)
                if i != len(description)-1:
                    raise RuntimeError('Description str does not lead to start grid',
                                       ' or track has intersections')
                break

            # assert description does not go beyond defined grid size
            assert current_index[0] < gridsize.cols
            assert current_index[1] < gridsize.rows

            next_node = Node(previous=current_node, entrydir=move_dir)
            grid[next_index[0]][next_index[1]] = next_node
            current_node = next_node
            current_index = next_index

        # process the linked list, replace with the following
        # straight segment = WE(EW), NS(SN)
        # curved segment = SE,SW,NE,NW, orientation of apex wrt center of grid
        for i in range(gridsize[1]):
            for j in range(gridsize[0]):
                node = grid[i][j]
                if node is None:
                    continue
                grid[i][j] = grid[i][j].to_name()

        self.grid = grid
        return

    def draw_track(self, img=None, show=False):
        ''' show a picture of the track '''
        color_side = (255, 0, 0)
        # boundary width / grid width
        deadzone = 0.087
        gs = int(self.resolution * self.scale)

        # prepare straight section (WE)
        straight = 255*np.ones([gs, gs, 3], dtype='uint8')
        straight = cv2.rectangle(
            straight, (0, 0), (gs-1, int(deadzone*gs)), color_side, -1)
        straight = cv2.rectangle(
            straight, (0, int((1-deadzone)*gs)), (gs-1, gs-1), color_side, -1)
        WE = straight

        # prepare turn section (SE)
        turn = 255*np.ones([gs, gs, 3], dtype='uint8')
        turn = cv2.rectangle(
            turn, (0, 0), (int(deadzone*gs), gs-1), color_side, -1)
        turn = cv2.rectangle(
            turn, (0, 0), (gs-1, int(deadzone*gs)), color_side, -1)
        turn = cv2.rectangle(
            turn, (0, 0), (int(0.5*gs), int(0.5*gs)), color_side, -1)
        turn = cv2.circle(turn, (int(0.5*gs), int(0.5*gs)),
                          int((0.5-deadzone)*gs), (255, 255, 255), -1)
        turn = cv2.circle(turn, (gs-1, gs-1), int(deadzone*gs), color_side, -1)
        SE = turn

        # prepare canvas
        rows = self.gridsize[0]
        cols = self.gridsize[1]
        if img is None:
            # white background
            # img = 255*np.ones([gs*rows,gs*cols,3],dtype='uint8')
            img = np.zeros([gs*rows, gs*cols, 3], dtype='uint8')
            img[:, :, 0] = 255
        lookup_table = {'SE': 0, 'SW': 270, 'NE': 90, 'NW': 180}
        for i in range(cols):
            for j in range(rows):
                signature = self.grid[i][rows-1-j]
                if signature is None:
                    continue

                if signature == 'WE':
                    img[j*gs:(j+1)*gs, i*gs:(i+1)*gs] = WE
                    continue
                elif signature == 'NS':
                    M = cv2.getRotationMatrix2D((gs/2, gs/2), 90, 1.01)
                    NS = cv2.warpAffine(WE, M, (gs, gs))
                    img[j*gs:(j+1)*gs, i*gs:(i+1)*gs] = NS
                    continue
                elif signature in lookup_table:
                    M = cv2.getRotationMatrix2D(
                        (gs/2, gs/2), lookup_table[signature], 1.01)
                    dst = cv2.warpAffine(SE, M, (gs, gs))
                    img[j*gs:(j+1)*gs, i*gs:(i+1)*gs] = dst
                    continue
                else:
                    print('err, unexpected track designation : ' + signature)

        # some rotation are not perfect and leave a black gap
        img = cv2.medianBlur(img, 5)
        return img

    # create a heuristic raceline
    # this function stores result in self.raceline
    # Note self.raceline takes u, a dimensionless variable that corresponds to
    # the control point on track
    # rance of u is (0,len(self.ctrl_pts) with 1 corresponding to the exit point out of
    # the starting grid,
    # both 0 and len(self.ctrl_pts) pointing to the entry ctrl point for the starting grid
    # and gives a pair of coordinates in METER
    def init_raceline(self, start: tuple[int, int], start_direction: Dir, offset=None):
        ''' Init a raceline from current track.
        Args:
            start: which grid to start from, e.g. (3,3), origin is at bottom left (0,0)
                    you MUST start on a straight section
            start_direction: which direction to ENTER start grid.
                note use the direction for ENTERING that grid element
                e.g. 'l' or 'd' for a NE oriented turn
            offset: np.array of size self.track_length_grid, lateral offset for each control grid
        '''
        self.ctrl_pts = []
        self.ctrl_pts_w = []
        # TODO continue the refactor
        origin_seq = None
        start_seq = None
        for i, seq in enumerate(self.grid_sequence):
            if start[0] == seq[0] and start[1] == seq[1]:
                start_seq = i
            if 0 == seq[0] and 0 == seq[1]:
                origin_seq = i
        # starting from [start], the sequence number for origin (0,0)
        self.origin_seq_no = (origin_seq - start_seq) % self.track_length_grid

        if offset is None:
            offset = np.zeros(self.track_length_grid)

        # provide exit direction given signature and entry direction
        lookup_table = {'WE': ['rr', 'll'], 'NS': ['uu', 'dd'], 'SE': [
            'ur', 'ld'], 'SW': ['ul', 'rd'], 'NE': ['dr', 'lu'], 'NW': ['ru', 'dl']}
        # provide correlation between direction (character) and directional vector
        lookup_table_dir = {'u': (0, 1), 'd': (
            0, -1), 'r': (1, 0), 'l': (-1, 0)}
        # provide right hand direction, this is for specifying offset direction
        lookup_table_right = {
            'u': (1, 0), 'd': (-1, 0), 'r': (0, -1), 'l': (0, 1)}
        # provide apex direction
        turn_offset_toward_center = {
            'SE': (1, -1), 'NE': (1, 1), 'SW': (-1, -1), 'NW': (-1, 1)}
        turns = ['SE', 'SW', 'NE', 'NW']

        def center(x, y):
            return [(x+0.5)*self.scale, (y+0.5)*self.scale]

        def left(x, y):
            return [(x)*self.scale, (y+0.5)*self.scale]

        def right(x, y):
            return [(x+1)*self.scale, (y+0.5)*self.scale]

        def up(x, y):
            return [(x+0.5)*self.scale, (y+1)*self.scale]

        def down(x, y):
            return [(x+0.5)*self.scale, (y)*self.scale]

        # direction of entry
        entry = start_direction
        current_coord = np.array(start, dtype='uint8')
        signature = self.grid[current_coord[0]][current_coord[1]]
        # find the previous signature, reverse entry to find ancestor
        # the precedent grid for start grid is also the final grid
        final_coord = current_coord - lookup_table_dir[start_direction]
        self.start_pos = ((0.5+start[0])*self.scale, (0.5+start[1])*self.scale)

        dire = lookup_table_dir[start_direction]
        self.start_dir = np.arctan2(dire[1], dire[0])

        # for referencing offset
        index = 0
        while True:
            signature = self.grid[current_coord[0]][current_coord[1]]

            # lookup exit direction
            for record in lookup_table[signature]:
                if record[0] == entry:
                    exit = record[1]
                    break

            # find the coordinate of the exit point
            # offset from grid center to centerpoint of exit boundary
            # go half a step from center toward exit direction
            exit_ctrl_pt = np.array(lookup_table_dir[exit], dtype='float')/2
            exit_ctrl_pt += current_coord
            exit_ctrl_pt += np.array([0.5, 0.5])
            # apply offset, offset range (-1,1)
            exit_ctrl_pt += offset[index] * \
                np.array(lookup_table_right[exit], dtype='float')/2
            index += 1

            exit_ctrl_pt *= self.scale
            self.ctrl_pts.append(exit_ctrl_pt.tolist())

            current_coord = current_coord + lookup_table_dir[exit]
            entry = exit

            last_signature = signature

            if all(start == current_coord):
                break

        # add end point to the beginning,
        # otherwise splprep will replace pts[-1] with pts[0] for a closed loop
        # This ensures that splev(u=0) gives us the beginning point
        pts = np.array(self.ctrl_pts)
        # start_point = np.array(self.ctrl_pts[0])
        # pts = np.vstack([pts,start_point])
        end_point = np.array(self.ctrl_pts[-1])
        pts = np.vstack([end_point, pts])

        # weights = np.array(self.ctrl_pts_w + [self.ctrl_pts_w[-1]])

        # s= smoothing factor
        # a good s value should be found in the range (m-sqrt(2*m),m+sqrt(2*m)),
        # m being number of datapoints
        m = len(self.ctrl_pts)+1
        smoothing_factor = 0.01*(m)
        tck, u = splprep(pts.T, u=np.linspace(
            0, self.track_length_grid, self.track_length_grid+1), s=smoothing_factor, per=1)
        # NOTE
        # tck, u = CubicSpline(np.linspace(0,self.track_length_grid,self.track_length_grid+1),pts)

        # this gives smoother result, but difficult to relate u to actual grid
        # tck, u = splprep(pts.T, u=None, s=0.0, per=1)
        self.u = u
        self.raceline = tck
        retval = self.generate_speed_profile()
        self.targetVfromU = speed_profile_fun = retval.speed_profile_fun
        self.max_v = retval.max_v
        self.min_v = retval.min_v

    def generate_speed_profile(self,
                               *,
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
        """
        g = 9.81

        # u values for control points
        uu = np.linspace(0, self.track_length_grid, n_steps+1)

        # let raceline curve be r(u)
        # dr = r'(u), parameterized with uu
        dr = np.array(splev(uu, self.raceline, der=1))
        # ddr = r''(u)
        ddr = np.array(splev(uu, self.raceline, der=2))

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

                (x_i, y_i) = splev(uu[i % n_steps], self.raceline, der=0)
                (x_i_1, y_i_1) = splev(uu[(i+1) %
                                          n_steps], self.raceline, der=0)
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

            (x_i, y_i) = splev(uu[i % n_steps], self.raceline, der=0)
            (x_i_1, y_i_1) = splev(uu[(i-1+n_steps) %
                                      n_steps], self.raceline, der=0)
            # distance between two steps
            ds = dist((x_i, y_i), (x_i_1, y_i_1))
            # print(ds)
            v3[(i-1+n_steps) % n_steps] = min((v3[i % n_steps] **
                                               2 + 2*a_lon*ds)**0.5, v2[(i-1+n_steps) % n_steps])
            # print(v3[(i-1+n_steps)%n_steps],v2[(i-1+n_steps)%n_steps])

        v3[-1] = v3[0]

        # when calling, make sure u is in range [0,len(self.ctrl_pts)]
        speed_profile_fun = interp1d(uu, v3, kind='cubic')

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

        return SpeedProfileOutput(speed_profile_fun, min_v, max_v)

    def save(self, filename=None):
        ''' Save raceline to pickle file'''
        if filename is None:
            filename = 'raceline.p'

        # assemble save data
        save = {}
        save['grid_sequence'] = self.grid_sequence
        save['scale'] = self.scale
        save['origin_seq_no'] = self.origin_seq_no
        save['track_length'] = self.track_length_grid
        save['raceline'] = self.raceline
        save['gridsize'] = self.gridsize
        save['resolution'] = self.resolution
        save['targetVfromU'] = self.targetVfromU
        save['track'] = self.grid
        save['min_v'] = self.min_v
        save['max_v'] = self.max_v
        save['start_pos'] = self.start_pos
        save['start_dir'] = self.start_dir

        full_filename = os.path.join(BASEDIR, 'buzzracer', 'data', filename)
        with open(full_filename, 'wb') as f:
            pickle.dump(save, f)
        self.print_ok(f'Track and raceline saved at {full_filename}')

    def load(self, filename=None):
        ''' Load quadratically smoothed raceline '''
        if filename is None:
            filename = 'raceline.p'
        try:
            full_path = os.path.join(BASEDIR, 'buzzracer', 'data', filename)
            with open(full_path, 'rb') as f:
                save = pickle.load(f)
        except FileNotFoundError:
            self.print_error(f"can't find saved raceline {filename}, run "
                             " `python -m buzzracer.scripts.qp_smooth [track_name]` first"
                             " Example track name: full"
                             )
            raise

        # Restore saved data
        # pylint: disable=attribute-defined-outside-init
        self.grid_sequence = save['grid_sequence']
        self.scale = save['scale']
        self.origin_seq_no = save['origin_seq_no']
        self.track_length_grid = save['track_length']
        self.raceline = save['raceline']
        self.gridsize = save['gridsize']
        # self.resolution = save['resolution']
        self.targetVfromU = save['targetVfromU']
        self.grid = save['track']
        self.min_v = save['min_v']
        self.max_v = save['max_v']
        self.start_pos = save['start_pos']
        self.start_dir = save['start_dir']
        self.x_limit = self.gridsize.cols*self.scale
        self.y_limit = self.gridsize.rows*self.scale
        # pylint: enable=attribute-defined-outside-init

        self.print_ok('Track and raceline loaded')
        self.reconstruct_raceline()
        self.prepare_discretized_raceline()
        return

    def calc_path_distance(self, u0, u1):
        ''' calculate distance '''
        s = 0
        steps = 10
        uu = np.linspace(u0, u1, steps)
        xx, yy = splev(uu, self.raceline, der=0)
        dx = np.diff(xx)
        dy = np.diff(yy)
        s = np.sum(np.sqrt(dx**2+dy**2))
        return s

    def calc_derivative(self, curve, ds):
        # find first and second derivative
        dr = []
        ddr = []
        n = curve.shape[0]
        for i in range(1, n-1):
            rl = curve[i-1, :]
            r = curve[i, :]
            rr = curve[i+1, :]
            points = [rl, r, rr]
            ((al, a, ar), (bl, b, br)) = self.lagrange_der(points, ds=[ds, ds])
            dr.append(al*rl+a*r+ar*rr)
            ddr.append(bl*rl+b*r+br*rr)
        dr = np.array(dr)
        ddr = np.array(ddr)
        dr = np.vstack([dr[0], dr, dr[-1]])
        ddr = np.vstack([ddr[0], ddr, ddr[-1]])
        return (dr, ddr)

    # right turn negative curvature
    def calc_curvature(self, dr_vec, ddr_vec):
        # ccw 90 deg
        A = np.array([[0, -1], [1, 0]])
        a = (A @ dr_vec.T).T
        b = ddr_vec
        curvature = np.sum(a*b, axis=1).flatten()
        return curvature

    def lagrange_der(self, points, ds=None):
        ''' Given three points, calculate first and second derivative 
        as a linear combination of the three points rl, r, rr.

        Args:
            points: Iterable of rl, r, rr, which stand for r_(k-1), r_k, r_(k+1)

        Returns:
            retval: 2*3, tuple ((al, a, ar),(bl, b, br))
            where f'@ r = al*rl + a*r + ar*rr
            where f''@r = bl*rl + b*r + br*rr
            ds, arc length between rl, r and r, rr
            if not specified, | r-rl | _2 will be used as approximation
        '''
        rl, r, rr = points

        def dist(x, y):
            return ((x[0]-y[0])**2 + (x[1]-y[1])**2)**0.5
        if ds is None:
            sl = -dist(rl, r)
            sr = dist(r, rr)
        else:
            sl = -ds[0]
            sr = ds[1]

        try:
            al = - sr/sl/(sl-sr)
            a = -(sl+sr)/sl/sr
            ar = -sl/sr/(sr-sl)

            bl = 2/sl/(sl-sr)
            b = 2/sl/sr
            br = 2/sr/(sr-sl)
        except Warning as e:
            print(e)

        return ((al, a, ar), (bl, b, br))

    def check_track_boundary(self, coord):
        ''' Check if a point is inside track boudnary

        Args:
            coord: (x,y)
        Returns:
            val: min distance to left/right boundary
        '''
        # figure out which grid the coord is in
        # grid coordinate, (col, row), col starts from left and row starts from bottom,
        # both indexed from 0
        nondim = np.array(np.array(coord)/self.scale//1, dtype=int)
        nondim[0] = np.clip(nondim[0], 0, len(self.grid)-1).astype(int)
        nondim[1] = np.clip(nondim[1], 0, len(self.grid[0])-1).astype(int)

        # e.g. 'WE','SE'
        grid_type = self.grid[nondim[0]][nondim[1]]

        # change ref frame to tile local ref frame
        x_local = coord[0]/self.scale - nondim[0]
        y_local = coord[1]/self.scale - nondim[1]

        # find the distance to track sides
        # boundary/wall width / grid side length
        deadzone = 0.087
        straights = ['WE', 'NS']
        turns = ['SE', 'SW', 'NE', 'NW']
        wl, wr = (0, 0)
        if grid_type in straights:
            if grid_type == 'WE':
                # track section is staight, arranged horizontally
                # remaining space on top (negative means coord outside track
                wl = y_local - deadzone
                wr = 1 - deadzone - y_local
            if grid_type == 'NS':
                # track section is staight, arranged vertically
                # remaining space on left (negative means coord outside track
                wl = x_local - deadzone
                wr = 1 - deadzone - x_local
        elif grid_type in turns:
            apex = None
            if grid_type == 'SE':
                apex = (1, 0)
            if grid_type == 'SW':
                apex = (0, 0)
            if grid_type == 'NE':
                apex = (1, 1)
            if grid_type == 'NW':
                apex = (0, 1)
            radius = ((x_local - apex[0])**2 + (y_local - apex[1])**2)**0.5
            wl = 1-deadzone-radius
            wr = radius - deadzone
        return min(wl, wr)

    def precise_track_boundary(self, coord, heading):
        ''' Given coordinate and heading, calculate precise boundary to left and right
        return a vector(dist_to_left, dist_to_right)'''
        heading = (heading + np.pi) % (2*np.pi) - np.pi
        # figure out which grid the coord is in
        # grid coordinate, (col, row), col starts from left and row starts from bottom,
        # both indexed from 0
        nondim = np.array(np.array(coord)/self.scale//1, dtype=int)
        nondim[0] = np.clip(nondim[0], 0, len(self.grid)-1).astype(int)
        nondim[1] = np.clip(nondim[1], 0, len(self.grid[0])-1).astype(int)

        # e.g. 'WE','SE'
        grid_type = self.grid[nondim[0]][nondim[1]]
        # NOTE grid_type may be None if coord is not on track

        # change ref frame to tile local ref frame
        x_local = coord[0]/self.scale - nondim[0]
        y_local = coord[1]/self.scale - nondim[1]

        # find the distance to track sides
        # boundary/wall width / grid side length
        deadzone = 0.087
        straights = ['WE', 'NS']
        turns = ['SE', 'SW', 'NE', 'NW']
        left = 0
        right = 0
        if grid_type in straights:
            if grid_type == 'WE':
                # track section is staight, arranged horizontally
                # remaining space on top (negative means coord outside track
                grid_down = y_local - deadzone
                grid_up = 1 - deadzone - y_local
                if (heading > -np.pi/2 and heading < np.pi/2):
                    left = grid_up / cos(heading)
                    right = grid_down / cos(heading)
                else:
                    left = - grid_down / cos(heading)
                    right = - grid_up / cos(heading)

            if grid_type == 'NS':
                # track section is staight, arranged vertically
                # remaining space on left (negative means coord outside track
                grid_left = x_local - deadzone
                grid_right = 1 - deadzone - x_local
                if (heading > 0 and heading < np.pi):
                    left = grid_left / sin(heading)
                    right = grid_right / sin(heading)
                else:
                    left = - grid_right / sin(heading)
                    right = - grid_left / sin(heading)
        elif grid_type in turns:
            step_size = 0.01

            # find left boundary
            left = 0.0
            flag_in_limit = True
            while flag_in_limit:
                left_point = (coord[0] + left * cos(heading+np.pi/2),
                              coord[1] + left * sin(heading+np.pi/2))
                flag_in_limit = self.check_track_boundary(left_point) > 0
                left += step_size

            # find right boundary
            right = 0.0
            flag_in_limit = True
            while flag_in_limit:
                right_point = (coord[0] + right * cos(heading-np.pi/2),
                               coord[1] + right * sin(heading-np.pi/2))
                flag_in_limit = self.check_track_boundary(right_point) > 0
                right += step_size

            # convert metric unit to dimensionless unit
            left /= self.scale
            right /= self.scale

        return (left*self.scale, right*self.scale)

    def draw_point_u(self, img, uu):
        ''' draw point corresponding to u, the parameter for race line '''
        x_new, y_new = splev(uu, self.raceline, der=0)

        for x, y in zip(x_new, y_new):
            img = self.draw_point(img, (x, y))
        return img

    def draw_raceline(self,  img=None, points=None, s_to_color=None):
        ''' draw the raceline from self.raceline 
        Args:
            img: Base image to draw onto
            points: List[tuple[x,y]] additional points to draw
            s_to_color: lambda: s: color(0-1), map progress to color
        Return:
            img: result image
        '''

        rows = self.gridsize[0]
        cols = self.gridsize[1]
        res = int(self.resolution*self.scale)

        # this gives smoother result, but difficult to relate u to actual grid
        # u_new = np.linspace(self.u.min(),self.u.max(),1000)

        # the range of u is len(self.ctrl_pts) + 1, since we copied one to the end
        # x_new and y_new are in non-dimensional grid unit
        u_new = np.linspace(0, self.track_length_grid, 1000)
        x_new, y_new = splev(u_new, self.raceline, der=0)
        # convert to visualization coordinate
        x_new *= self.resolution
        y_new *= self.resolution
        y_new = self.resolution*self.scale*rows - y_new

        if img is None:
            img = np.zeros([res*rows, res*cols, 3], dtype='uint8')

        pts = np.vstack([x_new, y_new]).T
        # for polylines, pts = pts.reshape((-1,1,2))
        pts = pts.reshape((-1, 2))
        pts = pts.astype(int)
        # render different color based on speed
        # slow - red, fast - green (BGR)

        def s2c(s):
            return (self.sToV(s)-self.min_v)/(self.max_v-self.min_v)
        if s_to_color is None:
            s_to_color = s2c

        def get_color(s):
            return (0, int(s_to_color(s)*255), int(255-255*s_to_color(s)))
        for i in range(len(u_new)-1):
            s = self.uToS(u_new[i] % self.track_length_grid)
            color = get_color(s)
            img = cv2.line(img, tuple(pts[i]), tuple(
                pts[i+1]), color=color, thickness=3)

        # plot reference points
        # img = cv2.polylines(img, [pts], isClosed=True, color=lineColor, thickness=3)
        if points is not None:
            for point in points:
                x = point[0]
                y = point[1]
                x *= self.resolution
                y *= self.resolution
                y = self.resolution*self.scale*rows - y

                img = cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)

        return img

    def local_trajectory(self, state, wheelbase=90e-3, return_u=False):
        # TODO refactor here onwards
        ''' Given state of the car,
        find the closest point on raceline to center of FRONT axle
        calculate the lateral offset ( in meters), this will be reported as offset, 
        which can be added directly to raceline orientation
        (after multiplied with an aggressiveness coefficient)
        to obtain desired front wheel orientation calculate the local derivative
        coord should be referenced from the origin(bottom left(edited)) of the track, in meters
        negative offset means coord is to the right of the raceline, viewing from raceline 
        init direction
        wheelbase is needed to calculate the local trajectory closes to the front axle instead 
        of the old axle
        '''
        # figure out which grid the coord is in
        coord = np.array([state.x, state.y])
        heading = state.heading
        # find the coordinate of center of front axle
        coord[0] += wheelbase*cos(heading)
        coord[1] += wheelbase*sin(heading)
        # grid coordinate, (col, row), col starts from left and row starts from bottom,
        # both indexed from 0
        # coord should be given in meters
        nondim = np.array((coord/self.scale)//1, dtype=int)

        # distance squared, not need to find distance here
        def dist_2(a, b):
            return (a[0]-b[0])**2+(a[1]-b[1])**2

        def dist(u):
            return dist_2(splev(u % self.track_length_grid, self.raceline), coord)
        # last_u is the seq found last time, which should be a good estimate of where to start
        # disable this functionality since it doesn't handle multiple cars
        self.last_u = None
        if self.last_u is None:
            # the seq here starts from origin
            seq = -1
            # figure out which u this grid corresponds to
            for i, grid in enumerate(self.grid_sequence):
                if nondim[0] == grid[0] and nondim[1] == grid[1]:
                    seq = i
                    break

            if seq == -1:
                print('error, coord not on track, x = %.2f, y=%.2f' %
                      (coord[0], coord[1]))
                return None

            # the grid that contains the coord
            # print("in grid : " + str(self.grid_sequence[seq]))

            # find the closest point to the coord
            # because we wrapped the end point to the beginning of sample point,
            # we need to add this offset
            # Now seq would correspond to u in raceline,
            # i.e. allow us to locate the raceline at that section
            seq += self.origin_seq_no
            seq %= self.track_length_grid

            # this gives a close, usually preceding raceline point,
            # this does not give the closest ctrl point
            # due to smoothing factor
            # print("neighbourhood raceline pt " + str(splev(seq,self.raceline)))

            # determine which end is the coord closer to,
            # since seq points to the previous control point,
            # not necessarily the closest one
            if dist(seq+1) < dist(seq):
                seq += 1
            if dist(seq-1) < dist(seq):
                seq -= 1
        else:
            seq = self.last_u

        # Goal: find the point on raceline closest to coord
        # i.e. find x that minimizes dist(x)
        # we know x will be close to seq

        # easy method
        # brute force, This takes 77% of runtime.
        # lt.s('minimize_scalar')
        # res = minimize_scalar(dist,bounds=[seq-0.6,seq+0.6],method='Bounded')
        # lt.e('minimize_scalar')

        # improved method: from observation, dist(x) is quadratic in proximity of seq
        # we assume it to be ax^3 + bx^2 + cx + d and
        # formulate this minimization as a linalg problem
        # sample some points to build the trinomial simulation
        self.debug['seq'] = seq
        iv = np.array([-0.6, -0.3, 0, 0.3, 0.6])+seq
        # formulate linear problem
        A = np.vstack([iv**3, iv**2, iv, [1, 1, 1, 1, 1]]).T
        # B = np.mat([dist(x0), dist(x1), dist(x2)]).T
        B = dist(iv).T
        # abc = np.linalg.solve(A,B)
        abc = np.linalg.lstsq(A, B, rcond=-1)[0]
        a = abc[0]
        b = abc[1]
        c = abc[2]
        d = abc[3]
        def poly(x):
            return a*x*x*x + b*x*x + c*x + d
        fit = minimize(poly, x0=seq, method='L-BFGS-B', bounds=((seq-0.6, seq+0.6),))
        min_fun_x = fit.x[0]
        self.last_u = min_fun_x % self.track_length_grid

        min_fun_val = float(fit.fun)

        raceline_point = splev(min_fun_x %
                               self.track_length_grid, self.raceline)
        # raceline_point = splev(res.x,self.raceline)

        der = splev(min_fun_x % self.track_length_grid, self.raceline, der=1)
        # der = splev(res.x,self.raceline,der=1)

        # calculate whether offset is ccw or cw
        # achieved by finding cross product of vec(raceline_orientation) and vec(ctrl_pnt->test_pnt)
        # then find sin(theta)
        # negative offset means car is to the right of the trajectory
        vec_raceline = (der[0], der[1])
        vec_offset = coord - raceline_point
        cross_theta = np.cross(vec_raceline, vec_offset)

        vec_curvature = splev(min_fun_x %
                              self.track_length_grid, self.raceline, der=2)
        norm_curvature = np.linalg.norm(vec_curvature)
        # gives right sign for omega,
        # this is indep of track direction since it's calculated based off vehicle orientation
        # cross_curvature = np.cross((cos(heading),sin(heading)),vec_curvature)
        cross_curvature = der[0]*vec_curvature[1]-der[1]*vec_curvature[0]

        # return target velocity
        request_velocity = self.targetVfromU( min_fun_x % self.track_length_grid)

        retval = LocalTrajOutput(ref_point=raceline_point,
                                 lateral_err=copysign( abs(min_fun_val)**0.5, cross_theta),
                                 heading_err=atan2(der[1], der[0]),
                                 curvature=copysign( norm_curvature, cross_curvature),
                                 v_target=request_velocity,
                                 progress=self.uToS(min_fun_x % self.track_length_grid)
                                 )
        return retval

    # create two function to map between u(raceline parameter)<->s(distance along racelien)
    # also create mapping between s -> v_ref
    # also create raceline_s, raceline parameterized with s
    def reconstruct_raceline(self):
        s_vec = [0]
        n_steps = 1000
        uu = np.linspace(0, self.track_length_grid, n_steps+1)
        def dist(a, b):
            return ((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5
        path_len = 0
        for i in range(n_steps):
            (x_i, y_i) = splev(uu[i % n_steps], self.raceline, der=0)
            (x_i_1, y_i_1) = splev(uu[(i+1) % n_steps], self.raceline, der=0)
            # distance between two steps
            ds = dist((x_i, y_i), (x_i_1, y_i_1))
            path_len += ds
            s_vec.append(path_len)

        ss = np.array(s_vec)
        vv = self.targetVfromU(uu % self.track_length_grid)

        # using interp1d functions can cause some overhead
        # when absolute speed is needed, use lookup tables
        # this may lose some accuracy but with larger n_step
        # and moderate change in velocity this should not be an issue
        self.sToV_lut = lambda x: self.v_lut[bisect(self.s_lut, x)]
        self.s_lut = ss
        self.v_lut = vv

        self.uToS = interp1d(uu, ss, kind='cubic')
        self.sToU = interp1d(ss, uu, kind='cubic')
        self.sToV = interp1d(ss, vv, kind='cubic')
        self.raceline_len_m = path_len
        # print("verify u and s mapping accuracy")
        # ss_remap = self.uToS(self.sToU(ss))
        # print("mean error in s %.5f m "%(np.mean(np.abs(ss-ss_remap))))
        # print("max error in s %.5f m "%(np.max(np.abs(ss-ss_remap))))

        # convert self.raceline(parameterized w.r.t. u)
        # to self.raceline_s (parameterized w.r.t. s, distance along path)
        rr = splev(uu % self.track_length_grid, self.raceline)
        tck, _ = splprep(rr, u=ss, s=0, per=1)
        self.raceline_s = tck

        def _norm(x):
            return np.linalg.norm(x, axis=0)

        xx = np.linspace(0, self.raceline_len_m, self.discretized_raceline_len)
        dr = np.array(splev(xx, self.raceline_s, der=1))
        # ddr = r''(u)
        ddr = np.array(splev(xx, self.raceline_s, der=2))
        def _norm(x):
            return np.linalg.norm(x, axis=0)
        # radius of curvature can be calculated as R = |y'|^3/sqrt(|y'|^2*|y''|^2-(y'*y'')^2)
        # gives right sign for omega,
        # this is indep of track direction since it's calculated based off vehicle orientation
        # for magnitude only
        # curvature_vec = 1.0/(_norm(dr)**3/(_norm(dr)**2*_norm(ddr)
        #                                    ** 2 - np.sum(dr*ddr, axis=0)**2)**0.5)
        dx = dr[0]
        dy = dr[1]
        ddx = ddr[0]
        ddy = ddr[1]
        curvature_vec = (dx * ddy - ddx * dy) / (dx**2 + dy**2)**1.5
        tck, _ = splprep([curvature_vec], u=xx, s=0, per=1)
        self.curvature_s = lambda s: splev(s % self.raceline_len_m, tck)[0]
        return

    # get future reference point for dynamic MPC
    # Inputs:
    # state: vehicle state, same as in self.local_trajectory()
    # p : lookahead steps
    # dt : time between each lookahead steps

    # Return:
    # xref : np array of size (p+1)*2,
    # there are p+1 entries because xref0 is the ref point for current location,
    # and then there are p projection points
    # psi_ref : reference heading at the reference points, size (p+1)*2
    # v_ref : reference heading at the reference points, size (p+1)*2
    # valid : a boolean indicating whether the function was able to find a valid result
    # The function first finds a point on trajectory closest to vehicle location
    # with local_trajectory(), then find p points down the trajectory that are spaced vk * dt
    # apart in path length. vk is the reference velocity at those points

    def get_ref_point(self, state, p, dt, reverse=False):
        t = self.t

        t.s()
        if reverse:
            self.print_error('reverse is not implemented')
        # set wheelbase to 0 to get point closest to vehicle CG
        t.s('local traj')
        retval = self.local_trajectory(
            state, wheelbase=0.102/2.0, return_u=True)
        t.e('local traj')
        if retval is None:
            return None, None, False

        # parse return value from local_trajectory
        (local_ctrl_pnt, offset, orientation, curvature, v_target, u0) = retval
        if isnan(orientation):
            return None, None, False

        # calculate s value for projection ref points
        t.s('find s')
        s0 = self.uToS(u0).item()
        v0 = self.targetVfromU(u0 % self.track_length_grid).item()
        der = splev(u0 % self.track_length_grid, self.raceline, der=1)
        heading0 = atan2(der[1], der[0])
        t.e('find s')

        t.s('curvature')
        def _norm(x):
            return np.linalg.norm(x, axis=0)
        # gives right sign for omega,
        # this is indep of track direction since it's calculated based off vehicle orientation

        dr = np.array(splev(u0 % self.track_length_grid, self.raceline, der=1))
        ddr = vec_curvature = np.array(
            splev(u0 % self.track_length_grid, self.raceline, der=2))
        cross_curvature = der[0]*vec_curvature[1]-der[1]*vec_curvature[0]
        curvature = 1.0/(_norm(dr)**3/(_norm(dr)**2*_norm(ddr)
                         ** 2 - np.sum(dr*ddr, axis=0)**2)**0.5)

        t.e('curvature')

        # curvature needs to be signed to indicate whether signage target angular velocity
        # a cross product gives right signage for omega,
        # this is indep of track direction since it's calculated based off vehicle orientation
        cross_curvature = der[0]*vec_curvature[1]-der[1]*vec_curvature[0]

        # k_vec.append(norm_curvature)
        # k_sign_vec.append(cross_curvature)
        k_vec = curvature
        k_sign_vec = cross_curvature

        s_vec = [s0]
        v_vec = [v0]
        heading_vec = [heading0]
        k_vec = [curvature]
        k_sign_vec = [cross_curvature]

        u_vec = [u0]

        t.s('main loop')
        for k in range(1, p+1):
            s_k = s_vec[-1] + v_vec[-1] * dt
            s_vec.append(s_k)
            # find ref velocity for projection ref points
            # TODO adjust ref velocity for current vehicle velocity
            # v_k = self.targetVfromU(u_k%self.track_length_grid)
            # v_k = self.sToV(s_k%self.raceline_len_m)
            v_k = self.sToV_lut(s_k % self.raceline_len_m)
            v_vec.append(v_k)
        t.e('main loop')

        # u_vec = np.array(u_vec)%self.track_length_grid
        # find ref heading for projection ref points
        t.s('psi')
        # der = np.array(splev(u_vec,self.raceline,der=1))
        s_vec = np.array(s_vec) % self.raceline_len_m
        der = np.array(splev(s_vec, self.raceline_s, der=1))
        # heading_k = atan2(der[1],der[0])
        # heading_vec.append(heading_k)
        t.e('psi')
        # find ref coordinates for projection ref points

        t.s('coord')
        coord_vec = np.array(splev(s_vec, self.raceline_s)).T
        t.e('coord')

        t.s('K')

        # norm_curvature = np.linalg.norm(vec_curvature,axis=1)
        dr = np.array(splev(s_vec, self.raceline_s, der=1))
        ddr = vec_curvature = np.array(splev(s_vec, self.raceline_s, der=2))

        curvature = 1.0/(_norm(dr)**3/(_norm(dr)**2*_norm(ddr)
                         ** 2 - np.sum(dr*ddr, axis=0)**2)**0.5)

        # curvature needs to be signed to indicate whether signage target angular velocity
        # a cross product gives right signage for omega,
        # this is indep of track direction since it's calculated based off vehicle orientation
        cross_curvature = der[0, :]*vec_curvature[1, :] - \
            der[1, :]*vec_curvature[0, :]

        # k_vec.append(norm_curvature)
        # k_sign_vec.append(cross_curvature)
        k_vec = curvature
        k_sign_vec = cross_curvature

        # TODO check dimension
        k_signed_vec = np.copysign(k_vec, k_sign_vec)

        x, y, heading, vf, vs, omega = state
        e_heading = ((heading - heading0) + pi/2.0) % (2*pi) - pi/2.0
        t.e('K')

        t.e()
        # return offset, e_heading, np.array(v_vec),np.array(k_signed_vec), np.array(coord_vec),True
        return offset, e_heading, np.array(v_vec), np.array(k_signed_vec), np.array(coord_vec), True

    def get_ref_x_y_vheading(self, state, p, dt, reverse=False):
        t = self.t

        t.s()
        if reverse:
            self.print_error('reverse is not implemented')
        # set wheelbase to 0 to get point closest to vehicle CG
        t.s('local traj')
        retval = self.local_trajectory(
            state, wheelbase=0.102/2.0, return_u=True)
        t.e('local traj')
        if retval is None:
            return None, None, False

        # parse return value from local_trajectory
        (local_ctrl_pnt, offset, orientation, curvature, v_target, u0) = retval
        if isnan(orientation):
            return None, None, False

        # calculate s value for projection ref points
        t.s('find s')
        s0 = self.uToS(u0).item()
        v0 = self.targetVfromU(u0 % self.track_length_grid).item()
        der = splev(u0 % self.track_length_grid, self.raceline, der=1)
        heading0 = atan2(der[1], der[0])
        t.e('find s')

        t.s('curvature')
        def _norm(x): 
            return np.linalg.norm(x, axis=0)
        # gives right sign for omega,
        # this is indep of track direction since it's calculated based off vehicle orientation

        dr = np.array(splev(u0 % self.track_length_grid, self.raceline, der=1))
        ddr = vec_curvature = np.array(
            splev(u0 % self.track_length_grid, self.raceline, der=2))
        cross_curvature = der[0]*vec_curvature[1]-der[1]*vec_curvature[0]
        curvature = 1.0/(_norm(dr)**3/(_norm(dr)**2*_norm(ddr)
                         ** 2 - np.sum(dr*ddr, axis=0)**2)**0.5)

        t.e('curvature')

        # curvature needs to be signed to indicate whether signage target angular velocity
        # a cross product gives right signage for omega,
        # this is indep of track direction since it's calculated based off vehicle orientation
        cross_curvature = der[0]*vec_curvature[1]-der[1]*vec_curvature[0]

        # k_vec.append(norm_curvature)
        # k_sign_vec.append(cross_curvature)
        k_vec = curvature
        k_sign_vec = cross_curvature

        u_vec = [u0]
        s_vec = [s0]
        k_vec = [curvature]
        k_sign_vec = [cross_curvature]

        v_vec = [v0]
        xy_vec = [splev(s0 % self.raceline_len_m, self.raceline_s)]

        t.s('main loop')
        for k in range(1, p+1):
            s_k = s_vec[-1] + v_vec[-1] * dt
            s_vec.append(s_k)
            # find ref velocity for projection ref points
            # TODO adjust ref velocity for current vehicle velocity
            # v_k = self.targetVfromU(u_k%self.track_length_grid)
            # v_k = self.sToV(s_k%self.raceline_len_m)
            v_k = self.sToV_lut(s_k % self.raceline_len_m)
            v_vec.append(v_k)

            xy_vec.append(splev(s_k % self.raceline_len_m, self.raceline_s))

        t.e('main loop')

        # u_vec = np.array(u_vec)%self.track_length_grid
        # find ref heading for projection ref points
        t.s('psi')
        # der = np.array(splev(u_vec,self.raceline,der=1))
        s_vec = np.array(s_vec) % self.raceline_len_m
        der = np.array(splev(s_vec, self.raceline_s, der=1))
        heading_vec = np.arctan2(der[1, :], der[0, :])
        t.e('psi')
        # find ref coordinates for projection ref points

        t.s('coord')
        coord_vec = np.array(splev(s_vec, self.raceline_s)).T
        t.e('coord')

        t.s('K')

        # norm_curvature = np.linalg.norm(vec_curvature,axis=1)
        dr = np.array(splev(s_vec, self.raceline_s, der=1))
        ddr = vec_curvature = np.array(splev(s_vec, self.raceline_s, der=2))

        curvature = 1.0/(_norm(dr)**3/(_norm(dr)**2*_norm(ddr)
                         ** 2 - np.sum(dr*ddr, axis=0)**2)**0.5)

        # curvature needs to be signed to indicate whether signage target angular velocity
        # a cross product gives right signage for omega, this is indep of track direction
        # since it's calculated based off vehicle orientation
        cross_curvature = der[0, :]*vec_curvature[1, :] - \
            der[1, :]*vec_curvature[0, :]

        # k_vec.append(norm_curvature)
        # k_sign_vec.append(cross_curvature)
        k_vec = curvature
        k_sign_vec = cross_curvature

        # TODO check dimension
        k_signed_vec = np.copysign(k_vec, k_sign_vec)

        x, y, heading, vf, vs, omega = state
        e_heading = ((heading - heading0) + pi/2.0) % (2*pi) - pi/2.0
        t.e('K')

        t.e()
        return np.array(xy_vec), np.array(v_vec), np.array(heading_vec)

    def predict_opponent(self, state, p, dt, reverse=False):
        ''' Predict an opponent car's future trajectory, assuming they are on ref raceline
            and will remain there, traveling at current speed
        Args:
            state: opponent vehicle state, same as in self.local_trajectory()
            p: lookahead steps
            dt: time between each lookahead steps

        Returns:
            xref: np array of size(p+1)*2, there are p+1 entries because xref0 is the ref point
            for current location, and then there are p projection points
            valid: a boolean indicating whether the function was able to find a valid result
            The function first finds a point on trajectory closest to vehicle location with
            local_trajectory(), then find p points down the trajectory that are spaced vk * dt apart
            in path length. vk is the reference velocity at those points
        '''
        if reverse:
            self.print_error('reverse is not implemented')
        # set wheelbase to 0 to get point closest to vehicle CG
        retval = self.local_trajectory(
            state, wheelbase=0.102/2.0, return_u=True)
        if retval is None:
            return None, None, False

        # parse return value from local_trajectory
        (local_ctrl_pnt, offset, orientation, curvature, v_target, u0) = retval
        if isnan(orientation):
            return None, None, False

        # calculate s value for projection ref points
        s0 = self.uToS(u0).item()
        # use optimal velocity
        # v0 = self.targetVfromU(u0%self.track_length_grid).item()
        # use actual velocity
        v0 = state[3]

        def _norm(x):
            return np.linalg.norm(x, axis=0)

        s_vec = [s0]
        v_vec = [v0]

        for _ in range(1, p+1):
            s_k = s_vec[-1] + v_vec[-1] * dt
            s_vec.append(s_k)
            # find ref velocity for projection ref points
            # TODO adjust ref velocity for current vehicle velocity

            # v_k = self.sToV_lut(s_k%self.raceline_len_m)
            # NOTE assume constant velocity
            v_k = v0
            v_vec.append(v_k)

        # find ref heading for projection ref points
        s_vec = np.array(s_vec) % self.raceline_len_m
        # find ref coordinates for projection ref points
        coord_vec = np.array(splev(s_vec, self.raceline_s)).T

        return coord_vec

    # draw a point on canvas at coord
    def draw_point(self, img, coord, color=(0, 0, 0)):
        src = self.m2canvas(coord)
        img = cv2.circle(img, src, 3, color, -1)

        return img

    def draw_points(self, img, coord_vec, color=(0, 0, 0)):
        for coord in coord_vec:
            src = self.m2canvas(coord)
            img = cv2.circle(img, src, 3, color, -1)
        return img
