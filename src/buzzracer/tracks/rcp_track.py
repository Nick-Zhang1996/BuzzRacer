''' Subclass of Track for RCP style modular track.
Desperately need cleanup and refactoring '''

from __future__ import annotations
import os
import pickle
from math import sin, cos
from enum import Enum
from typing import NamedTuple
import logging
from dataclasses import dataclass, replace

import cv2
import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt


from buzzracer.common import BASEDIR, get_logger
from buzzracer.tracks.track import Track, TrackConfig
from buzzracer.tracks.curvilinear_track import CurvilinearTrack, CurvilinearTrackData

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


logger = get_logger('RCPTrack')


@dataclass(kw_only=True)
class RCPTrackConfig(TrackConfig):
    gridsize: GridSize
    ''' (rows, cols), grid size of the track '''
    grid: list[list[Node | str | None]]
    track_length_grid: int = 0
    ''' Total grid length of the track'''
    grid_sequence: list[tuple[int, int]]
    ''' List of the grid (row, col) that defines the track'''
    x_limit: float
    ''' X-direction bound in meters, i.e. width of track'''
    y_limit: float
    ''' Y-direction bound in meters, i.e. height of track'''
    scale: float = 0.6
    ''' Edge length of one grid in meters (default 0.6m)'''


@dataclass
class RCPTrackState:
    example: int = 1


@dataclass
class RCPTrackRaceline:
    raceline_s: np.ndarray
    raceline_len_m: float
    start_pos: tuple
    start_dir: float


class RCPTrack(CurvilinearTrack):
    def __init__(self, config: RCPTrackConfig):
        CurvilinearTrack.__init__(self, config)
        self.config: RCPTrackConfig
        self.rcp_raceline: RCPTrackRaceline
        self.data: CurvilinearTrackData

    # Example factory for a track
    @staticmethod
    def example_factory():
        track_size = GridSize(6, 4)
        config = RCPTrack.build_config('uuurrullurrrdddddluulddl', track_size)
        track = RCPTrack(config)
        rcp_raceline = track.build_raceline((3, 3), 'd', offset=None)
        r_vec, left, right = track.process_rcp_raceline(rcp_raceline)
        data = track.build_track(r_vec, left, right)
        track.rcp_raceline = rcp_raceline
        track.data = data
        return track

    @staticmethod
    def build_config(description: str,
                     gridsize: GridSize,
                     start_grid: tuple[int, int] = (0, 0),
                     scale: float = 0.6) -> RCPTrackConfig:
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
        Return:
            RCPTrackConfig object
        '''

        track_length_grid = len(description)
        grid_sequence = []

        x_limit = gridsize.cols*scale
        y_limit = gridsize.rows*scale

        grid = [[None for _ in range(gridsize.rows)]
                for _ in range(gridsize.cols)]

        current_index = start_grid
        grid_sequence.append(current_index)
        current_node = grid[start_grid[0]][start_grid[1]] = Node()
        for i, dir_char in enumerate(description):
            move_dir = Dir.from_char(dir_char)
            current_node.set_exit(move_dir)
            next_index = Dir.move(current_index, move_dir)
            grid_sequence.append(next_index)

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

        return RCPTrackConfig(resolution=200,
                              discretized_raceline_len=1024,
                              x_limit=x_limit,
                              y_limit=y_limit,
                              scale=0.6,
                              gridsize=gridsize,
                              track_length_grid=track_length_grid,
                              grid_sequence=grid_sequence,
                              grid=grid)

    def build_track(self, r_vec, left_width, right_width):
        """ Build a Curvilinear Track.
        Args:
            r_vec: np.ndarray (N, 2) Reference points for curve.
            left_width: np.ndarray (N,) Half width from ref curve to left boundary
            right_width: np.ndarray (N,) Half width from ref curve to right boundary
        Return:
            CurvilinearTrackData Object
        """
        config = self.config
        data = CurvilinearTrack.build_track(self, r_vec, left_width, right_width)
        new_data = replace(data, x_min=0, x_max=config.x_limit,
                           y_min=0, y_max=config.y_limit)
        return new_data

    def m2canvas(self, coord):
        return Track.m2canvas(self, coord)

    def draw_track(self, img=None):
        config = self.config
        color_side = (255, 0, 0)
        # boundary width / grid width
        deadzone = 0.087
        gs = int(config.resolution * config.scale)

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
        rows = config.gridsize[0]
        cols = config.gridsize[1]
        if img is None:
            # white background
            # img = 255*np.ones([gs*rows,gs*cols,3],dtype='uint8')
            img = np.zeros([gs*rows, gs*cols, 3], dtype='uint8')
            img[:, :, 0] = 255
        lookup_table = {'SE': 0, 'SW': 270, 'NE': 90, 'NW': 180}
        for i in range(cols):
            for j in range(rows):
                signature = config.grid[i][rows-1-j]
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

    def build_raceline(self, start: tuple[int, int], start_direction: Dir, offset=None):
        ''' Build a raceline from current track.
        Args:
            config: RCPTrackConfig object
            start: which grid to start from, e.g. (3,3), origin is at bottom left (0,0)
                    you MUST start on a straight section
            start_direction: which direction to ENTER start grid from the finishing grid.
                note use the direction for ENTERING that grid element
                e.g. 'l' or 'd' for a NE oriented turn
            offset: np.array of size self.track_length_grid, lateral offset for each control grid
        '''
        config = self.config
        ctrl_pts = []
        # starting from [start], the sequence number for origin (0,0)

        if offset is None:
            offset = np.zeros(config.track_length_grid)

        # provide exit direction given signature and entry direction
        lookup_table = {'WE': ['rr', 'll'], 'NS': ['uu', 'dd'], 'SE': [
            'ur', 'ld'], 'SW': ['ul', 'rd'], 'NE': ['dr', 'lu'], 'NW': ['ru', 'dl']}
        # provide correlation between direction (character) and directional vector
        lookup_table_dir = {'u': (0, 1), 'd': (
            0, -1), 'r': (1, 0), 'l': (-1, 0)}
        # provide right hand direction, this is for specifying offset direction
        lookup_table_right = {
            'u': (1, 0), 'd': (-1, 0), 'r': (0, -1), 'l': (0, 1)}

        # direction of entry
        entry_dir = start_direction
        current_coord = np.array(start, dtype='uint8')
        signature = config.grid[current_coord[0]][current_coord[1]]
        # find the previous signature, reverse entry to find ancestor
        # the precedent grid for start grid is also the final grid
        start_pos = ((0.5+start[0])*config.scale, (0.5+start[1])*config.scale)

        dire = lookup_table_dir[start_direction]
        start_dir = np.arctan2(dire[1], dire[0])

        # for referencing offset
        index = 0
        while True:
            signature = config.grid[current_coord[0]][current_coord[1]]

            # lookup exit direction
            for record in lookup_table[signature]:
                if record[0] == entry_dir:
                    exit_dir = record[1]
                    break

            # find the coordinate of the exit point
            # offset from grid center to centerpoint of exit boundary
            # go half a step from center toward exit direction
            exit_ctrl_pt = np.array(lookup_table_dir[exit_dir], dtype='float')/2
            exit_ctrl_pt += current_coord
            exit_ctrl_pt += np.array([0.5, 0.5])
            # apply offset, offset range (-1,1)
            exit_ctrl_pt += offset[index] * np.array(lookup_table_right[exit_dir], dtype='float')/2
            index += 1

            exit_ctrl_pt *= config.scale
            ctrl_pts.append(exit_ctrl_pt.tolist())

            current_coord = current_coord + lookup_table_dir[exit_dir]
            entry_dir = exit_dir

            if all(start == current_coord):
                break

        # add end point to the beginning,
        # otherwise splprep will replace pts[-1] with pts[0] for a closed loop
        # This ensures that splev(u=0) gives us the beginning point
        pts = np.array(ctrl_pts)
        # start_point = np.array(ctrl_pts[0])
        # pts = np.vstack([pts,start_point])
        end_point = np.array(ctrl_pts[-1])
        pts = np.vstack([end_point, pts])

        # weights = np.array(ctrl_pts_w + [ctrl_pts_w[-1]])

        # s= smoothing factor
        # a good s value should be found in the range (m-sqrt(2*m),m+sqrt(2*m)),
        # m being number of datapoints
        m = len(ctrl_pts)+1
        smoothing_factor = 0.01*(m)
        # pylint: disable-next=unbalanced-tuple-unpacking
        tck, _ = splprep(pts.T,
                         u=np.linspace(0, config.track_length_grid, config.track_length_grid+1),
                         s=smoothing_factor,
                         per=1)
        # NOTE
        # tck, u = CubicSpline(np.linspace(0,config.track_length_grid,config.track_length_grid+1),pts)

        # this gives smoother result, but difficult to relate u to actual grid
        # tck, u = splprep(pts.T, u=None, s=0.0, per=1)
        raceline_s, raceline_len_m = Track.reparam_raceline(tck, config.track_length_grid)
        return RCPTrackRaceline(raceline_s=raceline_s,
                                raceline_len_m=raceline_len_m,
                                start_pos=start_pos,
                                start_dir=start_dir)

    def save(self, filename=None):
        ''' Save raceline to pickle file'''
        if filename is None:
            filename = 'raceline.p'

        # assemble save data
        save = {}
        save['config'] = self.config
        save['data'] = self.data
        save['raceline'] = self.rcp_raceline

        full_filename = os.path.join(BASEDIR, 'assets', filename)
        with open(full_filename, 'wb') as f:
            pickle.dump(save, f)
        logger.info(f'Track and raceline saved at {full_filename}')

    def load(self, filename=None):
        ''' Load quadratically smoothed raceline '''
        if filename is None:
            filename = 'raceline.p'
        try:
            full_path = os.path.join(BASEDIR, 'assets', filename)
            with open(full_path, 'rb') as f:
                save = pickle.load(f)
        except FileNotFoundError:
            logger.error(f"can't find saved raceline {filename}, run "
                         " `[uv run] python scripts/qp_smooth.py [track_name]` first"
                         " Example track name: full"
                         )
            raise

        self.config = save['config']
        self.data = save['data']
        self.rcp_raceline = save['raceline']

        logger.info('Track and raceline loaded')
        return

    def calc_path_distance(self, u0, u1):
        ''' calculate distance '''
        s = 0
        steps = 10
        uu = np.linspace(u0, u1, steps)
        xx, yy = splev(uu, self.rcp_raceline, der=0)
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

    def coord_is_in_track(self, coord):
        ''' Check if a point is inside track boudnary

        Args:
            coord: (x,y)
        Returns:
            val: min distance to left/right boundary
        '''
        # figure out which grid the coord is in
        # grid coordinate, (col, row), col starts from left and row starts from bottom,
        # both indexed from 0
        config = self.config
        nondim = np.array(np.array(coord)/config.scale//1, dtype=int)
        nondim[0] = np.clip(nondim[0], 0, len(config.grid)-1).astype(int)
        nondim[1] = np.clip(nondim[1], 0, len(config.grid[0])-1).astype(int)

        # e.g. 'WE','SE'
        grid_type = config.grid[nondim[0]][nondim[1]]

        # change ref frame to tile local ref frame
        x_local = coord[0]/config.scale - nondim[0]
        y_local = coord[1]/config.scale - nondim[1]

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
        config = self.config
        heading = (heading + np.pi) % (2*np.pi) - np.pi
        # DEBUG - Use simple in/out checking. This is more precise, but more expensive
        step_size = 0.01
        # find left boundary
        left = 0.0
        flag_in_limit = True
        while flag_in_limit:
            left_point = (coord[0] + left * cos(heading+np.pi/2),
                          coord[1] + left * sin(heading+np.pi/2))
            flag_in_limit = self.coord_is_in_track(left_point) > 0
            left += step_size

        # find right boundary
        right = 0.0
        flag_in_limit = True
        while flag_in_limit:
            right_point = (coord[0] + right * cos(heading-np.pi/2),
                           coord[1] + right * sin(heading-np.pi/2))
            flag_in_limit = self.coord_is_in_track(right_point) > 0
            right += step_size

        # convert metric unit to dimensionless unit
        left /= config.scale
        right /= config.scale

        return (left*config.scale, right*config.scale)

        # Find the grid for coord
        # grid coordinate, (col, row), col starts from left and row starts from bottom,
        nondim = np.array(np.array(coord)/config.scale//1, dtype=int)
        nondim[0] = np.clip(nondim[0], 0, len(config.grid)-1).astype(int)
        nondim[1] = np.clip(nondim[1], 0, len(config.grid[0])-1).astype(int)

        # e.g. 'WE','SE'
        grid_type = config.grid[nondim[0]][nondim[1]]
        # NOTE grid_type may be None if coord is not on track

        # Coordinate in the local grid, unit: grid size
        x_local = coord[0]/config.scale - nondim[0]
        y_local = coord[1]/config.scale - nondim[1]

        # Find the distance to track sides
        # boundary/wall width / grid side length
        deadzone = 0.087  # Width of track boundary padding
        straights = ['WE', 'NS']
        turns = ['SE', 'SW', 'NE', 'NW']
        left = 0
        right = 0
        # NOTE vulnerable if heading = 0 or np.pi/2
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
                flag_in_limit = self.coord_is_in_track(left_point) > 0
                left += step_size

            # find right boundary
            right = 0.0
            flag_in_limit = True
            while flag_in_limit:
                right_point = (coord[0] + right * cos(heading-np.pi/2),
                               coord[1] + right * sin(heading-np.pi/2))
                flag_in_limit = self.coord_is_in_track(right_point) > 0
                right += step_size

            # convert metric unit to dimensionless unit
            left /= config.scale
            right /= config.scale

        return (left*config.scale, right*config.scale)

    def draw_point_u(self, img, uu):
        ''' draw point corresponding to u, the parameter for race line '''
        x_new, y_new = splev(uu, self.rcp_raceline, der=0)

        for x, y in zip(x_new, y_new):
            img = self.draw_point(img, (x, y))
        return img

    def draw_raceline(self, raceline, bound, img=None, points=None, speed_profile=None):
        ''' draw the raceline from self.raceline
        Args:
            raceline: tck object of splprep
            bound: Upper bound of parameter for raceline
            img: Base image to draw onto
            points: List[tuple[x,y]] additional points to draw
            s_to_color: lambda: s: color(0-1), map progress to color
        Return:
            img: result image
        '''

        config = self.config
        rows = config.gridsize[0]
        cols = config.gridsize[1]
        res = int(config.resolution*config.scale)

        # this gives smoother result, but difficult to relate u to actual grid
        # u_new = np.linspace(self.u.min(),self.u.max(),1000)

        # the range of u is len(self.ctrl_pts) + 1, since we copied one to the end
        # x_new and y_new are in non-dimensional grid unit
        ss = np.linspace(0, bound, 1000)
        x_new, y_new = splev(ss, raceline, der=0)
        # convert to visualization coordinate
        x_new *= config.resolution
        y_new *= config.resolution
        y_new = config.resolution*config.scale*rows - y_new

        if img is None:
            img = np.zeros([res*rows, res*cols, 3], dtype='uint8')

        pts = np.vstack([x_new, y_new]).T
        # for polylines, pts = pts.reshape((-1,1,2))
        pts = pts.reshape((-1, 2))
        pts = pts.astype(int)
        # render different color based on speed
        # slow - red, fast - green (BGR)

        if speed_profile is not None:
            def s_to_color(s):
                p = speed_profile
                return (p.v_fun(s)-p.min_v)/(p.max_v-p.min_v)
        else:
            def s_to_color(s):
                del s
                return 0

        def get_color(s):
            return (0, int(s_to_color(s)*255), int(255-255*s_to_color(s)))

        for i in range(len(ss)-1):
            color = get_color(ss[i])
            img = cv2.line(img, tuple(pts[i]), tuple(pts[i+1]), color=color, thickness=3)

        # plot reference points
        # img = cv2.polylines(img, [pts], isClosed=True, color=lineColor, thickness=3)
        if points is not None:
            for point in points:
                x = point[0]
                y = point[1]
                x *= self.config.resolution
                y *= self.config.resolution
                y = self.config.resolution*self.config.scale*rows - y

                img = cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)

        return img

    def draw_point(self, img, coord, color=(0, 0, 0)):
        """ draw a point on canvas at coord"""
        src = self.m2canvas(coord)
        img = cv2.circle(img, src, 3, color, -1)

        return img

    def draw_points(self, img, coord_vec, color=(0, 0, 0)):
        for coord in coord_vec:
            src = self.m2canvas(coord)
            img = cv2.circle(img, src, 3, color, -1)
        return img

    def process_rcp_raceline(self, raceline: RCPTrackRaceline):
        """ Given an RCPTrackRaceline object.
        Sample reference points, and distance to left/right boundary with create_boundary().
        Args:
            raceline: rcp track raceline
        Returns:
            r_vec: (N,2) Raceline ref points
            left: (N, ) Distance to left boundary
            right: (N, ) Distance to right boundary """
        ss = np.linspace(0, raceline.raceline_len_m, self.config.discretized_raceline_len)
        # (N, 2)
        r_vec = np.array(splev(ss, raceline.raceline_s, der=0)).T
        # (2, N)
        dr_vec = np.array(splev(ss, raceline.raceline_s, der=1))
        heading_vec = np.arctan2(dr_vec[1], dr_vec[0])
        bdry = self.create_boundary(r_vec, heading_vec)
        # Add smoothing. Occassionally boundary can extend to other grids
        # Remove the spike
        left = bdry[:, 0]
        right = bdry[:, 1]
        for i in range(left.shape[0]-1):
            if np.abs(left[i] - left[i+1]) > 0.5:
                left[i+1] = left[i]
            if np.abs(right[i] - right[i+1]) > 0.5:
                right[i+1] = right[i]

        return r_vec, left, right

    # RCPTrack is now a CurvilinearTrack, the parent class implementation will be more efficient
    # @deprecated
    # def __local_trajectory(self, state, wheelbase=90e-3):
    #     """ Given the state of the car, provide geometry information of the raceline.
    #     Args:
    #         state: CartesianState
    #     Return:
    #         LocalTrajOutput
    #         .ref_point: Closest point on raceline
    #         .lateral_err: Lateral error. Left deviation is positive.
    #         .heading_err: Orientation error from raceline tangent. CCW positive
    #         .curvature: Signed curvature, CCW positive
    #         .v_target: Reference speed at ref_point
    #         .progress: Curve length along raceline in Frenet frame.
    #     """
    #     # TODO refactor here onwards
    #     # figure out which grid the coord is in
    #     coord = np.array([state.x, state.y])
    #     heading = state.heading
    #     # find the coordinate of center of front axle
    #     coord[0] += wheelbase*cos(heading)
    #     coord[1] += wheelbase*sin(heading)
    #     # grid coordinate, (col, row), col starts from left and row starts from bottom,
    #     # both indexed from 0
    #     # coord should be given in meters
    #     nondim = np.array((coord/self.config.scale)//1, dtype=int)

    #     # distance squared, not need to find distance here
    #     def dist_2(a, b):
    #         return (a[0]-b[0])**2+(a[1]-b[1])**2

    #     def dist(u):
    #         return dist_2(splev(u % self.config.track_length_grid, self.rcp_raceline), coord)
    #     # last_u is the seq found last time, which should be a good estimate of where to start
    #     # disable this functionality since it doesn't handle multiple cars
    #     self.last_u = None
    #     if self.last_u is None:
    #         # the seq here starts from origin
    #         seq = -1
    #         # figure out which u this grid corresponds to
    #         for i, grid in enumerate(self.config.grid_sequence):
    #             if nondim[0] == grid[0] and nondim[1] == grid[1]:
    #                 seq = i
    #                 break

    #         if seq == -1:
    #             print('error, coord not on track, x = %.2f, y=%.2f' %
    #                   (coord[0], coord[1]))
    #             return None

    #         # the grid that contains the coord
    #         # print("in grid : " + str(self.grid_sequence[seq]))

    #         # find the closest point to the coord
    #         # because we wrapped the end point to the beginning of sample point,
    #         # we need to add this offset
    #         # Now seq would correspond to u in raceline,
    #         # i.e. allow us to locate the raceline at that section
    #         seq += self.config.origin_seq_no
    #         seq %= self.config.track_length_grid

    #         # this gives a close, usually preceding raceline point,
    #         # this does not give the closest ctrl point
    #         # due to smoothing factor
    #         # print("neighbourhood raceline pt " + str(splev(seq,self.raceline)))

    #         # determine which end is the coord closer to,
    #         # since seq points to the previous control point,
    #         # not necessarily the closest one
    #         if dist(seq+1) < dist(seq):
    #             seq += 1
    #         if dist(seq-1) < dist(seq):
    #             seq -= 1
    #     else:
    #         seq = self.last_u

    #     # Goal: find the point on raceline closest to coord
    #     # i.e. find x that minimizes dist(x)
    #     # we know x will be close to seq

    #     # easy method
    #     # brute force, This takes 77% of runtime.
    #     # lt.s('minimize_scalar')
    #     # res = minimize_scalar(dist,bounds=[seq-0.6,seq+0.6],method='Bounded')
    #     # lt.e('minimize_scalar')

    #     # improved method: from observation, dist(x) is quadratic in proximity of seq
    #     # we assume it to be ax^3 + bx^2 + cx + d and
    #     # formulate this minimization as a linalg problem
    #     # sample some points to build the trinomial simulation
    #     iv = np.array([-0.6, -0.3, 0, 0.3, 0.6])+seq
    #     # formulate linear problem
    #     A = np.vstack([iv**3, iv**2, iv, [1, 1, 1, 1, 1]]).T
    #     # B = np.mat([dist(x0), dist(x1), dist(x2)]).T
    #     B = dist(iv).T
    #     # abc = np.linalg.solve(A,B)
    #     abc = np.linalg.lstsq(A, B, rcond=-1)[0]
    #     a = abc[0]
    #     b = abc[1]
    #     c = abc[2]
    #     d = abc[3]

    #     def poly(x):
    #         return a*x*x*x + b*x*x + c*x + d
    #     fit = minimize(poly, x0=seq, method='L-BFGS-B', bounds=((seq-0.6, seq+0.6),))
    #     min_fun_x = fit.x[0]
    #     self.last_u = min_fun_x % self.config.track_length_grid

    #     min_fun_val = float(fit.fun)

    #     raceline_point = splev(min_fun_x %
    #                            self.config.track_length_grid, self.rcp_raceline)
    #     # raceline_point = splev(res.x,self.raceline)

    #     der = splev(min_fun_x % self.config.track_length_grid, self.rcp_raceline, der=1)
    #     # der = splev(res.x,self.raceline,der=1)

    #     # calculate whether offset is ccw or cw
    #     # achieved by finding cross product of vec(raceline_orientation) and vec(ctrl_pnt->test_pnt)
    #     # then find sin(theta)
    #     # negative offset means car is to the right of the trajectory
    #     vec_raceline = (der[0], der[1])
    #     vec_offset = coord - raceline_point
    #     cross_theta = np.cross(vec_raceline, vec_offset)

    #     vec_curvature = splev(min_fun_x %
    #                           self.config.track_length_grid, self.rcp_raceline, der=2)
    #     norm_curvature = np.linalg.norm(vec_curvature)
    #     # gives right sign for omega,
    #     # this is indep of track direction since it's calculated based off vehicle orientation
    #     # cross_curvature = np.cross((cos(heading),sin(heading)),vec_curvature)
    #     cross_curvature = der[0]*vec_curvature[1]-der[1]*vec_curvature[0]

    #     # return target velocity
    #     request_velocity = self.targetVfromU(min_fun_x % self.track_length_grid)
    #     left, right = self.precise_track_boundary((state.x, state.y), state.heading)

    #     retval = LocalTrajOutput(ref_point=raceline_point,
    #                              lateral_err=copysign(abs(min_fun_val)**0.5, cross_theta),
    #                              raceline_dir=atan2(der[1], der[0]),
    #                              curvature=copysign(norm_curvature, cross_curvature),
    #                              v_target=request_velocity,
    #                              progress=self.uToS(min_fun_x % self.track_length_grid),
    #                              left_margin=left,
    #                              right_margin=right
    #                              )
    #     return retval
