# a track defined by a spline
import numpy as np
from math import cos, sin, atan2

from scipy.interpolate import splprep, splev

from buzzracer.common import *
from buzzracer.tracks.track import Track


class CurvilinearTrack(Track):
    def __init__(self, main, config):
        Track.__init__(self, main, config)
        # default parameters, to be override
        self.width = 0.8
        self.resolution = 200

        # derived class should override the constructor,
        # call self.build_continuous_track(r) to create a curvilinear track
        # r.shape == (n,2), and consistes of the discretized track centerline
        # see NascarTrack.py for example
        self.discretized_raceline_len = 1024
        return

    def draw_raceline(self, img):
        return img

    # state: x,y,theta,vf,vs,omega
    # x,y referenced from skidpad frame
    def local_trajectory(self, state, ccw=True, wheelbase=108e-3):
        x = state[0]
        y = state[1]
        heading = state[2]
        vf = state[3]
        vs = state[4]
        omega = state[5]

        # find the coordinate of center of front axle
        x += wheelbase*cos(heading)
        y += wheelbase*sin(heading)

        # TODO optimize this
        dxx = self.r[:, 0]-x
        dyy = self.r[:, 1]-y
        index = np.argmin(dxx**2+dyy**2)
        raceline_point = (self.r[index])

        # find offset
        # positive offset means car is to the left of the trajectory(need to turn right)
        dr = self.r[index+1] - self.r[index]
        track_to_car = (x-self.r[index, 0], y-self.r[index, 1])
        offset = np.cross(dr/np.linalg.norm(dr), track_to_car).item()

        raceline_orientation = atan2(dr[1], dr[0])

        signed_curvature = splev(self.ss[index], self.curvature)[0].item()

        # reference point on raceline,lateral offset, tangent line orientation, curvature(signed, ccw+)
        return (raceline_point, offset, raceline_orientation, signed_curvature, 2.0)

    # return true if vehicle is unsalvageably outside of the track
    # for use by Watchdog to terminate an experiment
    def is_outside(self, coord):
        state = (*coord, 0, 0, 0, 0)
        _, offset, _, _, _ = self.local_trajectory(state, wheelbase=0)
        return offset > (self.width/2)*1.5

    def build_continuous_track(self, r):
        self.r = r
        assert (len(self.r.shape) == 2)
        assert (self.r.shape[1] == 2)
        n = self.r.shape[0]
        xx = r[:, 0]
        yy = r[:, 1]

        s = 0
        ss = [s]
        for i in range(n):
            s += ((xx[(i+1) % n]-xx[i])**2 + (yy[(i+1) % n]-yy[i])**2)**0.5
            ss.append(s)
        self.ss = ss
        self.raceline_len_m = s
        self.r = np.vstack([self.r, self.r[-1]])

        tck, u = splprep(self.r.T, u=ss, s=0, per=1)
        self.raceline_s = tck

        # let raceline curve be r(u)
        # dr = r'(u), parameterized with xx/u
        dr = np.array(splev(ss, self.raceline_s, der=1))
        # ddr = r''(u)
        ddr = np.array(splev(ss, self.raceline_s, der=2))
        def _norm(x): return np.linalg.norm(x, axis=0)
        # radius of curvature can be calculated as R = |y'|^3/sqrt(|y'|^2*|y''|^2-(y'*y'')^2)
        curvature = 1.0/(_norm(dr)**3/(_norm(dr)**2*_norm(ddr)
                         ** 2 - np.sum(dr*ddr, axis=0)**2)**0.5)
        self.curvature, u = splprep(curvature.reshape(1, -1), u=ss, s=0, per=1)
        s_vec = self.ss
        # n*2
        ss = np.linspace(0, self.raceline_len_m, 3000)
        r_vec = np.array(splev(ss, self.raceline_s, der=0))
        dr_vec = np.array(splev(ss, self.raceline_s, der=1))
        self.phi = np.arctan2(dr_vec[1, :], dr_vec[0, :])
        lateral = np.vstack(
            [np.cos(self.phi+np.pi/2), np.sin(self.phi+np.pi/2)]).T
        # boundary
        upper = r_vec.T + lateral * self.width/2
        lower = r_vec.T - lateral * self.width/2

        self.x_min = np.min(np.hstack([upper[:, 0], lower[:, 0]])) - 0.1
        self.x_max = np.max(np.hstack([upper[:, 0], lower[:, 0]])) + 0.1
        self.y_min = np.min(np.hstack([upper[:, 1], lower[:, 1]])) - 0.1
        self.y_max = np.max(np.hstack([upper[:, 1], lower[:, 1]])) + 0.1

        # shift track to first quadrant, x,y>0
        '''
        self.x_limit = x_max - x_min
        self.y_limit = y_max - y_min
        upper[:,0] -= x_min
        upper[:,1] -= y_min
        lower[:,0] -= x_min
        lower[:,1] -= y_min
        r_vec[:,0] -= x_min
        r_vec[:,1] -= y_min
        '''

        self.r_vec = r_vec
        self.upper = upper
        self.lower = lower

        # self.raceline_len_m = s_vec[-1]
        # self.raceline_s = self.build_spline(r_vec)
        # self.upper_fun = self.build_spline(upper)
        # self.lower_fun = self.build_spline(lower)

        '''
        plt.plot(upper[:,0],upper[:,1])
        plt.plot(lower[:,0],lower[:,1])
        plt.plot(r_vec[0,:],r_vec[1,:],'o')
        plt.show()
        '''
        self.prepare_discretized_raceline()

    def prepare_discretized_raceline(self):
        ss = np.linspace(0, self.raceline_len_m, self.discretized_raceline_len)
        rr = splev(ss % self.raceline_len_m, self.raceline_s, der=0)
        drr = splev(ss % self.raceline_len_m, self.raceline_s, der=1)
        heading_vec = np.arctan2(drr[1], drr[0])

        # parameter, distance along track
        self.ss = ss
        self.raceline_points = np.array(rr)
        self.r = self.raceline_points.T
        self.raceline_headings = heading_vec

        # describe track boundary as offset from raceline
        self.raceline_left_boundary = np.ones_like(ss)*self.width/2
        self.raceline_right_boundary = np.ones_like(ss)*self.width/2
        self.discretized_raceline = np.vstack(
            [self.raceline_points, self.raceline_headings, self.raceline_left_boundary, self.raceline_right_boundary]).T
        return

    def m2canvas(self, coord):
        x_new = int(
            (np.clip(coord[0], self.x_min, self.x_max)-self.x_min) * self.resolution)
        y_new = int(
            (self.y_max - np.clip(coord[1], self.y_min, self.y_max)) * self.resolution)
        return (x_new, y_new)

    # draw a picture of the track
    def draw_track(self):
        x_pix = int((self.x_max - self.x_min)*self.resolution)
        y_pix = int((self.y_max - self.y_min)*self.resolution)
        # height, width
        img = 255*np.ones([y_pix, x_pix, 3], dtype=np.uint8)
        img = self.draw_polyline(
            self.upper, img, lineColor=(0, 0, 0), thickness=2)
        img = self.draw_polyline(
            self.lower, img, lineColor=(0, 0, 0), thickness=2)
        return img

    def precise_track_boundary(self, coord, heading):
        state = (coord[0], coord[1], heading, 0, 0, 0)
        raceline_point, offset, raceline_orientation, signed_curvature, _ = self.local_trajectory(
            state)
        left = self.width/2 - offset
        right = self.width/2 + offset
        return (left, right)


if __name__ == '__main__':
    pass
