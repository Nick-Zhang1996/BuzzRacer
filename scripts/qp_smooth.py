''' Smooth path to minimize curvature norm with quadratic programming. '''
# pylint: disable=all
# following paper "A quadratic programming approach to path smoothing"

from math import pi, radians, tan
import os.path
import argparse
from time import time
from dataclasses import replace
from deprecated import deprecated
import logging

import numpy as np
import cv2
import cvxopt
from scipy.interpolate import splev, splprep, interp1d
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from buzzracer.common import print_ok, print_info, print_error, BASEDIR
from buzzracer.tracks.rcp_track import RCPTrackRaceline, RCPTrack
from buzzracer.tracks.curvilinear_track import CurvilinearTrack
from buzzracer.tracks.track_factory import TrackFactory
from buzzracer.tracks.track import Track

logging.basicConfig(level=logging.INFO)


class BezierSpline:
    def __init__(self, break_pts):
        self.N = break_pts.shape[0]
        assert break_pts.shape == (self.N, 2)
        self.break_pts = break_pts
        """ Break points on spline"""
        self.P = BezierSpline.bezier_spline(break_pts)
        """ Array of control points, shape (n,2,5) """

    @staticmethod
    def bezier_spline(break_pts):
        '''
        Generate a bezier spline matching derivative estimated from lagrange interpolation
        Args:
            break_pts: (n,2)
        Return:
            vector function, domain [0,len(points)]
        '''
        break_pts = np.array(break_pts).T

        # calculate first and second derivative
        # w.r.t. ds, estimated with 2-norm
        df = []
        ddf = []
        n = break_pts.shape[1]
        for i in range(n):
            rl = break_pts[:, (i-1) % n]
            r = break_pts[:, (i) % n]
            rr = break_pts[:, (i+1) % n]
            points = [rl, r, rr]

            ((al, a, ar), (bl, b, br)) = lagrange_der(points)
            df.append(al*rl + a*r + ar*rr)
            ddf.append(bl*rl + b*r + br*rr)

        P = []
        for i in range(n):
            # generate bezier spline segments
            rl = break_pts[:, (i) % n]
            r = break_pts[:, (i+1) % n]
            section_P = BezierSpline.bezier_curve(
                [rl, r], [df[i], df[(i+1) % n]], [ddf[i], ddf[(i+1) % n]], ds=None)
            # NOTE testing
            def B(t, p): return (1-t)**5*p[0] + 5*t*(1-t)**4*p[1] + 10*t**2*(
                1-t)**3*p[2] + 10*t**3*(1-t)**2*p[3] + 5*t**4*(1-t)*p[4] + t**5*p[5]
            x_i = B(0, section_P[:, 0])
            y_i = B(0, section_P[:, 1])
            x_f = B(1, section_P[:, 0])
            y_f = B(1, section_P[:, 1])
            assert np.isclose(x_i, rl[0], atol=1e-5) and np.isclose(y_i, rl[1], atol=1e-5) and np.isclose(
                x_f, r[0], atol=1e-5) and np.isclose(y_f, r[1], atol=1e-5)

            P.append(section_P)

        # NOTE verify P dimension n*2*5
        return np.array(P)

    @staticmethod
    def bezier_curve(r, dr, ddr, ds=None):
        '''
        construct a fifth order bezier curve passing through endpoints r
        matching first and second derivative dr, ddr
        Args:
            r (2,2) = rl (start point dim (2,)), rr (end point dim (2,))
            dr (2,2),  taken w.r.t. arc length s
            ddr (2,2), taken w.r.t. arc length s
            ds: arc length between the endpoints
        Return:
            Control points (6,2)
        '''
        rl, rr = r
        drl, drr = dr
        ddrl, ddrr = ddr

        def dist(x, y):
            return ((x[0]-y[0])**2 + (x[1]-y[1])**2)**0.5
        if ds is None:
            ds = dist(rl, rr)

        # two sets of equations, one for x, one for y
        # bx = np.array([rl[0],rr[0],drl[0],drr[0],ddrl[0],ddrr[0]]).T
        # by = np.array([rl[1],rr[1],drl[1],drr[1],ddrl[1],ddrr[1]]).T

        # dr = dr/ds = dr/dt * dt/ds
        # we want dB/dt = dr/dt = dr(input) * ds/dt = dr * ds(between two endpoints)
        bx = np.array([rl[0], rr[0], drl[0]*ds, drr[0] *
                      ds, ddrl[0]*ds*ds, ddrr[0]*ds*ds]).T
        by = np.array([rl[1], rr[1], drl[1]*ds, drr[1] *
                      ds, ddrl[1]*ds*ds, ddrr[1]*ds*ds]).T
        b = np.vstack([bx, by]).T

        # x_x = P0_x, P1_x ... P5_x
        # x_y = P0_y, P1_y ... P5_y
        A = [[1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1],
             [-5, 5, 0, 0, 0, 0],
             [0, 0, 0, 0, -5, 5],
             [20, -40, 20, 0, 0, 0],
             [0, 0, 0, 20, -40, 20]]
        A = np.array(A)

        try:
            # solve for control points
            P = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            print_error("can't solve bezier Curve")

        return P

    def resample_bezier(self, new_n):
        """ Resample Bezier spline defined by break_pts with equal arc distance.
        Args:
            break_pts: break points
            new_n: number of break points on the new path
        Return:
            BezierSpline
          """
        # generate bezier spline
        break_pts = self.break_pts
        N = self.N

        # resample with equal arc distance
        # NOTE this seems to introduce instability
        arc_len = [0]
        # we have N+1 points here
        for i in range(1, N+1):
            s = QpSmooth.arc_len(self.eval, i-1, i)
            arc_len.append(s)
        uu = np.linspace(0, N, N+1)
        arc_len = np.array(arc_len)
        arc_len = np.cumsum(arc_len)
        s2u = interp1d(arc_len, uu)
        # raceline(0) and raceline(arc_len) are the same points, discard the latter
        ss = np.linspace(0, arc_len[-1], new_n+1)[:-1]
        uu = s2u(ss)

        # uu += np.hstack([0,np.random.rand(new_n-2)/3,0])
        new_break_pts = []
        for u in uu:
            new_break_pts.append(self.eval(u).flatten())

        # regenerate spline
        new_break_pts = np.array(new_break_pts)
        assert new_break_pts.shape == (new_n, 2)
        return BezierSpline(new_break_pts)

    def eval(self, u):
        '''
        u (iterable): parameter, domain [0,n], where n is number of break points in spline generation
        '''
        P = self.P
        u = np.array(u).reshape(-1, 1)
        n = len(P)
        assert (u >= 0).all()
        assert (u <= n).all()

        def B(t, p): return (1-t)**5*p[0] + 5*t*(1-t)**4*p[1] + 10*t**2*(
            1-t)**3*p[2] + 10*t**3*(1-t)**2*p[3] + 5*t**4*(1-t)*p[4] + t**5*p[5]

        try:
            r = [[B(uu % 1, np.array(P[int(uu[0]) % n, :, 0])), B(uu %
                                                                  1, np.array(P[int(uu[0]) % n, :, 1]))] for uu in u]
        except Warning as e:
            print(e)

        return np.array(r)


def lagrange_der(points, ds=None):
    '''
    Given three points, calculate first and second derivative as
    a linear combination of the three points rl, r, rr,
    which stand for r_(k-1), r_k, r_(k+1)
    where f'@r = al*rl + a*r + ar*rr
    where f''@r = bl*rl + b*r + br*rr
    if not specified, |r-rl|_2 will be used as approximation
    Args:
        Points: List of 3 points, each point of dim (2,)
        ds: arc length between rl,r and r, rr, list of (2)
    Return:
        (2,3) tuple
        ((al, a, ar),
        (bl, b, br))
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


class QpSmooth:

    @staticmethod
    def arc_len(fun, ui, uf):
        '''Calculate arc length of <x,y> = fun(u) from ui to uf'''
        steps = 20
        uu = np.linspace(ui, uf, steps)
        s = 0
        last_x, last_y = fun(ui).flatten()
        for i in range(steps):
            x, y = fun(uu[i]).flatten()
            s += ((x-last_x)**2 + (y-last_y)**2)**0.5
            last_x, last_y = x, y
        return s

    @staticmethod
    def curvature_jac(spline: BezierSpline):
        '''
        Calculate variance of curvature w.r.t. break point variation
        correspond to equation 6 in paper
        '''
        break_pts = spline.break_pts.T
        N = break_pts.shape[1]
        assert break_pts.shape == (2, N)
        A = np.array([[0, -1], [1, 0]])

        def raceline_fun(u):
            return spline.eval(u).flatten()
        ds = []
        for i in range(N):
            # TODO flatten()?
            ds.append(QpSmooth.arc_len(raceline_fun, i, (i+1)))

        # calculate first and second derivative
        # w.r.t. ds
        dr_vec = []
        ddr_vec = []
        # (N,3)
        # see eq 1
        alfa_vec = []
        # see eq 2
        beta_vec = []
        # see eq 6
        x_vec = []
        # see eq 3
        k_vec = []
        # normal vector
        n_vec = []

        # calculate terms in eq 6
        for i in range(N):
            # rl -> r_k-1
            rl = break_pts[:, (i-1) % N]
            # r -> r_k
            r = break_pts[:, (i) % N]
            # rr -> r_k+1
            rr = break_pts[:, (i+1) % N]
            points = [rl, r, rr]
            sl = ds[(i-1) % N]
            sr = ds[(i) % N]

            ((al, a, ar), (bl, b, br)) = lagrange_der(points, ds=(sl, sr))
            dr = al*rl + a*r + ar*rr
            ddr = bl*rl + b*r + br*rr

            dr_vec.append(dr)
            ddr_vec.append(ddr)

            alfa_vec.append([al, a, ar])
            beta_vec.append([bl, b, br])

            n = A @ dr
            n_vec.append(n.T)

        for i in range(N):
            # curvature at this characteristic point
            k = np.dot(A @ dr_vec[i], ddr_vec[i])
            xl = np.dot(A @ dr_vec[i], beta_vec[i][0] * n_vec[(i-1) % N])
            xl += np.dot(ddr_vec[i], alfa_vec[i][0] * A @ n_vec[(i-1) % N])

            x = beta_vec[i][1] + \
                np.dot(ddr_vec[i], alfa_vec[i][1] * A @ n_vec[i])

            xr = np.dot(A @ dr_vec[i], beta_vec[i][2] * n_vec[(i+1) % N])
            xr += np.dot(ddr_vec[i], alfa_vec[i][2] * A @ n_vec[(i+1) % N])

            k_vec.append(k)
            x_vec.append([xl, x, xr])

        # assemble matrix K, C, Ds
        x_vec = np.array(x_vec)
        k_vec = np.array(k_vec)

        K = np.array(k_vec).reshape(N, 1)
        C = np.zeros([N, N])
        C[0, 0] = x_vec[0, 1]
        C[0, 1] = x_vec[0, 2]
        C[0, -1] = x_vec[0, 0]

        C[-1, -2] = x_vec[-1, 0]
        C[-1, -1] = x_vec[-1, 1]
        C[-1, 0] = x_vec[-1, 2]

        for i in range(1, N-1):
            C[i, i-1] = x_vec[i, 0]
            C[i, i] = x_vec[i, 1]
            C[i, i+1] = x_vec[i, 2]

        C = np.array(C)

        # NOTE Ds is not simply ds
        # it is a helper for trapezoidal rule
        Ds = np.array(ds[:-2]) + np.array(ds[1:-1])
        Ds = np.hstack([ds[0], Ds, ds[-1]])

        Ds = 0.5*np.array(np.diag(Ds))

        # Debug variables
        # curvature
        # k = k_vec
        # ddr = ddr_vec

        # Tangent vector
        dr = np.array(dr_vec)
        # normal vector
        n = np.array(n_vec).reshape(-1, 2)

        return K, C, Ds, dr, n

    @staticmethod
    def convert_to_spline(spline):
        """Convert Bezier spline to a scipy.splprep B spline """
        steps = 100
        uu = np.linspace(0, spline.N, steps)
        r = spline.eval(uu).reshape(-1, 2).T
        # s = smoothing factor
        # per = loop/period
        tck, u = splprep(r, u=np.linspace(0, spline.N, steps), s=0, per=1)
        return tck

    def save_track_image(self):
        img_track = self.draw_track()
        # img_track = super().draw_raceline(img=img_track, points=self.break_pts)
        # do not show break points
        img_track = super().draw_raceline(img=img_track, points=[])
        if (self.img_filename is not None):
            filename = os.path.join(self.save_dir, self.img_filename)
            cv2.imwrite(filename, img_track)
            print(f'saved at {filename}')
        img_track_rgb = cv2.cvtColor(img_track.copy(), cv2.COLOR_BGR2RGB)
        plt.imshow(img_track_rgb)
        plt.show()
        return

    def test_lagrange_der(self):
        # generate three points
        points = np.array([[-1, -1], [2, 2], [5, 5]])
        rl, r, rr = points
        ((al, a, ar), (bl, b, br)) = self.lagrange_der(points)
        df = al*rl + a*r + ar*rr
        ddf = bl*rl + b*r + br*rr
        print(df)
        print(ddf)
        x = points[:, 0]
        y = points[:, 1]
        plt.plot(x, y)
        plt.show()
        return

    # test bezier curve
    # @deprecated
    # def test_bezier_curve(self):
    #     # generate a batch of points following a unit circle
    #     u = np.linspace(0, (2*pi)/17.0*14, 17)
    #     xx = np.cos(u)
    #     yy = np.sin(u)
    #     points = np.vstack([xx, yy]).T

    #     tic = time()
    #     # just test w/ an outlier
    #     points[4, :] = [0, 1.2]
    #     P = self.bezier_spline(points)

    #     u_close = np.linspace(0, points.shape[0], 1000)
    #     r = self.eval_bezier_spline(P, u_close)
    #     print(time()-tic)

    #     B_x = r[:, 0]
    #     B_y = r[:, 1]

    #     plt.plot(points[:, 0], points[:, 1], 'bo')
    #     plt.plot(B_x, B_y)

    #     plt.show()

    #     return

    # plot Bezier curve based raceline on a track map
    # @deprecated
    # def test_track(self):
    #     # prepare the full racetrack
    #     self.prepareTrack()
    #     # use control points as bezier breakpoints
    #     # generate bezier spline
    #     self.P = self.bezier_spline(self.ctrl_pts)
    #     self.u_max = len(self.ctrl_pts)
    #     self.raceline_fun = lambda u: self.eval_bezier_spline(self.P, u)
    #     # render
    #     img_track = self.draw_track()
    #     img_track = self.draw_raceline(img=img_track)
    #     plt.imshow(img_track)
    #     plt.show()

    # @deprecated
    # def test_curvature_jac(self, param=0):
    #     # prepare the full racetrack
    #     self.prepareTrack()
    #     self.break_pts = self.ctrl_pts

    #     # use control points as bezier breakpoints
    #     # generate bezier spline
    #     self.P = self.bezier_spline(self.break_pts)
    #     self.u_max = len(self.break_pts)
    #     self.raceline_fun = lambda u: self.eval_bezier_spline(self.P, u)

    #     K, C, Ds = self.curvature_jac()

    #     # verify J(X) = (K+CX).T Ds (K+CX)
    #     # test model with random variation

    #     # calculate J(0), without K,C,D
    #     N = self.u_max
    #     ds = self.ds
    #     k = self.k
    #     J = 0
    #     # trapezoidal rule
    #     J += 0.5*k[0]**2*ds[0]
    #     for i in range(1, N-1):
    #         J += 0.5*k[i]**2 * (ds[i-1]+ds[i])
    #     J += 0.5*k[-1]**2*ds[-1]

    #     X = np.zeros([N, 1])
    #     M_temp = K + C @ X
    #     J_m = M_temp.T @ Ds @ M_temp

    #     # error sum
    #     # print("err sum, un perturbed ")
    #     # print(np.sum(np.abs(J - J_m)))

    #     # calculate J(X) with random perturbation X
    #     delta_max = 1e-2
    #     # X = (np.random.random([N,1])-0.5)/0.5*delta_max
    #     X = np.zeros([N, 1])
    #     X[param, 0] = delta_max
    #     # move break points in tangential direction by X
    #     n = np.array(self.n).reshape(-1, 2)
    #     perturbed_break_pts = np.array(self.break_pts)
    #     for i in range(N):
    #         perturbed_break_pts[i, :] += n[i]*X[i]

    #     # visually verify the pertuabation is reasonable
    #     # i.e. tangential to path and small in magnitude
    #     old_pts = np.array(self.break_pts)
    #     plt.plot(old_pts[:, 0], old_pts[:, 1], 'ro')
    #     new_pts = perturbed_break_pts
    #     plt.plot(new_pts[:, 0], new_pts[:, 1], 'b*')
    #     plt.show()
    #     print('curvature before')
    #     print(self.k[param])

    #     # show raceline
    #     img_track = self.draw_track()
    #     img_track = self.draw_raceline(img=img_track)
    #     plt.imshow(img_track)
    #     plt.show()

    #     # calculate predicted new J(X) with
    #     # J(X) = (K+CX).T Ds (K+CX)
    #     M_temp = K + C @ X
    #     J_pm = M_temp.T @ Ds @ M_temp

    #     # calculate new J(X) without matrices
    #     # this requires re-generation of the Bezier Spline
    #     self.break_pts = new_pts
    #     self.P = self.bezier_spline(self.break_pts)
    #     self.raceline_fun = lambda u: self.eval_bezier_spline(self.P, u)
    #     # need this to calculate new ds and k
    #     K, C, Ds = self.curvature_jac()
    #     ds = self.ds
    #     k = self.k
    #     J_p = 0
    #     # trapezoidal rule
    #     J_p += 0.5*k[0]**2*ds[0]
    #     for i in range(1, N-1):
    #         J_p += 0.5*k[i]**2 * (ds[i-1]+ds[i])
    #     J_p += 0.5*k[-1]**2*ds[-1]
    #     # print("err sum, random perturbation ")
    #     # print(np.sum(np.abs(J_p - J_pm)))
    #     # 0.2 error
    #     # are we at least going in right direction?
    #     print('qualitative verification')
    #     # print(np.sum(np.abs(J_p - J_m)))
    #     print('ground truth')
    #     print(J_p-J)

    #     print('jacobian prediction')
    #     # print(np.sum(np.abs(J_p - J_pm)))
    #     print(J_pm[0, 0]-J)
    #     print('curvature after')
    #     print(self.k[param])

    #     # show raceline
    #     img_track = self.draw_track()
    #     img_track = self.draw_raceline(img=img_track)
    #     plt.imshow(img_track)
    #     plt.show()

    #     # render
    #     '''
    #     img_track = self.draw_track()
    #     img_track = self.draw_raceline(img=img_track)
    #     plt.imshow(img_track)
    #     plt.show()
    #     '''

    #     # 1.0 is ideal
    #     return (J_p-J)/(J_pm[0, 0]-J)

    def draw_raceline(self, raceline: BezierSpline, track, img):
        assert isinstance(raceline, BezierSpline)
        ss = np.linspace(0, raceline.N, 1000)
        points = raceline.eval(ss).reshape((-1, 2))
        return track.draw_polyline(points, img)

    def optimize_raceline(self,
                          raceline: RCPTrackRaceline,
                          *,
                          track=None,
                          max_iter=20,
                          offset=0,
                          visualize=False,
                          visualize_final_result=True,
                          save_gif=False,
                          save_steps=False,
                          ):
        ''' Optimize path and save to pickle file
        Args:
            max_iter: max iteration for path smoothing
            offset: offset from lateral constraints, higher means more room left
            visualize: if True, visualize each iteration in plt
            visualize_final_result: if True, visualize final result
            save_gif: if True, save a gif of the optimization
            save_steps: if True, save each optimization step as a png
        Return:
        '''

        # use control points as initial bezier breakpoints
        # elg. for full track there are 24 points
        # break_pts = np.array(raceline.control_points)
        # re-sample path, get more break points
        # N = len(break_pts)*3
        N = 200
        ss = np.linspace(0, raceline.raceline_len_m, N)
        break_pts = np.array(splev(ss, raceline.raceline_s, der=0)).T
        spline = BezierSpline(break_pts)
        spline = spline.resample_bezier(N)
        print_info('Had %d break points, resample to %d' % (len(break_pts), N))

        if save_gif:
            gifimages = []
            # self.gifimages.append(Image.fromarray(cv2.cvtColor(self.img_track.copy(),cv2.COLOR_BGR2RGB)))

        for iter_count in range(max_iter):
            print_ok('iter: %d' % (iter_count,))
            # re-sample reference points before every iteration
            spline = spline.resample_bezier(N)

            K, C, Ds, dr, n = QpSmooth.curvature_jac(spline)

            # assemble matrices in QP
            # NOTE ignored W, W=I
            P_qp = 2 * C.T @ Ds @ C
            q_qp = np.transpose(K.T @ Ds @ C + K.T @ Ds @ C)

            # assemble constrains
            # as in Gx <= h

            # track boundary
            # h = [F..., R...], split into two vec
            h1 = []
            h2 = []
            left = []
            right = []
            delta_max = 5e-2
            for i in range(N):
                coord = spline.break_pts[i]
                heading = np.arctan2(dr[i, 1], dr[i, 0])
                L, R = track.precise_track_boundary(coord, heading)
                # L, R = track.old_check_track_boundary(
                #     coord, heading, delta_max=delta_max, offset=offset)
                # TODO unclip lower bound
                left.append(L)
                right.append(R)
                h1.append(np.clip(L-offset, 0, delta_max))
                h2.append(np.clip(R-offset, 0, delta_max))

            h = np.array(h1+h2)
            G = np.vstack([np.identity(N), -np.identity(N)])

            # curvature constrain
            # CX <= Kmax - K
            # min radius allowed from kinematic constraints
            Rmin = 0.102/tan(radians(18))
            Kmax = 1.0/Rmin
            Kmin = -1.0/Rmin
            h3 = Kmax - K
            h3 = h3.flatten()
            h4 = -(Kmin - K)
            h4 = h4.flatten()
            h = np.hstack([h, h3, h4])
            G = np.vstack([G, C, -C])
            print_info('min radius = %.2f' % np.min(np.abs(1.0/K)))

            assert G.shape[1] == N
            assert G.shape[0] == 4*N
            assert h.shape[0] == 4*N

            # optimize
            P_qp = cvxopt.matrix(P_qp)
            q_qp = cvxopt.matrix(q_qp)
            G = cvxopt.matrix(G)
            h = cvxopt.matrix(h)
            cvxopt.solvers.options['show_progress'] = False
            sol = cvxopt.solvers.qp(P_qp, q_qp, G, h)

            variance = sol['x']
            # verify Gx <= h
            # print("h-GX, should be positive")
            constrain_met = np.array(h) - np.array(G) @ np.array(variance)
            assert constrain_met.all()

            # verify K do not violate constrain
            # assert (Kmax-K >0).all()
            # assert (K-Kmin >0).all()

            # check terminal condition
            print_info('max variation %.2f' % (np.max(np.abs(variance))))
            if np.max(np.abs(variance)) < 0.1*delta_max:
                print_ok('terminal condition met')
                break

            # apply changes to break points
            # move break points in tangential direction by variance vector
            n = np.array(n).reshape(-1, 2)
            perturbed_break_pts = np.array(spline.break_pts)
            for i in range(N):
                perturbed_break_pts[i, :] += n[i]*variance[i]

            if save_steps or visualize or save_gif:
                img_track = track.draw_track()
                img_track = self.draw_raceline(spline, track, img=img_track)
                A = np.array([[0, -1], [1, 0]])  # ccw 90 deg
                left_boundary = []
                right_boundary = []
                for i, p in enumerate(spline.break_pts):
                    img_track = track.draw_point(img_track, p)
                    n = A @ dr[i]
                    left_boundary.append(p + left[i] * n / np.linalg.norm(n))
                    right_boundary.append(p - right[i] * n / np.linalg.norm(n))
                left_boundary = np.array(left_boundary)
                right_boundary = np.array(right_boundary)
                for p in np.vstack([left_boundary, right_boundary]):
                    img_track = track.draw_point(img_track, p)

            if save_steps:
                filename = os.path.join(BASEDIR, 'outputs', f'qp_smooth_iter{iter_count}.png')
                cv2.imwrite(filename, img_track)
                print(f'iteration image saved at {filename}')
            if visualize:
                img_track_rgb = cv2.cvtColor(img_track.copy(), cv2.COLOR_BGR2RGB)
                plt.imshow(img_track_rgb)
                plt.show()
            if save_gif:
                gifimages.append(Image.fromarray(
                    cv2.cvtColor(img_track.copy(), cv2.COLOR_BGR2RGB)))

            spline = BezierSpline(perturbed_break_pts)

        tck = QpSmooth.convert_to_spline(spline)
        raceline_s, raceline_len_m = Track.reparam_raceline(tck, spline.N)

        # self.verify_speed_profile(speed_profile_fun=speed_profile_fun)
        if save_gif:
            print_info('saving gif.. This may take a while')
            log_no = 0
            gif_filename = './qpOpt'+str(log_no)+'.gif'
            self.gifimages[0].save(fp=gif_filename, format='GIF',
                                   append_images=gifimages, save_all=True, duration=600, loop=0)
            print_ok('gif saved at '+gif_filename)

        if visualize_final_result:
            img_track = track.draw_track()
            img_track = self.draw_raceline(spline, track, img=img_track)
            img_track_rgb = cv2.cvtColor(img_track.copy(), cv2.COLOR_BGR2RGB)
            plt.imshow(img_track_rgb)
            plt.show()
        return RCPTrackRaceline(raceline_s=raceline_s,
                                raceline_len_m=raceline_len_m,
                                start_pos=raceline.start_pos,
                                start_dir=raceline.start_dir)


def _apply_safety_margin(left, right, safety_margin):
    left = np.clip(left - safety_margin, 0.0, None)
    right = np.clip(right - safety_margin, 0.0, None)
    return left, right


def _rebuild_boundary_only(track: RCPTrack, safety_margin: float):
    data = track.data
    bdry = track.create_boundary(data.r_vec, data.phi_vec)
    left = bdry[:, 0]
    right = bdry[:, 1]

    for i in range(left.shape[0] - 1):
        if np.abs(left[i] - left[i + 1]) > 0.5:
            left[i + 1] = left[i]
        if np.abs(right[i] - right[i + 1]) > 0.5:
            right[i + 1] = right[i]

    left, right = _apply_safety_margin(left, right, safety_margin)
    _update_track_boundary_widths(track, left, right)


def _update_track_boundary_widths(track: RCPTrack, left: np.ndarray, right: np.ndarray):
    data = track.data
    lateral = np.column_stack([np.cos(data.phi_vec + np.pi / 2),
                               np.sin(data.phi_vec + np.pi / 2)])
    left_boundary_vec = data.r_vec + lateral * left[:, np.newaxis]
    right_boundary_vec = data.r_vec - lateral * right[:, np.newaxis]
    discretized_raceline = data.discretized_raceline.copy()
    discretized_raceline[:, 3] = left
    discretized_raceline[:, 4] = right

    track.data = replace(data,
                         left_width_vec=left,
                         right_width_vec=right,
                         discretized_raceline=discretized_raceline,
                         left_boundary_vec=left_boundary_vec,
                         right_boundary_vec=right_boundary_vec)


def _render_loaded_track_image(track: RCPTrack, selected_idx: int | None = None):
    img = track.draw_track()
    rl = track.rcp_raceline
    img = track.draw_raceline(rl.raceline_s, rl.raceline_len_m, img=img)
    data = track.data
    img = track.draw_polyline(data.left_boundary_vec, img=img)
    img = track.draw_polyline(data.right_boundary_vec, img=img)
    if selected_idx is not None:
        center = track.m2canvas(data.r_vec[selected_idx])
        left_pt = track.m2canvas(data.left_boundary_vec[selected_idx])
        right_pt = track.m2canvas(data.right_boundary_vec[selected_idx])
        img = cv2.line(img, left_pt, right_pt, (0, 255, 255), 4)
        img = cv2.circle(img, left_pt, 6, (0, 140, 255), -1)
        img = cv2.circle(img, right_pt, 6, (0, 140, 255), -1)
        img = cv2.circle(img, center, 7, (0, 200, 0), -1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _show_loaded_track(track: RCPTrack):
    fig, ax = plt.subplots()
    ax.imshow(_render_loaded_track_image(track))
    ax.set_title('Loaded track verification')
    ax.axis('off')
    plt.show()


def _launch_boundary_margin_editor(track: RCPTrack):
    data = track.data
    left = data.left_width_vec.copy()
    right = data.right_width_vec.copy()
    saved = {'left': left.copy(), 'right': right.copy()}
    step_m = float(np.median(np.diff(data.s_vec)))
    sigma_m = max(step_m * 6.0, 0.05)
    radius_m = sigma_m * 3.0
    width_max = float(max(np.max(left), np.max(right), track.config.scale) + 0.15)
    state = {'selected_idx': 0, 'syncing': False, 'dirty': False, 'action': None}

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.32)
    image_artist = ax.imshow(_render_loaded_track_image(track, selected_idx=0))
    ax.axis('off')

    save_ax = fig.add_axes([0.15, 0.24, 0.12, 0.045])
    discard_ax = fig.add_axes([0.30, 0.24, 0.12, 0.045])
    progress_ax = fig.add_axes([0.15, 0.16, 0.75, 0.03])
    left_ax = fig.add_axes([0.15, 0.10, 0.75, 0.03])
    right_ax = fig.add_axes([0.15, 0.04, 0.75, 0.03])

    save_button = Button(save_ax, 'save')
    discard_button = Button(discard_ax, 'discard')
    progress_slider = Slider(progress_ax, 'progress [m]',
                             0.0, float(data.raceline_len_m), valinit=0.0)
    left_slider = Slider(left_ax, 'left [m]', 0.0, width_max, valinit=float(left[0]))
    right_slider = Slider(right_ax, 'right [m]', 0.0, width_max, valinit=float(right[0]))

    s_vec = data.s_vec.copy()
    total_len = float(data.raceline_len_m)

    def _update_title():
        selected_progress = s_vec[state['selected_idx']]
        status = 'unsaved' if state['dirty'] else 'saved'
        ax.set_title(
            f'Boundary margin editor [{status}]  progress={selected_progress:.3f} m'
            f'  sigma={sigma_m:.3f} m')

    def _sync_margin_sliders(idx: int):
        state['syncing'] = True
        left_slider.set_val(float(left[idx]))
        right_slider.set_val(float(right[idx]))
        state['syncing'] = False

    def _redraw():
        _update_title()
        image_artist.set_data(_render_loaded_track_image(track, state['selected_idx']))
        fig.canvas.draw_idle()

    def _selected_idx_from_progress(progress_m: float):
        wrapped_progress = progress_m % total_len
        idx = int(np.argmin(np.abs(s_vec - wrapped_progress)))
        return idx

    def _gaussian_weights(center_idx: int):
        dist = np.abs(s_vec - s_vec[center_idx])
        dist = np.minimum(dist, total_len - dist)
        weights = np.zeros_like(dist)
        mask = dist <= radius_m
        weights[mask] = np.exp(-0.5 * (dist[mask] / sigma_m) ** 2)
        return weights

    def _apply_delta(target: np.ndarray, center_idx: int, new_value: float):
        delta = new_value - target[center_idx]
        if np.isclose(delta, 0.0):
            return target
        updated = np.clip(target + delta * _gaussian_weights(center_idx), 0.0, None)
        return updated

    def _on_progress_change(progress_m):
        idx = _selected_idx_from_progress(progress_m)
        state['selected_idx'] = idx
        _sync_margin_sliders(idx)
        _redraw()

    def _on_left_change(new_value):
        if state['syncing']:
            return
        idx = state['selected_idx']
        left[:] = _apply_delta(left, idx, float(new_value))
        _update_track_boundary_widths(track, left, right)
        state['dirty'] = True
        _redraw()

    def _on_right_change(new_value):
        if state['syncing']:
            return
        idx = state['selected_idx']
        right[:] = _apply_delta(right, idx, float(new_value))
        _update_track_boundary_widths(track, left, right)
        state['dirty'] = True
        _redraw()

    def _on_save(_event):
        state['action'] = 'save'
        state['dirty'] = False
        plt.close(fig)

    def _on_discard(_event):
        left[:] = saved['left']
        right[:] = saved['right']
        _update_track_boundary_widths(track, left, right)
        state['action'] = 'discard'
        state['dirty'] = False
        _sync_margin_sliders(state['selected_idx'])
        plt.close(fig)

    def _on_close(_event):
        if state['action'] == 'save':
            print_info('fine-tuned boundary margins accepted')
            return
        if state['action'] == 'discard':
            print_info('boundary edits discarded')
            return
        if state['dirty']:
            left[:] = saved['left']
            right[:] = saved['right']
            _update_track_boundary_widths(track, left, right)
            print_info('closing editor without saving; unsaved boundary edits discarded')
        state['action'] = 'discard'

    progress_slider.on_changed(_on_progress_change)
    left_slider.on_changed(_on_left_change)
    right_slider.on_changed(_on_right_change)
    save_button.on_clicked(_on_save)
    discard_button.on_clicked(_on_discard)
    fig.canvas.mpl_connect('close_event', _on_close)
    _sync_margin_sliders(0)
    _update_title()
    plt.show()
    return state['action'] == 'save'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'track_name', nargs='?', choices=TrackFactory.available_track_names())
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--speed-only', action='store_true',
        help='load the saved track and rebuild the full track data')
    mode_group.add_argument(
        '--boundary-only', action='store_true',
        help='load the saved track and edit only boundary-related track data')
    args = parser.parse_args()

    if args.speed_only or args.boundary_only:
        track = TrackFactory.build('saved')
    else:
        if args.track_name is None:
            parser.error('track_name is required unless --speed-only or --boundary-only is set')
        # optimize and save
        main = QpSmooth()
        track = TrackFactory.build(args.track_name)
        track.rcp_raceline = main.optimize_raceline(track.rcp_raceline, track=track, offset=0.15)

    safety_margin = 0.08
    if args.boundary_only:
        pass
    else:
        r_vec, left, right = track.process_rcp_raceline(track.rcp_raceline)
        left, right = _apply_safety_margin(left, right, safety_margin)
        track.data = track.build_track(r_vec, left, right)

    _launch_boundary_margin_editor(track)
    track.save()

    # verify results: load and show
    load_track: RCPTrack = TrackFactory.build('saved')
    print('-----------------')
    print_info('testing loading')
    load_track.load()
    _show_loaded_track(load_track)
