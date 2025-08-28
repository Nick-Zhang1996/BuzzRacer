''' Simulate vehicle dynamics in Curvilinear/Frenet reference frame '''
# NOTE this module requires extensive re-writing, skipping for now
# pylint: disable=all

from math import sin, cos

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import  minimize
from scipy.interpolate import  splev

from buzzracer.extensions.simulator import Simulator


def wrap(val):
    '''
    wrap angle to [-pi,pi]
    '''
    return (val + np.pi) % (2*np.pi) - np.pi


class CurvilinearSimulator(Simulator):
    ''' Simulate vehicle dynamics in Curvilinear/Frenet reference frame .

        point mass model

        Attirbutes:
            states: car.sim_states = (s, v, n, phi)
            s: progress along raceline/reference curve
            v: velocity
            n: lateral offset from ref curve, left positive
            phi: heading from ref curve tangend, ccw positive
            control:  (ay, ax)
            ax: acceleration in heading(phi) direction
            ay: acceleration in lateral direction (left positive)
            ay is before ax to follow convention of steering before throttle

        uses car.sim_states as internal state
    '''

    def __init__(self):
        super().__init__()
        self.track = self.main.track

    def init(self):
        super().init()

        self.cars = self.main.cars
        CurvilinearSimulator.dt = self.main.dt
        for car in self.cars:
            self.add_car(car)
        self.main.new_state_update.set()

    def add_car(self, car):
        '''
        Initialize a car

        car.states =  (x,y,heading,v_forward,v_sideway,omega)
        car.sim_states = (s, v, n, phi)
        s: progress along raceline/reference curve
        v: velocity
        n: lateral offset from ref curve, left positive
        phi: heading from ref curve tangend, ccw positive
        '''
        #x, y, heading, v_forward, v_sideway, omega = car.states
        curv = self.cart2_curv(car.states)
        car.sim_states = curv

        car.state_dim = 4
        car.control_dim = 2

    def cart2_curv(self, cart, guess_s=None):
        """transform cartesian states to curvilinear states relies on
        self.track.raceline_s.

        [cart]: (x,y,heading,v_forward,v_sideway,omega)
        [guess_s]: estimated s
        [return]: (s,v,n,phi)

        """
        x, y, heading, v_forward, _, _ = cart

        # dist = lambda s: np.linalg.norm(np.array(splev(s%self.track.raceline_len_m,self.track.raceline_s,der=0)) - np.array([x,y]))
        def dist(s):
            val = np.linalg.norm(np.array(splev(
                s % self.track.raceline_len_m, self.track.raceline_s, der=0)).flatten() - np.array([x, y]))
            return val

        if (guess_s is None):
            # initial guess to avoid local minima
            xx = np.linspace(0.0, self.track.raceline_len_m, 10)
            yy = [dist(x) for x in xx]
            guess_s = xx[np.argmin(yy)]
            ds = 2*self.track.raceline_len_m/10
            fit = minimize(dist, x0=guess_s, method='L-BFGS-B',
                           bounds=((guess_s-ds, guess_s+ds),))
        else:
            fit = minimize(dist, x0=guess_s, method='L-BFGS-B',
                           bounds=((guess_s-0.2, guess_s+0.2),))

        s = fit.x[0]

        r = np.array(splev(s % self.track.raceline_len_m,
                     self.track.raceline_s, der=0))
        dr = np.array(splev(s % self.track.raceline_len_m,
                      self.track.raceline_s, der=1))
        dr = dr/np.linalg.norm(dr)
        n = np.cross(dr, np.array([x, y]) - r)
        # ignore sideway velocity
        v = v_forward
        phi = wrap(heading - np.arctan2(dr[1], dr[0]))
        return np.array([s, v, n, phi])

    # DEBUG
    def debug_plot(self, cart):
        x, y, _ = cart

        def dist(s):
            val = np.linalg.norm(np.array(splev(
                s % self.track.raceline_len_m, self.track.raceline_s, der=0)).flatten() - np.array([x, y]))
            return val
        xx = np.linspace(-1.0, self.track.raceline_len_m, 1000)
        yy = [dist(x) for x in xx]
        plt.plot(xx, yy)

        xx = np.linspace(-1.0, self.track.raceline_len_m, 10)
        yy = [dist(x) for x in xx]
        plt.plot(xx, yy, 'o')
        plt.show()
        return

    def curv2_cart(self, curv):
        """transform curvilinear states to cartesian states.

        [curv]: (s,v,n,phi)
        [return]: (x,y,heading,v_forward,v_sideway,omega)

        """
        s, v, n, phi = curv.flatten()
        r = np.array(splev(s % self.track.raceline_len_m,
                     self.track.raceline_s, der=0))
        dr = np.array(splev(s % self.track.raceline_len_m,
                      self.track.raceline_s, der=1))
        dr = dr/np.linalg.norm(dr)

        # ccw 90 deg
        A = np.array([[0, -1], [1, 0]])
        x, y = r + (A @ dr)*n
        ref_heading = np.arctan2(dr[1], dr[0])
        heading = wrap(phi + ref_heading)
        v_forward = v
        v_sideway = 0.0
        omega = 0.0
        return np.array([x, y, heading, v_forward, v_sideway, omega])

    def curvature(self, s):
        """get signed curvature of raceline at s, ccw positive."""
        # TODO if this is a bottleneck, fit curvature(s) as a cubic fun
        # curvature = interp1d(ss,curvature(ss),kind='cubic')

        # radius of curvature can be calculated as R = |y'|^3/sqrt(|y'|^2*|y''|^2-(y'*y'')^2)
        # r = np.array(splev(s % self.track.raceline_len_m,
        #              self.track.raceline_s, der=0))
        dr = np.array(splev(s % self.track.raceline_len_m,
                      self.track.raceline_s, der=1))
        ddr = np.array(splev(s % self.track.raceline_len_m,
                       self.track.raceline_s, der=2))

        def _norm(x):
            return np.linalg.norm(x)
        dr_norm = _norm(dr)
        curvature = 1.0/(dr_norm**3/(dr_norm**2*_norm(ddr) **
                         2 - np.sum(dr*ddr, axis=0)**2)**0.5)
        sign = np.cross(dr.T, ddr.T)
        if (np.isnan(curvature)):
            # self.print_warning('curvature is nan, likely because curvature is exactly 0')
            curvature = 0.0
        return np.copysign(curvature, sign)

    @staticmethod
    def advance_point_mass_dynamics(curv_states, control, dt):
        s, v, n, phi = curv_states
        k_s = self.curvature(s)
        ay, ax = control
        dsdt = v*cos(phi)/(1-n*k_s)
        dvdt = ax
        dndt = v*sin(phi)
        dphidt = ay/v - k_s*dsdt

        if (dt is None):
            dt = CurvilinearSimulator.dt

        dx = np.array([dsdt, dvdt, dndt, dphidt])*dt
        return curv_states + dx

    @staticmethod
    def advance_dynamics(car_states, control, car, dt):
        """ignore car_states, update car.sim_states with control and optional
        [dt]

        [return] cartesian states corresponding to updated
        car.sim_states

        """
        car.sim_states = self.advance_point_mass_dynamics(
            car.sim_states, control, dt)

        return self.curv2_cart(car.sim_states)
