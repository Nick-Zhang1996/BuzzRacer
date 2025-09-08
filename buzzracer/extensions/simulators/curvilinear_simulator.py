''' Simulate vehicle dynamics in Curvilinear/Frenet reference frame '''
# NOTE this module requires extensive re-writing, skipping for now
# pylint: disable=all

from math import sin, cos

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import  splev

from buzzracer.extensions.simulator import Simulator


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
    def advance_dynamics(car_states, control, car, dt):
        """ignore car_states, update car.sim_states with control and optional
        [dt]

        [return] cartesian states corresponding to updated
        car.sim_states

        """
        car.sim_states = self.advance_point_mass_dynamics(
            car.sim_states, control, dt)

        return self.curv2_cart(car.sim_states)
