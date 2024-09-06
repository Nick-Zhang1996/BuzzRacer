# Simulation in Curvilinear ref frame
# uses car.sim_states as internal state

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from math import sin,cos,tan,radians,degrees,pi,atan
from threading import Event,Lock
from scipy.optimize import minimize_scalar,minimize,brentq
from scipy.interpolate import splprep, splev,CubicSpline,interp1d

from common import *
from extension import Simulator

def wrap(val):
    '''
    wrap angle to [-pi,pi]
    '''
    return (val + np.pi) % (2*np.pi) - np.pi

class CurvilinearSimulator(Simulator):
    '''
        point mass model
        states = x = car.sim_states = (s, v, n, phi)
        s: progress along raceline/reference curve
        v: velocity
        n: lateral offset from ref curve, left positive
        phi: heading from ref curve tangend, ccw positive
        control = u = (ay, ax)
        ax: acceleration in heading(phi) direction
        ay: acceleration in lateral direction (left positive)
        ay is before ax to follow convention of steering before throttle
    '''
    def __init__(self,main):
        super().__init__(main)
        self.track = self.main.track

    def init(self):
        super().init()

        self.cars = self.main.cars
        CurvilinearSimulator.dt = self.main.dt
        for car in self.cars:
            self.addCar(car)
        self.main.new_state_update.set()

    def addCar(self,car):
        '''
        initialize a car
        car.states =  (x,y,heading,v_forward,v_sideway,omega)
        car.sim_states = (s, v, n, phi)
        s: progress along raceline/reference curve
        v: velocity
        n: lateral offset from ref curve, left positive
        phi: heading from ref curve tangend, ccw positive
        '''
        x,y,heading,v_forward,v_sideway,omega = car.states
        curv = self.cart2Curv(car.states)
        car.sim_states = curv

        car.state_dim = 4
        car.control_dim = 2

    def cart2Curv(self, cart, guess_s=None):
        return CurvilinearSimulator.cart2CurvTrack(cart,self.track,guess_s)

    @staticmethod
    def cart2CurvTrack(cart, track,guess_s=None):
        '''
            transform cartesian states to curvilinear states
            relies on track.raceline_s
            [cart]: (x,y,heading,v_forward,v_sideway,omega)
            [guess_s]: estimated s
            [return]: (s,v,n,phi)
        '''
        x,y,heading,v_forward,v_sideway,omega = cart

        #dist = lambda s: np.linalg.norm(np.array(splev(s%track.raceline_len_m,track.raceline_s,der=0)) - np.array([x,y]))
        def dist(s):
            val = np.linalg.norm(np.array(splev(s%track.raceline_len_m,track.raceline_s,der=0)).flatten() - np.array([x,y]))
            return val

        if (guess_s is None):
            # initial guess to avoid local minima
            xx = np.linspace(0.0, track.raceline_len_m,50)
            yy = [dist(x) for x in xx]
            guess_s = xx[np.argmin(yy)]
            ds = 2*track.raceline_len_m/10
            fit = minimize(dist, x0=guess_s, method='L-BFGS-B', bounds=((guess_s-ds,guess_s+ds),))
        else:
            fit = minimize(dist, x0=guess_s, method='L-BFGS-B', bounds=((guess_s-0.2,guess_s+0.2),))

        s = fit.x[0]

        r = np.array(splev(s%track.raceline_len_m, track.raceline_s, der=0))
        dr = np.array(splev(s%track.raceline_len_m, track.raceline_s, der=1))
        dr = dr/np.linalg.norm(dr)
        n = np.cross(dr, np.array([x,y]) - r)
        # ignore sideway velocity
        v = v_forward
        phi = wrap(heading - np.arctan2(dr[1],dr[0]))
        return np.array([s,v,n,phi])

    # DEBUG
    def debugPlot(self,cart):
        x,y,heading,v_forward,v_sideway,omega = cart
        def dist(s):
            val = np.linalg.norm(np.array(splev(s%self.track.raceline_len_m,self.track.raceline_s,der=0)).flatten() - np.array([x,y]))
            return val
        xx = np.linspace(-1.0, self.track.raceline_len_m,1000)
        yy = [dist(x) for x in xx]
        plt.plot(xx,yy)

        xx = np.linspace(-1.0, self.track.raceline_len_m,10)
        yy = [dist(x) for x in xx]
        plt.plot(xx,yy,'o')
        plt.show()
        return

    def curv2Cart(self,curv):
        return CurvilinearSimulator.curv2CartTrack(curv,self.track)

    @staticmethod
    def curv2CartTrack(curv,track):
        '''
            transform curvilinear states to cartesian states
            [curv]: (s,v,n,phi)
            [return]: (x,y,heading,v_forward,v_sideway,omega)
        '''
        s,v,n,phi = curv.flatten()
        r = np.array(splev(s%track.raceline_len_m, track.raceline_s, der=0))
        dr = np.array(splev(s%track.raceline_len_m, track.raceline_s, der=1))
        dr = dr/np.linalg.norm(dr)

        # ccw 90 deg
        A = np.array([[0,-1],[1,0]])
        x,y = r + (A @ dr)*n
        ref_heading = np.arctan2(dr[1],dr[0])
        heading = wrap(phi + ref_heading)
        v_forward = v
        v_sideway = 0.0
        omega = 0.0
        return np.array([x,y,heading, v_forward, v_sideway, omega])

    @staticmethod
    def curvatureTrack(s,track):
        r = np.array(splev(s%track.raceline_len_m, track.curvature_fun, der=0))
        return r.item()

    def curvature(self,s):
        r = np.array(splev(s%self.track.raceline_len_m, self.track.curvature_fun, der=0))
        return r.item()

    def _curvature(self,s):
        '''
        get signed curvature of raceline at s, ccw positive
        '''
        # TODO if this is a bottleneck, fit curvature(s) as a cubic fun
        # curvature = interp1d(ss,curvature(ss),kind='cubic')

        # radius of curvature can be calculated as R = |y'|^3/sqrt(|y'|^2*|y''|^2-(y'*y'')^2)
        r = np.array(splev(s%self.track.raceline_len_m, self.track.raceline_s, der=0))
        dr = np.array(splev(s%self.track.raceline_len_m, self.track.raceline_s, der=1))
        ddr = np.array(splev(s%self.track.raceline_len_m, self.track.raceline_s, der=2))
        _norm = lambda x:np.linalg.norm(x)
        dr_norm = _norm(dr)
        curvature = 1.0/(dr_norm**3/(dr_norm**2*_norm(ddr)**2 - np.sum(dr*ddr,axis=0)**2)**0.5)
        sign = np.cross(dr.T,ddr.T)
        if (np.isnan(curvature)):
            #self.print_warning('curvature is nan, likely because curvature is exactly 0')
            curvature = 0.0
        return np.copysign(curvature, sign)

    @staticmethod
    def advancePointMassDynamics(curv_states, control, dt,track):
        s,v,n,phi = curv_states
        k_s = CurvilinearSimulator.curvatureTrack(s,track)
        ay,ax = control
        dsdt = v*cos(phi)/(1-n*k_s)
        dvdt = ax
        dndt = v*sin(phi)
        dphidt = ay/v - k_s*dsdt

        if (dt is None):
            dt = CurvilinearSimulator.dt

        dx = np.array([dsdt, dvdt, dndt, dphidt])*dt
        '''
        if (np.linalg.norm(dx[:3]) > 1.0):
            breakpoint()
        '''
        return curv_states + dx

    def advanceDynamics(self, car_states, control, car, dt=None):
        '''
        ignore car_states, update car.sim_states with control and optional [dt]
        [return] cartesian states corresponding to updated car.sim_states
        '''

        # DEBUG
        '''
        check1 = np.linalg.norm(self.cart2Curv(car_states,guess_s = car.sim_states[0]) - car.sim_states)
        check2 = np.linalg.norm(self.curv2Cart(self.cart2Curv(car_states,guess_s = car.sim_states[0])) - car_states)
        if (check1 > 0.001 or check2 > 0.001):
            print('inconsistency in coord frame transformation')
            print(check1,check2)
            print('card_states: ', car_states)
            print('curv_states: ', car.sim_states)
            print('card -> curv: ', self.cart2Curv(car_states))
            print('card -> curv -> card: ', self.curv2Cart(self.cart2Curv(car_states)))
            print('curv -> card: ', self.curv2Cart(car.sim_states))
            print('curv -> card -> curv ', self.cart2Curv(self.curv2Cart(car.sim_states)))
            breakpoint()
        '''

        car.sim_states = CurvilinearSimulator.advancePointMassDynamics(car.sim_states, control, dt,self.track)

        return self.curv2Cart(car.sim_states)
