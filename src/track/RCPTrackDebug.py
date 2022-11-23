# rarely used or obsolete methods for RCPTrack
import numpy as np
import os.path
from numpy import isclose
import matplotlib.pyplot as plt
from math import atan2,radians,degrees,sin,cos,pi,tan,copysign,asin,acos,isnan
from scipy.interpolate import splprep, splev,CubicSpline,interp1d
from scipy.optimize import minimize_scalar,minimize,brentq
from scipy.integrate import solve_ivp
from time import sleep,time
import cv2
from PIL import Image
import pickle
from bisect import bisect

from common import *
from util.timeUtil import execution_timer
from track.RCPTrack import RCPTrack

class RCPTrackDebug(RCPTrack):
    def __init__(self,main,config=None):
        super().__init__(main,config)
    # calculate curvature cost, among other things
    def cost(self,k):
        self.cost_count += 1
        if (False and self.cost_count % 100 == 0):
            self.K = k
            self.verify()
            bdy = self.boundaryClearanceVector(k)
            plt.plot(bdy)
            plt.show()

        # save a checkpoint
        if (False and self.cost_count % 1000 == 0):
            self.resolveLogname()
            output = open(self.logFilename,'wb')
            pickle.dump(k,output)
            output.close()
            print("checkpoint %d saved"%self.log_no)

        # part 1: curvature norm
        k = np.array(k)
        p1_cost = np.sum(k**2)
        # part 2: smoothness
        # relative importance of smoothness w.r.t curvature
        alfa = 1.0
        p2_cost = np.sum(np.abs(np.diff(k)))
        total_cost = p1_cost + alfa*p2_cost

        #print("call %d, cost = %.5f"%(self.cost_count,total_cost))
        #print("p1/p2 = %.2f"%(p1_cost/p2_cost))
        return total_cost

    def minimizeCurvatureRoutine(self,):
        steps = 100
        self.steps=steps
        # initialize an initial raceline for reference
        print("base raceline")
        self.prepareTrack()
        # discretize the initial raceline
        self.discretizePath(steps)

        # NOTE the reconstructed path's end deviate from original by around 5cm
        self.verify()
        # let's use it as a starting point for now
        K0 = self.K

        # DEBUG sensitivity analysis
        '''
        eps = 1e-5
        k = self.K
        k -= 3*eps
        self.verify(k)
        tmp = self.boundaryClearanceVector(k)
        plt.plot(tmp)
        plt.show()
        '''


        # steps = 1000
        # optimize on curvature norm
        # var: 
        # K(s), (steps,) vector of curvature along path
        # the parameterization variable is defined such that
        # s[0] = path start, s[steps-1] = path end
        # TODO uniform ds is enforced in each optimization iteration but the magnitude may change if the path length shrinks/expands
        # NOTE for now ds is fixed at self.ds

        # cost:
        # curvature norm, sum(|K|)
        # constrains:
        # must not cross track boundary (inequality)
        # curvature at start and finish must agree (loop) (equality)

        # assemble constrains
        wheelbase = 102e-3
        max_steering = radians(25)
        R_min = wheelbase / tan(max_steering)
        K_max = 1.0/R_min
        # track boundary
        cons = [{'type': 'ineq', 'fun': self.boundaryClearanceVector}]
        cons.append({'type': 'eq', 'fun': lambda x: x[-1]-x[0]})

        cons = tuple(cons)

        # bounds
        # part 1: hard on Ki < C, no super tight curves
        # how tight depends on vehicle wheelbase and steering angle
        bnds = tuple([(-K_max,K_max) for i in range(steps)])

        self.cost_count = 0
        res = minimize(self.cost,K0,method='SLSQP', jac='2-point',constraints=cons,bounds=bnds,options={'maxiter':1000,'eps':1e-10} )
        print(res)
        # verify again
        self.K = res.x
        print(self.K)
        self.verify(steps)

    # verify that we can restore x,y coordinate from K(s)/curvature path distance space
    def verify(self,K=None):
        # convert from K(s) space to X,Y(s) space using Fresnel integral
        # state variable X,Y,Heading
        if K is None:
            K = self.K
        K = interp1d(self.S,self.K)
        def kensel(s,x):
            return [ cos(x[2]), sin(x[2]), K(s)]

        s_span = [0,self.S[-1]]
        x0 = (self.x0,self.y0,self.phi0)
        # error 0.02, lateral error
        #sol = solve_ivp(kensel,s_span, x0, method='DOP853',t_eval = self.S )
        # error 0.02, longitudinal error
        sol = solve_ivp(kensel,s_span, x0, method='LSODA',t_eval = self.S )

        # plot results
        # original path
        steps = 1000
        u = np.linspace(0,self.u[-1],steps)
        x,y = splev(u,self.raceline,der=0)
        # quantify error
        error = ((x[-1]-sol.y[0,-1])**2 + (y[-1]-sol.y[1,-1])**2)**0.5
        print("error %.2f"%error)

        plt.plot(x,y)
        # regenerated path
        plt.plot(sol.y[0],sol.y[1])
        plt.show()
                
        return

    def verifySpeedProfile(self,n_steps=1000):
        # calculate theoretical lap time
        mu = 10.0/9.81
        g = 9.81
        t_total = 0
        path_len = 0
        xx = np.linspace(0,self.track_length_grid,n_steps+1)
        dist = lambda a,b: ((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5
        v3 = self.v3(xx)
        for i in range(n_steps):
            (x_i, y_i) = splev(xx[i%n_steps], self.raceline, der=0)
            (x_i_1, y_i_1) = splev(xx[(i+1)%n_steps], self.raceline, der=0)
            # distance between two steps
            ds = dist((x_i, y_i),(x_i_1, y_i_1))
            path_len += ds
            t_total += ds/v3[i%n_steps]

        print_info("Theoretical value:")
        print_info("\t top speed = %.2fm/s"%max(v3))
        print_info("\t total time = %.2fs"%t_total)
        print_info("\t path len = %.2fm"%path_len)

        # get direct distance from two u
        distuu = lambda u1,u2: dist(splev(u1, self.raceline, der=0),splev(u2, self.raceline, der=0))

        vel_vec = []
        ds_vec = []
        xx = np.linspace(0,self.track_length_grid,n_steps+1)

        # get velocity at each point
        for i in range(n_steps):
            # tangential direction
            tan_dir = splev(xx[i], self.raceline, der=1)
            tan_dir = np.array(tan_dir/np.linalg.norm(tan_dir))
            vel_now = self.v3(xx[i]%len(self.ctrl_pts)) * tan_dir
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

            theta = np.arctan2(vel_vec[i,1],vel_vec[i,0])
            theta_vec.append(theta)

            dtheta = np.arctan2(vel_vec[i+1,1],vel_vec[i+1,0]) - theta
            dtheta = (dtheta+np.pi)%(2*np.pi)-np.pi
            dtheta_vec.append(dtheta)

            speed = np.linalg.norm(vel_vec[i])
            next_speed = np.linalg.norm(vel_vec[i+1])
            v_vec.append(speed)

            dt = distuu(xx[i],xx[i+1])/speed
            dt_vec.append(dt)

            lat_acc_vec.append(speed*dtheta/dt)
            lon_acc_vec.append((next_speed-speed)/dt)

        dt_vec = np.array(dt_vec)
        lon_acc_vec = np.array(lon_acc_vec)
        lat_acc_vec = np.array(lat_acc_vec)

        # get acc_vector, track frame
        dt_vec2 = np.vstack([dt_vec,dt_vec]).T
        acc_vec = np.diff(vel_vec,axis=0)
        acc_vec = acc_vec / dt_vec2

        # plot acceleration vector cloud
        # with x,y axis being vehicle frame, x lateral
        '''
        p0, = plt.plot(lat_acc_vec,lon_acc_vec,'*',label='data')

        # draw the traction circle
        cc = np.linspace(0,2*np.pi)
        circle = np.vstack([np.cos(cc),np.sin(cc)])*mu*g
        p1, = plt.plot(circle[0,:],circle[1,:],label='1g')
        plt.gcf().gca().set_aspect('equal','box')
        plt.xlim(-12,12)
        plt.ylim(-12,12)
        plt.xlabel('Lateral Acceleration')
        plt.ylabel('Longitudinal Acceleration')
        plt.legend(handles=[p0,p1])
        plt.show()

        p0, = plt.plot(theta_vec,label='theta')
        p1, = plt.plot(v_vec,label='v')
        p2, = plt.plot(dtheta_vec,label='dtheta')
        acc_mag_vec = (acc_vec[:,0]**2+acc_vec[:,1]**2)**0.5
        p0, = plt.plot(acc_mag_vec,'*',label='acc vec2mag')
        p1, = plt.plot((lon_acc_vec**2+lat_acc_vec**2)**0.5,label='acc mag')

        p2, = plt.plot(lon_acc_vec,label='longitudinal')
        p3, = plt.plot(lat_acc_vec,label='lateral')
        plt.legend(handles=[p0,p1])
        plt.show()
        '''
        print("theoretical laptime %.2f"%t_total)

        self.reconstructRaceline()
        return t_total


class Track():

    def isOutside(self,coord):
