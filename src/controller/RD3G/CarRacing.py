import os
import numpy as np
from time import time
from math import sin,cos,tan,atan,radians,degrees
from PIL import Image
from scipy import interpolate
import scipy.sparse # sparse matrix operations
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from matplotlib.animation import FuncAnimation
import matplotlib.image as mpimg
from scipy.ndimage import rotate
from scipy.interpolate import splprep, splev,CubicSpline,interp1d

from common import *
from util.TimeUtil import TimeUtil
from controller.RD3G.src.build.car_racing import CarRacing as cpp_CarRacing
from controller.RD3G.ResidualGame import ResidualGame

from controller.RD3G.SymbolicDynamics import SymbolicDynamics,MultiAgentSymbolicDynamics
from extension.simulator.CurvilinearSimulator import CurvilinearSimulator
import sympy

def wrap(val):
    '''
    wrap angle to [-pi,pi]
    '''
    return (val + np.pi) % (2*np.pi) - np.pi

# two car racing game
# uses curvilinear model
class CarRacing(ResidualGame):
    USE_CPP = True
    FORCE_PYTHON_SOLVER = False
    def __init__(self):
        super().__init__()

        # animation/visualization related
        self.sprite_visualization = False # True would use car images instead of boaxes

        # point mass model
        # u_i = [ay, ax]
        # x_i = [s, v, n, phi]
        # s: progress along raceline/reference curve
        # v: velocity
        # n: lateral offset from ref curve, left positive
        # phi: heading from ref curve tangend, ccw positive
        # ay: acceleration in lateral direction (left positive)
        # ax: acceleration in heading(phi) direction
        # ay is before ax to follow convention of steering before throttle
        # collision constraint: [(si-sj)/dx]**2 + [(ni-nj)/dy]**2 >= 1
        # agent count: N, time step: 1..T+1
        # X (game state) = concatenated state, first by agent, then by time)
        # state p of agent i at time k: X[k,i,p] or X.flatten()[k*N*m + i*m + p]
        # U (control) = concatenated control  dim: T*N*m
        # control p of agent i at time k: U[k,i,p] or U.flatten()[k*N*m + i*m + p]
        '''
        x_i_k: 1..T, T*N*n  NOTE starts from 1
        u_i_k: 0..T-1, T*N*m
        lamda_i_k: 0..T-1 T*N*n
        mu_k_i_j: 1..T T*N*N NOTE starts from 1
        '''

        # Problem formulation
        # decision variables:
        self.N = 2
        self.T = 30
        self.dt = dt = 0.05
        # dimension of x and u for single agent
        self.n = 4
        self.m = 2

        self.tolerance = 5e-4
        self.iterations = 30

        # collision definition
        # (x-x)T h_Qh (x-x) < C
        self.h_Qh = np.diag([-1,0,-1,0])
        self.car_size = 0.15

        self.track = None
        self.img_track = None

        # bounds for visualization
        #self.visual_x_lim = [-1.5,3.5]
        #self.visual_y_lim = [-0.5,2.5]


        #if (self.sprite_visualization):
        #self.car_scale = 0.0005/2
        #self.car_img_vec = [mpimg.imread('./resources/porsche_green.png'),mpimg.imread('./resources/porsche_orange.png'),mpimg.imread('./resources/porsche_blue.png')]


        # initial state,
        #self.x0 = np.array([[0.3, 1.0, 0.15, 0], [0, 1.1, -0.2, radians(0)]])
        self.x0 = np.zeros((self.N, self.n))

        # step cost parameters
        # NOTE this lambda fun needs to be implemented in c++
        # s,v,n,phi
        self.J_x_ref_fun = lambda i:np.array([0,1.0,0,0])
        self.J_Qr = np.diag([0,0.1,0,0])
        self.J_Q = np.diag([0,0,0.4,0.4])
        self.J_R = np.eye(self.m)*0.001
        self.guess = np.zeros((self.T,self.N,self.m))

        #self.print_debug_enable()


    def setup(self):
        # subclass responsible for loading cpp/eigen module
        # and setting x0
        if (self.USE_CPP or self.CPP_DEBUG):
            self.cpp = cpp_CarRacing(self.N, self.T, self.dt, self.rho, self.rho_b, self.bc_a, self.bc_b, self.tolerance, self.backtracking_max_iter, self.J_Qr, self.J_Q, self.J_R, self.h_Qh, self.car_size)
            ss = np.linspace(0, self.track.raceline_len_m, 1024)
            #curvature_vec = self.track.curvature_fun(ss)
            curvature_vec = np.array(splev(ss%self.track.raceline_len_m, self.track.curvature_fun, der=0)).flatten()
            self.cpp.set_curvature_vector(np.vstack([ss,curvature_vec]).T)

    def _visualize(self,U,X=None):
        # NOTE not needed
        return
        if (X is None):
            X = np.vstack([self.x0[np.newaxis,:,:],self.rollout(self.x0,U)])
        fig, ax = plt.subplots()

        '''
        # draw trajectory in Frenet frame
        for i in range(self.N):
            xx = X[:,i,0]
            yy = X[:,i,2]
            plt.plot(-yy,xx,'*-')
        '''
        # convert frenet coordinate to cartesian
        # (s,v,n,phi) -> (x,y,heading, v)
        car_cart_states = []
        for k in range(self.T):
            this_states = []
            for i in range(self.N):
                this_states.append(self.curv2Cart(X[k,i]))
            car_cart_states.append(this_states)
        car_cart_states = np.array(car_cart_states)
        # draw track
        L,W,_ = self.img_track.shape
        ax.imshow(self.img_track, extent=[self.track.x_min, self.track.x_max, self.track.y_min, self.track.y_max])
        for i in range(self.N):
            ax.plot(car_cart_states[:,i,0], car_cart_states[:,i,1])

        ax.set_aspect('equal', adjustable='box')
        return fig

    def _animation(self,U,X=None,gif_prefix=''):
        ''' build a gif animation'''
        if X is None:
            X = np.vstack([self.x0[np.newaxis,:,:],self.rollout(self.x0,U)])
        fig, ax = plt.subplots()
        # draw track
        L,W,_ = self.img_track.shape
        ax.imshow(self.img_track, extent=[self.track.x_min, self.track.x_max, self.track.y_min, self.track.y_max])

        if (self.sprite_visualization):
            car_scale = self.car_scale
            # draw car sprite
            car_pose_vec = []
            for states in X:
                car_pose_vec.append( [ self.curv2Cart(states[i]) for i in range(self.N) ])

            im_vec = []
            for i in range(self.N):
                rotated_car_img = np.clip(rotate(self.car_img_vec[i%len(self.car_img_vec)],degrees(car_pose_vec[0][i][2]),reshape=True), 0.0, 1.0)
                L,W,_ = rotated_car_img.shape
                im = ax.imshow(rotated_car_img, extent=[car_pose_vec[0][i][0]-W*car_scale, car_pose_vec[0][i][0]+W*car_scale, car_pose_vec[0][i][1]-L*car_scale, car_pose_vec[0][i][1]+L*car_scale])
                im_vec.append(im)

            def update(frame):
                for i in range(self.N):
                    rotated_car_img = np.clip(rotate(self.car_img_vec[i%len(self.car_img_vec)],degrees(car_pose_vec[frame][i][2]),reshape=True), 0.0, 1.0)
                    L,W,_ = rotated_car_img.shape
                    im_vec[i].set_data(rotated_car_img)
                    im_vec[i].set_extent((car_pose_vec[frame][i][0]-W*car_scale, car_pose_vec[frame][i][0]+W*car_scale, car_pose_vec[frame][i][1]-L*car_scale, car_pose_vec[frame][i][1]+L*car_scale))
                return im_vec
        else:
            car_pos_vec = []
            car_angle_vec = []
            box_vec = []
            circle_vec = []
            color_vec = ['red','green','blue','black']
            color_vec = [color_vec[i%len(color_vec)] for i in range(self.N)]
            # prepare smoothed animation
            for i,color in zip(range(self.N),color_vec):
                # interpolate for smooth graphics
                #tt = np.linspace(0,self.T*self.dt,50)
                tt = np.linspace(0,self.dt*self.T,self.T+1)
                # for plt.Rectangle, we offset position so this corresponds to top left corner
                # also flip x axis
                car_pose_vec = []
                for states in X:
                    car_pose_vec.append( [ self.curv2Cart(states[i]) for i in range(self.N) ])
                car_pose_vec = np.array(car_pose_vec)
                xx = car_pose_vec[:,i,0]
                yy = car_pose_vec[:,i,1]
                angle = car_pose_vec[:,i,2]

                xx_fun = interpolate.interp1d(tt,xx)
                yy_fun = interpolate.interp1d(tt,yy)
                angle_fun = interpolate.interp1d(tt,angle)

                pos_vec = np.vstack([xx_fun(tt),yy_fun(tt)]).T
                angle_vec = angle_fun(tt)/np.pi*180.0
                car_angle_vec.append(angle_vec)
                car_pos_vec.append(pos_vec)
                # width, height
                box_vec.append(plt.Rectangle(pos_vec[0]-np.array([0.2/2, 0.1/2]), 0.2, 0.1,angle=angle_vec[0], color=color,rotation_point='center'))
                circle_vec.append(plt.Circle(pos_vec[0], radius=(0.3/2),  color=color, fill=False))

            def update(frame):
                for i in range(self.N):
                    box_vec[i].set_xy(car_pos_vec[i][frame]-np.array([0.2/2, 0.1/2]))
                    box_vec[i].set_angle(car_angle_vec[i][frame])
                    circle_vec[i].set_center(car_pos_vec[i][frame])
                return box_vec
            # Add the boxes to the plot
            for box in box_vec:
                ax.add_patch(box)
            for circ in circle_vec:
                ax.add_patch(circ)


        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(*self.visual_x_lim)
        ax.set_ylim(*self.visual_y_lim)

        # Create the animation
        anim = FuncAnimation(fig, update, frames=self.T, blit=True)

        #gif_filename = self.resolveLogname(logPrefix=gif_prefix)
        #anim.save(gif_filename, writer='pillow')
        plt.show()
        self._snapshots(U,X)

    # save snapshots
    def _snapshots(self,U,X=None,png_prefix=''):
        ''' build a gif animation'''
        if X is None:
            X = np.vstack([self.x0[np.newaxis,:,:],self.rollout(self.x0,U)])
        fig, ax = plt.subplots()
        # draw track
        L,W,_ = self.img_track.shape
        ax.imshow(self.img_track, extent=[self.track.x_min, self.track.x_max, self.track.y_min, self.track.y_max])

        car_scale = self.car_scale
        # draw car sprite
        car_pose_vec = []
        for states in X:
            car_pose_vec.append( [ self.curv2Cart(states[i]) for i in range(self.N) ])

        im_vec = []
        for frame in range(0,self.T,7):
            for i in range(self.N):
                rotated_car_img = np.clip(rotate(self.car_img_vec[i%len(self.car_img_vec)],degrees(car_pose_vec[frame][i][2]),reshape=True), 0.0, 1.0)
                L,W,_ = rotated_car_img.shape
                im = ax.imshow(rotated_car_img, extent=[car_pose_vec[frame][i][0]-W*car_scale, car_pose_vec[frame][i][0]+W*car_scale, car_pose_vec[frame][i][1]-L*car_scale, car_pose_vec[frame][i][1]+L*car_scale])
                im_vec.append(im)

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(*self.visual_x_lim)
        ax.set_ylim(*self.visual_y_lim)

        #gif_filename = self.resolveLogname(logPrefix=gif_prefix)
        #anim.save(gif_filename, writer='pillow')
        plt.show()


    ''' --------  math functions and their derivatives ------ '''
    def J(self,x_k,u_k_i,i):
        '''
        step cost for an agent, given x,u
        x_k.shape (N*n) x_k_i = [x,y,vx,vy]
        u_k_i.shape (m) u_k_i = [ax, ay]
        i: agent id
        '''
        if (self.USE_CPP):
            return self.cpp.J(x_k,u_k_i,i)
        j = 1-i
        val =  (x_k[i]-self.J_x_ref_fun(i)).T @ self.J_Qr @ (x_k[i]-self.J_x_ref_fun(i)) + x_k[i].T @ self.J_Q @ x_k[i] + u_k_i.T @ self.J_R @ u_k_i
        if (i==0):
            val += -(x_k[i,0] - x_k[j,0])
        if (self.CPP_DEBUG):
            alt = self.cpp.J(x_k,u_k_i,i)
            if (np.linalg.norm(alt-val)>1e-4):
                breakpoint()
        return val

    # dJi dxi
    def dJi_dxi(self,x_k,u_k_i,i):
        if (self.USE_CPP):
            return self.cpp.dJi_dxi(x_k,u_k_i,i)
        val = 2* (x_k[i]-self.J_x_ref_fun(i)).T @ self.J_Qr + 2*x_k[i].T @ self.J_Q
        if (i==0):
            val += -np.array([1,0,0,0])
        if (self.CPP_DEBUG):
            alt = self.cpp.dJi_dxi(x_k,u_k_i,i)
            if (np.linalg.norm(alt-val)>1e-4):
                breakpoint()
        return val

    # NOTE obsolete
    # dJi dxj
    def dJi_dxj(self,x_k,u_k_i,i,j):
        if (self.USE_CPP):
            return self.cpp.dJi_dxj(x_k,u_k_i,i,j)
        val = 0
        if (self.CPP_DEBUG):
            alt = self.cpp.dJi_dxj(x_k,u_k_i,i,j)
            if (np.linalg.norm(alt-val)>1e-4):
                breakpoint()
        return val

    def dJi_du(self,x_k,u_k_i,i):
        if (self.USE_CPP):
            return self.cpp.dJi_du(x_k,u_k_i,i)
        val = 2* u_k_i.T @ self.J_R
        if (self.CPP_DEBUG):
            alt = self.cpp.dJi_du(x_k,u_k_i,i)
            if (np.linalg.norm(alt-val)>1e-4):
                breakpoint()
        return val

    # dJ^i / dxi dxi
    def dJi_dxi_dxi(self,x_k,u,i):
        if (self.USE_CPP):
            return self.cpp.dJi_dxi_dxi(x_k,u,i)
        val = 2*self.J_Qr + 2*self.J_Q
        if (self.CPP_DEBUG):
            alt = self.cpp.dJi_dxi_dxi(x_k,u,i)
            if (np.linalg.norm(alt-val)>1e-4):
                breakpoint()
        return val

    # dJi / dxi dxj
    def dJi_dxi_dxj(self, x_k, u_k_i, i, j):
        return 0
    def dJi_dxj_dxj(self, x_k, u_k_i, i, j):
        return 0
    def dJi_dudu(self, x_k, u_k_i, i):
        return 2*self.J_R

    def buildDynamicsJacobian(self):
        ''' find dfdx, dfdu with symbolic math, note this finds df/dx, not dx+/dx '''
        dyn = SymbolicDynamics(self.n, self.m)

        # below is almost verbatim copy of f(x,u,i)
        s,v,n,phi = dyn.x
        ay,ax = dyn.u

        k_s = sympy.symbols(f'k_s') # NOTE external variable
        #k = lambda s: splev(s,self.track.curvature)[0].item()
        # k_s = self.track.curvature_fun(x0)
        #k_s = k(s)

        dsdt = v*sympy.cos(phi)/(1-n*k_s)
        dvdt = ax
        dndt = v*sympy.sin(phi)
        dphidt = ay/v - k_s*dsdt
        dyn.f = [dsdt, dvdt, dndt, dphidt]
        dyn.symDerF()

        print(f'dfdx = {dyn.dfdx}')
        print(f'dfdu = {dyn.dfdu}')
        return

    def f(self,x,u,i):
        # u_i = [ay, ax]
        # x_i = [s, v, n, phi]
        s,v,n,phi = x
        ay,ax = u
        #k_s = self.track.curvature_fun(s%self.track.raceline_len_m)
        k_s = CurvilinearSimulator.curvatureTrack(s,self.main.track)

        dsdt = v*cos(phi)/(1-n*k_s)
        dvdt = ax
        dndt = v*sin(phi)
        dphidt = ay/v - k_s*dsdt
        dx = np.array([dsdt, dvdt, dndt, dphidt])

        return x+dx*self.dt

    def df_dx(self,x,u,i):
        x0,x1,x2,x3 = x
        u0, u1 = u
        #k_s = self.track.curvature_fun(x0%self.track.raceline_len_m)
        k_s = CurvilinearSimulator.curvatureTrack(x0,self.main.track)

        dfdx = np.array([[0, cos(x3)/(-k_s*x2 + 1), k_s*x1*cos(x3)/(-k_s*x2 + 1)**2, -x1*sin(x3)/(-k_s*x2 + 1)], [0, 0, 0, 0], [0, sin(x3), 0, x1*cos(x3)], [0, -k_s*cos(x3)/(-k_s*x2 + 1) - u0/x1**2, -k_s**2*x1*cos(x3)/(-k_s*x2 + 1)**2, k_s*x1*sin(x3)/(-k_s*x2 + 1)]])
        val = np.eye(4) + dfdx*self.dt
        if (self.DEBUG):
            num = jacobianNumerical(lambda xx:self.f(xx,u,i), x,dim=self.n)
            assert (np.linalg.norm(num-val)<1e-4)
        return val

    def df_du(self,x,u,i):
        x0,x1,x2,x3 = x
        dfdu = np.array([[0, 0], [0, 1], [0, 0], [1/x1, 0]])
        val = dfdu*self.dt
        if (self.DEBUG):
            num = jacobianNumerical(lambda uu:self.f(x,uu,i), u,dim=self.n)
            assert (np.linalg.norm(num-val)<1e-4)
        return val

    # collision definition is similar to Double Integrator, car is an "ellipsis"
    def h(self, x_i, x_j):
        ''' car distance larger than sqrt(7) normalized '''
        if (self.USE_CPP):
            return self.cpp.h(x_i,x_j)
        val = -( (x_i[0]-x_j[0]) )**2 - (x_i[2]-x_j[2])**2 + self.car_size**2
        if (self.CPP_DEBUG):
            alt = self.cpp.h(x_i,x_j)
            if (np.linalg.norm(alt-val)>1e-4):
                breakpoint()
        return val

    def dh_dxi(self,x_i,x_j):
        if (self.USE_CPP):
            return self.cpp.dh_dxi(x_i,x_j)
        val =  2*(x_i-x_j).T @ self.h_Qh
        if (self.CPP_DEBUG):
            alt = self.cpp.dh_dxi(x_i,x_j)
            if (np.linalg.norm(alt-val)>1e-4):
                breakpoint()
        return val
    def dh_dxj(self,x_i,x_j):
        if (self.USE_CPP):
            return self.cpp.dh_dxj(x_i,x_j)
        val = 2*(x_j-x_i).T @ self.h_Qh
        if (self.CPP_DEBUG):
            alt = self.cpp.dh_dxj(x_i,x_j)
            if (np.linalg.norm(alt-val)>1e-4):
                breakpoint()
        return val
    def dh_dxi_dxi(self,x_i,x_j):
        if (self.USE_CPP):
            return self.cpp.dh_dxi_dxi(x_i,x_j)
        val =  2*self.h_Qh.T
        if (self.CPP_DEBUG):
            alt = self.cpp.dh_dxi_dxi(x_i,x_j)
            if (np.linalg.norm(alt-val)>1e-4):
                breakpoint()
        return val
    def dh_dxj_dxi(self,x_i,x_j):
        if (self.USE_CPP):
            return self.cpp.dh_dxj_dxi(x_i,x_j)
        val = -2* self.h_Qh.T
        if (self.CPP_DEBUG):
            alt = self.cpp.dh_dxj_dxi(x_i,x_j)
            if (np.linalg.norm(alt-val)>1e-4):
                breakpoint()
        return val
    def dh_dxi_dxj(self,x_i,x_j):
        if (self.USE_CPP):
            return self.cpp.dh_dxi_dxj(x_i,x_j)
        val = -2*self.h_Qh.T
        if (self.CPP_DEBUG):
            alt = self.cpp.dh_dxi_dxj(x_i,x_j)
            if (np.linalg.norm(alt-val)>1e-4):
                breakpoint()
        return val
    def dh_dxj_dxj(self,x_i,x_j):
        if (self.USE_CPP):
            return self.cpp.dh_dxj_dxj(x_i,x_j)
        val = 2*self.h_Qh.T
        if (self.CPP_DEBUG):
            alt = self.cpp.dh_dxj_dxj(x_i,x_j)
            if (np.linalg.norm(alt-val)>1e-4):
                breakpoint()
        return val

    def testAnimation(self):
        u_ref = np.zeros((self.T,self.N,self.m))
        x_ref = self.rollout(self.x0,u_ref)
        full_x_ref = np.vstack([self.x0[np.newaxis,:,:],x_ref])
        self._animation(u_ref,full_x_ref)
        #self._visualize(u_ref,full_x_ref)
        #plt.show()

    def cart2Curv(self, cart, guess_s=None):
        '''
            transform cartesian states to curvilinear states
            relies on self.track.raceline_s
            [cart]: (x,y,heading,v_forward,v_sideway,omega)
            [guess_s]: estimated s
            [return]: (s,v,n,phi)
        '''
        x,y,heading,v_forward,v_sideway,omega = cart

        #dist = lambda s: np.linalg.norm(np.array(splev(s%self.track.raceline_len_m,self.track.raceline_s,der=0)) - np.array([x,y]))
        def dist(s):
            val = np.linalg.norm(np.array(splev(s%self.track.raceline_len_m,self.track.raceline_s,der=0)).flatten() - np.array([x,y]))
            return val

        if (guess_s is None):
            # initial guess to avoid local minima
            xx = np.linspace(0.0, self.track.raceline_len_m,10)
            yy = [dist(x) for x in xx]
            guess_s = xx[np.argmin(yy)]
            ds = 2*self.track.raceline_len_m/10
            fit = minimize(dist, x0=guess_s, method='L-BFGS-B', bounds=((guess_s-ds,guess_s+ds),))
        else:
            fit = minimize(dist, x0=guess_s, method='L-BFGS-B', bounds=((guess_s-0.2,guess_s+0.2),))

        s = fit.x[0]

        r = np.array(splev(s%self.track.raceline_len_m, self.track.raceline_s, der=0))
        dr = np.array(splev(s%self.track.raceline_len_m, self.track.raceline_s, der=1))
        dr = dr/np.linalg.norm(dr)
        n = np.cross(dr, np.array([x,y]) - r)
        # ignore sideway velocity
        v = v_forward
        phi = wrap(heading - np.arctan2(dr[1],dr[0]))
        return np.array([s,v,n,phi])

    def curv2Cart(self, curv):
        '''
            transform curvilinear states to cartesian states
            [curv]: (s,v,n,phi)
            [return]: (x,y,heading,v_forward,v_sideway,omega)
        '''
        s,v,n,phi = curv.flatten()
        r = np.array(splev(s%self.track.raceline_len_m, self.track.raceline_s, der=0))
        dr = np.array(splev(s%self.track.raceline_len_m, self.track.raceline_s, der=1))
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
    def final(self):
        super().final()





if __name__=="__main__":
    main = CarRacing()
    main.buildDynamicsJacobian()
    main.setup()
    main.solve(save_gif=False,visualize=True,animate=True)
    main.final()
    #main.testAnimation()

