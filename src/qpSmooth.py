# smooth path with quadratic programming
# following paper "A quadratic programming approach to path smoothing"

import cv2
import cvxopt
import warnings
import pickle
import numpy as np
from scipy.interpolate import interp1d
from math import pi,isclose,radians,cos,sin,atan2,tan
from scipy.interpolate import splprep, splev,CubicSpline,interp1d

from PIL import Image
import matplotlib.pyplot as plt

from time import time
from common import *
from track.RCPTrack import RCPTrack


class QpSmooth(RCPTrack):
    # track: RCPTrack object
    def __init__(self):
        RCPTrack.__init__(self)
        warnings.simplefilter("error")
        return

    # given three points, calculate first and second derivative as a linear combination of the three points rl, r, rr, which stand for r_(k-1), r_k, r_(k+1)
    # return: 2*3, tuple
    #       ((al, a, ar),
    #        (bl, b, br))
    # where f'@r = al*rl + a*r + ar*rr
    # where f''@r = bl*rl + b*r + br*rr
    # ds, arc length between rl,r and r, rr 
    # if not specified, |r-rl|_2 will be used as approximation
    def lagrangeDer(self,points,ds=None):
        rl,r,rr = points
        dist = lambda x,y:((x[0]-y[0])**2 + (x[1]-y[1])**2)**0.5
        if ds is None:
            sl = -dist(rl,r)
            sr = dist(r,rr)
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

        return ((al,a,ar),(bl,b,br))

    # construct a fifth order bezier curve passing through endpoints r
    # matching first and second derivative dr, ddr
    # r (2,2)
    # dr (2,2),  taken w.r.t. arc length s
    # ddr (2,2), taken w.r.t. arc length s 
    # ds: arc length between the endpoints
    def bezierCurve(self,r,dr,ddr,ds=None):
        rl,rr = r
        drl,drr = dr
        ddrl,ddrr = ddr

        dist = lambda x,y:((x[0]-y[0])**2 + (x[1]-y[1])**2)**0.5
        if ds is None:
            ds = dist(rl,rr)

        # two sets of equations, one for x, one for y
        #bx = np.array([rl[0],rr[0],drl[0],drr[0],ddrl[0],ddrr[0]]).T
        #by = np.array([rl[1],rr[1],drl[1],drr[1],ddrl[1],ddrr[1]]).T

        # dr = dr/ds = dr/dt * dt/ds
        # we want dB/dt = dr/dt = dr(input) * ds/dt = dr * ds(between two endpoints)
        bx = np.array([rl[0],rr[0],drl[0]*ds,drr[0]*ds,ddrl[0]*ds*ds,ddrr[0]*ds*ds]).T
        by = np.array([rl[1],rr[1],drl[1]*ds,drr[1]*ds,ddrl[1]*ds*ds,ddrr[1]*ds*ds]).T
        b = np.vstack([bx,by]).T

        # x_x = P0_x, P1_x ... P5_x
        # x_y = P0_y, P1_y ... P5_y
        A = [[ 1, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 1],
             [-5, 5, 0, 0, 0, 0],
             [ 0, 0, 0, 0,-5, 5],
             [20,-40,20,0, 0, 0],
             [0 , 0, 0,20,-40,20]]
        A = np.array(A)

        try:
            sol = np.linalg.solve(A,b)
        except np.linalg.LinAlgError:
            print_error("can't solve bezier Curve")

        # return the control points
        P = sol
        return P

    # generate a bezier spline matching derivative estimated from lagrange interpolation
    # break_pts.shape = (n,2)
    # return: vector function, domain [0,len(points)]
    def bezierSpline(self,break_pts):
        break_pts = np.array(break_pts).T

        # calculate first and second derivative
        # w.r.t. ds, estimated with 2-norm
        df = []
        ddf = []
        n = break_pts.shape[1]
        for i in range(n):
            rl = break_pts[:,(i-1)%n]
            r  = break_pts[:,(i)%n]
            rr = break_pts[:,(i+1)%n]
            points = [rl, r, rr]
            
            ((al,a,ar),(bl,b,br)) = self.lagrangeDer(points)
            df.append( al*rl + a*r + ar*rr)
            ddf.append(bl*rl + b*r + br*rr)

        P = []
        for i in range(n):
            # generate bezier spline segments
            rl = break_pts[:,(i)%n]
            r  = break_pts[:,(i+1)%n]
            section_P = self.bezierCurve([rl,r],[df[i],df[(i+1)%n]],[ddf[i],ddf[(i+1)%n]],ds=None)
            # NOTE testing
            B = lambda t,p: (1-t)**5*p[0] + 5*t*(1-t)**4*p[1] + 10*t**2*(1-t)**3*p[2] + 10*t**3*(1-t)**2*p[3] + 5*t**4*(1-t)*p[4] + t**5*p[5]
            x_i = B(0,section_P[:,0])
            y_i = B(0,section_P[:,1])
            x_f = B(1,section_P[:,0])
            y_f = B(1,section_P[:,1])
            assert np.isclose(x_i,rl[0],atol=1e-5) and np.isclose(y_i,rl[1],atol=1e-5) and np.isclose(x_f,r[0],atol=1e-5) and np.isclose(y_f,r[1],atol=1e-5)

            P.append(section_P)

        # NOTE verify P dimension n*2*5
        return np.array(P)

    # P: array of control points, shape n*2*5
    # u (iterable): parameter, domain [0,n], where n is number of break points in spline generation

    def evalBezierSpline(self,P,u):
        u = np.array(u).reshape(-1,1)
        n = len(P)
        assert (u>=0).all()
        assert (u<=n).all()

        B = lambda t,p: (1-t)**5*p[0] + 5*t*(1-t)**4*p[1] + 10*t**2*(1-t)**3*p[2] + 10*t**3*(1-t)**2*p[3] + 5*t**4*(1-t)*p[4] + t**5*p[5]

        try:
            r = [ [B(uu%1,np.array(P[int(uu)%n,:,0])),B(uu%1,np.array(P[int(uu)%n,:,1]))] for uu in u]
        except Warning as e:
            print(e)

        return np.array(r)

    # calculate arc length of <x,y> = fun(u) from ui to uf
    def arcLen(self,fun,ui,uf):
        steps = 20
        uu = np.linspace(ui,uf,steps)
        s = 0
        last_x,last_y = fun(ui).flatten()
        for i in range(steps):
            x,y = fun(uu[i]).flatten()
            s += ((x-last_x)**2 + (y-last_y)**2)**0.5
            last_x,last_y = x,y
        return s

    # calculate variance of curvature w.r.t. break point variation
    # correspond to equation 6 in paper
    def curvatureJac(self):
        break_pts = np.array(self.break_pts).T
        # u_max is also number of break points
        N = self.u_max
        A = np.array([[0,-1],[1,0]])


        # prepare ds vector with initial raceline
        # s[i] = arc distance r_i to r_{i+1}
        # NOTE maybe more accurately this is ds
        ds = []
        fun = lambda x:self.raceline_fun(x).flatten()
        for i in range(N):
            ds.append(self.arcLen(fun,i,(i+1)))

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
            rl = break_pts[:,(i-1)%N]
            # r -> r_k
            r  = break_pts[:,(i)%N]
            # rr -> r_k+1
            rr = break_pts[:,(i+1)%N]
            points = [rl, r, rr]
            sl = ds[(i-1)%N]
            sr = ds[(i)%N]
            
            ((al,a,ar),(bl,b,br)) = self.lagrangeDer(points,ds=(sl,sr))
            dr = al*rl + a*r + ar*rr
            ddr = bl*rl + b*r + br*rr

            dr_vec.append(dr)
            ddr_vec.append(ddr)

            alfa_vec.append( [al,a,ar])
            beta_vec.append( [bl,b,br])

            n = A @ dr
            n_vec.append(n.T)

        for i in range(N):
            # curvature at this characteristic point
            k = np.dot(A @ dr_vec[i], ddr_vec[i])
            xl = np.dot(A @ dr_vec[i], beta_vec[i][0] * n_vec[(i-1)%N])
            xl += np.dot(ddr_vec[i], alfa_vec[i][0] * A @ n_vec[(i-1)%N])

            x = beta_vec[i][1] + np.dot(ddr_vec[i], alfa_vec[i][1] * A @ n_vec[i])
            
            xr = np.dot(A @ dr_vec[i], beta_vec[i][2] * n_vec[(i+1)%N])
            xr += np.dot(ddr_vec[i], alfa_vec[i][2] * A @ n_vec[(i+1)%N])

            k_vec.append(k)
            x_vec.append([xl,x,xr])

        # assemble matrix K, C, Ds
        x_vec = np.array(x_vec)
        k_vec = np.array(k_vec)

        K = np.array(k_vec).reshape(N,1)
        C = np.zeros([N,N])
        C[0,0] = x_vec[0,1]
        C[0,1] = x_vec[0,2]
        C[0,-1] = x_vec[0,0]

        C[-1,-2] = x_vec[-1,0]
        C[-1,-1] = x_vec[-1,1]
        C[-1,0] = x_vec[-1,2]

        for i in range(1,N-1):
            C[i,i-1] = x_vec[i,0]
            C[i,i] = x_vec[i,1]
            C[i,i+1] = x_vec[i,2]

        C = np.array(C)

        # NOTE Ds is not simply ds
        # it is a helper for trapezoidal rule
        Ds = np.array(ds[:-2]) + np.array(ds[1:-1])
        Ds = np.hstack([ds[0], Ds, ds[-1]])

        Ds = 0.5*np.array(np.diag(Ds))

        self.ds = ds
        self.k = k_vec
        self.n = np.array(n_vec).reshape(-1,2)
        self.dr = np.array(dr_vec)
        self.ddr = ddr_vec

        return K, C, Ds



    # override RCPTrack function
    # draw raceline with Bezier curve
    def drawRaceline(self,lineColor=(0,0,255), img=None):
        rows = self.gridsize[0]
        cols = self.gridsize[1]
        res = self.resolution

        # this gives smoother result, but difficult to relate u to actual grid
        #u_new = np.linspace(self.u.min(),self.u.max(),1000)

        # the range of u is len(self.ctrl_pts) + 1, since we copied one to the end
        # x_new and y_new are in non-dimensional grid unit
        # NOTE add new function here
        u_new = np.linspace(0,len(self.break_pts),1000)
        xy = self.raceline_fun(u_new).reshape(-1,2)
        x_new = xy[:,0]
        y_new = xy[:,1]

        # convert to visualization coordinate
        x_new /= self.scale
        x_new *= self.resolution
        y_new /= self.scale
        y_new *= self.resolution
        y_new = self.resolution*rows - y_new

        if img is None:
            img = np.zeros([res*rows,res*cols,3],dtype='uint8')

        pts = np.vstack([x_new,y_new]).T
        # for polylines, pts = pts.reshape((-1,1,2))
        pts = pts.reshape((-1,2))
        pts = pts.astype(int)
        # render different color based on speed
        # slow - red, fast - green (BGR)
        v2c = lambda x: int((x-self.min_v)/(self.max_v-self.min_v)*255)
        getColor = lambda v:(0,v2c(v),255-v2c(v))
        for i in range(len(u_new)-1):
            #img = cv2.line(img, tuple(pts[i]),tuple(pts[i+1]), color=getColor(self.targetVfromU(u_new[i]%(self.break_pts.shape[0]))), thickness=3) 
            # ignore color for now
            img = cv2.line(img, tuple(pts[i]),tuple(pts[i+1]), color=(0,255,0), thickness=3) 

        # solid color
        #img = cv2.polylines(img, [pts], isClosed=True, color=lineColor, thickness=3) 
        for point in self.break_pts:
            x = point[0]
            y = point[1]
            x /= self.scale
            x *= self.resolution
            y /= self.scale
            y *= self.resolution
            y = self.resolution*rows - y
            
            img = cv2.circle(img, (int(x),int(y)), 5, (0,0,255),-1)

        return img

    # input:
    # coord=r=(x,y) unit:m
    # normal direction vector n, NOTE |n|!=1
    # return:
    # F,R such that r+F*n and r-R*n are boundaries of the track
    # F,R will be bounded by delta_max
    def checkTrackBoundary(self,coord,n,delta_max):
        # since we use 1/sin and 1/cos
        # if n[0]or n[1] = 0, then there's numerical instability
        # we use a dirty workaround that when they're too small we force them to be radians(0.1)
        if (abs(n[0])<0.01):
            n = (0.01,n[1])
        if (abs(n[1])<0.01):
            n = (n[0],0.01)

        # figure out which grid the coord is in
        # grid coordinate, (col, row), col starts from left and row starts from bottom, both indexed from 0
        nondim= np.array(np.array(coord)/self.scale//1,dtype=int)
        nondim[0] = np.clip(nondim[0],0,len(self.track)-1).astype(int)
        nondim[1] = np.clip(nondim[1],0,len(self.track[0])-1).astype(int)

        # e.g. 'WE','SE'
        grid_type = self.track[nondim[0]][nondim[1]]

        # change ref frame to tile local ref frame
        x_local = coord[0]/self.scale - nondim[0]
        y_local = coord[1]/self.scale - nondim[1]

        # find the distance to track sides
        # boundary/wall width / grid side length
        # for a flush fit in visualization
        # use 0.087*2
        deadzone = 0.087 * 3.0
        straights = ['WE','NS']
        turns = ['SE','SW','NE','NW']
        if grid_type in straights:
            if grid_type == 'WE':
                # track section is staight, arranged horizontally
                sin_val = n[1]/((n[0]**2+n[1]**2)**0.5)
                if (n[1]>0):
                    # pointing upward
                    F = (1 - deadzone - y_local)/sin_val
                    R = (y_local - deadzone)/sin_val
                else:
                    R = -(1 - deadzone - y_local)/sin_val
                    F = -(y_local - deadzone)/sin_val
            elif grid_type == 'NS':
                # track section is staight, arranged vertically
                cos_val = n[0]/((n[0]**2+n[1]**2)**0.5)
                if (n[0]>0):
                    # pointing rightward
                    F = (1 - deadzone - x_local)/cos_val
                    R = (x_local - deadzone)/cos_val
                else:
                    R = -(1 - deadzone - x_local)/cos_val
                    F = -(x_local - deadzone)/cos_val
        elif grid_type in turns:
            norm = ((n[0]**2+n[1]**2)**0.5)
            if grid_type == 'SE':
                apex = (1,0)
                dot = np.dot(n,(-0.5**0.5,0.5**0.5))
            elif grid_type == 'SW':
                apex = (0,0)
                dot = np.dot(n,(0.5**0.5,0.5**0.5))
            elif grid_type == 'NE':
                apex = (1,1)
                dot = np.dot(n,(-0.5**0.5,-0.5**0.5))
            elif grid_type == 'NW':
                apex = (0,1)
                dot = np.dot(n,(0.5**0.5,-0.5**0.5))

            radius = ((x_local - apex[0])**2 + (y_local - apex[1])**2)**0.5
            if (dot > 0):
                # pointing radially outward
                F = (1 - deadzone - radius)/dot*norm
                R = (radius - deadzone)/dot*norm
            else:
                R = -(1 - deadzone - radius)/dot*norm
                F = -(radius - deadzone)/dot*norm

        # if the point given already violates constrain, then F, R may <0
        # NOTE maybe raise a warning?
        F = max(F,0)
        R = max(R,0)

        return min(F*self.scale,delta_max), min(R*self.scale,delta_max)

    # convert raceline to a B spline to reuse old code for velocity generation and localTrajectory, since they expect a spline object
    def convertToSpline(self):
        # sample entire path
        steps = 100
        N = len(self.break_pts)
        uu = np.linspace(0,N,steps)
        r = self.raceline_fun(uu).reshape(-1,2).T
        # s = smoothing factor
        # per = loop/period
        tck, u = splprep(r, u=np.linspace(0,self.track_length_grid,steps),s=0,per=1) 

        self.u = u
        self.raceline = tck
        u_new = np.linspace(0,self.track_length_grid,steps)
        x_new, y_new = splev(u_new, self.raceline)

        self.generateSpeedProfile()
        self.verifySpeedProfile()
        img_track = self.drawTrack()
        #img_track = super().drawRaceline(img=img_track, points=self.break_pts)
        # do not show break points
        img_track = super().drawRaceline(img=img_track, points=[])
        plt.imshow(img_track)
        plt.show()
        return


    def testLagrangeDer(self):
        # generate three points
        points = np.array([[-1,-1],[2,2],[5,5]])
        rl,r,rr = points
        ((al,a,ar),(bl,b,br)) = self.lagrangeDer(points)
        df = al*rl + a*r + ar*rr
        ddf = bl*rl + b*r + br*rr
        print(df)
        print(ddf)
        x = points[:,0]
        y = points[:,1]
        plt.plot(x,y)
        plt.show()
        return

    # test bezier curve
    def testBezierCurve(self):
        # generate a batch of points following a unit circle
        u = np.linspace(0,(2*pi)/17.0*14,17)
        xx = np.cos(u)
        yy = np.sin(u)
        points = np.vstack([xx,yy]).T

        tic = time()
        # just test w/ an outlier
        points[4,:] = [0,1.2]
        P = self.bezierSpline(points)

        u_close = np.linspace(0,points.shape[0],1000)
        r = self.evalBezierSpline(P,u_close)
        print(time()-tic)

        B_x = r[:,0]
        B_y = r[:,1]

        plt.plot(points[:,0],points[:,1],'bo')
        plt.plot(B_x,B_y)

        plt.show()

        return

    # plot Bezier curve based raceline on a track map
    def testTrack(self):
        # prepare the full racetrack
        self.prepareTrack()
        # use control points as bezier breakpoints
        # generate bezier spline
        self.P = self.bezierSpline(self.ctrl_pts)
        self.u_max = len(self.ctrl_pts)
        self.raceline_fun = lambda u:self.evalBezierSpline(self.P,u)
        # render
        img_track = self.drawTrack()
        img_track = self.drawRaceline(img=img_track)
        plt.imshow(img_track)
        plt.show()

    def testCurvatureJac(self,param=0):
        # prepare the full racetrack
        self.prepareTrack()
        self.break_pts = self.ctrl_pts

        # use control points as bezier breakpoints
        # generate bezier spline
        self.P = self.bezierSpline(self.break_pts)
        self.u_max = len(self.break_pts)
        self.raceline_fun = lambda u:self.evalBezierSpline(self.P,u)

        K, C, Ds = self.curvatureJac()
        '''
        # NOTE verify matrix for compatible dimension
        print("N")
        print(self.u_max)
        print("K")
        print(K.shape)
        print("C")
        print(C.shape)
        print("Ds")
        print(Ds.shape)
        '''

        # verify J(X) = (K+CX).T Ds (K+CX)
        # test model with random variation

        # calculate J(0), without K,C,D
        N = self.u_max
        ds = self.ds
        k = self.k
        J = 0
        # trapezoidal rule
        J += 0.5*k[0]**2*ds[0]
        for i in range(1,N-1):
            J += 0.5*k[i]**2 * (ds[i-1]+ds[i])
        J += 0.5*k[-1]**2*ds[-1]

        X = np.zeros([N,1])
        M_temp = K + C @ X
        J_m = M_temp.T @ Ds @ M_temp

        # error sum
        #print("err sum, un perturbed ")
        #print(np.sum(np.abs(J - J_m)))

        # calculate J(X) with random perturbation X
        delta_max = 1e-2
        #X = (np.random.random([N,1])-0.5)/0.5*delta_max
        X = np.zeros([N,1])
        X[param,0] = delta_max
        # move break points in tangential direction by X
        n = np.array(self.n).reshape(-1,2)
        perturbed_break_pts = np.array(self.break_pts)
        for i in range(N):
            perturbed_break_pts[i,:] += n[i]*X[i]

        # visually verify the pertuabation is reasonable
        # i.e. tangential to path and small in magnitude
        old_pts = np.array(self.break_pts)
        plt.plot(old_pts[:,0],old_pts[:,1],'ro')
        new_pts = perturbed_break_pts
        plt.plot(new_pts[:,0],new_pts[:,1],'b*')
        plt.show()
        print("curvature before")
        print(self.k[param])

        # show raceline
        img_track = self.drawTrack()
        img_track = self.drawRaceline(img=img_track)
        plt.imshow(img_track)
        plt.show()

        # calculate predicted new J(X) with 
        # J(X) = (K+CX).T Ds (K+CX)
        M_temp = K + C @ X
        J_pm = M_temp.T @ Ds @ M_temp

        # calculate new J(X) without matrices
        # this requires re-generation of the Bezier Spline
        self.break_pts = new_pts
        self.P = self.bezierSpline(self.break_pts)
        self.raceline_fun = lambda u:self.evalBezierSpline(self.P,u)
        # need this to calculate new ds and k
        K, C, Ds = self.curvatureJac()
        ds = self.ds
        k = self.k
        J_p = 0
        # trapezoidal rule
        J_p += 0.5*k[0]**2*ds[0]
        for i in range(1,N-1):
            J_p += 0.5*k[i]**2 * (ds[i-1]+ds[i])
        J_p += 0.5*k[-1]**2*ds[-1]
        #print("err sum, random perturbation ")
        #print(np.sum(np.abs(J_p - J_pm)))
        # 0.2 error
        # are we at least going in right direction?
        print("qualitative verification")
        #print(np.sum(np.abs(J_p - J_m)))
        print("ground truth")
        print(J_p-J)

        print("jacobian prediction")
        #print(np.sum(np.abs(J_p - J_pm)))
        print(J_pm[0,0]-J)
        print("curvature after")
        print(self.k[param])

        # show raceline
        img_track = self.drawTrack()
        img_track = self.drawRaceline(img=img_track)
        plt.imshow(img_track)
        plt.show()




        # render
        '''
        img_track = self.drawTrack()
        img_track = self.drawRaceline(img=img_track)
        plt.imshow(img_track)
        plt.show()
        '''

        # 1.0 is ideal
        return (J_p-J)/(J_pm[0,0]-J)

    # resample path defined in raceline_fun
    # new_n: number of break points on the new path
    def resamplePath(self,new_n):
        # generate bezier spline
        P = self.bezierSpline(self.break_pts)
        N = len(self.break_pts)

        self.raceline_fun = lambda u:self.evalBezierSpline(P,u)

        # show initial raceline
        '''
        print("showing initial raceline BEFORE resampling")
        img_track = self.drawTrack()
        img_track = self.drawRaceline(img=img_track)
        plt.imshow(img_track)
        plt.show()
        '''

        # resample with equal arc distance
        # NOTE this seems to introduce instability
        arc_len = [0]
        # we have N+1 points here
        for i in range(1,N+1):
            s = self.arcLen(self.raceline_fun,i-1,i)
            arc_len.append(s)
        uu = np.linspace(0,N,N+1)
        arc_len = np.array(arc_len)
        arc_len = np.cumsum(arc_len)
        s2u = interp1d(arc_len,uu)
        ss = np.linspace(0,arc_len[-1],new_n+1)
        uu = s2u(ss)

        # resample in parameter space
        #uu = np.linspace(0,N,new_n+1)

        #spacing = N/new_n
        # raceline(0) and raceline(N) are the same points
        # if we include both we would have numerical issues
        uu = uu[:-1]
        #uu += np.hstack([0,np.random.rand(new_n-2)/3,0])
        new_break_pts =[]
        for u in uu:
            new_break_pts.append(self.raceline_fun(u).flatten())

        # regenerate spline
        self.break_pts = np.array(new_break_pts)
        P = self.bezierSpline(self.break_pts)
        N = len(self.break_pts)
        self.raceline_fun = lambda u:self.evalBezierSpline(P,u)

        '''
        print("showing initial raceline AFTER resampling")
        img_track = self.drawTrack()
        img_track = self.drawRaceline(img=img_track)
        plt.imshow(img_track)
        plt.show()
        '''

    # optimize path and save to pickle file
    def optimizePath(self):
        # initialize
        # prepare the full racetrack
        self.prepareTrack()

        # use control points as initial bezier breakpoints
        # for full track there are 24 points
        self.break_pts = np.array(self.ctrl_pts)

        # re-sample path, get more break points
        new_N = len(self.break_pts)*3
        print_info("Had %d break points, resample to %d"%(len(self.break_pts),new_N))
        self.resamplePath(new_N)

        # save a gif of the optimization process
        self.saveGif = False

        if self.saveGif:
            self.gifimages = []
            #self.gifimages.append(Image.fromarray(cv2.cvtColor(self.img_track.copy(),cv2.COLOR_BGR2RGB)))

        max_iter = 20
        for iter_count in range(max_iter):

            # TODO re-sample break points before every iteration
            self.resamplePath(new_N)

            # generate bezier spline
            self.P = self.bezierSpline(self.break_pts)
            self.u_max = len(self.break_pts)
            N = self.u_max

            self.raceline_fun = lambda u:self.evalBezierSpline(self.P,u)

            # show raceline
            print_ok("iter: %d"%(iter_count,))
            if self.saveGif:
                img_track = self.drawTrack()
                img_track = self.drawRaceline(img=img_track)
                #plt.imshow(img_track)
                #plt.show()
                self.gifimages.append(Image.fromarray(cv2.cvtColor(img_track.copy(),cv2.COLOR_BGR2RGB)))

            K, C, Ds = self.curvatureJac()

            # assemble matrices in QP
            # NOTE ignored W, W=I
            P_qp = 2 * C.T @ Ds @ C
            q_qp = np.transpose(K.T @ Ds @ C + K.T @ Ds @ C)

            # assemble constrains
            # as in Gx <= h

            # track boundary
            # h = [F..., R...], split into two vec
            h1 =  []
            h2 =  []
            delta_max = 5e-2
            for i in range(N):
                coord = self.break_pts[i]
                F,R = self.checkTrackBoundary(coord,self.n[i],delta_max)
                h1.append(F)
                h2.append(R)

            h = np.array(h1+h2)
            G = np.vstack([np.identity(N),-np.identity(N)])

            # curvature constrain
            # CX <= Kmax - K
            # min radius allowed
            Rmin = 0.102/tan(radians(18))
            Kmax = 1.0/Rmin
            Kmin = -1.0/Rmin
            h3 = Kmax - K
            h3 = h3.flatten()
            h4 = -(Kmin - K)
            h4 = h4.flatten()
            h = np.hstack([h,h3,h4])
            G = np.vstack([G,C,-C])
            print_info("min radius = %.2f"%np.min(np.abs(1.0/K)))

            assert G.shape[1]==N
            assert G.shape[0]==4*N
            assert h.shape[0]==4*N


            # optimize
            P_qp = cvxopt.matrix(P_qp)
            q_qp = cvxopt.matrix(q_qp)
            G = cvxopt.matrix(G)
            h = cvxopt.matrix(h)
            cvxopt.solvers.options['show_progress'] = False
            sol = cvxopt.solvers.qp(P_qp, q_qp, G, h)

            # DEBUG
            # verify Gx <= h is not violated
            #print(sol)
            #print(sol['x'])

            variance = sol['x']
            # verify Gx <= h
            #print("h-GX, should be positive")
            constrain_met = np.array(h) - np.array(G) @ np.array(variance)
            assert constrain_met.all()

            # verify K do not violate constrain
            # FIXME this is not met
            #assert (Kmax-K >0).all()
            #assert (K-Kmin >0).all()

            # check terminal condition
            print_info("max variation %.2f"%(np.max(np.abs(variance))))
            if np.max(np.abs(variance))<0.1*delta_max:
                print_ok("terminal condition met")
                break

            # apply changes to break points
            # move break points in tangential direction by variance vector
            n = np.array(self.n).reshape(-1,2)
            perturbed_break_pts = np.array(self.break_pts)
            for i in range(N):
                perturbed_break_pts[i,:] += n[i]*variance[i]

            self.break_pts = perturbed_break_pts

        if self.saveGif:
            print_info("saving gif.. This may take a while")
            self.log_no = 0
            gif_filename = "./qpOpt"+str(self.log_no)+".gif"
            self.gifimages[0].save(fp=gif_filename,format='GIF',append_images=self.gifimages,save_all=True,duration = 600,loop=0)
            print_ok("gif saved at "+gif_filename)

        img_track = self.drawTrack()
        img_track = self.drawRaceline(img=img_track)
        plt.imshow(img_track)
        plt.show()

        self.convertToSpline()
        self.save()


if __name__ == "__main__":
    # optimize and save
    qp = QpSmooth()
    val = qp.optimizePath()

    # load and show
    fulltrack = RCPTrack()
    print("-----------------")
    print_info("testing loading")
    fulltrack.load()
    img_track = fulltrack.drawTrack()
    img_track = fulltrack.drawRaceline(img=img_track,points=[])
    plt.imshow(img_track)
    plt.show()
