# a track defined by a spline
import os
from common import *
import cv2
import cvxopt
import numpy as np
from math import cos,sin,pi,atan2,radians,degrees,tan
from scipy.interpolate import splprep, splev,CubicSpline,interp1d
import matplotlib.pyplot as plt
import pickle
import types

from track.Track import Track

class CurvilinearTrack(Track):
    def __init__(self,main,config):
        Track.__init__(self,main,config)
        # default parameters, to be overriden
        self.width = 0.6
        self.resolution = 200

        # derived class should override the constructor, 
        # call self.buildContinuousTrack(r) to create a curvilinear track
        # r.shape == (n,2), and consistes of the discretized track centerline
        # see NascarTrack.py for example
        self.discretized_raceline_len = 1024

        self.max_v = 4
        self.min_v = 0
        self.save_dir = './'


        return

    def createBoundary(self, raceline_s, raceline_len_m):
        # generate left/right boundary from self.raceline_left_boundary_*
        ss = np.linspace(0,raceline_len_m, self.discretized_raceline_len)
        rr = np.array(splev(ss,raceline_s,der=0))
        boundary = np.array([self.preciseTrackBoundary(r,0) for r in rr.T])

        left_boundary_fun, _ = splprep(boundary[:,0].reshape(1,-1), u=ss,s=0,per=1) 
        right_boundary_fun, _ = splprep(boundary[:,1].reshape(1,-1), u=ss,s=0,per=1) 
        return left_boundary_fun, right_boundary_fun

    # for plotting raceline during qpSmooth procedure
    def drawBezierRaceline(self,img,P,u_max,break_pts=None):
        u_new = np.linspace(0,u_max,1000)
        xy = self.evalBezierSpline(P,u_new).reshape(-1,2)
        pts = np.array([ self.m2canvas(coord) for coord in xy ]).astype(int)
        for i in range(len(u_new)-1):
            img = cv2.line(img, tuple(pts[i]),tuple(pts[i+1]), color=(0,255,0), thickness=3) 

        #img = cv2.polylines(img, [pts], isClosed=True, color=lineColor, thickness=3) 
        if (not break_pts is None):
            for point in break_pts:
                x = point[0]
                y = point[1]
                x,y = self.m2canvas(point)
                img = cv2.circle(img, (int(x),int(y)), 5, (0,0,255),-1)

        return img


    # depends on self.raceline_s, self.min_v
    def drawRaceline(self,img):
        ss = np.linspace(0,self.raceline_len_m,self.discretized_raceline_len)
        rr = np.array(splev(ss,self.raceline_s,der=0))
        plot_points = np.array([ self.m2canvas(coord) for coord in rr.T ]).astype(int)
        # render different color based on speed
        # slow - red, fast - green (BGR)
        v2c = lambda x: int((x-self.min_v)/(self.max_v-self.min_v)*255)
        getColor = lambda v:(0,v2c(v),255-v2c(v))
        for i in range(len(plot_points)-1):
            img = cv2.line(img, tuple(plot_points[i]),tuple(plot_points[i+1]), color=getColor(self.raceline_speed_s(ss[i])), thickness=3) 
            #img = cv2.line(img, tuple(plot_points[i]),tuple(plot_points[i+1]), color=(0,255,0), thickness=3) 
        return img

    #state: x,y,theta,vf,vs,omega
    # x,y referenced from skidpad frame
    def localTrajectory(self,state,ccw=True,wheelbase = 108e-3):
        x = state[0]
        y = state[1]
        heading = state[2]
        vf = state[3]
        vs = state[4]
        omega = state[5]

        # find the coordinate of center of front axle
        x += wheelbase*cos(heading)
        y += wheelbase*sin(heading)

        dxx = self.r[:,0]-x
        dyy = self.r[:,1]-y
        index = np.argmin(dxx**2+dyy**2)
        raceline_point = (self.r[index])

        # find offset
        # positive offset means car is to the left of the trajectory(need to turn right)
        dr = self.r[(index+1)%self.discretized_raceline_len] - self.r[index]
        track_to_car = (x-self.r[index,0], y-self.r[index,1])
        offset = np.cross(dr/np.linalg.norm(dr),track_to_car).item()

        raceline_orientation = atan2(dr[1],dr[0])

        signed_curvature = splev(self.ss[index],self.curvature_fun)[0].item()

        # reference point on raceline,lateral offset, tangent line orientation, curvature(signed, ccw+), recommended velocity
        return (raceline_point,offset,raceline_orientation,signed_curvature,2.0)

    # return true if vehicle is unsalvageably outside of the track
    # for use by Watchdog to terminate an experiment
    def isOutside(self,coord):
        x = coord[0]
        y = coord[1]
        state = (*coord,0,0,0,0)
        dxx = self.r[:,0]-coord[0]
        dyy = self.r[:,1]-coord[1]
        index = np.argmin(dxx**2+dyy**2)
        raceline_point = (self.r[index])

        # find offset
        # positive offset means car is to the left of the trajectory(need to turn right)
        dr = self.r[(index+1)%self.discretized_raceline_len] - self.r[index]
        track_to_car = (x-self.r[index,0], y-self.r[index,1])
        offset = np.cross(dr/np.linalg.norm(dr),track_to_car).item()
        s = self.ss[index]

        retval = ( offset > splev(s,self.raceline_left_boundary_fun)[0].item()*1.5 ) or ( -offset > splev(s,self.raceline_right_boundary_fun)[0].item()*1.5 )
        return retval

    def isOutsideCurv(self,curv_coord):
        # curv_coord: s,v,n,omega
        offset = curv_coord[2]
        s = curv_coord[0]
        retval = ( offset > splev(s,self.raceline_left_boundary_fun)[0].item()) or ( -offset > splev(s,self.raceline_right_boundary_fun)[0].item() )
        return retval

    def buildContinuousTrack(self,r):
        assert (len(r.shape) == 2)
        assert (r.shape[1] == 2)
        n = r.shape[0]
        xx = r[:,0]
        yy = r[:,1]

        s = 0
        ss = [s]
        for i in range(n):
            s += ((xx[(i+1)%n]-xx[i])**2 +(yy[(i+1)%n]-yy[i])**2 )**0.5
            ss.append(s)
        raceline_len_m = s
        r = np.vstack([r,r[-1]])

        raceline_s, u = splprep(r.T, u=ss,s=0,per=1)

        # let raceline curve be r(u)
        # dr = r'(u), parameterized with xx/u
        dr = np.array(splev(ss,raceline_s,der=1))
        # ddr = r''(u)
        ddr = np.array(splev(ss,raceline_s,der=2))
        _norm = lambda x:np.linalg.norm(x,axis=0)
        # radius of curvature can be calculated as R = |y'|^3/sqrt(|y'|^2*|y''|^2-(y'*y'')^2)
        curvature = 1.0/(_norm(dr)**3/(_norm(dr)**2*_norm(ddr)**2 - np.sum(dr*ddr,axis=0)**2)**0.5)
        curvature_fun, u = splprep(curvature.reshape(1,-1), u=ss,s=0,per=1)
        # n*2
        ss = np.linspace(0,raceline_len_m,self.discretized_raceline_len)
        r_vec = np.array(splev(ss,raceline_s,der=0))
        dr_vec = np.array(splev(ss,raceline_s,der=1))
        phi = np.arctan2(dr_vec[1,:], dr_vec[0,:])
        lateral = np.vstack([np.cos(phi+np.pi/2), np.sin(phi+np.pi/2)]).T

        # boundary
        upper = r_vec.T + lateral * self.width/2
        lower = r_vec.T - lateral * self.width/2


        # discretized raceline center progress
        self.ss = ss
        # centerline positions, correspond to self.ss (n*2)
        self.r = self.r_vec =  r_vec.T
        self.raceline_points = r_vec
        self.curvature_fun = curvature_fun
        # heading
        self.raceline_headings = phi
        # raceline length in meters
        self.raceline_len_m = raceline_len_m
        # spline for centerline(raceline) defined on [0,raceline_len_m]
        self.raceline_s = raceline_s
        # discretized position vector for left/right boundary
        self.raceline_left_boundary_points = upper
        self.raceline_right_boundary_points = lower

        self.x_min = np.min( np.hstack([upper[:,0],lower[:,0]]) ) - 0.1
        self.x_max = np.max( np.hstack([upper[:,0],lower[:,0]]) ) + 0.1
        self.y_min = np.min( np.hstack([upper[:,1],lower[:,1]]) ) - 0.1
        self.y_max = np.max( np.hstack([upper[:,1],lower[:,1]]) ) + 0.1

        # left boundary distance to centerline
        self.raceline_left_boundary = np.ones_like(ss)*self.width/2
        self.raceline_right_boundary = np.ones_like(ss)*self.width/2

        self.raceline_left_boundary_fun, _ = splprep(self.raceline_left_boundary.reshape(1,-1), u=ss,s=0,per=1) 
        self.raceline_right_boundary_fun, _ = splprep(self.raceline_right_boundary.reshape(1,-1), u=ss,s=0,per=1) 
        self.discretized_raceline = np.vstack([self.raceline_points,self.raceline_headings, self.raceline_left_boundary, self.raceline_right_boundary]).T

        #self.upper_fun = self.buildSpline(upper)
        #self.lower_fun = self.buildSpline(lower)

        self.raceline_speed_s = interp1d(ss,np.ones_like(ss),kind='cubic')

        '''
        plt.plot(upper[:,0],upper[:,1])
        plt.plot(lower[:,0],lower[:,1])
        plt.plot(r_vec[0,:],r_vec[1,:],'o')
        plt.show()
        '''
        return


    def m2canvas(self,coord):
        x_new = int((np.clip(coord[0],self.x_min,self.x_max)-self.x_min) * self.resolution)
        y_new = int((self.y_max - np.clip(coord[1],self.y_min,self.y_max)) * self.resolution)
        return (x_new,y_new)

    # draw a picture of the track
    def drawTrack(self):
        x_pix = int((self.x_max - self.x_min)*self.resolution)
        y_pix = int((self.y_max - self.y_min)*self.resolution)
        # height, width
        img = 255*np.ones([y_pix,x_pix,3],dtype=np.uint8)
        img = self.drawPolyline(self.raceline_left_boundary_points,img,lineColor=(0,0,0),thickness=3)
        img = self.drawPolyline(self.raceline_right_boundary_points,img,lineColor=(0,0,0),thickness=3)
        return img

    def preciseTrackBoundary(self,coord,heading):
        x = coord[0]
        y = coord[1]
        dxx = self.r[:,0]-x
        dyy = self.r[:,1]-y
        index = np.argmin(dxx**2+dyy**2)

        # find offset
        # positive offset means car is to the left of the trajectory(need to turn right)
        dr = self.r[(index+2)%self.discretized_raceline_len] - self.r[index]
        track_to_car = (x-self.r[index,0], y-self.r[index,1])
        offset = np.cross(dr/np.linalg.norm(dr),track_to_car).item()
        s = self.ss[index]
        left =  splev(s,self.raceline_left_boundary_fun)[0].item() - offset
        right = splev(s,self.raceline_right_boundary_fun)[0].item() + offset
        return (left,right)

    # optimize path and save to pickle file
    def optimizePath(self,*,max_iter=20, offset=0, mu=0.8,acc_max_fun=lambda x:8.0, dec_max_fun=lambda x:3.3, visualize=False,save_gif=False,save_steps=False,):
        self.print_info('Minimum curvature smoothing')
        # use control points as initial bezier breakpoints
        # for full track there are 24 points

        new_N = 60
        step = int(self.discretized_raceline_len / new_N)
        # break_pts.shape == n*2
        break_pts = self.r[0::step,:]
        self.print_info(f'subsample to {self.r.shape[0]} break points')
        #self.resamplePath(60)

        if save_gif:
            self.gifimages = []
            #self.gifimages.append(Image.fromarray(cv2.cvtColor(self.img_track.copy(),cv2.COLOR_BGR2RGB)))

        for iter_count in range(max_iter):

            # re-sample reference points before every iteration
            break_pts = self.resamplePath(break_pts, new_N)
            N = len(break_pts)

            # generate bezier spline
            P = self.bezierSpline(break_pts)

            print_ok("iter: %d"%(iter_count,))
            img_track = self.drawTrack()
            img_track = self.drawBezierRaceline(img_track,P,len(break_pts),break_pts)
            if (save_steps):
                filename = os.path.join(self.save_dir,f'iter{iter_count}.png')
                cv2.imwrite(filename,img_track)
                print(f'iteration image saved at {filename}')


            if (visualize):
                img_track_rgb = cv2.cvtColor(img_track.copy(),cv2.COLOR_BGR2RGB)
                plt.imshow(img_track_rgb)
                plt.show()
            if (save_gif):
                self.gifimages.append(Image.fromarray(cv2.cvtColor(img_track.copy(),cv2.COLOR_BGR2RGB)))

            K, C, Ds, n = self.curvatureJac(P,break_pts)

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
                coord = break_pts[i]
                F,R = self.checkTrackBoundary(coord,n[i],delta_max,offset)
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
            # n = np.array(n).reshape(-1,2)
            perturbed_break_pts = np.array(break_pts)
            for i in range(N):
                perturbed_break_pts[i,:] += n[i]*variance[i]

            break_pts = perturbed_break_pts

        raceline_s, raceline_len_m = self.convertToSpline(P,break_pts)
        ss = np.linspace(0,raceline_len_m, self.discretized_raceline_len)

        dr = np.array(splev(ss,raceline_s,der=1))
        # ddr = r''(u)
        ddr = np.array(splev(ss,raceline_s,der=2))
        heading = np.arctan2(dr[1,:], dr[0,:])
        _norm = lambda x:np.linalg.norm(x,axis=0)
        # radius of curvature can be calculated as R = |y'|^3/sqrt(|y'|^2*|y''|^2-(y'*y'')^2)
        curvature = 1.0/(_norm(dr)**3/(_norm(dr)**2*_norm(ddr)**2 - np.sum(dr*ddr,axis=0)**2)**0.5)
        curvature_fun, u = splprep(curvature.reshape(1,-1), u=ss,s=0,per=1)

        # create boundary
        left_boundary_fun, right_boundary_fun = self.createBoundary(raceline_s, raceline_len_m)

        # continuous raceline
        self.raceline_s = raceline_s
        self.raceline_len_m = raceline_len_m
        self.raceline_left_boundary_fun = left_boundary_fun
        self.raceline_right_boundary_fun = right_boundary_fun
        self.curvature_fun = curvature_fun

        # discretized raceline
        self.ss = ss
        self.r = np.array(splev(self.ss, raceline_s, der=0)).T
        self.curvature = curvature
        self.raceline_points = self.r.T
        self.raceline_headings = heading
        self.raceline_left_boundary = np.array(splev(self.ss, left_boundary_fun, der=0))
        self.raceline_right_boundary = np.array(splev(self.ss, right_boundary_fun, der=0))

        lateral = np.vstack([np.cos(heading+np.pi/2), np.sin(heading+np.pi/2)]).T

        self.raceline_left_boundary_points = self.r + lateral * self.raceline_left_boundary.T
        self.raceline_right_boundary_points = self.r - lateral * self.raceline_right_boundary.T
        self.discretized_raceline = np.vstack([self.raceline_points,self.raceline_headings, self.raceline_left_boundary, self.raceline_right_boundary]).T

        retval = self.generateSpeedProfile(mu=mu,acc_max_fun=acc_max_fun, dec_max_fun=dec_max_fun)
        self.raceline_speed_s = self.sToV = speed_profile_fun = retval['speed_profile_fun']
        self.max_v = retval['max_v']
        self.min_v = retval['min_v']


        self.verifySpeedProfile(speed_profile_fun = speed_profile_fun)
        if save_gif:
            print_info("saving gif.. This may take a while")
            self.log_no = 0
            gif_filename = "./qpOpt"+str(self.log_no)+".gif"
            self.gifimages[0].save(fp=gif_filename,format='GIF',append_images=self.gifimages,save_all=True,duration = 600,loop=0)
            print_ok("gif saved at "+gif_filename)

        img_track = self.drawTrack()
        img_track = self.drawRaceline(img=img_track)
        img_track = self.drawBezierRaceline(img_track,P,N,break_pts)
        img_track_rgb = cv2.cvtColor(img_track.copy(),cv2.COLOR_BGR2RGB)

        plt.imshow(img_track_rgb)
        plt.show()

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

    # requires self.raceline_s, self.raceline_len_m
    def generateSpeedProfile(self, *, mu=1.0, acc_max_fun=lambda x:10.0, dec_max_fun=lambda x:10.0, show=False):
        '''
        generate speed profile given traction constraints, braking/acceleration limit
        [mu]: coefficient of friction for the radius of traction circle. maximum traction = mu*g
        [acc_max_fun]: function (velocity) => maximum acceleration available from motor. Given velocity, provide maximum acceleration available, for miniz ~3.3m/s2
        [dec_max_fun]: same as acc_max_fun, for deceleration ~4.5
        '''
        g = 10.0

        # generate velocity profile
        # u values for control points
        xx = np.linspace(0,self.raceline_len_m,self.discretized_raceline_len+1)

        # let raceline curve be r(u)
        # dr = r'(u), parameterized with xx/u
        dr = np.array(splev(xx,self.raceline_s,der=1))
        # ddr = r''(u)
        ddr = np.array(splev(xx,self.raceline_s,der=2))
        _norm = lambda x:np.linalg.norm(x,axis=0)
        # radius of curvature can be calculated as R = |y'|^3/sqrt(|y'|^2*|y''|^2-(y'*y'')^2)
        curvature = 1.0/(_norm(dr)**3/(_norm(dr)**2*_norm(ddr)**2 - np.sum(dr*ddr,axis=0)**2)**0.5)

        # first pass, based on lateral acceleration
        v1 = (mu*g/curvature)**0.5

        dist = lambda a,b: ((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5
        # second pass, based on engine capacity and available longitudinal traction
        # start from the index with lowest speed
        min_xx = np.argmin(v1)
        v2 = np.zeros_like(v1)
        v2[min_xx] = v1[min_xx]
        for i in range(min_xx,min_xx+self.discretized_raceline_len):
            # lateral acc at next step if the car mainains speed
            a_lat = v2[i%self.discretized_raceline_len]**2*curvature[(i+1)%self.discretized_raceline_len]

            # is there available traction for acceleration?
            if ((mu*g)**2-a_lat**2)>0:
                a_lon_available_traction = ((mu*g)**2-a_lat**2)**0.5
                # constrain with motor capacity
                a_lon = min(acc_max_fun(v2[i%self.discretized_raceline_len]),a_lon_available_traction)

                (x_i, y_i) = splev(xx[i%self.discretized_raceline_len], self.raceline_s, der=0)
                (x_i_1, y_i_1) = splev(xx[(i+1)%self.discretized_raceline_len], self.raceline_s, der=0)
                # distance between two steps
                ds = dist((x_i, y_i),(x_i_1, y_i_1))
                # assume vehicle accelerate uniformly between the two steps
                v2[(i+1)%self.discretized_raceline_len] =  min((v2[i%self.discretized_raceline_len]**2 + 2*a_lon*ds)**0.5,v1[(i+1)%self.discretized_raceline_len])
            else:
                v2[(i+1)%self.discretized_raceline_len] =  v1[(i+1)%self.discretized_raceline_len]

        v2[-1]=v2[0]
        # third pass, backwards for braking
        min_xx = np.argmin(v2)
        v3 = np.zeros_like(v1)
        v3[min_xx] = v2[min_xx]
        for i in np.linspace(min_xx,min_xx-self.discretized_raceline_len,self.discretized_raceline_len+2):
            i = int(i)
            a_lat = v3[i%self.discretized_raceline_len]**2*curvature[(i-1+self.discretized_raceline_len)%self.discretized_raceline_len]
            a_lon_available_traction = abs((mu*g)**2-a_lat**2)**0.5
            a_lon = min(dec_max_fun(v3[i%self.discretized_raceline_len]),a_lon_available_traction)
            #print(a_lon)

            (x_i, y_i) = splev(xx[i%self.discretized_raceline_len], self.raceline_s, der=0)
            (x_i_1, y_i_1) = splev(xx[(i-1+self.discretized_raceline_len)%self.discretized_raceline_len], self.raceline_s, der=0)
            # distance between two steps
            ds = dist((x_i, y_i),(x_i_1, y_i_1))
            #print(ds)
            v3[(i-1+self.discretized_raceline_len)%self.discretized_raceline_len] =  min((v3[i%self.discretized_raceline_len]**2 + 2*a_lon*ds)**0.5,v2[(i-1+self.discretized_raceline_len)%self.discretized_raceline_len])
            #print(v3[(i-1+self.discretized_raceline_len)%self.discretized_raceline_len],v2[(i-1+self.discretized_raceline_len)%self.discretized_raceline_len])
            pass

        v3[-1]=v3[0]

        # when callingmake sure u is in range [0,len(self.ctrl_pts)]
        speed_profile_fun = interp1d(xx,v3,kind='cubic')
        #self.v1 = interp1d(xx,v1,kind='cubic')
        #self.v2 = interp1d(xx,v2,kind='cubic')
        #self.v3 = interp1d(xx,v3,kind='cubic')

        max_v = max(v3)
        min_v = min(v3)

        # debug target v curve fitting
        #p0, = plt.plot(xx,v3,'*',label='original')
        #xxx = np.linspace(0,self.track_length_grid,10*self.discretized_raceline_len)
        #sampleV = self.raceline_speed_s(xxx)
        #p1, = plt.plot(xxx,sampleV,label='fitted')
        #plt.legend(handles=[p0,p1])
        #plt.show()

        # three pass of velocity profile
        if (show):
            p0, = plt.plot(curvature, label='curvature')
            p1, = plt.plot(v1,label='1st pass')
            p2, = plt.plot(v2,label='2nd pass')
            p3, = plt.plot(v3,label='3rd pass')
            plt.legend(handles=[p1,p2,p3])
            plt.show()

        return {'speed_profile_fun':speed_profile_fun, 'min_v': min_v, 'max_v':max_v}

    # convert raceline to a B spline to reuse old code for velocity generation and localTrajectory, since they expect a spline object
    # result save at self.raceline
    def convertToSpline(self, P, break_pts):
        # sample entire path
        uu = np.linspace(0,len(break_pts),self.discretized_raceline_len)
        r = self.evalBezierSpline(P,uu).reshape(-1,2).T
        # s = smoothing factor
        # per = loop/period
        fun = lambda x:self.evalBezierSpline(P,x).flatten()
        N = len(break_pts)
        new_raceline_len = self.arcLen(fun,0,N)
        self.print_info(f'Reference length {self.raceline_len_m} -> {new_raceline_len}')

        raceline_s, u = splprep(r, u=np.linspace(0,new_raceline_len,self.discretized_raceline_len),s=0,per=1) 

        return raceline_s, new_raceline_len


    # calculate variance of curvature w.r.t. break point variation
    # correspond to equation 6 in paper
    def curvatureJac(self,P,break_pts):
        N = len(break_pts)
        break_ptsT = np.array(break_pts).T
        A = np.array([[0,-1],[1,0]])


        # prepare ds vector with initial raceline
        # s[i] = arc distance r_i to r_{i+1}
        # NOTE maybe more accurately this is ds
        ds = []
        fun = lambda x:self.evalBezierSpline(P,x).flatten()
        for i in range(N):
            ds.append(self.arcLen(fun,i,(i+1)))

        # calculate first and second derivative
        # w.r.t. ds
        dr_vec = []
        ddr_vec = []
        # (N=len(break_pts),3)
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
            rl = break_ptsT[:,(i-1)%N]
            # r -> r_k
            r  = break_ptsT[:,(i)%N]
            # rr -> r_k+1
            rr = break_ptsT[:,(i+1)%N]
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

        #self.ds = ds
        #self.k = k_vec
        n = np.array(n_vec).reshape(-1,2)
        #self.dr = np.array(dr_vec)
        #self.ddr = ddr_vec

        return K, C, Ds, n

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


    # resample path defined in raceline_fun
    # new_n: number of break points on the new path
    def resamplePath(self, break_pts, new_n):
        # generate bezier spline
        P = self.bezierSpline(break_pts)
        N = len(break_pts)

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
            s = self.arcLen(lambda x:self.evalBezierSpline(P,x),i-1,i)
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
            new_break_pts.append(self.evalBezierSpline(P,u).flatten())

        # regenerate spline
        break_pts = np.array(new_break_pts)
        return break_pts

        '''
        print("showing initial raceline AFTER resampling")
        img_track = self.drawTrack()
        img_track = self.drawRaceline(img=img_track)
        plt.imshow(img_track)
        plt.show()
        '''
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

    def checkTrackBoundary(self,coord,n,delta_max,offset=0):
        '''
            # input:
            # coord: r=(x,y) unit:m
            # n: normal direction vector n, NOTE |n|!=1
            # delta_max: upper bound for returned value
            # offset: offset from positive boundary
            # return:
            # F,R such that r+F*n and r-R*n are boundaries of the track
        '''

        L,R = self.preciseTrackBoundary(coord,heading=0)
        L = max(L-offset,0)
        R = max(R-offset,0)
        return (L,R)

    # uses self.raceline_len_m, self.raceline_s
    def verifySpeedProfile(self,*,speed_profile_fun, mu = 0.7,show_traction_circle=True):
        # calculate theoretical lap time
        g = 9.81
        t_total = 0
        path_len = 0
        n_steps = self.discretized_raceline_len
        xx = np.linspace(0,self.raceline_len_m,n_steps+1)
        dist = lambda a,b: ((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5
        vv = speed_profile_fun(xx)
        for i in range(n_steps):
            (x_i, y_i) = splev(xx[i%n_steps], self.raceline_s, der=0)
            (x_i_1, y_i_1) = splev(xx[(i+1)%n_steps], self.raceline_s, der=0)
            # distance between two steps
            ds = dist((x_i, y_i),(x_i_1, y_i_1))
            path_len += ds
            t_total += ds/(vv[i%n_steps]+vv[(i+1)%n_steps])*2

        print_info("Theoretical value:")
        print_info("\t top speed = %.2fm/s"%max(vv))
        print_info("\t total time = %.2fs"%t_total)
        print_info("\t path len = %.2fm"%path_len)

        # cartesian distance from two u(parameter)
        distuu = lambda u1,u2: dist(splev(u1, self.raceline_s, der=0),splev(u2, self.raceline_s, der=0))

        vel_vec = []
        ds_vec = []

        # get velocity at each point
        for i in range(n_steps):
            # tangential direction
            tan_dir = splev(xx[i], self.raceline_s, der=1)
            tan_dir = np.array(tan_dir/np.linalg.norm(tan_dir))
            vel_now = vv[i] * tan_dir
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
        if (show_traction_circle):
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

            p0, = plt.plot(vv,label='speed')
            plt.legend(handles=[p0])
            plt.show()
        print("theoretical laptime %.2f"%t_total)
        return t_total

if __name__ == "__main__":
    pass
