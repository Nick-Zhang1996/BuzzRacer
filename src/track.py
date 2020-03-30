#!/usr/bin/python

# this file contains all data structure and algorithm to :
# describe an RCP track
# describe a raceline within the track
# provide desired trajectory(raceline) given a car's location within thte track

# TODO:
# 2 Setup experiment to run the car at different speed in a circular path to find yaw setpoint
# implement yaw damping
# implement EKF to estimate velocity and yaw rate
# 3 add velocity trajectory
#  add velocity PI control
# 1 enable real time tuning of parameters
# runfile

import numpy as np
from numpy import isclose
import matplotlib.pyplot as plt
from math import atan2,radians,degrees,sin,cos,pi,tan,copysign,asin,acos,isnan
from scipy.interpolate import splprep, splev,CubicSpline,interp1d
from scipy.optimize import minimize_scalar,minimize,brentq
from time import sleep,time
from timeUtil import execution_timer
import cv2
from PIL import Image
from car import Car

# debugging
K_vec = [] # curvature
steering_vec = []
sim_omega_vec = []
sim_log_vec = {}



class Node:
    def __init__(self,previous=None,entrydir=None):
        # entry direction
        self.entry = entrydir
        # previous node
        self.previous = previous
        # exit direction
        self.exit = None
        self.next = None
        return
    def setExit(self,exit=None):
        self.exit = exit
        return
    def setEntry(self,entry=None):
        self.entry = entry
        return

class RCPtrack:
    def __init__(self):
        # resolution : pixels per grid side length
        self.resolution = 150
        # for calculating derivative and integral of offset
        # for PID to use
        self.offset_history = []
        self.offset_timestamp = []

    def initTrack(self,description, gridsize, scale,savepath=None):
        # build a track and save it
        # description: direction to go to reach next grid u(p), r(ight),d(own), l(eft)
        # e.g For a track like this                    
        #         /-\            
        #         | |            
        #         \_/            
        # The trajectory description, starting from the bottom left corner (origin) would be
        # uurrddll (cw)(string), it does not matter which direction is usd

        # gridsize (rows, cols), size of thr track
        # savepath, where to store the track file

        # scale : meters per grid width (for RCP roughly 0.565m)
        self.scale = scale
        self.gridsize = gridsize
        self.track_length = len(description)
        self.grid_sequence = []
        
        grid = [[None for i in range(gridsize[0])] for j in range(gridsize[1])]

        current_index = np.array([0,0])
        self.grid_sequence.append(list(current_index))
        grid[0][0] = Node()
        current_node = grid[0][0]
        lookup_table_dir = {'u':(0,1),'d':(0,-1),'r':(1,0),'l':(-1,0) }

        for i in range(len(description)):
            current_node.setExit(description[i])

            previous_node = current_node
            if description[i] in lookup_table_dir:
                current_index += lookup_table_dir[description[i]]
                self.grid_sequence.append(list(current_index))
            else:
                print("error, unexpected value in description")
                exit(1)
            if all(current_index == [0,0]):
                grid[0][0].setEntry(description[i])
                # if not met, description does not lead back to origin
                assert i==len(description)-1
                break

            # assert description does not go beyond defined grid
            assert (current_index[0]<gridsize[1]) & (current_index[1]<gridsize[0])

            current_node = Node(previous=previous_node, entrydir=description[i])
            grid[current_index[0]][current_index[1]] = current_node

        #grid[0][0].setEntry(description[-1])


        # process the linked list, replace with the following
        # straight segment = WE(EW), NS(SN)
        # curved segment = SE,SW,NE,NW
        lookup_table = { 'WE':['rr','ll'],'NS':['uu','dd'],'SE':['ur','ld'],'SW':['ul','rd'],'NE':['dr','lu'],'NW':['ru','dl']}
        for i in range(gridsize[1]):
            for j in range(gridsize[0]):
                node = grid[i][j]
                if node == None:
                    continue

                signature = node.entry+node.exit
                for entry in lookup_table:
                    if signature in lookup_table[entry]:
                        grid[i][j] = entry

                if grid[i][j] is Node:
                    print('bad track description: '+signature)
                    exit(1)

        self.track = grid
        # TODO add save pickle function
        return 


    def drawTrack(self, img=None,show=False):
        # show a picture of the track
        # resolution : pixels per grid length
        color_side = (255,0,0)
        # boundary width / grid width
        deadzone = 0.087
        gs = self.resolution

        # prepare straight section (WE)
        straight = 255*np.ones([gs,gs,3],dtype='uint8')
        straight = cv2.rectangle(straight, (0,0),(gs-1,int(deadzone*gs)),color_side,-1)
        straight = cv2.rectangle(straight, (0,int((1-deadzone)*gs)),(gs-1,gs-1),color_side,-1)
        WE = straight

        # prepare turn section (SE)
        turn = 255*np.ones([gs,gs,3],dtype='uint8')
        turn = cv2.rectangle(turn, (0,0),(int(deadzone*gs),gs-1),color_side,-1)
        turn = cv2.rectangle(turn, (0,0),(gs-1,int(deadzone*gs)),color_side,-1)
        turn = cv2.rectangle(turn, (0,0), (int(0.5*gs),int(0.5*gs)),color_side,-1)
        turn = cv2.circle(turn, (int(0.5*gs),int(0.5*gs)),int((0.5-deadzone)*gs),(255,255,255),-1)
        turn = cv2.circle(turn, (gs-1,gs-1),int(deadzone*gs),color_side,-1)
        SE = turn

        # prepare canvas
        rows = self.gridsize[0]
        cols = self.gridsize[1]
        if img is None:
            img = 255*np.ones([gs*rows,gs*cols,3],dtype='uint8')
        lookup_table = {'SE':0,'SW':270,'NE':90,'NW':180}
        for i in range(cols):
            for j in range(rows):
                signature = self.track[i][rows-1-j]
                if signature == None:
                    continue

                if (signature == 'WE'):
                    img[j*gs:(j+1)*gs,i*gs:(i+1)*gs] = WE
                    continue
                elif (signature == 'NS'):
                    M = cv2.getRotationMatrix2D((gs/2,gs/2),90,1.01)
                    NS = cv2.warpAffine(WE,M,(gs,gs))
                    img[j*gs:(j+1)*gs,i*gs:(i+1)*gs] = NS
                    continue
                elif (signature in lookup_table):
                    M = cv2.getRotationMatrix2D((gs/2,gs/2),lookup_table[signature],1.01)
                    dst = cv2.warpAffine(SE,M,(gs,gs))
                    img[j*gs:(j+1)*gs,i*gs:(i+1)*gs] = dst
                    continue
                else:
                    print("err, unexpected track designation : " + signature)

        # some rotation are not perfect and leave a black gap
        img = cv2.medianBlur(img,5)
        '''
        if show:
            plt.imshow(img)
            plt.show()
        '''

        return img
    
    def loadTrackfromFile(self,filename,newtrack,gridsize):
        # load a track 
        self.track = newtrack
        self.gridsize = gridsize

        return

    # this function stores result in self.raceline
    # seq_no: labeling the starting grid as 0, progressing through the raceline direction, the sequence number of (0,0) grid, i.e., bottom left. In other words, how many grids are between the starting grid and the origin? If starting gtid is origin grid, then seq_no is zero
    # Note self.raceline takes u, a dimensionless variable that corresponds to the control point on track
    # rance of u is (0,len(self.ctrl_pts) with 1 corresponding to the exit point out of the starting grid,
    # both 0 and len(self.ctrl_pts) pointing to the entry ctrl point for the starting grid
    # and gives a pair of coordinates in METER
    def initRaceline(self,start, start_direction,seq_no,offset=None, filename=None):
        #init a raceline from current track, save if specified 
        # start: which grid to start from, e.g. (3,3)
        # start_direction: which direction to go. 
        #note use the direction for ENTERING that grid element 
        # e.g. 'l' or 'd' for a NE oriented turn
        # NOTE you MUST start on a straight section
        self.ctrl_pts = []
        self.ctrl_pts_w = []
        self.origin_seq_no = seq_no
        if offset is None:
            offset = np.zeros(self.track_length)

        # provide exit direction given signature and entry direction
        lookup_table = { 'WE':['rr','ll'],'NS':['uu','dd'],'SE':['ur','ld'],'SW':['ul','rd'],'NE':['dr','lu'],'NW':['ru','dl']}
        # provide correlation between direction (character) and directional vector
        lookup_table_dir = {'u':(0,1),'d':(0,-1),'r':(1,0),'l':(-1,0) }
        # provide right hand direction, this is for specifying offset direction 
        lookup_table_right = {'u':(1,0),'d':(-1,0),'r':(0,-1),'l':(0,1) }
        # provide apex direction
        turn_offset_toward_center = {'SE':(1,-1),'NE':(1,1),'SW':(-1,-1),'NW':(-1,1)}
        turns = ['SE','SW','NE','NW']

        center = lambda x,y : [(x+0.5)*self.scale,(y+0.5)*self.scale]

        left = lambda x,y : [(x)*self.scale,(y+0.5)*self.scale]
        right = lambda x,y : [(x+1)*self.scale,(y+0.5)*self.scale]
        up = lambda x,y : [(x+0.5)*self.scale,(y+1)*self.scale]
        down = lambda x,y : [(x+0.5)*self.scale,(y)*self.scale]

        # direction of entry
        entry = start_direction
        current_coord = np.array(start,dtype='uint8')
        signature = self.track[current_coord[0]][current_coord[1]]
        # find the previous signature, reverse entry to find ancestor
        # the precedent grid for start grid is also the final grid
        final_coord = current_coord - lookup_table_dir[start_direction]
        last_signature = self.track[final_coord[0]][final_coord[1]]

        # for referencing offset
        index = 0
        while (1):
            signature = self.track[current_coord[0]][current_coord[1]]

            # lookup exit direction
            for record in lookup_table[signature]:
                if record[0] == entry:
                    exit = record[1]
                    break

            # 0~0.5, 0 means no offset at all, 0.5 means hitting apex 
            apex_offset = 0.2

            # find the coordinate of the exit point
            # offset from grid center to centerpoint of exit boundary
            # go half a step from center toward exit direction
            exit_ctrl_pt = np.array(lookup_table_dir[exit],dtype='float')/2
            exit_ctrl_pt += current_coord
            exit_ctrl_pt += np.array([0.5,0.5])
            # apply offset, offset range (-1,1)
            exit_ctrl_pt += offset[index]*np.array(lookup_table_right[exit],dtype='float')/2
            index += 1

            exit_ctrl_pt *= self.scale
            self.ctrl_pts.append(exit_ctrl_pt.tolist())

            current_coord = current_coord + lookup_table_dir[exit]
            entry = exit

            last_signature = signature

            if (all(start==current_coord)):
                break

        # add end point to the beginning, otherwise splprep will replace pts[-1] with pts[0] for a closed loop
        # This ensures that splev(u=0) gives us the beginning point
        pts=np.array(self.ctrl_pts)
        #start_point = np.array(self.ctrl_pts[0])
        #pts = np.vstack([pts,start_point])
        end_point = np.array(self.ctrl_pts[-1])
        pts = np.vstack([end_point,pts])

        #weights = np.array(self.ctrl_pts_w + [self.ctrl_pts_w[-1]])

        # s= smoothing factor
        #a good s value should be found in the range (m-sqrt(2*m),m+sqrt(2*m)), m being number of datapoints
        m = len(self.ctrl_pts)+1
        smoothing_factor = 0.01*(m)
        tck, u = splprep(pts.T, u=np.linspace(0,len(pts)-1,len(pts)), s=smoothing_factor, per=1) 
        #NOTE 
        #tck, u = CubicSpline(np.linspace(0,len(pts)-1,len(pts)),pts) 

        # this gives smoother result, but difficult to relate u to actual grid
        #tck, u = splprep(pts.T, u=None, s=0.0, per=1) 
        #self.u = u
        self.raceline = tck

        # friction factor
        mu = 10.0/9.81
        g = 9.81
        n_steps = 1000
        self.n_steps = n_steps
        # maximum longitudinial acceleration available from motor, given current longitudinal speed
        # actually around 3.3
        acc_max_motor = lambda x:10
        dec_max_motor = lambda x:10
        # generate velocity profile
        # u values for control points
        xx = np.linspace(0,len(pts)-1,n_steps+1)
        #curvature = splev(xx,self.raceline,der=2)
        #curvature = np.linalg.norm(curvature,axis=0)

        # let raceline curve be r(u)
        # dr = r'(u), parameterized with xx/u
        dr = np.array(splev(xx,self.raceline,der=1))
        # ddr = r''(u)
        ddr = np.array(splev(xx,self.raceline,der=2))
        _norm = lambda x:np.linalg.norm(x,axis=0)
        # radius of curvature can be calculated as R = |y'|^3/sqrt(|y'|^2*|y''|^2-(y'*y'')^2)
        curvature = 1.0/(_norm(dr)**3/(_norm(dr)**2*_norm(ddr)**2 - np.sum(dr*ddr,axis=0)**2)**0.5)

        # first pass, based on lateral acceleration
        v1 = (mu*g/curvature)**0.5

        dist = lambda a,b: ((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5
        # second pass, based on engine capacity and available longitudinal traction
        # start from the smallest index
        min_xx = np.argmin(v1)
        v2 = np.zeros_like(v1)
        v2[min_xx] = v1[min_xx]
        for i in range(min_xx,min_xx+n_steps):
            a_lat = v1[i%n_steps]**2*curvature[(i+1)%n_steps]
            a_lon_available_traction = abs((mu*g)**2-a_lat**2)**0.5
            a_lon = min(acc_max_motor(v2[i%n_steps]),a_lon_available_traction)

            (x_i, y_i) = splev(xx[i%n_steps], self.raceline, der=0)
            (x_i_1, y_i_1) = splev(xx[(i+1)%n_steps], self.raceline, der=0)
            # distance between two steps
            ds = dist((x_i, y_i),(x_i_1, y_i_1))
            # assume vehicle accelerate uniformly between the two steps
            v2[(i+1)%n_steps] =  min((v2[i%n_steps]**2 + 2*a_lon*ds)**0.5,v1[(i+1)%n_steps])

        v2[-1]=v2[0]
        # third pass, backwards for braking
        min_xx = np.argmin(v2)
        v3 = np.zeros_like(v1)
        v3[min_xx] = v2[min_xx]
        for i in np.linspace(min_xx,min_xx-n_steps,n_steps+2):
            i = int(i)
            a_lat = v2[i%n_steps]**2*curvature[(i-1+n_steps)%n_steps]
            a_lon_available_traction = abs((mu*g)**2-a_lat**2)**0.5
            a_lon = min(dec_max_motor(v3[i%n_steps]),a_lon_available_traction)
            #print(a_lon)

            (x_i, y_i) = splev(xx[i%n_steps], self.raceline, der=0)
            (x_i_1, y_i_1) = splev(xx[(i-1+n_steps)%n_steps], self.raceline, der=0)
            # distance between two steps
            ds = dist((x_i, y_i),(x_i_1, y_i_1))
            #print(ds)
            v3[(i-1+n_steps)%n_steps] =  min((v3[i%n_steps]**2 + 2*a_lon*ds)**0.5,v2[(i-1+n_steps)%n_steps])
            #print(v3[(i-1+n_steps)%n_steps],v2[(i-1+n_steps)%n_steps])
            pass

        v3[-1]=v3[0]

        # call with self.targetVfromU(u) alwayos u is in range [0,len(self.ctrl_pts)]
        self.targetVfromU = interp1d(xx,v3,kind='cubic')
        self.v1 = interp1d(xx,v1,kind='cubic')
        self.v2 = interp1d(xx,v2,kind='cubic')

        self.max_v = max(v3)
        self.min_v = min(v3)

        # debug target v curve fitting
        #p0, = plt.plot(xx,v3,'*',label='original')
        #xxx = np.linspace(0,len(pts)-1,10*n_steps)
        #sampleV = self.targetVfromU(xxx)
        #p1, = plt.plot(xxx,sampleV,label='fitted')
        #plt.legend(handles=[p0,p1])
        #plt.show()


        #p0, = plt.plot(curvature, label='curvature')
        #p1, = plt.plot(v1,label='1st pass')
        #p2, = plt.plot(v2,label='2nd pass')
        #p3, = plt.plot(v3,label='3rd pass')
        #plt.legend(handles=[p0,p1,p2,p3])
        #plt.show()

        # calculate theoretical lap time
        t_total = 0
        for i in range(n_steps):
            (x_i, y_i) = splev(xx[i%n_steps], self.raceline, der=0)
            (x_i_1, y_i_1) = splev(xx[(i+1)%n_steps], self.raceline, der=0)
            # distance between two steps
            ds = dist((x_i, y_i),(x_i_1, y_i_1))
            t_total += ds/v3[i%n_steps]
        print("top speed = %.2fm/s"%max(v3))
        print("total time = %.2fs"%t_total)

        # calculate acceleration vector
        tt = np.linspace(0,t_total,n_steps)
        dt = t_total/n_steps
        # reference u as we go around the track
        # this is for splev, float
        u_now = 0.0
        # reference index
        # this is for indexing xx,v3 array, int
        i_now = 0
        # map from u space to i space
        u2i = lambda x:int(float(x)/(len(pts)-1)*(n_steps))%n_steps
        # get direct distance from two u
        distuu = lambda u1,u2: dist(splev(u1, self.raceline, der=0),splev(u2, self.raceline, der=0))

        xx = np.linspace(0,len(pts)-1,n_steps+1)
        vel_now = v3[0] * np.array(splev(xx[0], self.raceline, der=1))

        vel_vec = []
        
        # traverse through one lap in equal distance time step, this is different from the traverse in equial distance
        for j in range(n_steps):
            # tangential direction
            tan_dir = splev(u_now, self.raceline, der=1)
            tan_dir = np.array(tan_dir/np.linalg.norm(tan_dir))
            vel_now = self.v1(u_now%len(self.ctrl_pts)) * tan_dir
            vel_vec.append(vel_now)

            # get u and i corresponding to next time step
            func = lambda x:distuu(u_now,x)-self.v1(u_now%len(self.ctrl_pts))*dt
            bound_low = u_now
            #bound_high = u_now+len(pts)/self.n_steps*self.max_v/self.min_v
            bound_high = u_now+len(pts)/9
            assert(func(bound_low)*func(bound_high)<0)
            u_now = brentq(func,bound_low,bound_high)

        vel_vec = np.array(vel_vec)

        lat_acc_vec = []
        dtheta_vec = []
        theta_vec = []
        mean_v_vec = []
        for j in range(n_steps-1):
            #mean_v = np.linalg.norm(vel_vec[j]) + np.linalg.norm(vel_vec[j+1])
            #mean_v /= 2
            theta = np.arctan2(vel_vec[j,1],vel_vec[j,0])
            theta_vec.append(theta)

            dtheta = np.arctan2(vel_vec[j+1,1],vel_vec[j+1,0]) - theta
            dtheta = (dtheta+np.pi)%(2*np.pi)-np.pi
            dtheta_vec.append(dtheta)

            mean_v = np.linalg.norm(vel_vec[j])
            mean_v_vec.append(mean_v)

            lat_acc_vec.append(mean_v*dtheta/dt)

        #acc_vec = np.diff(vel_vec,axis=0)/dt

        #plt.plot(acc_vec[:,0],acc_vec[:,1],'*')

        p0, = plt.plot(theta_vec,label='theta')
        #p1, = plt.plot(np.diff(theta_vec)/dt,label='dtheta_diff')
        #p1, = plt.plot(mean_v_vec/np.average(mean_v_vec),label='v')
        p2, = plt.plot(dtheta_vec/np.average(dtheta_vec),label='dtheta')
        p3, = plt.plot(np.array(lat_acc_vec)/10.0,label='lateral')
        plt.legend(handles=[p0,p2,p3])

        # draw the traction circle
        #cc = np.linspace(0,2*np.pi)
        #circle = np.vstack([np.cos(cc),np.sin(cc)])*mu*g
        #plt.plot(circle[0,:],circle[1,:])
        #plt.gcf().gca().set_aspect('equal','box')
        plt.show()

        return t_total
    
    # draw the raceline from self.raceline
    def drawRaceline(self,lineColor=(0,0,255), img=None):

        rows = self.gridsize[0]
        cols = self.gridsize[1]
        res = self.resolution

        # this gives smoother result, but difficult to relate u to actual grid
        #u_new = np.linspace(self.u.min(),self.u.max(),1000)

        # the range of u is len(self.ctrl_pts) + 1, since we copied one to the end
        # x_new and y_new are in non-dimensional grid unit
        u_new = np.linspace(0,len(self.ctrl_pts),1000)
        x_new, y_new = splev(u_new, self.raceline, der=0)
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
        pts = pts.astype(np.int)
        # render different color based on speed
        # slow - red, fast - green (BGR)
        v2c = lambda x: int((x-self.min_v)/(self.max_v-self.min_v)*255)
        getColor = lambda v:(0,v2c(v),255-v2c(v))
        for i in range(len(u_new)-1):
            img = cv2.line(img, tuple(pts[i]),tuple(pts[i+1]), color=getColor(self.targetVfromU(u_new[i]%len(self.ctrl_pts))), thickness=3) 

        # solid color
        #img = cv2.polylines(img, [pts], isClosed=True, color=lineColor, thickness=3) 
        for point in self.ctrl_pts:
            x = point[0]
            y = point[1]
            x /= self.scale
            x *= self.resolution
            y /= self.scale
            y *= self.resolution
            y = self.resolution*rows - y
            
            img = cv2.circle(img, (int(x),int(y)), 5, (0,0,255),-1)

        return img

    # draw ONE arrow, unit: meter, coord sys: dimensioned
    # source: source of arrow, in meter
    # orientation, radians from x axis, ccw positive
    # length: in pixels, though this is only qualitative
    def drawArrow(self,source, orientation, length, color=(0,0,0),thickness=2, img=None, show=False):

        if (length>1):
            length = int(length)
        else:
            pass
            '''
            if show:
                plt.imshow(img)
                plt.show()
            '''
            return img

        rows = self.gridsize[0]
        cols = self.gridsize[1]
        res = self.resolution

        src = self.m2canvas(source)
        if (src is None):
            print("drawArrow err -- point outside canvas")
            return img
        #test_pnt = self.m2canvas(test_pnt)

        if img is None:
            img = np.zeros([res*rows,res*cols,3],dtype='uint8')

    
        # y-axis positive direction in real world and cv plotting is reversed
        dest = (int(src[0] + cos(orientation)*length),int(src[1] - sin(orientation)*length))

        #img = cv2.circle(img,test_pnt , 3, color,-1)

        img = cv2.circle(img, src, 3, (0,0,0),-1)
        img = cv2.line(img, src, dest, color, thickness) 
            

        '''
        if show:
            plt.imshow(img)
            plt.show()
        '''

        return img

    def loadRaceline(self,filename=None):
        pass

    def saveRaceline(self,filename):
        pass

    def optimizeRaceline(self):
        pass
    def setResolution(self,res):
        self.resolution = res
        return
    # given state of robot
    # find the closest point on raceline to center of FRONT axle
    # calculate the lateral offset (in meters), this will be reported as offset, which can be added directly to raceline orientation (after multiplied with an aggressiveness coefficient) to obtain desired front wheel orientation
    # calculate the local derivative
    # coord should be referenced from the origin (bottom left(edited)) of the track, in meters
    # negative offset means coord is to the right of the raceline, viewing from raceline init direction
    # wheelbase is needed to calculate the local trajectory closes to the front axle instead of the old axle
    def localTrajectory(self,state,wheelbase=90e-3):
        # figure out which grid the coord is in
        coord = np.array([state[0],state[1]])
        heading = state[2]
        # find the coordinate of center of front axle
        coord[0] += wheelbase*cos(heading)
        coord[1] += wheelbase*sin(heading)
        heading = state[2]
        # grid coordinate, (col, row), col starts from left and row starts from bottom, both indexed from 0
        # coord should be given in meters
        nondim= np.array((coord/self.scale)//1,dtype=np.int)

        # the seq here starts from origin
        seq = -1
        # figure out which u this grid corresponds to 
        for i in range(len(self.grid_sequence)):
            if nondim[0]==self.grid_sequence[i][0] and nondim[1]==self.grid_sequence[i][1]:
                seq = i
                break

        if seq == -1:
            print("error, coord not on track")
            return None

        # the grid that contains the coord
        #print("in grid : " + str(self.grid_sequence[seq]))

        # find the closest point to the coord
        # because we wrapped the end point to the beginning of sample point, we need to add this offset
        # Now seq would correspond to u in raceline, i.e. allow us to locate the raceline at that section
        seq += self.origin_seq_no
        seq %= self.track_length

        # this gives a close, usually preceding raceline point, this does not give the closest ctrl point
        # due to smoothing factor
        #print("neighbourhood raceline pt " + str(splev(seq,self.raceline)))

        # distance squared, not need to find distance here
        dist_2 = lambda a,b: (a[0]-b[0])**2+(a[1]-b[1])**2
        fun = lambda u: dist_2(splev(u%len(self.ctrl_pts),self.raceline),coord)
        # determine which end is the coord closer to, since seq points to the previous control point,
        # not necessarily the closest one
        if fun(seq+1) < fun(seq):
            seq += 1

        # Goal: find the point on raceline closest to coord
        # i.e. find x that minimizes fun(x)
        # we know x will be close to seq

        # easy method
        #brute force, This takes 77% of runtime. 
        #lt.s('minimize_scalar')
        #res = minimize_scalar(fun,bounds=[seq-0.6,seq+0.6],method='Bounded')
        #lt.e('minimize_scalar')

        # improved method: from observation, fun(x) is quadratic in proximity of seq
        # we assume it to be ax^3 + bx^2 + cx + d and formulate this minimization as a linalg problem
        # sample some points to build the trinomial simulation
        iv = np.array([-0.6,-0.3,0,0.3,0.6])+seq
        # formulate linear problem
        A = np.vstack([iv**3, iv**2,iv,[1,1,1,1,1]]).T
        #B = np.mat([fun(x0), fun(x1), fun(x2)]).T
        B = fun(iv).T
        #abc = np.linalg.solve(A,B)
        abc = np.linalg.lstsq(A,B)[0]
        a = abc[0]
        b = abc[1]
        c = abc[2]
        d = abc[3]
        fun = lambda x : a*x*x*x + b*x*x + c*x + d
        fit = minimize(fun, x0=seq, method='L-BFGS-B', bounds=((seq-0.6,seq+0.6),))
        min_fun_x = fit.x[0]

        min_fun_val = fit.fun[0]
        # find min val
        #x = min_fun_x = (-b+(b*b-3*a*c)**0.5)/(3*a)
        #if (seq-0.6<x<seq+0.6):
        #    min_fun_val = a*x*x*x + b*x*x + c*x + d
        #else:
        #    # XXX this is a bit sketchy, maybe none of them is right
        #    x = (-b+(b*b-3*a*c)**0.5)/(3*a)
        #    min_fun_val = a*x*x*x + b*x*x + c*x + d

        '''
        xx = np.linspace(seq-0.6,seq+0.6)
        plt.plot(xx,fun(xx),'b--')
        plt.plot(iv,fun(iv),'bo')
        plt.plot(xx,a*xx**3+b*xx**2+c*xx+d,'r-')
        plt.plot(iv,a*iv**3+b*iv**2+c*iv+d,'ro')
        plt.show()
        '''

        #lt.track('x err', abs(min_fun_x-res.x))
        #lt.track('fun err',abs(min_fun_val-res.fun))
        #print('x err', abs(min_fun_x-res.x))
        #print('fun err',abs(min_fun_val-res.fun))

        raceline_point = splev(min_fun_x%len(self.ctrl_pts),self.raceline)
        #raceline_point = splev(res.x,self.raceline)

        der = splev(min_fun_x%len(self.ctrl_pts),self.raceline,der=1)
        #der = splev(res.x,self.raceline,der=1)

        if (False):
            print("Seek local trajectory")
            print("u = "+str(min_fun_x))
            print("dist = "+str(min_fun_val**0.5))
            print("closest point on track: "+str(raceline_point))
            print("closest point orientation: "+str(degrees(atan2(der[1],der[0]))))

        # calculate whether offset is ccw or cw
        # achieved by finding cross product of vec(raceline_orientation) and vec(ctrl_pnt->test_pnt)
        # then find sin(theta)
        # negative offset means car is to the right of the trajectory
        vec_raceline = (der[0],der[1])
        vec_offset = coord - raceline_point
        cross_theta = np.cross(vec_raceline,vec_offset)


        vec_curvature = splev(min_fun_x%len(self.ctrl_pts),self.raceline,der=2)
        norm_curvature = np.linalg.norm(vec_curvature)
        # gives right sign for omega, this is indep of track direction since it's calculated based off vehicle orientation
        cross_curvature = np.cross((cos(heading),sin(heading)),vec_curvature)

        # return target velocity
        request_velocity = self.targetVfromU(min_fun_x%len(self.ctrl_pts))

        # reference point on raceline,lateral offset, tangent line orientation, curvature(signed), v_target(not implemented)
        return (raceline_point,copysign(abs(min_fun_val)**0.5,cross_theta),atan2(der[1],der[0]),copysign(norm_curvature,cross_curvature),request_velocity)

# conver a world coordinate in meters to canvas coordinate
    def m2canvas(self,coord):

        rows = self.gridsize[0]
        cols = self.gridsize[1]
        res = self.resolution

        x_new, y_new = coord[0], coord[1]
        # x_new and y_new are converted to non-dimensional grid unit
        x_new /= self.scale
        y_new /= self.scale
        if (x_new>cols or y_new>rows):
            return None

        # convert to visualization coordinate
        x_new *= self.resolution
        x_new = int(x_new)
        y_new *= self.resolution
        # y-axis positive direction in real world and cv plotting is reversed
        y_new = int(self.resolution*rows - y_new)
        return (x_new, y_new)

# draw the vehicle (one dot with two lines) onto a canvas
# coord: location of the dor, in meter (x,y)
# heading: heading of the vehicle, radians from x axis, ccw positive
#  steering : steering of the vehicle, left positive, in radians, w/ respect to vehicle heading
# NOTE: this function modifies img, if you want to recycle base img, send img.copy()
    #def drawCar(self, coord, heading,steering, img):
    def drawCar(self, img, state, steering):
        # check if vehicle is outside canvas
        x,y,heading, vf_lf, vs_lf, omega_lf = state
        coord = (x,y)
        src = self.m2canvas(coord)
        if src is None:
            print("Can't draw car -- outside track")
            return img
        # draw vehicle, orientation as black arrow
        img =  self.drawArrow(coord,heading,length=30,color=(0,0,0),thickness=5,img=img)

        # draw steering angle, orientation as red arrow
        img = self.drawArrow(coord,heading+steering,length=20,color=(0,0,255),thickness=4,img=img)

        return img

# draw traction circle, a circle representing 1g (or as specified), and a red dot representing current acceleration in vehicle frame
    def drawAcc(acc,img):
        pass

    # update car state with bicycle model, no slip
    # dt: time, in sec
    # state: (x,y,theta), np array
    # x,y: coordinate of car origin(center of rear axle)
    # theta, car heading, in rad, ref from x axis
    # beta: steering angle, left positive, in rad
    # return new state (x,y,theta)
    def updateCar(self,dt,state,throttle,beta,v_override=None):
        # wheelbase, in meter
        # heading of pi/2, i.e. vehile central axis aligned with y axis,
        # means theta = 0 (the x axis of car and world frame is aligned)
        coord = state['coord']
        heading = state['heading']
        vf = state['vf']
        vs = state['vs']
        omega = state['omega']

        theta = state['heading'] - pi/2
        L = 90e-3
        # NOTE if side slip is ever modeled, update ds
        ds = vf*dt
        dtheta = ds*tan(beta)/L

        #new_v = max(vf+(throttle-0.2)*4*dt,0)
        # NOTE add a time constant
        #new_v = max((throttle-0.16779736)/0.05022026,0)
        new_v = v_override
        dvf = new_v - vf
        dvs = 0
        # specific to vehicle frame (x to right of rear axle, y to forward)
        if (beta==0):
            dx = 0
            dy = ds
        else:
            dx = - L/tan(beta)*(1-cos(dtheta))
            dy =  abs(L/tan(beta)*sin(dtheta))

        #print(dx,dy)
        # specific to world frame
        dX = dx*cos(theta)-dy*sin(theta)
        dY = dx*sin(theta)+dy*cos(theta)

        acc_x = vf+dvf - vf*cos(dtheta) - vs*sin(dtheta)
        acc_y = vs+dvs - vs*cos(dtheta) - vf*sin(dtheta)

        state['coord'] = (state['coord'][0]+dX,state['coord'][1]+dY)
        state['heading'] += dtheta
        state['vf'] = vf + dvf
        state['vs'] = vs + dvs
        state['omega'] = dtheta/dt
        # in new vehicle frame
        state['acc'] = (acc_x,acc_y)
        return state
    


    
if __name__ == "__main__":

    # initialize track and raceline, multiple tracks are defined here, you may choose any one

    # full RCP track
    # row, col
    fulltrack = RCPtrack()
    track_size = (6,4)
    fulltrack.initTrack('uuurrullurrrdddddluulddl',track_size, scale=0.565)
    # add manual offset for each control points
    adjustment = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    adjustment[0] = -0.2
    adjustment[1] = -0.2
    #bottom right turn
    adjustment[2] = -0.2
    adjustment[3] = 0.5
    adjustment[4] = -0.2

    #bottom middle turn
    adjustment[6] = -0.2

    #bottom left turn
    adjustment[9] = -0.2

    # left L turn
    adjustment[12] = 0.5
    adjustment[13] = 0.5

    adjustment[15] = -0.5
    adjustment[16] = 0.5
    adjustment[18] = 0.5

    adjustment[21] = 0.35
    adjustment[22] = 0.35

    # start coord, direction, sequence number of origin
    # pick a grid as the starting grid, this doesn't matter much, however a starting grid in the middle of a long straight helps
    # to find sequence number of origin, start from the start coord(seq no = 0), and follow the track, each time you encounter a new grid it's seq no is 1+previous seq no. If origin is one step away in the forward direction from start coord, it has seq no = 1
    #fulltrack.initRaceline((3,3),'d',10,offset=adjustment)
    #fulltrack.initRaceline((3,3),'d',10)


    # another complex track
    #alter = RCPtrack()
    #alter.initTrack('ruurddruuuuulddllddd',(6,4),scale=1.0)
    #alter.initRaceline((3,3),'u')

    # simple track, one loop
    simple = RCPtrack()
    simple.initTrack('uurrddll',(3,3),scale=0.565)
    #simple.initRaceline((0,0),'l',0)

    # current track setup in mk103
    mk103 = RCPtrack()
    mk103.initTrack('uuruurddddll',(5,3),scale=0.565)

    # heuristic manually generated adjustment
    manual_adj = [0,0,0,0,0,0,0,0,0,0,0,0]
    manual_adj[4] = -0.5
    manual_adj[8] = -0.5
    manual_adj[9] = 0
    manual_adj[10] = -0.5
    manual_adj = np.array(manual_adj)
    # optimized with SLSQP, 2.62s
    slsqp_adj = np.array([ 3.76377694e-01,  5.00000000e-01,  4.98625816e-01,  5.00000000e-01,
                5.00000000e-01,  4.56985291e-01,  3.91826908e-03, -6.50918621e-18,
                        5.00000000e-01,  5.00000000e-01,  5.00000000e-01,  3.16694602e-01])

    mk103.initRaceline((2,2),'d',4,offset=manual_adj)
    exit(0)
    img_track = mk103.drawTrack()
    img_track = mk103.drawRaceline(img=img_track)
    plt.imshow(cv2.cvtColor(img_track,cv2.COLOR_BGR2RGB))
    plt.show()

    # select a track
    s = mk103
    # visualize raceline
    img_track = s.drawTrack()
    img_track = s.drawRaceline(img=img_track)

    porsche_setting = {'wheelbase':90e-3,
                     'max_steer_angle_left':radians(27.1),
                     'max_steer_pwm_left':1150,
                     'max_steer_angle_right':radians(27.1),
                     'max_steer_pwm_right':1850,
                     'serial_port' : None,
                     'max_throttle' : 0.5}
    # porsche 911
    car = Car(porsche_setting)

    # given a starting simulation location, find car control and visualize it
    # for RCP track
    coord = (3.6*0.565,3.5*0.565)
    # for simple track
    # coord = (2.5*0.565,1.5*0.565)

    # for mk103 track
    coord = (0.5*0.565,1.7*0.565)
    heading = pi/2
    # be careful here
    reverse = False
    throttle,steering,valid,debug_dict = car.ctrlCar([coord[0],coord[1],heading,0,0,0],s)
    # should be x,y,heading,vf,vs,omega, i didn't implement the last two
    #s.state = np.array([coord[0],coord[1],heading,0,0,0])
    sim_states = {'coord':coord,'heading':heading,'vf':throttle,'vs':0,'omega':0}
    #print(throttle,steering,valid)

    #img_track_car = s.drawCar(coord,heading,steering,img_track.copy())
    gifimages = []

    state = np.array([sim_states['coord'][0],sim_states['coord'][1],sim_states['heading'],0,0,sim_states['omega']])
    img_track_car = s.drawCar(img_track.copy(),state,steering)
    cv2.imshow('car',img_track_car)
    # prepare save gif
    saveGif = False
    if saveGif:
        gifimages.append(Image.fromarray(cv2.cvtColor(img_track_car,cv2.COLOR_BGR2RGB)))

    max_acc = 0
    sim_dt = 0.01
    sim_log_vec['omega'] = []
    sim_log_vec['vf'] = []
    sim_log_vec['v_target'] = []
    v_override = 0
    for i in range(620):
        # update car
        sim_states = s.updateCar(sim_dt,sim_states,throttle,steering,v_override=v_override)
        sim_log_vec['omega'].append(sim_states['omega'])

        state = np.array([sim_states['coord'][0],sim_states['coord'][1],sim_states['heading'],sim_states['vf'],0,sim_states['omega']])
        throttle,steering,valid,debug_dict = car.ctrlCar(state,s,reverse=reverse)

        if (len(sim_log_vec['v_target'])>0):
            v_override = sim_log_vec['v_target'][-1]

        sim_log_vec['vf'].append(debug_dict['vf'])
        sim_log_vec['v_target'].append(debug_dict['v_target'])
        img_track_car = s.drawCar(img_track.copy(),state,steering)
        #img_track_car = s.drawAcc(sim_state['acc'],img_track_car)
        #print(sim_states['acc'])
        acc = sim_states['acc']
        acc_mag = (acc[0]**2+acc[1]**2)**0.5

        cv2.imshow('car',img_track_car)
        if saveGif:
            gifimages.append(Image.fromarray(cv2.cvtColor(img_track_car,cv2.COLOR_BGR2RGB)))
        k = cv2.waitKey(int(sim_dt/0.001)) & 0xFF
        if k == ord('q'):
            break

    cv2.destroyAllWindows()
    if saveGif:
        gifimages[0].save(fp="./mk103new.gif",format='GIF',append_images=gifimages,save_all=True,duration = 50,loop=0)

    #p0, = plt.plot(sim_log_vec['vf'],label='vf')
    #p1, = plt.plot(sim_log_vec['v_target'],label='v_target')
    #plt.show()

