#!/usr/bin/python

# this file contains all data structure and algorithm to :
# describe an RCP track
# describe a raceline within the track
# provide desired trajectory(raceline) given a car's location within thte track

import numpy as np
from numpy import isclose
import matplotlib.pyplot as plt
from math import atan2,radians,degrees,sin,cos,pi,tan,copysign,asin,acos
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize_scalar
from time import sleep
from timeUtil import execution_timer
import cv2

t = execution_timer(False)
lt = execution_timer(True)


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

# for coordinate transformation
# 3 coord frame:
# 1. Inertial, world frame, as in vicon
# 2. Track frame, local world frame, used in RCPtrack as world frame
# 3. Car frame, origin at rear axle center, y pointing forward, x to right
class TF:
    def __init__(self):
        pass

    def euler2q(self,roll,pitch,yaw):
        q = np.array([ cos(roll/2)*cos(pitch/2)*cos(yaw/2)+sin(roll/2)*sin(pitch/2)*sin(yaw/2), -cos(roll/2)*sin(pitch/2)*sin(yaw/2)+cos(pitch/2)*cos(yaw/2)*sin(roll/2), cos(roll/2)*cos(yaw/2)*sin(pitch/2) + sin(roll/2)*cos(pitch/2)*sin(yaw/2), cos(roll/2)*cos(pitch/2)*sin(yaw/2) - sin(roll/2)*cos(yaw/2)*sin(pitch/2)])
        return q

    def q2euler(self,q):
        R = self.q2R(q)
        roll,pitch,yaw = self.R2euler(R)
        return (roll,pitch,yaw)

    # given unit quaternion, find corresponding rotation matrix (passive)
    def q2R(self,q):
        assert(isclose(np.linalg.norm(q),1,atol=0.001))
        Rq = [[q[0]**2+q[1]**2-q[2]**2-q[3]**2, 2*q[1]*q[2]+2*q[0]*q[3], 2*q[1]*q[3]-2*q[0]*q[2]],\
           [2*q[1]*q[2]-2*q[0]*q[3],  q[0]**2-q[1]**2+q[2]**2-q[3]**2,    2*q[2]*q[3]+2*q[0]*q[1]],\
           [2*q[1]*q[3]+2*q[0]*q[2],  2*q[2]*q[3]-2*q[0]*q[1], q[0]**2-q[1]**2-q[2]**2+q[3]**2]]
        Rq = np.matrix(Rq)
        return Rq

    # given euler angles, find corresponding rotation matrix (passive)
    # roll, pitch, yaw, (in reverse sequence, yaw is applied first, then pitch applied to intermediate frame)
    # all in radians
    def euler2R(self, roll,pitch,yaw):
        '''
        R = [[ c2*c3, c2*s3, -s2],
        ...  [s1*s2*s3-c1*s3, s1*s2*s3+c1*c3, c2*s1],
        ...  [c1*s2*c3+s1*s3, c1*s2*s3-s1*c3, c2*c1]]
        '''

        R = [[ cos(pitch)*cos(yaw), cos(pitch)*sin(yaw), -sin(pitch)],\
          [sin(roll)*sin(pitch)*sin(yaw)-cos(roll)*sin(yaw), sin(roll)*sin(pitch)*sin(yaw)+cos(roll)*cos(yaw), cos(pitch)*sin(roll)],\
          [cos(roll)*sin(pitch)*cos(yaw)+sin(roll)*sin(yaw), cos(roll)*sin(pitch)*sin(yaw)-sin(roll)*cos(yaw), cos(pitch)*cos(roll)]]
        R = np.matrix(R)
        return R
    # euler angle from R, in rad, roll,pitch,yaw
    def R2euler(self,R):
        roll = atan2(R[1,2],R[2,2])
        pitch = -asin(R[0,2])
        yaw = atan2(R[0,1],R[0,0])
        return (roll,pitch,yaw)
        
    # given pose of T(track frame) in W(vicon world frame), and pose of B(car body frame) in W,
    # find pose of B in T
    # T = [q,x,y,z], (7,) np.array
    # everything in W frame unless noted, vec_B means in B basis, e.g.
    # a_R_b denotes a passive rotation matrix that transforms from b to a
    # vec_a = a_R_b * vec_b
    def reframe(self,T, B):
        # TB = OB - OT
        OB = np.matrix(B[-3:]).T
        OT = np.matrix(T[-3:]).T
        TB = OB - OT
        T_R_W = self.q2R(T[:4])
        B_R_W = self.q2R(B[:4])

        # coord of B origin in T, in T basis
        TB_T = T_R_W * TB
        # in case we want full pose, just get quaternion from the rotation matrix below
        B_R_T = B_R_W * np.linalg.inv(T_R_W)
        (roll,pitch,yaw) = self.R2euler(B_R_T)

        # x,y, heading
        return (TB_T[0,0],TB_T[1,0],yaw+pi/2)

class RCPtrack:
    def __init__(self):
        # resolution : pixels per grid side length
        self.resolution = 200

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
        color_side = (250,0,0)
        # boundary width / grid width
        deadzone = 0.09
        gs = self.resolution

        # prepare straight section (WE)
        straight = 255*np.ones([gs,gs,3],dtype='uint8')
        straight = cv2.rectangle(straight, (0,0),(gs-1,int(deadzone*gs)),color_side,-1)
        straight = cv2.rectangle(straight, (0,int((1-deadzone)*gs)),(gs-1,gs-1),color_side,-1)
        WE = straight

        # prepare straight section (SE)
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
        if show:
            plt.imshow(img)
            plt.show()

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

            '''
            if signature in turns:
                if last_signature in turns:
                    # double turn U turn or S turn (chicane)
                    # do not remove previous control point (never added)

                    new_ctrl_point = np.array(center(current_coord[0],current_coord[1])) + apex_offset*np.array(turn_offset_toward_center[signature])*self.scale

                    # to reduce the abruptness of the turn and bring raceline closer to apex, add a control point at apex
                    pre_apex_ctrl_pnt = np.array(self.ctrl_pts[-1])
                    self.ctrl_pts_w[-1] = 1
                    post_apex_ctrl_pnt = new_ctrl_point

                    mid_apex_ctrl_pnt = 0.5*(pre_apex_ctrl_pnt+post_apex_ctrl_pnt)
                    self.ctrl_pts.append(mid_apex_ctrl_pnt.tolist())
                    self.ctrl_pts_w.append(18)

                    self.ctrl_pts.append(new_ctrl_point.tolist())
                    self.ctrl_pts_w.append(1)
                else:
                    # one turn, or first turn element in a S or U turn
                    # remove previous control point
                    #del self.ctrl_pts[-1]
                    #del self.ctrl_pts_w[-1]
                    new_ctrl_point = np.array(center(current_coord[0],current_coord[1])) + apex_offset*np.array(turn_offset_toward_center[signature])*self.scale
                    self.ctrl_pts.append(new_ctrl_point.tolist())
                    self.ctrl_pts_w.append(1)

            else:
                # straights

                # exit point only
                #exit_ctrl_pt = np.array(lookup_table_dir[exit],dtype='float')/2
                #exit_ctrl_pt += current_coord
                #exit_ctrl_pt += np.array([0.5,0.5])
                #exit_ctrl_pt *= self.scale
                #self.ctrl_pts.append(exit_ctrl_pt.tolist())
                #self.ctrl_pts_w.append(1.2)

                # center point
                #exit_ctrl_pt = np.array(lookup_table_dir[exit],dtype='float')/2
                exit_ctrl_pt = np.array([0.5,0.5])
                exit_ctrl_pt += current_coord
                exit_ctrl_pt *= self.scale
                self.ctrl_pts.append(exit_ctrl_pt.tolist())
                self.ctrl_pts_w.append(1)
            '''

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
        smoothing_factor = 0.015*(m)
        tck, u = splprep(pts.T, u=np.linspace(0,len(pts)-1,len(pts)), s=smoothing_factor, per=1) 

        # this gives smoother result, but difficult to relate u to actual grid
        #tck, u = splprep(pts.T, u=None, s=0.0, per=1) 
        #self.u = u

        self.raceline = tck

        return

    # draw the raceline from self.raceline
    def drawRaceline(self,lineColor=(0,0,255), img=None, show=False):

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
        pts = pts.reshape((-1,1,2))
        pts = pts.astype(np.int)
        img = cv2.polylines(img, [pts], isClosed=True, color=lineColor, thickness=3) 
        for point in self.ctrl_pts:
            x = point[0]
            y = point[1]
            x /= self.scale
            x *= self.resolution
            y /= self.scale
            y *= self.resolution
            y = self.resolution*rows - y
            
            img = cv2.circle(img, (int(x),int(y)), 5, (255,0,0),-1)

        if show:
            plt.imshow(img)
            plt.show()

        return img

    # draw ONE arrow, unit: meter, coord sys: dimensioned
    # source: source of arrow, in meter
    # orientation, radians from x axis, ccw positive
    # length: in pixels, though this is only qualitative
    def drawArrow(self,source, orientation, length, color=(0,0,0),thickness=2, img=None, show=False):

        if (length>1):
            length = int(length)
        else:
            if show:
                plt.imshow(img)
                plt.show()
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
            

        if show:
            plt.imshow(img)
            plt.show()

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
    # given the coordinate of robot
    # find the closest point on raceline
    # calculate the offset (in meters), this will be reported as offset, which can be added directly to raceline orientation (after multiplied with an aggressiveness coefficient) to obtain desired front wheel orientation
    # calculate the local derivative
    # coord should be referenced from the origin (bottom left(edited)) of the track, in meters
    # negative offset means coord is to the right of the raceline, viewing from raceline init direction
    def localTrajectory(self,coord):
        lt.s()
        # figure out which grid the coord is in
        lt.s('which grid')
        coord = np.array(coord)
        # grid coordinate, (col, row), col starts from left and row starts from bottom, both indexed from 0
        # coord should be given in meters
        nondim= np.array((coord/self.scale)//1,dtype=np.int)
        lt.e('which grid')

        lt.s('grid2u')
        seq = -1
        # figure out which u this grid corresponds to 
        for i in range(len(self.grid_sequence)):
            if nondim[0]==self.grid_sequence[i][0] and nondim[1]==self.grid_sequence[i][1]:
                seq = i
                break

        lt.e('grid2u')

        if seq == -1:
            print("error, coord not on track")
            lt.e()
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
        lt.s('dataprep') # 27% runtime
        dist_2 = lambda a,b: (a[0]-b[0])**2+(a[1]-b[1])**2
        fun = lambda u: dist_2(splev(u,self.raceline),coord)
        # determine which end is the coord closer to, since seq points to the previous control point,
        # not necessarily the closest one
        if fun(seq+1) < fun(seq):
            seq += 1
        lt.e('dataprep')

        #search around the control point closest to coord

        #brute force,  This takes 77% of runtime. 
        '''
        lt.s('minimize_scalar')
        res = minimize_scalar(fun,bounds=[seq-0.6,seq+0.6],method='Bounded')
        #xx = np.linspace(seq-0.6,seq+0.6)
        #plt.plot(xx,fun(xx))
        #plt.show()
        lt.e('minimize_scalar')
        '''

        # improved: from observation fun(x) is quadratic in proximity of seq
        # we assume it to be ax2 + bx + c and formulate this as a linalg problem
        lt.s('quadrature sim')
        #x0 = seq-0.6
        #x1 = seq
        #x2 = seq+0.6
        #iv = np.array([x0,x1,x2])
        iv = np.array([-0.6,0,0.6])+seq
        #A = np.mat([[x0**2, x0, 1],[x1**2, x1, 1],[x2**2, x2, 1]])
        A = np.vstack([iv**2,iv,[1,1,1]]).T
        #B = np.mat([fun(x0), fun(x1), fun(x2)]).T
        B = fun(iv).T
        abc = np.linalg.solve(A,B)
        a = abc[0]
        b = abc[1]
        c = abc[2]
        min_fun_x = -b/2/a
        min_fun_val = c/a-b**2/(4*a**2)
        lt.e('quadrature sim')
        #lt.track('x err', abs(min_fun_x-res.x))
        #lt.track('fun err',abs(min_fun_val-res.fun))





        lt.s('splev val')
        raceline_point = splev(min_fun_x,self.raceline)
        lt.e('splev val')

        lt.s('splev der')
        der = splev(min_fun_x,self.raceline,der=1)
        lt.e('splev der')

        if (False):
            print("Seek local trajectory")
            print("u = "+str(min_fun_x))
            print("dist = "+str(min_fun_val**0.5))
            print("closest point on track: "+str(raceline_point))
            print("closest point orientation: "+str(degrees(atan2(der[1],der[0]))))

        # calculate whether offset is ccw or cw
        # achieved by finding cross product of vec(raceline_orientation) and vec(ctrl_pnt->test_pnt)
        # then find sin(theta)
        # negative offset means car is ot the right
        lt.s('final')
        vec_raceline = (der[0],der[1])
        vec_offset = coord - raceline_point
        cross_theta = np.cross(vec_raceline,vec_offset)
        lt.e('final')

        lt.e()
        return (raceline_point,copysign(min_fun_val**0.5,cross_theta),atan2(der[1],der[0]))

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
# NOTE: this function modifies img, if you want to recycle base img, sent img.copy()
    def drawCar(self, coord, heading,steering, img):
        # check if vehicle is outside canvas
        src = self.m2canvas(coord)
        if src is None:
            print("Can't draw car -- outside track")
            return img
        # draw vehicle, orientation as black arrow
        img =  self.drawArrow(coord,heading,length=30,color=(0,0,0),thickness=5,img=img)

        # draw steering angle, orientation as red arrow
        img = self.drawArrow(coord,heading+steering,length=20,color=(255,0,0),thickness=4,img=img)

        return img

# given world coordinate of the vehicle, provide throttle and steering output
# throttle -1.0,1.0
# reverse: true if running in opposite direction of raceline init direction
# steering as an angle in radians, UNTRIMMED, left positive
# valid: T/F, if the car can be controlled here, if this is false, then throttle will be set to 0
    def ctrlCar(self,coord,heading,reverse=False):
        global t
        t.s()

        t.s('localTrajectory()')
        retval = self.localTrajectory(coord)
        t.e('localTrajectory()')
        if retval is None:
            return (0,0,False)

        (local_ctrl_pnt,offset,orientation) = retval
        if reverse:
            offset = -offset
            orientation += pi

        # how much to compensate for per meter offset from track
        # 5 deg per cm offset XXX the maximum allowable offset here is a bit too large

        t.s('ctrl math')
        if (abs(offset) > 0.3):
            ret = (0,0,False)
        else:
            ctrl_ratio = 5.0/180*pi/0.01
            # sign convention for offset: - requires left steering(+)
            steering = orientation-heading - offset * ctrl_ratio
            steering = (steering+pi)%(2*pi) -pi
            # handle edge case, unwrap ( -355 deg turn -> +5 turn)
            if (steering>radians(24.5)):
                steering = radians(24.5)
            elif (steering<-radians(24.5)):
                steering = -radians(24.5)

            throttle = 0.24
            ret =  (throttle,steering,True)

        t.e('ctrl math')
        t.e()
        return ret

    # update car state with bicycle model, no slip
    # dt: time, in sec
    # v: velocity of rear wheel, in m/s
    # state: (x,y,theta), np array
    # x,y: coordinate of car origin(center of rear axle)
    # theta, car heading, in rad, ref from x axis
    # beta: steering angle, left positive, in rad
    # return new state (x,y,theta)
    def updateCar(self,dt,v,state,beta):
        # wheelbase, in meter
        # heading of pi/2, i.e. vehile central axis aligned with y axis,
        # means theta = 0 (the x axis of car and world frame is aligned)
        theta = state[2] - pi/2
        L = 98e-3
        dr = v*dt
        dtheta = dr*tan(beta)/L
        # specific to vehicle frame (x to right of rear axle, y to forward)
        if (beta==0):
            dx = 0
            dy = dr
        else:
            dx = - L/tan(beta)*(1-cos(dtheta))
            dy =  abs(L/tan(beta)*sin(dtheta))
        #print(dx,dy)
        # specific to world frame
        dX = dx*cos(theta)-dy*sin(theta)
        dY = dx*sin(theta)+dy*cos(theta)
        return state+np.array([dX,dY,dtheta])

def show(img):
    plt.imshow(img)
    plt.show()
    return
    
if __name__ == "__main__":
    # test tf
    '''
    tf = TF()
    c = lambda x:cos(radians(x))
    s = lambda x:sin(radians(x))
    w_ob = (2+4*s(30)+3*c(30), 6+4*c(30)-3*s(30), 0)
    w_ot = (2,6,0)
    q_b = tf.euler2q(radians(180),0,radians(30))
    q_t = tf.euler2q(radians(180),0,radians(60))
    T = np.hstack([q_t,np.array(w_ot)])
    B = np.hstack([q_b,np.array(w_ob)])
    print(tf.reframe(T,B)) # should give 4,3,radians(30)
    '''

    s = RCPtrack()
    # example: simple track
    #s.initTrack('uurrddll',(3,3),scale=1.0)
    #s.initRaceline((0,0),'l')

    # another sample track
    #s.initTrack('ruurddruuuuulddllddd',(6,4),scale=1.0)
    #s.initRaceline((3,3),'u')

    # MK111 track
    # row, col
    track_size = (6,4)
    s.initTrack('uuurrullurrrdddddluulddl',track_size, scale=0.565)
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

    # start coord, direction, sequence number of origin(which u gives the exit point for origin)
    s.initRaceline((3,3),'d',10,offset=adjustment)

    # use new track
    #s.initTrack('ruurddruuuuulddllddd',(6,4),scale=1.0)
    #s.initRaceline((3,3),'u')

    s.initTrack('uurrddll',(3,3),scale=0.565)
    s.initRaceline((0,0),'l',0)

    # visualize raceline
    img_track = s.drawTrack()
    img_track = s.drawRaceline(img=img_track)

    # given a starting location, find car control and visualize it
    coord = (3.6*0.565,3.5*0.565)
    # for simple track
    coord = (2.5*0.565,1.5*0.565)
    heading = pi/2
    throttle,steering,valid = s.ctrlCar(coord,heading)
    s.state = np.array([coord[0],coord[1],heading])
    #print(throttle,steering,valid)

    img_track_car = s.drawCar(coord,heading,steering,img_track.copy())
    showobj = plt.imshow(img_track)

    # 100 iteration steps
    for i in range(100):
        # update car
        s.state = s.updateCar(dt=0.1,v=throttle,state=s.state,beta=steering)

        throttle,steering,valid = s.ctrlCar((s.state[0],s.state[1]),s.state[2],reverse=True)
        print(i,throttle,steering,valid)
        #img_track_car = s.drawCar((s.state[0],s.state[1]),s.state[2],steering,img_track.copy())
        #showobj.set_data(img_track_car)
        #plt.draw()
        #plt.pause(0.01)

    t.summary()
    lt.summary()


'''
    # generate test point array, in meters (x,y)
    # we'll visualize where the wheel should point if vehicle is positioned in these points
    x = np.arange(0,track_size[1]*s.scale,0.05)
    y = np.arange(0,track_size[0]*s.scale,0.05)
    xx,yy = np.meshgrid(x,y)
    pnt = np.hstack([xx.reshape(-1,1),yy.reshape(-1,1)])

    correction_coeff = radians(1)*100*7
    # should probably vectorize this...
    for i in range(pnt.shape[0]):
        # test function, testpoint in meters
        testpoint = (pnt[i,0],pnt[i,1])
        # agressiveness of adjustment
        # radians per meter offset
        # i.e. 3 degree per cm offset
        (local_ctrl_pnt,offset,orientation) = s.localTrajectory(testpoint)

        # if abs(offset*correction_coeff)>45deg, we cannot correct for the deviation
        # this corresponds to 15cm offset
        if (abs(offset*correction_coeff) > radians(45)):
            # visualize on-track reference point
            img_track = s.drawArrow(testpoint, orientation+offset*correction_coeff, (50/10*(10-abs(offset)*100)), img=img_track)

    show(img_track)
'''
