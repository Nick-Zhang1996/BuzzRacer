#this file contains all data structure and algorithm to :
# describe an RCP track
# describe a raceline within the track
# provide desired trajectory(raceline) given a car's location within thte track

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize_scalar
import cv2


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
        self.resolution = 100

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


    def drawTrack(self, img=None,show=True):
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

    def drawRaceline(self,lineColor=(0,0,255), img=None, show=True):

        rows = self.gridsize[0]
        cols = self.gridsize[1]
        res = self.resolution

        # this gives smoother result, but difficult to relate u to actual grid
        #u_new = np.linspace(self.u.min(),self.u.max(),1000)

        # the range of u is len(self.ctrl_pts) + 1, since we copied one to the end
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
    # find the closest point on track
    # find the offset
    # find the local derivative
    # coord should be referenced from the origin (bottom right) of the track
    def localTrajectory(self,coord):
        # figure out which grid the coord is in
        coord = np.array(coord)
        # grid coordinate
        nondim= np.array((coord/self.scale)//1,dtype=np.int)

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
        print("in grid : " + str(self.grid_sequence[seq]))

        # find the closest point to the coord
        # because we wrapped the end point to the beginning of sample point, we need to add thie offset
        seq += self.origin_seq_no
        seq %= self.track_length

        # this gives a close, usually preceding raceline point, this does not give the closest ctrl point
        # due to smoothing factor
        print("neighbourhood raceline pt " + str(splev(seq,self.raceline)))

        dist = lambda a,b: (a[0]-b[0])**2+(a[1]-b[1])**2
        fun = lambda u: dist(splev(u,self.raceline),coord)
        # determine which end is the coord closer to, since seq is the previous control point,
        # not necessarily the closest one
        if fun(seq+1) < fun(seq):
            seq += 1

        res = minimize_scalar(fun,bounds=[seq-0.6,seq+0.6],method='Bounded')
        print("u = "+str(res))
        print("closest point : "+str(splev(res.x,self.raceline)))
        print("closest point dev: "+str(splev(res.x,self.raceline,der=1)))

        return


def show(img):
    plt.imshow(img)
    plt.show()
    return
    
if __name__ == "__main__":
    s = RCPtrack()
    # simple track
    #s.initTrack('uurrddll',(3,3),scale=1.0)
    #s.initRaceline((0,0),'l')

    # Gu's track, randomly generated
    #s.initTrack('ruurddruuuuulddllddd',(6,4),scale=1.0)
    #s.initRaceline((3,3),'u')

    # MK111 track
    s.initTrack('uuurrullurrrdddddluulddl',(6,4), scale=1.0)
    adjustment = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    adjustment[0] = -0.2
    adjustment[1] = -0.2
    #bottom right turn
    adjustment[2] = -0.2
    adjustment[3] = 0.5
    adjustment[4] = -0.2

    #bottom middle turn
    adjustment[6] = -0.5

    #bottom left turn
    adjustment[9] = 0.5

    # left L turn
    adjustment[12] = 0.5
    adjustment[13] = 0.5

    adjustment[15] = -0.5
    adjustment[18] = 0.7

    adjustment[21] = 0.35
    adjustment[22] = 0.35

    # start coord, direction, sequence number of origin(which u gives the exit point for origin)
    s.initRaceline((3,3),'d',10,offset=adjustment)
    s.localTrajectory((2.5,4.0))

    img_track = s.drawTrack(show=False)
    img_raceline = s.drawRaceline(img=img_track)

    
