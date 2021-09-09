from Track import Track
import numpy as np
import cv2
from math import sin,cos

class EmptyTrack(Track):
    def __init__(self):
        self.resolution = 200
        self.gridsize = [1,1]
        self.scale = 1.0

        # draw empty track
    def drawTrack(self):
        gs = self.resolution
        img = 255*np.ones([gs,gs,3],dtype='uint8')
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
    def drawCar(self, img, state, steering):
        # check if vehicle is outside canvas
        # FIXME
        #x,y, v, heading, omega = state
        x,y,heading, vf_lf, vs_lf, omega_lf = state

        coord = (x,y)
        src = self.m2canvas(coord)
        if src is None:
            #print("Can't draw car -- outside track")
            return img
        # draw vehicle, orientation as black arrow
        img =  self.drawArrow(coord,heading,length=30,color=(0,0,0),thickness=5,img=img)

        # draw steering angle, orientation as red arrow
        img = self.drawArrow(coord,heading+steering,length=20,color=(0,0,255),thickness=4,img=img)

        return img

    # draw a point on canvas at coord
    def drawPoint(self, img, coord, color = (0,0,0)):
        src = self.m2canvas(coord)
        if src is None:
            #print("Can't draw point -- outside track")
            return img
        img = cv2.circle(img, src, 3, color,-1)

        return img

    # draw a circle on canvas at coord
    def drawCircle(self, img, coord, radius_m, color = (0,0,0)):
        src = self.m2canvas(coord)
        if src is None:
            #print("Can't draw point -- outside track")
            return img
        radius_pix = int(radius_m / self.scale * self.resolution)
        img = cv2.circle(img, src, radius_pix, color,-1)

        return img
# conver a world coordinate in meters to canvas coordinate
    def m2canvas(self,coord):

        rows = self.gridsize[0]
        cols = self.gridsize[1]
        res = self.resolution

        x_new, y_new = coord[0], coord[1]
        # x_new and y_new are converted to non-dimensional grid unit
        x_new /= self.scale
        y_new /= self.scale
        if (x_new>cols or y_new>rows or x_new<0 or y_new<0):
            return None

        # convert to visualization coordinate
        x_new *= self.resolution
        x_new = int(x_new)
        y_new *= self.resolution
        # y-axis positive direction in real world and cv plotting is reversed
        y_new = int(self.resolution*rows - y_new)
        return (x_new, y_new)
