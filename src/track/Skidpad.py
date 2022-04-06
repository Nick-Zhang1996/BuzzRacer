# a simulated skidpad
import cv2
import numpy as np
from math import cos,sin,pi,atan2,radians,degrees,tan
import matplotlib.pyplot as plt

from track.Track import Track
#from track.car import Car

class Skidpad(Track):
    def __init__(self):
        #super(Skidpad,self).__init__()
        self.resolution = 100
        return

    # initialize a skidpad, centered at origin
    def initSkidpad(self,radius, velocity, ccw=True):
        self.radius = radius
        self.velocity = velocity
        self.ccw = ccw
        return

    #state: x,y,theta,vf,vs,omega
    # x,y referenced from skidpad frame
    def localTrajectory(self,state,ccw=True):
        x = state[0]
        y = state[1]
        heading = state[2]
        vf = state[3]
        vs = state[4]
        omega = state[5]

        # find the coordinate of center of front axle
        #wheelbase = 98e-3
        wheelbase = 108e-3
        x += wheelbase*cos(heading)
        y += wheelbase*sin(heading)

        # find offset
        # positive offset means car is to the left of the trajectory(need to turn right)
        r = (x**2+y**2)**0.5
        offset = r-self.radius
        if ccw:
            offset = - offset

        # find closest point on track
        phase = atan2(y,x)
        raceline_point = (self.radius*cos(phase),self.radius*sin(phase))

        # line orientation
        if ccw:
            raceline_orientation = phase+pi/2
            signed_curvature = 1.0/self.radius
        else:
            raceline_orientation = phase-pi/2
            signed_curvature = -1.0/self.radius

        # reference point on raceline,lateral offset, tangent line orientation, curvature(signed)
        #print(phase,offset)
        return (raceline_point,offset,raceline_orientation,signed_curvature,self.velocity)

    # prepare a picture of the track
    def drawTrack(self):
        # resolution : pixels per meter
        res = self.resolution
        canvas = 255*np.ones([int(res*self.radius*4),int(res*self.radius*4),3],dtype='uint8')
        self.canvas_size = canvas.shape
        canvas = cv2.circle(canvas,self.m2canvas((0,0)),int(self.radius*res),(255,0,0),5)
        return canvas

# draw car on a track image prepared by drawTrack()
# draw the vehicle (one dot with two lines) onto a canvas
# coord: location of the dor, in meter (x,y)
# heading: heading of the vehicle, radians from x axis, ccw positive
#  steering : steering of the vehicle, left positive, in radians, w/ respect to vehicle heading
# NOTE: this function modifies img, if you want to recycle base img, sent img.copy()
    def drawCar(self,img, state, steering):
        x = state[0]
        y = state[1]
        heading = state[2]
        # check if vehicle is outside canvas
        coord = (x,y)
        src = self.m2canvas(coord)
        if src is None:
            print("Can't draw car -- outside track")
            return img
        # draw vehicle, orientation as black arrow
        img =  self.drawArrow(coord,heading,length=30,color=(0,0,0),thickness=5,img=img)
        '''img = cv2.imread('src/image.png',-1)'''

        # draw steering angle, orientation as red arrow
        img = self.drawArrow(coord,heading+steering,length=20,color=(0,0,255),thickness=4,img=img)
        # image rotation according to heading and steering angles
        '''
        height, width = img.shape[:2]
        center = (width/2, height/2)
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=heading+steering, scale=1)
        img = cv2.warpAffine(src=img, M=rotate_matrix, dsize=(width, height)) 
        '''      

        return img

    # draw ONE arrow, unit: meter, coord sys: dimensioned
    # coord: coordinate for source of arrow, in meter
    # orientation, radians from x axis, ccw positive
    # length: in pixels, though this is only qualitative
    def drawArrow(self,coord, orientation, length, color=(0,0,0),thickness=2, img=None, show=False):

        if (length>1):
            length = int(length)
        else:
            return img

        res = self.resolution

        src = self.m2canvas(coord)
        if (src is None):
            print("drawArrow err -- point outside canvas")
            return img

        # y-axis positive direction in real world and cv plotting is reversed
        dest = (int(src[0] + cos(orientation)*length),int(src[1] - sin(orientation)*length))

        img = cv2.circle(img, src, 3, (0,0,0),-1)
        img = cv2.line(img, src, dest, color, thickness) 
            
        return img

# conver a world coordinate in meters to canvas coordinate
    def m2canvas(self,coord):
        x_new = int(coord[0]*self.resolution + self.canvas_size[0]/2)
        y_new = self.canvas_size[1] - int(coord[1]*self.resolution + self.canvas_size[1]/2)
        if (y_new<0 or y_new>=self.canvas_size[1]) or (x_new<0 or x_new>=self.canvas_size[0]):
            return None
        else:
            return (x_new,y_new)


if __name__ == "__main__":
    pass
'''
if __name__ == "__main__":
    sp = Skidpad()
    sp.initSkidpad(radius=1,velocity=1)
    car = Car()
    x = 0.0
    y = 1.0
    theta = pi
    vf = 0
    vs = 0
    omega = 0
    sim_dt = 0.1
    state = (x,y,theta,vf,vs,omega)

    # visualize raceline
    img_track = sp.drawTrack()
    img_track_car = sp.drawCar(img_track.copy(),state,radians(20))
    img_track_car = cv2.cvtColor(img_track_car, cv2.COLOR_BGR2RGB)
    plt.imshow(img_track_car)
    plt.show()

    cv2.imshow('car',img_track_car)

    for i in range(200):
        throttle, steering, valid,debug = car.ctrlCar(state,sp)
        state = car.updateCar(state,throttle,steering,sim_dt)
        img_track_car = sp.drawCar(img_track.copy(),state,steering)
        print(state[3])
        cv2.imshow('car',img_track_car)
        k = cv2.waitKey(50) & 0xFF
        if k == ord('q'):
            break

    cv2.destroyAllWindows()
    pass

'''
