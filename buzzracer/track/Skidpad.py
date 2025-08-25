# a simulated skidpad
from common import *
import cv2
import numpy as np
from math import cos, sin, pi, atan2, radians, degrees, tan
from scipy.interpolate import splprep, splev, CubicSpline, interp1d
import matplotlib.pyplot as plt

from track.Track import Track
# from track.car import Car


class Skidpad(Track):
    def __init__(self, main, config):
        # super(Skidpad,self).__init__()
        Track.__init__(self, main, config)

        # default parameters, to be override
        self.radius = 2.0
        self.width = 0.5
        self.ccw = True
        self.resolution = 100
        # reference velocity, obsolete
        self.velocity = 1.0

        ConfigObject.__init__(self, config)

        # setup
        theta_vec = np.linspace(0, 2*np.pi, 1000)
        ss = theta_vec*self.radius
        rr = [np.cos(theta_vec)*self.radius, np.sin(theta_vec)*self.radius]
        tck, u = splprep(rr, u=ss, s=0, per=1)
        self.raceline_s = tck

        self.raceline_len_m = 2*np.pi*self.radius
        self.start_pos = (0, self.radius)
        self.start_dir = np.pi/2
        return

    def draw_raceline(self, img):
        return img

    # state: x,y,theta,vf,vs,omega
    # x,y referenced from skidpad frame
    def local_trajectory(self, state, ccw=True):
        x = state[0]
        y = state[1]
        heading = state[2]
        vf = state[3]
        vs = state[4]
        omega = state[5]

        # find the coordinate of center of front axle
        # wheelbase = 98e-3
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
        phase = atan2(y, x)
        raceline_point = (self.radius*cos(phase), self.radius*sin(phase))

        # line orientation
        if ccw:
            raceline_orientation = phase+pi/2
            signed_curvature = 1.0/self.radius
        else:
            raceline_orientation = phase-pi/2
            signed_curvature = -1.0/self.radius

        # reference point on raceline,lateral offset, tangent line orientation, curvature(signed)
        # print(phase,offset)
        return (raceline_point, offset, raceline_orientation, signed_curvature, self.velocity)

    # prepare a picture of the track
    def draw_track(self):
        # resolution : pixels per meter
        res = self.resolution
        canvas = 255*np.ones([int(res*self.radius*3),
                             int(res*self.radius*3), 3], dtype='uint8')
        self.canvas_size = canvas.shape
        canvas = cv2.circle(canvas, self.m2canvas((0, 0)),
                            int(self.radius*res), (255, 0, 0), 1)
        canvas = cv2.circle(canvas, self.m2canvas((0, 0)), int(
            (self.radius-self.width/2)*res), (255, 0, 0), 3)
        canvas = cv2.circle(canvas, self.m2canvas((0, 0)), int(
            (self.radius+self.width/2)*res), (255, 0, 0), 3)
        return canvas

# conver a world coordinate in meters to canvas coordinate
    def m2canvas(self, coord):
        x_new = int(coord[0]*self.resolution + self.canvas_size[0]/2)
        y_new = self.canvas_size[1] - \
            int(coord[1]*self.resolution + self.canvas_size[1]/2)
        if (y_new < 0 or y_new >= self.canvas_size[1]) or (x_new < 0 or x_new >= self.canvas_size[0]):
            return None
        else:
            return (x_new, y_new)

    def precise_track_boundary(self, coord, heading):
        r = (coord[0]**2 + coord[1]**2)**0.5
        phase = np.arctan2(coord[1], coord[0])
        rel_heading = heading - phase
        if (rel_heading > 0 and rel_heading < np.pi):
            # ccw
            left = r - (self.radius-self.width/2)
            right = (self.radius+self.width/2) - r
        else:
            # cw
            right = r - (self.radius-self.width/2)
            left = (self.radius+self.width/2) - r
        return (left, right)


if __name__ == '__main__':
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
    img_track = sp.draw_track()
    img_track_car = sp.draw_car(img_track.copy(),state,radians(20))
    img_track_car = cv2.cvtColor(img_track_car, cv2.COLOR_BGR2RGB)
    plt.imshow(img_track_car)
    plt.show()

    cv2.imshow('car',img_track_car)

    for i in range(200):
        throttle, steering, valid,debug = car.ctrl_car(state,sp)
        state = car.update_car(state,throttle,steering,sim_dt)
        img_track_car = sp.draw_car(img_track.copy(),state,steering)
        print(state[3])
        cv2.imshow('car',img_track_car)
        k = cv2.waitKey(50) & 0xFF
        if k == ord('q'):
            break

    cv2.destroyAllWindows()
    pass

'''
