from ctrlMppiWrapper import ctrlMppiWrapper
from RCPTrack import RCPtrack
from Track import Track
import matplotlib.pyplot as plt
from math import radians,degrees
from ctrlMppiWrapper import ctrlMppiWrapper
import matplotlib.pyplot as plt
import numpy as np


if __name__=="__main__":
    track = RCPtrack()
    #track.startPos = (0.6*3.5,0.6*1.75)
    #track.startDir = radians(90)
    track.load()

    porsche_setting = {'wheelbase':90e-3,
                     'max_steer_angle_left':radians(27.1),
                     'max_steer_pwm_left':1150,
                     'max_steer_angle_right':radians(27.1),
                     'max_steer_pwm_right':1850,
                     'serial_port' : '/dev/ttyUSB0',
                     'max_throttle' : 0.55}
    car_setting = porsche_setting
    car_setting['serial_port'] = None
    car = ctrlMppiWrapper(car_setting,0.01)
    car.discretized_raceline_len = 1024
    car.track = track
    car.prepareDiscretizedRaceline()
    background = car.createBoundary(show=True)

    # test random points
    def isOutsideTrack(point,heading,img):

        # find closest index
        car.discretized_raceline = car.discretized_raceline.flatten()
        n = car.discretized_raceline_len
        xx =  car.discretized_raceline[0::6]
        yy =  car.discretized_raceline[1::6]
        dist_sqr = (xx-point[0])**2 + (yy-point[1])**2
        closest_index = np.argmin(dist_sqr)

        # determine offset, left or right
        heading = car.discretized_raceline[closest_index*6+2]
        #print(degrees(heading))
        raceline_to_point_angle = np.arctan2(point[1]-yy[closest_index], point[0]-xx[closest_index])
        angle_diff = (raceline_to_point_angle - heading + np.pi) % (2*np.pi) - np.pi

        offset_from_raceline = ((point[1]-yy[closest_index])**2 + (point[0]-xx[closest_index])**2)**0.5

        left = car.discretized_raceline[closest_index*6 + 4]
        right = car.discretized_raceline[closest_index*6 + 5]

        if (angle_diff > 0):
            # left
            point_is_outside = left < offset_from_raceline
            print("left "+str(offset_from_raceline))
        else:
            point_is_outside = right < offset_from_raceline
            print("right "+str(offset_from_raceline))

        if point_is_outside:
            my_color = (0,100,100)
        else:
            my_color = (0,255,0)

        img = track.drawPoint(img,point,color = my_color)

        # draw ref point on track
        img = track.drawPoint(img,(xx[closest_index],yy[closest_index]),color = (0,0,0))
        return img
        

    x_base = np.linspace(0,track.scale*4).flatten()
    y_base = np.linspace(0,track.scale*6).flatten()

    xx_grid,yy_grid = np.meshgrid(x_base,y_base)
    xx_grid = xx_grid.flatten()
    yy_grid = yy_grid.flatten()

    #img = isOutsideTrack((0.1,0.1),0.0,img)
    #img = isOutsideTrack((0.3,0.3),0.0,img)
    img = background
    for x,y in zip(xx_grid,yy_grid):
        img = isOutsideTrack((x,y),0.0,img)
    plt.imshow(img)
    plt.show()

        
        




