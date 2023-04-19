# import cv2
import pygame
from pygame.locals import *
import math
from time import sleep,time
from common import *
from extension.Extension import Extension
from threading import Event
import pickle
import matplotlib.pyplot as plt
from math import degrees,radians
from PIL import Image
from xml.dom.minidom import parseString

from track import TrackFactory

class Visualization(Extension):
    def __init__(self,main):
        super().__init__(main)
        self.update_visualization = Event()
        self.update_freq = 100
        self.frame_dt = 1.0/self.update_freq
        # NOTE
        self.frame_dt = 0.0
        self.count = 0
        # default setting, will be overridden if defined in config
        self.car_graphics = False
        self.track = self.main.track

    def final(self):
        pygame.quit()

    def init(self,):
        self.visualization_ts = time()
        self.img_track = self.main.track.drawTrack()
        self.img_blank_track = self.img_track.copy()
        self.img_blank_track_with_obstacles = self.track.plotObstacles(self.img_track.copy())
        self.img_track = self.main.track.drawRaceline(img=self.img_track)
        
        pygame.init()

        config = parseString('<track>full</track>')
        config_track= config.getElementsByTagName('track')[0]
        self.track = TrackFactory(self,config_track)
        self.img_track = self.track.drawTrack()
        self.img_blank_track = self.img_track.copy()
        self.img_track_raceline = self.track.drawRaceline(img=self.img_track)
        self.background = pygame.surfarray.make_surface(self.img_track_raceline[:,:,::-1])
        self.background = pygame.transform.flip(self.background, False, True)
        self.background = pygame.transform.rotate(self.background, -90)
        self.screen = pygame.display.set_mode(self.background.get_size(), pygame.SCALED)
        self.screen.blit(self.background, (0,0))
        self.background = self.background.convert()

        #img = self.screen
        img = self.img_track.copy()
        
        for car in self.main.cars:
            self.drawCar(car)

        self.visualization_img = img
        
        # draw static components onto background
        self.drawControlStaticForAllCars()

        # refresh background
        self.screen.blit(self.background, (0,0))
        pygame.display.flip()

    def postInit(self,):
        self.saveBlankImg()


    # FIXME
    def saveBlankImg(self):
        img = self.img_blank_track_with_obstacles.copy()
        try:
            obstacles = self.main.cars[0].controller.obstacles
            # plot obstacles
            for obs in obstacles:
                img = self.main.track.drawCircle(img, obs, 0.1, color=(255,100,100))
        except AttributeError:
            pass
        
        with open("track_img.p",'wb') as f:
            print_info(self.prefix()+"saved raw track background")
            pickle.dump(img,f)

    # show image
    # do this last since controllers may need to alter the image
    def postUpdate(self,):
        if (self.update_visualization.is_set()):
            self.update_visualization.clear()
            self.visualization_ts = time()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.main.exit_request.set()
                elif event.type == pygame.KEYDOWN:
                    # q for quit
                    if event.key == pygame.K_q:
                        # first time q is presed, slow down
                        if not self.main.slowdown.isSet():
                            print_ok("slowing down, press q again to shutdown")
                            self.main.slowdown.set()
                            self.main.slowdown_ts = time()
                        else:
                            # second time, shut down
                            self.main.exit_request.set()
                    # p for pause
                    elif event.key == pygame.K_p:
                        self.print_info("Paused")
                        input("press Enter to continue")
                    # s for snapshot
                    elif event.key == pygame.K_s:
                        self.print_info("Requesting snapshot")
                        self.main.snapshot.takeSnapshot()

    def preUpdate(self,):
        # restrict update rate to 0.02s/frame, a rate higher than this can lead to frozen frames
        #print_info(self.prefix(), "preupdate %.1f"%(time()-self.visualization_ts))
        if (time()-self.visualization_ts > self.frame_dt):
            self.update_visualization.set()
        
        if (self.update_visualization.is_set()):
            for car in self.main.cars:
                self.drawCar(car)
            self.drawControlStaticForAllCars()
            self.drawControlForAllCars()
            self.track.plotObstacles()
            pygame.display.flip()

        self.screen.blit(self.background, (0,0))

    def final(self):
        img = self.img_track.copy()
        self.visualization_img = img
        self.update_visualization.set()
        #self.main.cars[0].controller.plotObstacles()
        #self.main.cars[0].controller.plotTrajectory()
        #img = self.visualization_img.copy()
        #filename = "./last_frame_" + self.main.algorithm + ".png"
        #cv2.imwrite(filename,img)
        #print_info(self.prefix()+"saved last frame at " + filename)

    def drawControlStaticForAllCars(self):
        offset = -10
        for car in self.main.cars:
            self.drawControlStatic(car, (-10,offset))
            offset += 60

    def drawControlStatic(self,car,coord):
        # draw car illustration
        x2 = coord[0] + 3.5
        y2 = coord[1] + 20
        
        image = pygame.image.load(car.params['rendering']).convert_alpha()
        size = image.get_size()
        scale = 40.0/size[1]/200.0*self.track.resolution/0.0461*size[0]
        scale /= 10000
        scale *= 0.65
        size = (size[0] * scale, size[1] * scale)
        image = pygame.transform.scale(image, size)
        image = pygame.transform.rotate(image, degrees(np.pi/2))
        self.background.blit(image, (x2,y2))
        
    def drawControlForAllCars(self):
        offset = -10
        for car in self.main.cars:
            self.drawControl(car, (-10,offset))
            offset += 60
        
    def drawControl(self,car,coord):
        # FIXME move static stuff to background since it doesn't change
        #x1 and y1 are the origin values -- need to be changed if origin changes
        x1 = coord[0] + 30
        y1 = coord[1]
        x,y,heading, vf_lf, vs_lf, omega_lf = car.states
        steering = car.steering
        throttle = car.throttle
        pygame.font.init()

        # Add steering bar
        pygame.draw.rect(self.screen, (0, 0, 255), pygame.Rect(x1 + 4, y1 + 25, 96, 15), 1)
        end_coordinate = int(50 - (steering * 100))
        temp_rect = pygame.Rect(x1 + 50, y1 + 25, end_coordinate - 50, 15)
        temp_rect.normalize()
        pygame.draw.rect(self.screen, (0, 255, 0), temp_rect)
        font = pygame.font.SysFont(None, 20)
        text = font.render('Steering', True, (0, 0, 0))
        self.screen.blit(text, (x1 + 104, y1 + 25))
        
        # Add Throttle bar
        pygame.draw.rect(self.screen, (0, 0, 255), pygame.Rect(x1 + 4, y1 + 45, 96, 15), 1)
        font = pygame.font.SysFont(None, 20)
        text = font.render('Throttle', True, (0, 0, 0))
        self.screen.blit(text, (x1 + 104, y1 + 45))
        throttle_end = int(50 + (72 * throttle))
        pygame.draw.rect(self.screen, (0, 255, 0), pygame.Rect(x1 + 50, y1 + 45, throttle_end - 50, 15))
        
    def drawAcceleration(self,car,coord):
        #x1 and y1 are the origin values -- need to be changed if origin changes
        x1 = coord[0]
        y1 = coord[1]
        x,y,heading, vf_lf, vs_lf, omega_lf = car.states
        steering = car.steering
        throttle = car.throttle

        # Add acceleration bar
        pygame.draw.circle(self.screen, (0, 0, 255), (x1 + 50, y1 + 80), 18, 1)
        pygame.draw.rect(self.screen, (255, 0, 0), (x1 + 104, y1 + 80, 80, 20))
        
        acc_x = ((np.square(vf_lf) - np.square(vs_lf)/(2*x)))
        acc_y = ((np.square(vf_lf) - np.square(vs_lf)/(2*y)))
        acc_x_scale = int(acc_x/3)
        acc_y_scale = int(acc_y/3)
        direction_x = 0
        direction_y = 0
        if (steering == 0):
            direction_x = (x1 + (50))
            direction_y = (y1 + (80 + (6 * acc_y_scale)))
        if (0 < steering):
            direction_x = (x1 + (50 + (6 * acc_x_scale)))
            direction_y = (y1 + (80 + (6 * acc_y_scale)))
        if(steering < 0):
            direction_x = (x1 + (50 - (6 * acc_x_scale)))
            direction_y = (y1 + (80 + (6 * acc_y_scale))) 

        pygame.draw.circle(self.screen, (0, 255, 0), (direction_x, direction_y), 3, -1)
        pygame.display.flip()
        
        

# draw the vehicle (one dot with two lines) onto a canvas
# coord: location of the dor, in meter (x,y)
# heading: heading of the vehicle, radians from x axis, ccw positive
#  steering : steering of the vehicle, left positive, in radians, w/ respect to vehicle heading
# NOTE: this function modifies img, if you want to recycle base img, send img.copy()
    def drawCar(self, car):
        #print("drawcar")
        x,y,heading, vf_lf, vs_lf, omega_lf = car.states
        throttle = car.throttle
        steering = car.steering
        coord = (x,y)
        src = self.main.track.m2canvas(coord)
        img = self.img_track.copy()
        if src is None:
            #print("Can't draw car -- outside track")
            return

        # overlay vehicle image, orientation as headed
        # significant performance impact
        self.overlayCarRendering(car)
        # if (self.car_graphics):
        #     self.overlayCarRendering(car)
        # else:
        #     # startpoint = (int(src[0]-15*abs(math.cos(heading))), int(src[1]-15*abs(math.sin(heading))))
        #     # endpoint = (int(src[0]+15*abs(math.cos(heading))), int(src[1]+15*abs(math.sin(heading))))
        #     # draw vehicle, orientation as black arrow
        #     # self.main.track.drawArrow(coord,heading,length=30,color=(0,0,0),thickness=5,img=img)
        #     pygame.draw.circle(self.screen, color=(0,0,0), center=src, radius=5)
        #     # draw steering angle, orientation as red arrow
        #     # self.main.track.drawArrow(coord,heading+steering,length=20,color=(0,0,255),thickness=4,img=img)

    
    def overlayCarRendering(self, car):
        x,y,heading, vf_lf, vs_lf, omega_lf = car.states
        coord = (x,y)
        src = self.main.track.m2canvas(coord)
        if (src is None):
            print("overlayCarRendering err -- coordinate outside canvas")
        self.overlayCarRenderingRaw(car,coord,heading)

    # TODO optimize this
    # overlay Car rendering at specified location in pixel coord, for plotting controls
    def overlayCarRenderingRaw(self,car, src,angle=np.pi/2):
        image = pygame.image.load(car.params['rendering']).convert_alpha()
        size = image.get_size()
        scale = 40.0/size[1]/200.0*self.track.resolution/0.0461*size[0]
        scale /= 15000
        size = (size[0] * scale, size[1] * scale)
        image = pygame.transform.scale(image, size)
        image = pygame.transform.rotate(image, degrees(angle))
        car_rect = image.get_rect()
        car_rect.center = self.track.m2canvas(src)
        self.screen.blit(image, car_rect)
        
        
        
    


