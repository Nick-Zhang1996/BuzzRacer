import cv2
from time import sleep,time
from common import *
from Extension import Extension
from threading import Event
import pickle
import matplotlib.pyplot as plt
class Visualization(Extension):
    def __init__(self,main):
        super().__init__(main)
        self.update_visualization = Event()
        self.update_freq = 100
        self.frame_dt = 1.0/self.update_freq
        # NOTE
        self.frame_dt = 0.0
        self.count = 0

    def final(self):
        cv2.destroyAllWindows()

    def init(self,):
        self.visualization_ts = time()
        self.img_track = self.main.track.drawTrack()
        #self.img_track = self.main.track.drawRaceline(img=self.img_track)
        img = self.img_track.copy()
        for car in self.main.cars:
            img = self.main.track.drawCar(img, car.states, car.steering)
            self.visualization_img = img
        cv2.imshow('experiment',img)
        cv2.waitKey(200)

    def postInit(self,):
        self.saveBlankImg()

    def saveBlankImg(self):
        img = self.img_track.copy()
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
            cv2.imshow('experiment',self.visualization_img)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                # first time q is presed, slow down
                if not self.main.slowdown.isSet():
                    print_ok("slowing down, press q again to shutdown")
                    self.main.slowdown.set()
                    self.main.slowdown_ts = time()
                else:
                    # second time, shut down
                    self.main.exit_request.set()

    def preUpdate(self,):
        # restrict update rate to 0.02s/frame, a rate higher than this can lead to frozen frames
        #print_info(self.prefix(), "preupdate %.1f"%(time()-self.visualization_ts))
        if (time()-self.visualization_ts > self.frame_dt):
            self.update_visualization.set()

        if (self.update_visualization.is_set()):
            img = self.img_track.copy()
            for car in self.main.cars:
                img = self.main.track.drawCar(img, car.states, car.steering)
                self.visualization_img = img

    def final(self):
        img = self.img_track.copy()
        self.visualization_img = img
        self.update_visualization.set()
        self.main.cars[0].controller.plotObstacles()
        self.main.cars[0].controller.plotTrajectory()
        img = self.visualization_img.copy()
        filename = "./last_frame_" + self.main.algorithm + ".png"
        cv2.imwrite(filename,img)
        print_info(self.prefix()+"saved last frame at " + filename)




