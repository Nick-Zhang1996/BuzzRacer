''' Create a visualization of car on track'''

import os
from time import time
from threading import Event
from math import degrees
import pickle

import cv2
from PIL import Image
import numpy as np

from buzzracer.common import BASEDIR
from buzzracer.extensions.extension import Extension, ExtensionConfig, ExtensionState


@Extension.register('visualization', ExtensionConfig, ExtensionState)
class Visualization(Extension):
    def __init__(self, config, state):
        super().__init__(config, state)
        self.update_visualization = Event()
        self.update_freq = 100
        self.frame_dt = 1.0/self.update_freq
        self.frame_dt = 0.0
        self.count = 0
        self.car_graphics = False
        ''' Use realistic cartoon image for car sprite'''
        self.track = self.main.track
        self.main.breakpoint = Event()
        self.visualization_ts: float = 0.0
        ''' clock time of last visualization update'''

        self.save_frames = Event()
        self.visualization_ts = time()

        self.img_track = None
        '''' Image of a track, with static visualization components like debuggint text '''
        self.img_blank_track = None
        ''' Blacnk image of the track '''
        self.img_blank_track_with_obstacles = None
        ''' Blacnk image of the track, with obstacles if any'''
        self.visualization_img = None
        ''' The current visualization image'''

    def init(self):
        img_track = self.main.track.draw_track()
        self.img_blank_track = img_track.copy()
        self.img_blank_track_with_obstacles = self.track.plot_obstacles(self.main.cars,
                                                                        img_track.copy())
        track = self.track
        img_track = self.track.draw_raceline(track.data.raceline_s,
                                             track.data.raceline_len_m,
                                             img=img_track)

        img = img_track.copy()
        for car in self.main.cars:
            filename = os.path.join(BASEDIR, 'assets', car.param.rendering)
            car.image = cv2.imread(filename, -1)
            if car.image is None:
                self.print_error(f'Failed to load car image from {filename}')
            img = self.draw_car(img, car)

        # draw static components onto background
        self.img_track = self.draw_control_static_for_all_cars(img_track)
        self.visualization_img = img
        cv2.imshow('experiment', img)
        cv2.waitKey(200)

    def post_init(self,):
        # self.save_blank_img()
        pass

    def save_blank_img(self):
        ''' Save the blank background as pickle dump'''
        img = self.img_blank_track_with_obstacles.copy()
        try:
            obstacles = self.main.cars[0].controller.obstacles
            # plot obstacles
            for obs in obstacles:
                img = self.main.track.draw_circle(
                    img, obs, 0.1, color=(255, 100, 100))
        except AttributeError:
            pass

        filename = os.path.join(BASEDIR, 'assets', 'track_img.p')
        with open(filename, 'wb') as f:
            self.print_info(f'saved raw track background at {filename}')
            pickle.dump(img, f)

    def post_update(self,):
        ''' Show visualization image
            Do this last since controllers may need to alter the image
        '''
        if self.update_visualization.is_set():
            self.update_visualization.clear()
            self.visualization_ts = time()
            cv2.imshow('experiment', self.visualization_img)

            k = cv2.waitKey(1) & 0xFF
            # q for quit
            if k == ord('q'):
                # first time q is presed, slow down
                if not self.main.slowdown.is_set():
                    self.print_ok('slowing down, press q again to shutdown')
                    self.main.slowdown.set()
                else:
                    # second time, shut down
                    self.main.exit_request.set()
            # p for pause
            elif k == ord('p'):
                self.print_info('Paused')
                input('press Enter to continue')
            # p for pause
            elif k == ord('b'):
                self.print_info('breakpoint')
                self.main.breakpoint.set()
            # s for snapshot
            elif k == ord('s'):
                self.print_info('Requesting snapshot')
                try:
                    self.main.snapshot.toggle_snapshot()
                except AttributeError:
                    self.print_warning('Snapshot module is not loaded')

    def pre_update(self,):
        # restrict update rate to 0.02s/frame, a rate higher than this can lead to frozen frames
        # print_info(self.prefix(), "preupdate %.1f"%(time()-self.visualization_ts))
        if time()-self.visualization_ts > self.frame_dt:
            self.update_visualization.set()

        if self.update_visualization.is_set():
            img = self.img_track.copy()
            for car in self.main.cars:
                img = self.draw_car(img, car)
            img = self.draw_control_for_all_cars(img)
            img = self.track.plot_obstacles(self.main.cars, img)
            self.visualization_img = img

    def final(self):
        cv2.destroyAllWindows()

    def draw_control_static_for_all_cars(self, img):
        ''' Draw static visualization components '''
        offset = -10
        for car in self.main.cars:
            img = self.draw_control_static(img, car, (-10, offset))
            offset += 60
        self.img_track = img
        return img

    def draw_control_static(self, img, car, coord):
        ''' Draw static visualization components '''
        # draw car illustration
        x2 = coord[0] + 20
        y2 = coord[1] + 50
        img = self.overlay_car_rendering_raw(img, car, (x2, y2))
        return img

    def draw_control_for_all_cars(self, img):
        ''' Draw control visualization components '''
        offset = -10
        for car in self.main.cars:
            # img = self.draw_acceleration(img, car, (0,0))
            img = self.draw_control(img, car, (-10, offset))
            offset += 60
        return img

    def draw_control(self, img, car, coord):
        ''' draw control, throttle/steering: [-1,1]'''
        def fmap(val, in_l, in_h, out_low, out_high):
            # out of bound flag
            oob = False
            if val < in_l:
                val = in_l
                oob = True
            elif val > in_h:
                val = in_h
                oob = True
            return (val-in_l)/(in_h-in_l)*(out_high-out_low)+out_low, oob

        # x1 and y1 are the origin values -- need to be changed if origin changes
        x1 = coord[0] + 30
        y1 = coord[1]
        # x, y, heading, vf_lf, vs_lf, omega_lf = car.state
        # Add steering bar
        steering, oob = fmap(car.steering, -car.param.max_steer_left,
                             car.param.max_steer_right, 100, 0)
        img = cv2.rectangle(img, (x1, y1 + 25),
                            (x1 + 100, y1 + 40), (0, 0, 255), 1)
        if oob:
            img = cv2.rectangle(img, (x1 + 50, y1 + 25),
                                (x1 + int(steering), y1 + 40), (0, 0, 255), -1)
        else:
            img = cv2.rectangle(img, (x1 + 50, y1 + 25),
                                (x1 + int(steering), y1 + 40), (0, 255, 0), -1)
        img = cv2.putText(img, 'Steering', (x1 + 104, y1 + 35),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        # Add Throttle bar
        throttle, oob = fmap(car.throttle, car.min_throttle,
                             car.max_throttle, 0, 100)
        img = cv2.rectangle(img, (x1, y1 + 45),
                            (x1 + 100, y1 + 60), (0, 0, 255), 1)
        if oob:
            img = cv2.rectangle(img, (x1 + 52, y1 + 45),
                                (x1 + int(throttle), y1 + 60), (0, 0, 255), -1)
        else:
            img = cv2.rectangle(img, (x1 + 52, y1 + 45),
                                (x1 + int(throttle), y1 + 60), (0, 255, 0), -1)
        img = cv2.putText(img, 'Throttle', (x1 + 104, y1 + 55),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        return img

    def draw_car(self, img, car):
        ''' Draw the vehicle (one dot with two lines) onto a canvas
        coord: location of the dor, in meter (x,y)
        heading: heading of the vehicle, radians from x axis, ccw positive
        steering : steering of the vehicle, left positive, in radians, w/ respect to vehicle heading
        NOTE: this function modifies img, if you want to recycle base img, send img.copy()
        '''
        x, y, heading = car.state[:3]
        steering = car.steering
        coord = (x, y)
        src = self.main.track.m2canvas(coord)
        if src is None:
            # print("Can't draw car -- outside track")
            return img
        # overlay vehicle image, orientation as headed
        # significant performance impact
        if self.car_graphics:
            img = self.overlay_car_rendering(img, car)
        else:
            # draw vehicle, orientation as black arrow
            img = self.main.track.draw_arrow(
                coord, heading, length=30, color=(0, 0, 0), thickness=5, img=img)
            # draw steering angle, orientation as red arrow
            img = self.main.track.draw_arrow(
                coord, heading+steering, length=20, color=(0, 0, 255), thickness=4, img=img)
        return img

    def overlay_car_rendering(self, img, car):
        x, y, heading = car.state[:3]
        coord = (x, y)
        src = self.main.track.m2canvas(coord)
        if src is None:
            print('overlay_car_rendering err -- coordinate outside canvas')
            return img
        return self.overlay_car_rendering_raw(img, car, src, heading)

    def overlay_car_rendering_raw(self, img, car, src, angle=np.pi/2):
        ''' Overlay Car rendering at specified location in pixel coord, for plotting controls '''
        height, width = car.image.shape[:2]
        center = (width/2, height/2)
        scale = 2.7/height*self.track.config.resolution*car.param.width
        rotate_matrix = cv2.getRotationMatrix2D(
            center=center, angle=degrees(angle), scale=scale)
        rotated_car = cv2.warpAffine(
            src=car.image, M=rotate_matrix, dsize=(width, height))
        overlay_t = Image.fromarray(
            cv2.cvtColor(rotated_car, cv2.COLOR_BGRA2RGBA))
        bg_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        bg_img = Image.alpha_composite(
            Image.new('RGBA', bg_img.size), bg_img.convert('RGBA'))
        x, y = (src[0]-width//2), (src[1]-height//2)

        bg_img.paste(overlay_t, (x, y), overlay_t)
        bg_img = np.array(bg_img, dtype=np.uint8)
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_RGBA2BGRA)

        return bg_img

    def get_current_frame(self) -> Image:
        return Image.fromarray(cv2.cvtColor(self.visualization_img, cv2.COLOR_BGR2RGB))
