import cv2
from time import sleep, time
from common import *
from extension.Extension import Extension
from threading import Event
import pickle
import matplotlib.pyplot as plt
from math import degrees, radians
from PIL import Image
from threading import Event


class Visualization(Extension):
    def __init__(self):
        super().__init__(handle_name='visualization')
        self.update_visualization = Event()
        self.update_freq = 100
        self.frame_dt = 1.0/self.update_freq
        # NOTE
        self.frame_dt = 0.0
        self.count = 0
        # default setting, will be overridden if defined in config
        self.car_graphics = False
        self.track = self.main.track
        self.main.breakpoint = Event()

    def final(self):
        cv2.destroyAllWindows()

    def init(self,):
        self.visualization_ts = time()
        self.img_track = self.main.track.draw_track()
        self.img_blank_track = self.img_track.copy()
        self.img_blank_track_with_obstacles = self.track.plot_obstacles(
            self.img_track.copy())
        self.img_track = self.main.track.draw_raceline(img=self.img_track)

        img = self.img_track.copy()
        for car in self.main.cars:
            car.image = cv2.imread(car.params['rendering'], -1)
            img = self.draw_car(img, car)

        # draw static components onto background
        self.img_track = self.draw_control_static_for_all_cars(self.img_track)
        self.visualization_img = img
        cv2.imshow('experiment', img)
        cv2.waitKey(200)

    def post_init(self,):
        self.save_blank_img()

    def save_blank_img(self):
        # img = self.img_blank_track.copy()
        img = self.img_blank_track_with_obstacles.copy()
        try:
            obstacles = self.main.cars[0].controller.obstacles
            # plot obstacles
            for obs in obstacles:
                img = self.main.track.draw_circle(
                    img, obs, 0.1, color=(255, 100, 100))
        except AttributeError:
            pass

        with open('track_img.p', 'wb') as f:
            print_info(self.prefix()+'saved raw track background')
            pickle.dump(img, f)

    # show image
    # do this last since controllers may need to alter the image
    def post_update(self,):
        if (self.update_visualization.is_set()):
            self.update_visualization.clear()
            self.visualization_ts = time()
            cv2.imshow('experiment', self.visualization_img)

            k = cv2.waitKey(1) & 0xFF
            # q for quit
            if k == ord('q'):
                # first time q is presed, slow down
                if not self.main.slowdown.isSet():
                    print_ok('slowing down, press q again to shutdown')
                    self.main.slowdown.set()
                    self.main.slowdown_ts = time()
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
                self.main.snapshot.take_snapshot()

    def pre_update(self,):
        # restrict update rate to 0.02s/frame, a rate higher than this can lead to frozen frames
        # print_info(self.prefix(), "preupdate %.1f"%(time()-self.visualization_ts))
        if (time()-self.visualization_ts > self.frame_dt):
            self.update_visualization.set()

        if (self.update_visualization.is_set()):
            img = self.img_track.copy()
            for car in self.main.cars:
                img = self.draw_car(img, car)
            img = self.draw_control_for_all_cars(img)
            img = self.track.plot_obstacles(img)
            self.visualization_img = img

    def final(self):
        img = self.img_track.copy()
        self.visualization_img = img
        self.update_visualization.set()
        # self.main.cars[0].controller.plot_obstacles()
        # self.main.cars[0].controller.plot_trajectory()
        # img = self.visualization_img.copy()
        # filename = "./last_frame_" + self.main.algorithm + ".png"
        # cv2.imwrite(filename,img)
        # print_info(self.prefix()+"saved last frame at " + filename)

    def draw_control_static_for_all_cars(self, img):
        offset = -10
        for car in self.main.cars:
            img = self.draw_control_static(img, car, (-10, offset))
            offset += 60
        self.img_track = img
        return img

    def draw_control_static(self, img, car, coord):
        # draw car illustration
        x2 = coord[0] + 20
        y2 = coord[1] + 50
        img = self.overlay_car_rendering_raw(img, car, (x2, y2))
        return img

    def draw_control_for_all_cars(self, img):
        offset = -10
        for car in self.main.cars:
            # img = self.draw_acceleration(img, car, (0,0))
            img = self.draw_control(img, car, (-10, offset))
            offset += 60
        return img

    def draw_control(self, img, car, coord):
        ''' draw control, throttle/steering: [-1,1]'''
        def bound(a, l, h):
            val = l if a < l else a
            return h if val > h else val

        def map(val, in_l, in_h, out_low, out_high):
            # out of bound flag
            oob = False
            if (val < in_l):
                val = in_l
                oob = True
            elif (val > in_h):
                val = in_h
                oob = True
            return (val-in_l)/(in_h-in_l)*(out_high-out_low)+out_low, oob

        # x1 and y1 are the origin values -- need to be changed if origin changes
        x1 = coord[0] + 30
        y1 = coord[1]
        x, y, heading, vf_lf, vs_lf, omega_lf = car.states
        # Add steering bar
        steering, oob = map(car.steering, -car.max_steering_left,
                            car.max_steering_right, 100, 0)
        img = cv2.rectangle(img, (x1, y1 + 25),
                            (x1 + 100, y1 + 40), (0, 0, 255), 1)
        if (oob):
            img = cv2.rectangle(img, (x1 + 50, y1 + 25),
                                (x1 + int(steering), y1 + 40), (0, 0, 255), -1)
        else:
            img = cv2.rectangle(img, (x1 + 50, y1 + 25),
                                (x1 + int(steering), y1 + 40), (0, 255, 0), -1)
        img = cv2.putText(img, 'Steering', (x1 + 104, y1 + 35),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        # Add Throttle bar
        throttle, oob = map(car.throttle, car.min_throttle,
                            car.max_throttle, 0, 100)
        img = cv2.rectangle(img, (x1, y1 + 45),
                            (x1 + 100, y1 + 60), (0, 0, 255), 1)
        if (oob):
            img = cv2.rectangle(img, (x1 + 52, y1 + 45),
                                (x1 + int(throttle), y1 + 60), (0, 0, 255), -1)
        else:
            img = cv2.rectangle(img, (x1 + 52, y1 + 45),
                                (x1 + int(throttle), y1 + 60), (0, 255, 0), -1)
        img = cv2.putText(img, 'Throttle', (x1 + 104, y1 + 55),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        return img

    def draw_acceleration(self, img, car, coord):
        # x1 and y1 are the origin values -- need to be changed if origin changes
        x1 = coord[0]
        y1 = coord[1]
        x, y, heading, vf_lf, vs_lf, omega_lf = car.states
        steering = car.steering
        throttle = car.throttle

        # Add acceleration bar
        img = cv2.circle(img, (x1 + 50, y1 + 80), 18, (0, 0, 255), 1)
        img = cv2.putText(img, 'Acceleration', (x1 + 104, y1 + 80),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        # img = cv2.circle(img, (x1 + 50, y1 + 80), 3, (0, 255, 0), -1)
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
        if (steering < 0):
            direction_x = (x1 + (50 - (6 * acc_x_scale)))
            direction_y = (y1 + (80 + (6 * acc_y_scale)))

        img = cv2.circle(img, (direction_x, direction_y), 3, (0, 255, 0), -1)
        return img

# draw the vehicle (one dot with two lines) onto a canvas
# coord: location of the dor, in meter (x,y)
# heading: heading of the vehicle, radians from x axis, ccw positive
#  steering : steering of the vehicle, left positive, in radians, w/ respect to vehicle heading
# NOTE: this function modifies img, if you want to recycle base img, send img.copy()
    def draw_car(self, img, car):
        x, y, heading, vf_lf, vs_lf, omega_lf = car.states
        throttle = car.throttle
        steering = car.steering
        coord = (x, y)
        src = self.main.track.m2canvas(coord)
        if src is None:
            # print("Can't draw car -- outside track")
            return img
        # overlay vehicle image, orientation as headed
        # significant performance impact
        if (self.car_graphics):
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
        x, y, heading, vf_lf, vs_lf, omega_lf = car.states
        coord = (x, y)
        src = self.main.track.m2canvas(coord)
        if (src is None):
            print('overlay_car_rendering err -- coordinate outside canvas')
            return img
        return self.overlay_car_rendering_raw(img, car, src, heading)

    # TODO optimize this
    # overlay Car rendering at specified location in pixel coord, for plotting controls
    def overlay_car_rendering_raw(self, img, car, src, angle=np.pi/2):
        # x,y,heading, vf_lf, vs_lf, omega_lf = car.states
        # coord = (x,y)
        # src = self.main.track.m2canvas(coord)
        # if (src is None):
        #    print("overlay_car_rendering err -- coordinate outside canvas")
        #    return img

        # image rotation according to heading and steering angles
        # heading = np.pi/2
        height, width = car.image.shape[:2]
        center = (width/2, height/2)
        # dynamic scale
        scale = 40.0/height/200.0*self.track.resolution/0.0461*car.width
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
