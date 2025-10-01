''' Visualize car on track with moderngl / OpenGL
    This replaces the OpenCV-based renderer Visualization by moving all drawing
    operations to the GPU.
'''

from __future__ import annotations
from typing import TYPE_CHECKING
import os
from math import degrees, sin, cos, radians
import pickle
from threading import Event, Thread

import moderngl
from moderngl import Texture
import moderngl_window
from moderngl_window import geometry
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from matplotlib import font_manager

from buzzracer.common import BASEDIR
from buzzracer.extensions.extension import Extension
if TYPE_CHECKING:
    from buzzracer.cars.car import Car


class VisualizationGL(Extension):
    def __init__(self):
        super().__init__(handle_name='visualization')
        self.update_visualization = Event()
        self.car_graphics = False
        ''' Use realistic cartoon image for car sprite'''
        self.track = self.main.track
        self.main.breakpoint = Event()

        self.img_track = None
        '''' Image of a track, with static visualization components like debuggint text '''
        self.img_blank_track = None
        ''' Blacnk image of the track '''
        self.img_blank_track_with_obstacles = None
        ''' Blacnk image of the track, with obstacles if any'''
        self.visualization_img = None
        ''' The current visualization image'''
        self.moderngl_thread = None
        self.car_images = {}

    def init(self):
        for car in self.main.cars:
            filename = os.path.join(BASEDIR, 'buzzracer', car.params.rendering)
            self.car_images[car] = cv2.imread(filename, -1)
        img_track = self.main.track.draw_track()
        self.img_blank_track = img_track.copy()
        self.img_blank_track_with_obstacles = self.track.plot_obstacles(
            img_track.copy())
        self.img_track = self.main.track.draw_raceline(img=img_track)
        # img = img_track.copy()
        # draw static components onto background
        # self.img_track = self.draw_control_static_for_all_cars(img_track)
        # self.visualization_img = img
        self.moderngl_thread = Thread(target=self._moderngl_thread_function, daemon=True)
        self.moderngl_thread.start()

    def _moderngl_thread_function(self):
        _WindowConfig.host = self
        # Don't really need args, but must provide a non-empty one so it doesn't
        # try to parse the actual sys.argv
        rows, cols = self.img_track.shape[:2]
        _WindowConfig.window_size = (cols, rows)
        moderngl_window.run_window_config(_WindowConfig, args=['-wnd', 'pyglet'])

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

        filename = os.path.join(BASEDIR, 'buzzracer', 'data', 'track_img.p')
        with open(filename, 'wb') as f:
            self.print_info(f'saved raw track background at {filename}')
            pickle.dump(img, f)

    def pre_update(self,):
        # img = self.img_track.copy()
        # for car in self.main.cars:
        #     img = self.draw_car(img, car)
        # img = self.draw_control_for_all_cars(img)
        # img = self.track.plot_obstacles(img)
        # self.visualization_img = img
        pass

    # --- opencg draw function for building background ---
    # nit: move to opengl

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

    def overlay_car_rendering_raw(self, img, car, src, angle=np.pi/2):
        ''' Overlay Car rendering at specified location in pixel coord, for plotting controls '''
        height, width = self.car_images[car].shape[:2]
        center = (width/2, height/2)
        # dynamic scale
        scale = 40.0/height/200.0*self.track.resolution/0.0461*car.params.width
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

    def get_background_img(self):
        # return np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        return self.img_track

    def get_car_img(self):
        return [car.image for car in self.main.cars]


class _WindowConfig(moderngl_window.WindowConfig):
    """
    A high-performance visualization extension using ModernGL.

    This class replaces the OpenCV-based renderer by moving all drawing
    operations to the GPU.
    """
    title = "BuzzRacer"
    resizable = False
    host = None
    ''' Access point to VisualizationGL instance to retrieve current car/track state '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # --- Shaders ---
        # A single, versatile program for drawing textured sprites or solid colors.
        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_position;
                in vec2 in_texcoord_0;
                out vec2 uv;
                uniform mat4 model; // Transform for object (translate, rotate, scale)
                uniform mat4 ortho; // Project from track frame to NDC
                void main() {
                    gl_Position = ortho * model * vec4(in_position, 0.0, 1.0);
                    uv = in_texcoord_0;
                }
            """,
            fragment_shader="""
                #version 330
                in vec2 uv;
                out vec4 fragColor;
                uniform sampler2D tex;
                uniform vec4 color;
                uniform int use_texture; // Flag to switch between texture and solid color
                void main() {
                    if (use_texture == 1) {
                        fragColor = texture(tex, uv);
                        // Discard transparent pixels for clean sprite rendering
                        if (fragColor.a < 0.1) discard;
                    } else {
                        fragColor = color;
                    }
                }
            """
        )
        self.ctx.gc_mode = 'context_gc'

        # --- Uniforms ---
        # Get locations of shader variables for quick access
        self.model_matrix_loc = self.prog['model']
        self.color_loc = self.prog['color']
        self.use_texture_loc = self.prog['use_texture']
        self.prog['tex'].value = 0  # Tell the shader to use texture unit 0

        # track_dim_pixel = self.host.img_track.shape[:2]
        track_dim_m = (self.host.track.x_limit, self.host.track.y_limit)
        # Project matrix from track frame to NDC
        self.ortho_matrix_loc = self.prog['ortho']

        # --- Geometry ---
        # Full window quad, track coordinate frame
        self.quad = geometry.quad_2d(size=track_dim_m, pos=(track_dim_m[0]/2, track_dim_m[1]/2))

        # --- Textures ---
        self.bg_texture = self.texture_from_image(self.host.get_background_img())
        self.text_texture = {text: self.texture_from_text(text)
                             for text in ['ST', 'TH']}

        self.car_textures = {}
        for car in self.host.main.cars:
            filename = os.path.join(BASEDIR, 'buzzracer', car.params.rendering)
            car_img = Image.open(filename).convert("RGBA")
            texture = self.ctx.texture(car_img.size, 4, car_img.tobytes())
            self.car_textures[car] = texture

    def texture_from_image(self, img):
        ''' Convert final image to an RGBA moderngl texture '''
        rgba = Image.fromarray(cv2.cvtColor(cv2.flip(img, 0), cv2.COLOR_BGR2RGB)).convert("RGBA")
        return self.ctx.texture(rgba.size, 4, rgba.tobytes())

    def texture_from_text(self, text, size=12):
        font_path = font_manager.findfont("DejaVu Sans")
        font = ImageFont.truetype(font_path, size=size)
        bbox = font.getbbox(text)
        width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        # render text to a Pillow image
        img = Image.new("RGBA", (width, height*2), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), text, font=font, fill=(0, 0, 0, 255))
        img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        # convert to numpy bytes for OpenGL
        tex = self.ctx.texture(img.size, 4, img.tobytes())
        tex.build_mipmaps()
        return tex

    def on_render(self, time: float, frame_time: float):
        """The main drawing method, called automatically every frame."""
        self.ctx.clear(0.1, 0.1, 0.1)
        # pylint: disable-next=no-member
        self.ctx.enable(moderngl.BLEND)
        self.ctx.viewport = (0, 0, self.window_size[0], self.window_size[1])

        # 1. Draw the background
        self.bg_texture.use(location=0)
        self.use_texture_loc.value = 1
        # width, height
        track_dim_m = (self.host.track.x_limit, self.host.track.y_limit)
        model = self.create_transform_matrix(pos=(0, 0),
                                             scale=(1, 1))
        self.model_matrix_loc.write(model.astype('f4').tobytes())

        ortho_mtx = self.ortho(0, track_dim_m[0], 0, track_dim_m[1])
        self.ortho_matrix_loc.write(ortho_mtx.astype('f4').tobytes())

        self.quad.render(self.prog)

        # 2. Draw obstacles (if any)
        # self.draw_obstacles()

        # 3. Draw each car and its dynamic UI
        for i, car in enumerate(self.host.main.cars):
            self.draw_car(car)
            self.draw_car_ui(car, i)

    def draw_obstacles(self):
        """Draws obstacles as solid color quads."""
        try:
            obstacles = self.main.cars[0].controller.obstacles
            self.use_texture_loc.value = 0  # Switch to solid color mode
            self.color_loc.value = (1.0, 0.4, 0.4, 1.0)  # Reddish color

            for obs in obstacles:
                pixel_pos = self.main.track.m2canvas(obs)
                pixel_radius = 0.1 * self.main.track.resolution
                model = self.create_transform_matrix(
                    pos=pixel_pos, scale=(pixel_radius * 2, pixel_radius * 2))
                self.model_matrix_loc.write(model.astype('f4'))
                self.quad.render(self.prog)
        except (AttributeError, IndexError):
            pass  # No obstacles to draw

    def track_to_pixel(self, track_coord: tuple[float, float]) -> tuple[int, int]:
        ''' Convert coordinate in track frame to pixel unit
        Args:
            track_coord: (x,y, ...) coordinate in track frame unit: meters
        Returns:
            pix_coord: (x,y) coordinate in pixel unit
        '''
        # track: (0,0), bottom left, (track.x_limit, track.y_limit)
        track = self.host.main.track
        x_pix = int(track_coord[0] / track.x_limit * self.window_size[0])
        y_pix = int(track_coord[1] / track.y_limit * self.window_size[1])
        return (x_pix, y_pix)

    def pixel_to_track(self, pix_coord: tuple[int, int]) -> tuple[float, float]:
        ''' Convert coordinate in pixel unit to track frame (m)
        Args:
            pix_coord: (x,y) coordinate in pixel unit
        Returns:
            track_coord: (x,y, ...) coordinate in track frame unit: meters
        '''
        # track: (0,0), bottom left, (track.x_limit, track.y_limit)
        track = self.host.main.track
        x_track = pix_coord[0] / self.window_size[0] * track.x_limit
        y_track = pix_coord[1] / self.window_size[1] * track.y_limit
        return (x_track, y_track)

    def draw_car_pose(self, car: Car, pose: tuple[float, ...]):
        ''' Draw car at specified pose
        Args:
            car: Car object, for finding correct car texture
            pose: tuple with (x,y,heading(rad), ... )
        '''

        self.car_textures[car].use(location=0)
        self.use_texture_loc.value = 1

        # Scale sprite based on the car's physical width in meters
        # car image is of size 616 * 442, with the actual car 586 * 242
        # car physical size is 0.18 * 0.08 -> image physical size 0.19 * 0.146
        car_quad = geometry.quad_2d(size=(0.19, 0.146))
        model = self.create_transform_matrix(
            pos=pose,
            rot=pose[2],
        )
        self.model_matrix_loc.write(model.astype('f4'))
        track_dim_m = (self.host.track.x_limit, self.host.track.y_limit)
        ortho_mtx = self.ortho(0, track_dim_m[0], 0, track_dim_m[1])
        self.ortho_matrix_loc.write(ortho_mtx.astype('f4').tobytes())
        car_quad.render(self.prog)

    def draw_car(self, car):
        """Draws the car's sprite."""
        self.draw_car_pose(car, car.state)

    def draw_car_ui(self, car, car_index):
        """Draws the dynamic steering and throttle bars for a car."""
        x1 = 100
        y1 = self.window_size[1] - car_index * 50
        red = (1, 0, 0, 1)
        green = (0, 1, 0, 1)

        # Helper to map a value from one range to another
        def fmap(val, in_l, in_h, out_low, out_high):
            oob = val < in_l or val > in_h
            val = max(in_l, min(in_h, val))
            return (val - in_l) / (in_h - in_l) * (out_high - out_low) + out_low, oob

        # --- Car Static---
        self.draw_car_pose(car, (*self.pixel_to_track((x1-80, y1 - 30)), radians(90)))
        # --- Text ---
        self.draw_text((x1-60, y1-20), self.text_texture['ST'])
        self.draw_text((x1-60, y1-40), self.text_texture['TH'])

        # --- Steering Bar ---
        s_val, s_oob = fmap(car.steering, -car.max_steering_left, car.max_steering_right, 1, 0)
        self.draw_prog_bar((x1, y1 - 20), s_val, color=red if s_oob else green)

        # --- Throttle Bar ---
        t_val, t_oob = fmap(car.throttle, car.min_throttle, car.max_throttle, 0, 1)
        self.draw_prog_bar((x1, y1 - 40), t_val, color=red if t_oob else green)

    def draw_text(self, pos: tuple[int, int], texture: Texture):

        texture.use(location=0)
        quad = geometry.quad_2d(size=texture.size, pos=pos)
        model = self.create_transform_matrix()
        ortho_mtx = self.ortho(0, self.window_size[0], 0, self.window_size[1])
        self.ortho_matrix_loc.write(ortho_mtx)
        self.model_matrix_loc.write(model)
        self.use_texture_loc.value = 1
        self.color_loc.value = (0, 0, 0, 1)
        quad.render(self.prog)

    def draw_prog_bar(self,
                      pos: tuple[int, int],
                      value: float,
                      width=100,
                      height=15,
                      color=(0, 1, 0, 1)):
        """Helper to draw a progress bar.
        Args:
            pos: (x,y) Progress bar center in pixel coord
            value: 0-1, value of the progress bar
            width: width of progress bar 
            height: height of progress bar 
            color: R,G,B,A, range 0-1
        """

        # Use pixel unit for non-physical objects

        # Background
        prog_bar_quad = geometry.quad_2d(size=(width, height), pos=pos)
        model = self.create_transform_matrix()
        ortho_mtx = self.ortho(0, self.window_size[0], 0, self.window_size[1])
        self.ortho_matrix_loc.write(ortho_mtx)
        self.model_matrix_loc.write(model)
        self.use_texture_loc.value = 0
        self.color_loc.value = (0, 0, 0, 1)
        prog_bar_quad.render(self.prog)

        # Bar
        bar_width = int(width*value)
        bar_pos = (pos[0] - width//2 + bar_width//2, pos[1])
        prog_bar_quad = geometry.quad_2d(size=(bar_width, height-2), pos=bar_pos)
        model = self.create_transform_matrix()
        ortho_mtx = self.ortho(0, self.window_size[0], 0, self.window_size[1])
        self.ortho_matrix_loc.write(ortho_mtx)
        self.model_matrix_loc.write(model)
        self.use_texture_loc.value = 0
        self.color_loc.value = color
        prog_bar_quad.render(self.prog)

    def on_key_event(self, key, action, modifiers):
        """Handles keyboard inputs."""
        if action != self.wnd.keys.ACTION_PRESS:
            return

        key_map = {
            self.wnd.keys.Q: 'quit',
            self.wnd.keys.P: 'pause',
            self.wnd.keys.B: 'breakpoint',
            self.wnd.keys.S: 'snapshot',
        }

        if key in key_map:
            command = key_map[key]
            if command == 'quit':
                if not self.host.main.slowdown.isSet():
                    print('Slowing down, press Q again to shutdown')
                    self.host.main.slowdown.set()
                else:
                    self.host.main.exit_request.set()
                    self.wnd.close()
                    # self.final()
            elif command == 'pause':
                print('Paused. Check console to continue.')
                input('Press Enter in the console to continue...')
            elif command == 'breakpoint':
                self.host.main.breakpoint.set()
            elif command == 'snapshot':
                try:
                    self.host.main.snapshot.toggle_snapshot()
                except AttributeError:
                    print('Warning: Snapshot module not loaded.')

    @staticmethod
    def create_transform_matrix(pos=(0, 0), rot=0, scale=(1, 1)):
        """Creates a 2D model matrix for position, rotation, and scale."""
        # input: (x,y,z, 1.0)
        cos_r, sin_r = cos(rot), sin(rot)
        # Transpose because opengl expect column-major, but order='C'
        return np.array([
            [scale[0] * cos_r, -scale[1] * sin_r, 0, pos[0]],
            [scale[0] * sin_r,  scale[1] * cos_r, 0, pos[1]],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype='f4').T.copy(order='C')

    @staticmethod
    def ortho(left, right, bottom, top, near=-1, far=1):
        # Creates an orthographic projection matrix
        # Transpose because opengl expect column-major, but order='C'
        return np.array((
            (2 / (right - left), 0, 0, -(right+left)/(right-left)),
            (0, 2 / (top - bottom), 0, -(top+bottom)/(top-bottom)),
            (0, 0, -2/(far-near), -(far+near)/(far-near)),
            (0, 0, 0, 1)
        ), dtype='f4').T.copy(order='C')

    def final(self):
        """Clean up GPU resources."""
        self.bg_texture.release()
        for tex in self.car_textures:
            if tex:
                tex.release()
        self.prog.release()
        self.quad.release()
