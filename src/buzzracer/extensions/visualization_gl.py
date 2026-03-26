''' Visualize car on track with moderngl / OpenGL
    This replaces the OpenCV-based renderer Visualization by moving all drawing
    operations to the GPU.
'''

from __future__ import annotations
from typing import TYPE_CHECKING, NamedTuple
from collections.abc import Iterable
import os
from math import degrees, sin, cos, radians
import pickle
from threading import Event, Thread
from functools import lru_cache
from time import sleep
import queue

import moderngl
from moderngl import Texture
import moderngl_window
from moderngl_window import geometry
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from matplotlib import font_manager
from deprecated import deprecated

from buzzracer.common import BASEDIR
from buzzracer.extensions.extension import Extension, ExtensionConfig, ExtensionState
from buzzracer.utilities.execution_timer import ExecutionTimer
if TYPE_CHECKING:
    from buzzracer.cars.car import Car
    from buzzracer.tracks.track import Track


class VisualizationGLConfig(ExtensionConfig):
    def __init__(self, main_config):
        super().__init__(main_config)
        self.show_car_info = False


@Extension.register('visualization', VisualizationGLConfig, ExtensionState)
class VisualizationGL(Extension):
    def __init__(self, config, state):
        super().__init__(config, state)
        self.t = ExecutionTimer(False)
        self.update_visualization = Event()
        self.show_car_info = True
        ''' Use realistic cartoon image for car sprite'''
        self.track = self.main.track
        self.main.breakpoint = Event()
        self.save_frames = Event()
        """ If set, save frames, never cleared"""
        self.new_frame = Event()
        """ new state available, instruct gl to save frame"""

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
        # Maintain two polylines so we always have full set of polylines to render
        # Otherwise if renderer is called when polyline hasn't been re-created,
        # then we the displayed polyline will flicker
        self.polylines: list[Polyline] = []
        self.new_polylines: list[Polyline] = []

    def init(self):
        for car in self.main.cars:
            filename = os.path.join(BASEDIR, 'assets', car.param.rendering)
            self.car_images[car] = cv2.imread(filename, -1)
        img = self.main.track.draw_track()
        self.img_blank_track = img.copy()
        self.img_blank_track_with_obstacles = self.track.plot_obstacles(img.copy())
        track: Track = self.main.track
        self.img_track = track.draw_raceline(
            track.data.raceline_s, track.data.raceline_len_m, img=img)
        # img = img_track.copy()
        # draw static components onto background
        # self.img_track = self.draw_control_static_for_all_cars(img_track)
        # self.visualization_img = img
        self.moderngl_thread = Thread(
            target=self._moderngl_thread_function, daemon=True)
        self.moderngl_thread.start()
        sleep(2)

    def post_update(self):
        self.polylines = self.new_polylines
        self.new_polylines = []
        if self.save_frames.is_set():
            self.new_frame.set()

    def get_current_frame(self) -> Image:
        try:
            while len(_WindowConfig.frame_queue) > 1:
                raw_pixels = _WindowConfig.frame_queue.get_nowait()
            raw_pixels = _WindowConfig.frame_queue.get_nowait()
            img = Image.frombytes('RGB', _WindowConfig.window_size, raw_pixels)
            # pylint: disable-next=no-member
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        except queue.Empty:
            return None

    def _moderngl_thread_function(self):
        _WindowConfig.host = self
        _WindowConfig.t = self.t
        _WindowConfig.frame_queue = queue.Queue()
        # Don't really need args, but must provide a non-empty one so it doesn't
        # try to parse the actual sys.argv
        rows, cols = self.img_track.shape[:2]
        _WindowConfig.window_size = (cols, rows)
        moderngl_window.run_window_config(
            _WindowConfig, args=['-wnd', 'pyglet'])

    def post_init(self,):
        # self.save_blank_img()
        pass

    def final(self):
        self.t.summary()

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

    # --- opencg draw function for building background ---

    def draw_polyline(self, points, color: tuple[float, ...] = (0, 0, 1, 1)):
        ''' Draw a polyline from multiple points
        Args:
            points: Iterable of (x,y) in track space (unit: m)
            lineColor: RGBA color, range (0,1)
        '''
        self.new_polylines.append(Polyline(points=points, color=color))

    @deprecated
    def draw_control_static_for_all_cars(self, img):
        ''' Draw static visualization components '''
        offset = -10
        for car in self.main.cars:
            img = self.draw_control_static(img, car, (-10, offset))
            offset += 60
        self.img_track = img
        return img

    @deprecated
    def draw_control_static(self, img, car, coord):
        ''' Draw static visualization components '''
        # draw car illustration
        x2 = coord[0] + 20
        y2 = coord[1] + 50
        img = self.overlay_car_rendering_raw(img, car, (x2, y2))
        return img

    @deprecated
    def overlay_car_rendering_raw(self, img, car, src, angle=np.pi/2):
        ''' Overlay Car rendering at specified location in pixel coord, for plotting controls '''
        height, width = self.car_images[car].shape[:2]
        center = (width/2, height/2)
        # dynamic scale
        scale = 40.0/height/200.0*self.track.config.resolution/0.0461*car.param.width
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


class Polyline(NamedTuple):
    points: Iterable
    ''' Vertices of the polyline of shape [N,2]'''
    color: tuple[float, float, float, float]
    ''' R,G,B,A (0-1) color of the line'''


class _WindowConfig(moderngl_window.WindowConfig):
    """
    A high-performance visualization extension using ModernGL.

    This class replaces the OpenCV-based renderer by moving all drawing
    operations to the GPU.
    """
    title = "BuzzRacer"
    resizable = False
    vsync = True
    host = None
    t = None  # ExecutionTimer instance
    frame_queue = None  # Frame queue
    window_size = None  # (cols, rows)

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
        track = self.host.track
        track_dim_m = (track.config.x_limit, track.config.y_limit)
        # Project matrix from track frame to NDC
        self.ortho_matrix_loc = self.prog['ortho']

        # --- Geometry ---
        # Full window quad, track coordinate frame
        self.quad = geometry.quad_2d(size=track_dim_m, pos=(
            track_dim_m[0]/2, track_dim_m[1]/2))
        self.unit_quad = geometry.quad_2d(size=(1.0, 1.0), pos=(0.0, 0.0))

        # --- Textures ---
        self.bg_texture = self.texture_from_image(
            self.host.get_background_img())
        self.text_texture = {text: self.texture_from_text(text)
                             for text in ['ST', 'TH']}

        self.car_textures = {}
        for car in self.host.main.cars:
            filename = os.path.join(BASEDIR, 'assets', car.param.rendering)
            car_img = Image.open(filename).convert("RGBA")
            texture = self.ctx.texture(car_img.size, 4, car_img.tobytes())
            self.car_textures[car] = texture
        _WindowConfig.transform_matrix = _WindowConfig.create_transform_matrix()
        self.host.window = self

    def texture_from_image(self, img):
        ''' Convert final image to an RGBA moderngl texture '''
        rgba = Image.fromarray(cv2.cvtColor(
            cv2.flip(img, 0), cv2.COLOR_BGR2RGB)).convert("RGBA")
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
        self.t.s()
        self.t.s('setup')
        self.ctx.clear(0.1, 0.1, 0.1)
        # pylint: disable-next=no-member
        self.ctx.enable(moderngl.BLEND)
        self.ctx.viewport = (0, 0, self.window_size[0], self.window_size[1])
        self.t.e('setup')

        self.t.s('background')
        # 1. Draw the background
        self.bg_texture.use(location=0)
        self.use_texture_loc.value = 1
        # width, height
        track = self.host.track
        track_dim_m = (track.config.x_limit, track.config.y_limit)
        model = self.update_transform_matrix(pos=(0, 0), scale=(1, 1))
        self.model_matrix_loc.write(model)

        ortho_mtx = self.ortho(0, track_dim_m[0], 0, track_dim_m[1])
        self.ortho_matrix_loc.write(ortho_mtx)

        self.quad.render(self.prog)
        self.t.e('background')

        # 2. Draw obstacles (if any)
        # self.draw_obstacles()

        # 3. Draw each car and its dynamic UI
        for i, car in enumerate(self.host.main.cars):
            self.t.s('car')
            self.draw_car(car)
            self.t.e('car')
            if self.host.show_car_info:
                self.t.s('car_ui')
                self.draw_car_ui(car, i)
                self.t.e('car_ui')
        self.t.s('polyline')
        for polyline in self.host.polylines:
            self.draw_polyline(polyline)
        self.t.e('polyline')
        self.t.s('save frame')
        if self.host.new_frame.is_set():
            raw_pixels = self.wnd.fbo.read(components=3)
            self.frame_queue.put(raw_pixels)
            self.host.new_frame.clear()
        self.t.e('save frame')
        self.t.e()

    def draw_polyline(self, polyline: Polyline):
        ''' Render points as a 1px polyline'''
        points = np.array(polyline.points, dtype='f4', order='C')

        vbo = self.ctx.buffer(points.tobytes())
        vao = self.ctx.vertex_array(self.prog, [(vbo, '2f', 'in_position')])
        self.use_texture_loc.value = 0  # Switch to solid color mode
        self.color_loc.value = polyline.color
        model = self.update_transform_matrix(pos=(0, 0), scale=(1, 1))
        self.model_matrix_loc.write(model)

        track = self.host.track
        track_dim_m = (track.config.x_limit, track.config.y_limit)
        ortho_mtx = self.ortho(0, track_dim_m[0], 0, track_dim_m[1])
        self.ortho_matrix_loc.write(ortho_mtx)
        self.ctx.line_width = 3.0
        # In your render loop:
        vao.render(mode=moderngl.Context.LINE_STRIP, vertices=points.shape[0])
        self.ctx.line_width = 1.0

    def draw_obstacles(self):
        """Draws obstacles as solid color quads."""
        try:
            obstacles = self.main.cars[0].controller.obstacles
            self.use_texture_loc.value = 0  # Switch to solid color mode
            self.color_loc.value = (1.0, 0.4, 0.4, 1.0)  # Reddish color

            for obs in obstacles:
                pixel_pos = self.main.track.m2canvas(obs)
                pixel_radius = 0.1 * self.main.track.config.resolution
                model = self.update_transform_matrix(
                    pos=pixel_pos, scale=(pixel_radius * 2, pixel_radius * 2))
                self.model_matrix_loc.write(model)
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
        x_pix = int(track_coord[0] / track.config.x_limit * self.window_size[0])
        y_pix = int(track_coord[1] / track.config.y_limit * self.window_size[1])
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
        x_track = pix_coord[0] / self.window_size[0] * track.config.x_limit
        y_track = pix_coord[1] / self.window_size[1] * track.config.y_limit
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
        model = self.update_transform_matrix(
            pos=pose,
            rot=pose[2],
            scale=(0.19, 0.146)
        )
        self.model_matrix_loc.write(model)
        track = self.host.track
        track_dim_m = (track.config.x_limit, track.config.y_limit)
        ortho_mtx = self.ortho(0, track_dim_m[0], 0, track_dim_m[1])
        self.ortho_matrix_loc.write(ortho_mtx)
        self.unit_quad.render(self.prog)

    def draw_car(self, car):
        """Draws the car's sprite."""
        self.draw_car_pose(car, car.state)

    def draw_car_ui(self, car, car_index):
        """Draws the dynamic steering and throttle bars for a car."""
        self.t.s('setup')
        x1 = 100
        y1 = self.window_size[1] - car_index * 50
        red = (1, 0, 0, 1)
        green = (0, 1, 0, 1)

        # Helper to map a value from one range to another
        def fmap(val, in_l, in_h, out_low, out_high):
            oob = val < in_l or val > in_h
            val = max(in_l, min(in_h, val))
            return (val - in_l) / (in_h - in_l) * (out_high - out_low) + out_low, oob

        self.t.e('setup')
        # --- Car Static---
        self.t.s('draw_car_pose')
        self.draw_car_pose(
            car, (*self.pixel_to_track((x1-80, y1 - 30)), radians(90)))
        self.t.e('draw_car_pose')
        # --- Text ---
        self.t.s('draw_text')
        self.draw_text((x1-60, y1-20), self.text_texture['ST'])
        self.draw_text((x1-60, y1-40), self.text_texture['TH'])
        self.t.e('draw_text')

        # --- Steering Bar ---
        self.t.s('draw_prog_bar')
        s_val, s_oob = fmap(
            car.steering, -car.param.max_steer_left, car.param.max_steer_right, 1, 0)
        self.draw_prog_bar((x1, y1 - 20), s_val, color=red if s_oob else green)

        # --- Throttle Bar ---
        t_val, t_oob = fmap(
            car.throttle, car.param.min_throttle, car.param.max_throttle, 0, 1)
        self.draw_prog_bar((x1, y1 - 40), t_val, color=red if t_oob else green)
        self.t.e('draw_prog_bar')

    def draw_text(self, pos: tuple[int, int], texture: Texture):

        texture.use(location=0)
        model = self.update_transform_matrix(pos=pos, scale=texture.size)
        ortho_mtx = self.ortho(0, self.window_size[0], 0, self.window_size[1])
        self.ortho_matrix_loc.write(ortho_mtx)
        self.model_matrix_loc.write(model)
        self.use_texture_loc.value = 1
        self.color_loc.value = (0, 0, 0, 1)
        self.unit_quad.render(self.prog)

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
        model = self.update_transform_matrix(pos=pos, scale=(width, height))
        self.model_matrix_loc.write(model)

        ortho_mtx = self.ortho(0, self.window_size[0], 0, self.window_size[1])
        self.ortho_matrix_loc.write(ortho_mtx)
        self.use_texture_loc.value = 0
        self.color_loc.value = (0, 0, 0, 1)
        self.unit_quad.render(self.prog)

        # Bar
        bar_width = int(width*value)
        bar_pos = (pos[0] - width//2 + bar_width//2, pos[1])
        model = self.update_transform_matrix(
            pos=bar_pos, scale=(bar_width, height-2))
        # ortho_mtx = self.ortho(0, self.window_size[0], 0, self.window_size[1])
        # self.ortho_matrix_loc.write(ortho_mtx)
        self.model_matrix_loc.write(model)
        # self.use_texture_loc.value = 0 # unchanged
        self.color_loc.value = color
        self.unit_quad.render(self.prog)

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
                if not self.host.main.slowdown.is_set():
                    print('Slowing down, press Q again to shutdown')
                    self.host.main.slowdown.set()
                else:
                    self.final()
                    self.wnd.close()
                    self.host.main.exit_request.set()
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
    def update_transform_matrix(pos=(0, 0), rot=0, scale=(1, 1)):
        """Creates a 2D model matrix for position, rotation, and scale."""
        # input: (x,y,z, 1.0)
        cos_r, sin_r = cos(rot), sin(rot)
        _WindowConfig.transform_matrix[0, 0] = scale[0] * cos_r
        _WindowConfig.transform_matrix[0, 1] = scale[0] * sin_r
        _WindowConfig.transform_matrix[1, 0] = -scale[1] * sin_r
        _WindowConfig.transform_matrix[1, 1] = scale[1] * cos_r
        _WindowConfig.transform_matrix[3, 0] = pos[0]
        _WindowConfig.transform_matrix[3, 1] = pos[1]
        return _WindowConfig.transform_matrix

    @staticmethod
    @lru_cache(maxsize=8)
    def ortho(left, right, bottom, top, near=-1, far=1):
        # Creates an orthographic projection matrix
        # Transpose because opengl expect column-major, but order='C'
        return np.array((
            (2 / (right - left), 0, 0, -(right+left)/(right-left)),
            (0, 2 / (top - bottom), 0, -(top+bottom)/(top-bottom)),
            (0, 0, -2/(far-near), -(far+near)/(far-near)),
            (0, 0, 0, 1)
        ), dtype='f4').T.copy(order='C').tobytes()

    def final(self):
        """Clean up GPU resources."""
        self.bg_texture.release()
        for tex in self.car_textures.values():
            if tex:
                tex.release()
        self.prog.release()
        self.quad.release()
