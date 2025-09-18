''' Visualize car on track with moderngl / OpenGL
    This replaces the OpenCV-based renderer Visualization by moving all drawing
    operations to the GPU.
'''

import os
from math import degrees, sin, cos
import pickle
from threading import Event, Thread

import moderngl
import moderngl_window
from moderngl_window import geometry
from moderngl_window.timers.clock import Timer
import numpy as np
from PIL import Image
import cv2

from buzzracer.common import BASEDIR
from buzzracer.extensions.extension import Extension


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

    def init(self):
        img_track = self.main.track.draw_track()
        self.img_blank_track = img_track.copy()
        self.img_blank_track_with_obstacles = self.track.plot_obstacles(
            img_track.copy())
        img_track = self.main.track.draw_raceline(img=img_track)

        img = img_track.copy()
        for car in self.main.cars:
            filename = os.path.join(BASEDIR, 'buzzracer', car.params.rendering)
            car.image = cv2.imread(filename, -1)
            if car.image is None:
                self.print_error(f'Failed to load car image from {filename}')
            # img = self.draw_car(img, car)

        # draw static components onto background
        self.img_track = self.draw_control_static_for_all_cars(img_track)
        self.visualization_img = img
        self.moderngl_thread = Thread(target=self._moderngl_thread_function, daemon=True)
        self.moderngl_thread.start()

    def _moderngl_thread_function(self):
        try:
            Window = moderngl_window.find_window_classes()[0]
            window = Window(title=_WindowConfig.title,
                            size=self.img_track.shape[:2])
            _WindowConfig.wnd = window
            timer = Timer()
            vis_instance = _WindowConfig(
                ctx=window.ctx, wnd=window, timer=timer, host=self)

            while not window.is_closing and not self.main.exit_request.is_set():
                current_time, frame_time = timer.next_frame()
                vis_instance.render(current_time, frame_time)
                window.swap_buffers()
        finally:
            if 'window' in locals() and not window.is_closing:
                window.destroy()
            self.print_debug("Visualization thread: Window destroyed.")

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
        height, width = car.image.shape[:2]
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
        return self.img_track

    def get_car_img(self):
        return [car.img for car in self.main.cars]


class _WindowConfig(moderngl_window.WindowConfig):
    """
    A high-performance visualization extension using ModernGL.

    This class replaces the OpenCV-based renderer by moving all drawing
    operations to the GPU.
    """
    title = "BuzzRacer"
    resizable = True
    # The window size will be set dynamically from the track image

    def __init__(self, ctx, wnd, timer, host):
        super().__init__(ctx=ctx, wnd=wnd, timer=timer)
        self.host = host
        ''' Access point to VisualizationGL instance to retrieve current car/track state '''

        # --- Shaders ---
        # A single, versatile program for drawing textured sprites or solid colors.
        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_vert;
                in vec2 in_uv;
                out vec2 uv;
                uniform mat4 model; // Transform for object (translate, rotate, scale)
                uniform mat4 view;  // Transform for camera (orthographic projection)
                void main() {
                    gl_Position = view * model * vec4(in_vert, 0.0, 1.0);
                    uv = in_uv;
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

        # --- Uniforms ---
        # Get locations of shader variables for quick access
        self.model_matrix_loc = self.prog['model']
        self.view_matrix_loc = self.prog['view']
        self.color_loc = self.prog['color']
        self.use_texture_loc = self.prog['use_texture']
        self.prog['tex'].value = 0  # Tell the shader to use texture unit 0

        # --- Geometry ---
        # A single unit quad is sufficient; we'll transform it for every object.
        self.quad = geometry.quad_2d(size=(1.0, 1.0))

        # --- Textures ---
        self.bg_texture = self.texture_from_image(self.host.get_background_img())

        self.car_textures = [self.texture_from_image(img) for img in self.host.get_car_img()]

        # --- View Matrix (Orthographic Projection) ---
        # This matrix maps your track's pixel coordinates directly to the screen space.
        width, height = self.window_size
        ortho_matrix = self.ortho(0, width, height, 0, -1, 1)
        self.view_matrix_loc.write(ortho_matrix.astype('f4'))

    def texture_from_image(self, img):
        ''' Convert final image to an RGBA moderngl texture '''
        # Alternatively load with Image.open() directly
        # car_img = Image.open(car.params.rendering).convert("RGBA")
        # texture = self.ctx.texture(car_img.size, 4, car_img.tobytes())
        rgba = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
        return self.ctx.texture(rgba.size, 4, rgba.tobytes())

    def render(self, time: float, frame_time: float):
        """The main drawing method, called automatically every frame."""
        self.ctx.clear(0.1, 0.1, 0.1)
        self.ctx.enable(moderngl.BLEND)

        # 1. Draw the background
        self.bg_texture.use(location=0)
        self.use_texture_loc.value = 1
        width, height = self.window_size
        model = self.create_transform_matrix(pos=(width / 2, height / 2), scale=(width, height))
        self.model_matrix_loc.write(model.astype('f4'))
        self.quad.render(self.prog)

        # 2. Draw obstacles (if any)
        # self.draw_obstacles()

        # 3. Draw each car and its dynamic UI
        for i, car in enumerate(self.host.main.cars):
            self.draw_car(car, i)
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

    def draw_car(self, car, car_index):
        """Draws the car's sprite."""
        if car_index >= len(self.car_textures) or self.car_textures[car_index] is None:
            return

        x, y, heading = car.state[:3]
        pixel_pos = self.host.main.track.m2canvas((x, y))
        if pixel_pos is None:
            return  # Car is off the track

        self.car_textures[car_index].use(location=0)
        self.use_texture_loc.value = 1

        # Scale sprite based on the car's physical width in meters
        pixel_width = car.params.width * self.host.main.track.resolution
        model = self.create_transform_matrix(
            pos=pixel_pos,
            rot=-degrees(heading),  # Y-axis is inverted in pixel coordinates vs math
            scale=(pixel_width * 1.5, pixel_width)  # TODO: Adjust aspect ratio if needed
        )
        self.model_matrix_loc.write(model.astype('f4'))
        self.quad.render(self.prog)

    def draw_car_ui(self, car, car_index):
        """Draws the dynamic steering and throttle bars for a car."""
        self.use_texture_loc.value = 0  # Solid color mode
        offset_y = -10 + car_index * 60
        x1, y1 = (-10 + 30), offset_y
        green, red = (0, 1, 0, 1), (1, 0, 0, 1)

        # Helper to map a value from one range to another
        def fmap(val, in_l, in_h, out_low, out_high):
            oob = val < in_l or val > in_h
            val = max(in_l, min(in_h, val))
            return (val - in_l) / (in_h - in_l) * (out_high - out_low) + out_low, oob

        # --- Steering Bar ---
        s_val, s_oob = fmap(car.steering, -car.max_steering_left, car.max_steering_right, 100, 0)
        self.draw_ui_bar((x1, y1 + 25), s_val, 50, red if s_oob else green)

        # --- Throttle Bar ---
        t_val, t_oob = fmap(car.throttle, car.min_throttle, car.max_throttle, 0, 100)
        center = 0 if car.min_throttle < 0 else 0  # Center point for bi-directional throttle
        self.draw_ui_bar((x1, y1 + 45), t_val, center, red if t_oob else green)

    def draw_ui_bar(self, pos, value, center, color, width=100, height=15):
        """Helper to draw a single UI bar."""
        self.color_loc.value = color
        fill_width = abs(value - center)
        fill_start_x = pos[0] + min(center, value)

        model = self.create_transform_matrix(
            pos=(fill_start_x + fill_width / 2, pos[1] + height / 2),
            scale=(fill_width, height)
        )
        self.model_matrix_loc.write(model.astype('f4'))
        self.quad.render(self.prog)

    def key_event(self, key, action, modifiers):
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
        cos_r, sin_r = cos(np.radians(rot)), sin(np.radians(rot))
        return np.array([
            [scale[0] * cos_r, -scale[1] * sin_r, 0, 0],
            [scale[0] * sin_r,  scale[1] * cos_r, 0, 0],
            [0, 0, 1, 0],
            [pos[0], pos[1], 0, 1]
        ], dtype='f4').T

    @staticmethod
    def ortho(left, right, bottom, top, near=-1, far=1):
        """Creates an orthographic projection matrix."""
        return np.array([
            [2 / (right - left), 0, 0, 0],
            [0, 2 / (top - bottom), 0, 0],
            [0, 0, -2 / (far - near), 0],
            [-(right + left) / (right - left), -(top + bottom) /
             (top - bottom), -(far + near) / (far - near), 1]
        ], dtype='f4').T

    def final(self):
        """Clean up GPU resources."""
        self.bg_texture.release()
        for tex in self.car_textures:
            if tex:
                tex.release()
        self.prog.release()
        self.quad.release()
