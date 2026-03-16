import numpy as np
# TODO add these to requirements.txt
import moderngl
import moderngl_window
from moderngl_window import geometry
from PIL import Image


def make_noise_texture(size=400):
    """Generate a 400x400 random RGB noise background"""
    arr = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def trajectory(t):
    """Example trajectory function for sprite (x, y, angle)"""
    x = 200 + 100 * np.cos(t * 0.5)
    y = 200 + 100 * np.sin(t * 0.5)
    angle = t * 30  # degrees
    return x, y, angle


class Visualizer(moderngl_window.WindowConfig):
    window_size = (400, 400)
    aspect_ratio = None
    title = "2D Car Visualization"
    resource_dir = "."

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Create geometry: fullscreen quad (for background) + sprite quad
        self.quad = geometry.quad_2d(size=(2.0, 2.0))
        self.sprite = geometry.quad_2d(size=(0.2, 0.2))  # sprite size relative to screen

        # Shaders for textured quad
        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_position;
                in vec2 in_texcoord_0;
                out vec2 uv;
                uniform vec2 translate;
                uniform float angle;
                uniform float scale;
                void main() {
                    // Apply rotation + translation for sprite
                    float c = cos(radians(angle));
                    float s = sin(radians(angle));
                    mat2 rot = mat2(c, -s, s, c);
                    vec2 pos = in_position * scale;
                    pos = rot * pos;
                    pos += translate;
                    gl_Position = vec4(pos, 0.0, 1.0);
                    uv = in_texcoord_0;
                }
            """,
            fragment_shader="""
                #version 330
                in vec2 uv;
                out vec4 fragColor;
                uniform sampler2D tex;
                void main() {
                    fragColor = texture(tex, uv);
                }
            """,
        )

        # Load background texture (noise)
        noise_img = make_noise_texture(400)
        self.bg_tex = self.ctx.texture(noise_img.size, 3, noise_img.tobytes())
        self.bg_tex.build_mipmaps()

        # Load sprite texture (e.g. car.png)
        sprite_img = Image.open("car.png").convert("RGBA")
        self.sprite_tex = self.ctx.texture(sprite_img.size, 4, sprite_img.tobytes())
        self.sprite_tex.build_mipmaps()

        # For time/animation
        self.time = 0.0
        self.timestamp_vec = []

    def on_render(self, time, frame_time):
        self.ctx.clear(0.1, 0.1, 0.1)
        self.timestamp_vec.append(frame_time)
        if (len(self.timestamp_vec)) > 2:
            print(f'freq = {1.0/np.mean(np.array(self.timestamp_vec))}')

        # Draw background (fullscreen quad, no transform)
        self.bg_tex.use(location=0)
        self.prog["translate"].value = (0.0, 0.0)
        self.prog["angle"].value = 0.0
        self.prog["scale"].value = 2.0  # cover whole screen
        self.quad.render(self.prog)

        # Update trajectory
        self.time += frame_time
        x, y, angle = trajectory(self.time)

        # Normalize x,y from [0,400] to [-1,1]
        tx = (x / 200.0) - 1.0
        ty = (y / 200.0) - 1.0

        # Draw sprite
        self.sprite_tex.use(location=0)
        self.prog["translate"].value = (tx, ty)
        self.prog["angle"].value = angle
        self.prog["scale"].value = 0.3  # sprite size
        self.sprite.render(self.prog)


if __name__ == "__main__":
    moderngl_window.run_window_config(Visualizer)
