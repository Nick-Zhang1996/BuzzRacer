"""Interactive GIF trimmer."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from PIL import Image, ImageSequence


@dataclass
class GifData:
    frames: list[Image.Image]
    durations_ms: list[int]
    disposals: list[int] | None
    loop: int
    transparency: int | None
    background: int | None
    comment: bytes | None
    size: tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trim a GIF with a simple slider UI.")
    parser.add_argument("gif_path", help="Path to the GIF to trim.")
    return parser.parse_args()


def load_gif(gif_path: Path) -> GifData:
    with Image.open(gif_path) as gif:
        frames: list[Image.Image] = []
        durations_ms: list[int] = []
        disposals: list[int | None] = []

        default_duration = int(gif.info.get("duration", 100))
        loop = int(gif.info.get("loop", 0))
        transparency = gif.info.get("transparency")
        background = gif.info.get("background")
        comment = gif.info.get("comment")
        size = gif.size

        for frame in ImageSequence.Iterator(gif):
            frames.append(frame.copy())
            durations_ms.append(int(frame.info.get("duration", default_duration)))
            disposals.append(getattr(frame, "disposal_method", frame.info.get("disposal")))

    if not frames:
        raise ValueError(f"No frames found in {gif_path}")

    disposal_values = None if any(value is None for value in disposals) else [int(value) for value in disposals]

    return GifData(
        frames=frames,
        durations_ms=durations_ms,
        disposals=disposal_values,
        loop=loop,
        transparency=transparency,
        background=background,
        comment=comment,
        size=size,
    )


def trimmed_output_path(gif_path: Path) -> Path:
    return gif_path.with_name(f"{gif_path.stem}_trimmed.gif")


class GifTrimmerUI:
    def __init__(self, gif_path: Path, gif_data: GifData):
        self.gif_path = gif_path
        self.gif_data = gif_data
        self.output_path = trimmed_output_path(gif_path)

        self.start_frame = 0
        self.end_frame = len(gif_data.frames) - 1
        self.preview_frame = 0

        width_px, height_px = gif_data.size
        dpi = 100
        fig_width = min(12, max(4, width_px / dpi))
        fig_height = min(10, max(3.5, height_px / dpi + 1.7))
        self.fig, self.ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        self.fig.subplots_adjust(left=0.08, right=0.92, top=0.90, bottom=0.28)

        self.ax.set_axis_off()
        self.image_artist = self.ax.imshow(self.frame_to_array(self.preview_frame))
        self.selection_text = self.ax.set_title("", fontsize=11)
        self.status_text = self.fig.text(0.08, 0.05, "", fontsize=10)

        slider_color = "#d7dee8"
        start_slider_ax = self.fig.add_axes([0.12, 0.17, 0.62, 0.03], facecolor=slider_color)
        end_slider_ax = self.fig.add_axes([0.12, 0.11, 0.62, 0.03], facecolor=slider_color)
        button_ax = self.fig.add_axes([0.77, 0.095, 0.15, 0.11])

        max_index = len(gif_data.frames) - 1
        self.start_slider = Slider(
            start_slider_ax,
            "Start",
            0,
            max_index,
            valinit=self.start_frame,
            valstep=1,
            valfmt="%0.0f",
        )
        self.end_slider = Slider(
            end_slider_ax,
            "End",
            0,
            max_index,
            valinit=self.end_frame,
            valstep=1,
            valfmt="%0.0f",
        )
        self.save_button = Button(button_ax, "Save\nTrimmed GIF")

        self.start_slider.on_changed(self.on_start_changed)
        self.end_slider.on_changed(self.on_end_changed)
        self.save_button.on_clicked(self.save_trimmed_gif)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.fig.canvas.mpl_connect("close_event", self.on_close)

        self.timer = self.fig.canvas.new_timer(interval=self.current_duration_ms())
        self.timer.add_callback(self.advance_preview)
        self.timer.start()
        self.refresh_view()

    def frame_to_array(self, frame_index: int) -> np.ndarray:
        return np.asarray(self.gif_data.frames[frame_index].convert("RGBA"))

    def current_duration_ms(self) -> int:
        return max(1, self.gif_data.durations_ms[self.preview_frame])

    def refresh_view(self) -> None:
        self.image_artist.set_data(self.frame_to_array(self.preview_frame))
        frame_count = self.end_frame - self.start_frame + 1
        self.selection_text.set_text(
            f"{self.gif_path.name} | frames {self.start_frame} to {self.end_frame} "
            f"({frame_count} selected) | preview {self.preview_frame}"
        )
        self.status_text.set_text(
            f"Output: {self.output_path.name}    Press 's' to save, 'q' to quit."
        )
        self.fig.canvas.draw_idle()

    def set_preview_frame(self, frame_index: int) -> None:
        self.preview_frame = int(np.clip(frame_index, self.start_frame, self.end_frame))
        self.refresh_view()

    def on_start_changed(self, value: float) -> None:
        new_start = int(value)
        if new_start > self.end_frame:
            self.end_slider.set_val(new_start)
            return
        self.start_frame = new_start
        if self.preview_frame < self.start_frame or self.preview_frame > self.end_frame:
            self.preview_frame = self.start_frame
        self.refresh_view()

    def on_end_changed(self, value: float) -> None:
        new_end = int(value)
        if new_end < self.start_frame:
            self.start_slider.set_val(new_end)
            return
        self.end_frame = new_end
        if self.preview_frame < self.start_frame or self.preview_frame > self.end_frame:
            self.preview_frame = self.start_frame
        self.refresh_view()

    def advance_preview(self) -> None:
        if self.preview_frame >= self.end_frame:
            self.preview_frame = self.start_frame
        else:
            self.preview_frame += 1
        self.refresh_view()
        self.timer.interval = self.current_duration_ms()
        self.timer.start()

    def save_trimmed_gif(self, _event=None) -> None:
        start = self.start_frame
        end = self.end_frame + 1

        selected_frames = [frame.copy() for frame in self.gif_data.frames[start:end]]
        selected_durations = self.gif_data.durations_ms[start:end]

        save_kwargs: dict[str, object] = {
            "format": "GIF",
            "save_all": True,
            "append_images": selected_frames[1:],
            "duration": selected_durations,
            "loop": self.gif_data.loop,
        }
        if self.gif_data.disposals is not None:
            save_kwargs["disposal"] = self.gif_data.disposals[start:end]
        if self.gif_data.transparency is not None:
            save_kwargs["transparency"] = self.gif_data.transparency
        if self.gif_data.background is not None:
            save_kwargs["background"] = self.gif_data.background
        if self.gif_data.comment is not None:
            save_kwargs["comment"] = self.gif_data.comment

        selected_frames[0].save(self.output_path, **save_kwargs)
        self.status_text.set_text(
            f"Saved {self.output_path.name} with frames {self.start_frame} to {self.end_frame}."
        )
        self.fig.canvas.draw_idle()

    def on_key_press(self, event) -> None:
        if event.key == "s":
            self.save_trimmed_gif()
        elif event.key == "q":
            plt.close(self.fig)

    def on_close(self, _event) -> None:
        self.timer.stop()

    def show(self) -> None:
        plt.show()


def main() -> None:
    args = parse_args()
    gif_path = Path(args.gif_path).expanduser().resolve()
    if not gif_path.is_file():
        raise FileNotFoundError(f"GIF not found: {gif_path}")

    gif_data = load_gif(gif_path)
    GifTrimmerUI(gif_path, gif_data).show()


if __name__ == "__main__":
    main()
