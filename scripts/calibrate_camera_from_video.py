#!/usr/bin/env python3
"""Calibrate a camera from an evenly sampled chessboard video.

The chessboard size is expressed as the number of *inner* corners.  For
example, a board with 10 by 7 squares has a (9, 6) inner-corner pattern.

Example:
    python scripts/calibrate_camera_from_video.py calibration.mp4 \
        --rotate 90-cw --square-size 0.024 --samples 40 \
        --output camera_calibration.npz

The resulting ``.npz`` contains the camera matrix, distortion coefficients,
per-image extrinsics, and an optimal new camera matrix.  ``--square-size``
sets the physical unit used for the translation vectors (metres in the example
above); it may be left at 1.0 when only undistortion is needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np


PATTERN_SIZE = (9, 6)  # (corners across, corners down), not squares.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Video containing the chessboard.")
    parser.add_argument(
        "--samples",
        type=int,
        default=30,
        help="Number of evenly spaced video frames to inspect (default: 30).",
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=1.0,
        help="Side length of one chessboard square, in any desired unit (default: 1).",
    )
    parser.add_argument(
        "--rotate",
        choices=("none", "90-cw", "90-ccw", "180"),
        default="none",
        help=(
            "Rotate every frame before calibration. Use this when the video is portrait "
            "but subsequent images will be landscape (default: none)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Calibration .npz path (default: <video stem>_calibration.npz).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open Matplotlib windows (useful for headless runs).",
    )
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be at least 1")
    if args.square_size <= 0:
        parser.error("--square-size must be positive")
    return args


def sampled_indices(frame_count: int, count: int) -> np.ndarray:
    """Return unique, approximately evenly spaced frame indices."""
    if frame_count < 1:
        raise ValueError("video has no readable frame count")
    return np.unique(np.linspace(0, frame_count - 1, min(count, frame_count), dtype=int))


def find_corners(gray: np.ndarray) -> np.ndarray | None:
    """Find and refine (9, 6) corners, using the robust SB detector if present."""
    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(
            gray, PATTERN_SIZE, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        )
        if found:
            return corners.astype(np.float32)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, flags)
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


def make_object_points(square_size: float) -> np.ndarray:
    """Create z=0 3-D points for one chessboard observation."""
    points = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    points[:, :2] = np.mgrid[0 : PATTERN_SIZE[0], 0 : PATTERN_SIZE[1]].T.reshape(-1, 2)
    return points * square_size


def rotate_frame(frame: np.ndarray, rotation: str) -> np.ndarray:
    """Return a frame in the orientation used by calibration and later use."""
    rotations = {
        "none": None,
        "90-cw": cv2.ROTATE_90_CLOCKWISE,
        "90-ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
        "180": cv2.ROTATE_180,
    }
    rotation_code = rotations[rotation]
    return frame if rotation_code is None else cv2.rotate(frame, rotation_code)


def plot_frames(
    frames: Iterable[np.ndarray], titles: Iterable[str], heading: str, max_columns: int = 5
) -> None:
    """Show every frame in one compact pyplot figure."""
    frames, titles = list(frames), list(titles)
    columns = min(max_columns, len(frames))
    rows = int(np.ceil(len(frames) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3 * rows), squeeze=False)
    figure.suptitle(heading)
    for axis, frame, title in zip(axes.flat, frames, titles):
        axis.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        axis.set_title(title)
        axis.axis("off")
    for axis in axes.flat[len(frames) :]:
        axis.axis("off")
    figure.tight_layout()


def main() -> None:
    args = parse_args()
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sampled_indices(frame_count, args.samples)
    object_template = make_object_points(args.square_size)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    detected_frames: list[np.ndarray] = []
    detected_indices: list[int] = []
    image_size: tuple[int, int] | None = None

    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            print(f"Skipping unreadable frame {index}.")
            continue
        frame = rotate_frame(frame, args.rotate)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = find_corners(gray)
        if corners is None:
            continue
        image_size = (frame.shape[1], frame.shape[0])
        annotated = frame.copy()
        cv2.drawChessboardCorners(annotated, PATTERN_SIZE, corners, True)
        object_points.append(object_template.copy())
        image_points.append(corners)
        detected_frames.append(annotated)
        detected_indices.append(int(index))

    capture.release()
    print(
        f"Detected the {PATTERN_SIZE} chessboard in "
        f"{len(detected_frames)}/{len(indices)} sampled frames (rotation: {args.rotate})."
    )
    if not detected_frames or image_size is None:
        raise RuntimeError("No chessboard corners were found. Check the pattern size and video.")
    if len(detected_frames) < 3:
        raise RuntimeError("Need at least three valid chessboard views for calibration.")

    titles = [f"frame {index}" for index in detected_indices]
    if not args.no_show:
        plot_frames(detected_frames, titles, "Detected chessboard corners")

    rms, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, distortion, image_size, alpha=1, newImgSize=image_size
    )
    output = args.output or args.video.with_name(f"{args.video.stem}_calibration.npz")
    np.savez(
        output,
        rms_reprojection_error=rms,
        image_size=np.asarray(image_size),
        pattern_size=np.asarray(PATTERN_SIZE),
        square_size=args.square_size,
        frame_rotation=args.rotate,
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion,
        rotation_vectors=np.asarray(rvecs),
        translation_vectors=np.asarray(tvecs),
        new_camera_matrix=new_camera_matrix,
        valid_roi=np.asarray(roi),
    )

    print(f"RMS reprojection error: {rms:.4f} pixels")
    print("camera_matrix (K):\n", camera_matrix)
    print("distortion_coefficients:\n", distortion.ravel())
    print("new_camera_matrix:\n", new_camera_matrix)
    print("valid_roi (x, y, width, height):", roi)
    print(f"Saved calibration matrices to {output}")

    # Demo: undistort each calibration frame using the matrices just computed.
    undistorted = [
        cv2.undistort(frame, camera_matrix, distortion, None, new_camera_matrix)
        for frame in detected_frames
    ]
    if not args.no_show:
        plot_frames(undistorted, titles, "Calibration frames after undistortion")
        plt.show()

    # Minimal reuse example for a separate image:
    # image = cv2.imread("another_frame.png")
    # corrected = cv2.undistort(image, camera_matrix, distortion, None, new_camera_matrix)
    # cv2.imwrite("another_frame_undistorted.png", corrected)


if __name__ == "__main__":
    main()
