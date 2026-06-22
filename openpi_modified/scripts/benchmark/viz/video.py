"""Value curve video generation — split-screen camera | real-time curve plot.

Adapted from openpi_modified/openpi_reward branch test_rl.py
create_value_curve_video(), with performance optimization using canvas buffer
instead of per-frame PNG encode/decode.
"""

from __future__ import annotations

import os

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from scripts.benchmark.viz.image_utils import to_uint8_array
from scripts.benchmark.viz.status_bar import draw_status_bar


def create_value_curve_video(
    images_head: np.ndarray,
    pred_values: np.ndarray,
    gt_values: np.ndarray,
    output_path: str,
    images_left: np.ndarray | None = None,
    images_right: np.ndarray | None = None,
    fps: int = 30,
    cam_width: int = 640,
    curve_width: int = 448,
    value_range: tuple[float, float] = (-1.0, 0.0),
    camera_layout: str = "base",
) -> str | None:
    """Create a split-screen video with camera images and real-time value curves.

    Left side: camera image(s) with status bar.
    Right side: matplotlib value curve updating per frame.

    Args:
        images_head: (T, H, W, C) head camera images in [-1, 1].
        pred_values: (T,) predicted values.
        gt_values: (T,) ground truth values.
        output_path: Path to save MP4 video.
        images_left: Optional (T, H, W, C) left wrist camera images.
        images_right: Optional (T, H, W, C) right wrist camera images.
        fps: Video frame rate.
        cam_width: Width to resize camera images to.
        curve_width: Width of curve plot area.
        value_range: (min, max) y-axis range for value curve.
        camera_layout: 'base' for single camera, 'triple' for three cameras.

    Returns:
        Path to generated video, or None if no images.
    """
    pred_values = np.asarray(pred_values).reshape(-1)
    gt_values = np.asarray(gt_values).reshape(-1)
    T = min(len(images_head), len(pred_values), len(gt_values))
    if T == 0:
        return None
    images_head = images_head[:T]
    pred_values = pred_values[:T]
    gt_values = gt_values[:T]

    cam_h = cam_width  # square for base camera
    status_h = 36
    curve_h = cam_h + status_h
    total_width = cam_width + curve_width
    total_height = cam_h + status_h

    mse_values = (pred_values - gt_values) ** 2

    # Setup matplotlib figure (reuse for efficiency)
    fig_width = curve_width / 100
    fig_height = curve_h / 100
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.12, top=0.9)

    max_time_sec = T / fps
    ax.set_xlim(0, max_time_sec)
    ax.set_ylim(value_range)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Value", fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.tick_params(labelsize=7)

    (line_pred,) = ax.plot([], [], "--", color="#d62728", label="Pred", linewidth=1.2)
    (line_gt,) = ax.plot([], [], "-", color="#1f77b4", label="GT", linewidth=1.2)
    (point_pred,) = ax.plot([], [], "o", color="#d62728", markersize=3)
    (point_gt,) = ax.plot([], [], "o", color="#1f77b4", markersize=3)
    ax.legend(loc="upper right", fontsize=7)

    frames = []
    skip = 2 if T > 300 else 1

    for t in range(T):
        if t % skip != 0 and t != T - 1:
            continue

        # Camera image
        img_head_uint8 = to_uint8_array(images_head[t])
        img_head_pil = Image.fromarray(img_head_uint8).resize((cam_width, cam_h))

        if camera_layout == "triple" and images_left is not None and images_right is not None:
            third_w = cam_width // 3
            img_l = Image.fromarray(to_uint8_array(images_left[t])).resize((third_w, cam_h))
            img_h = img_head_pil.resize((third_w, cam_h))
            img_r = Image.fromarray(to_uint8_array(images_right[t])).resize((third_w, cam_h))
            cam_img = Image.new("RGB", (cam_width, cam_h))
            cam_img.paste(img_l, (0, 0))
            cam_img.paste(img_h, (third_w, 0))
            cam_img.paste(img_r, (2 * third_w, 0))
        else:
            cam_img = img_head_pil

        # Status bar
        current_mse = float(mse_values[t]) if t < len(mse_values) else None
        status_bar = draw_status_bar(cam_img.width, t, float(pred_values[t]), float(gt_values[t]), current_mse)

        cam_with_status = Image.new("RGB", (cam_img.width, cam_img.height + status_h))
        cam_with_status.paste(status_bar, (0, 0))
        cam_with_status.paste(cam_img, (0, status_h))

        # Update curve
        time_sec = np.arange(t + 1) / fps
        line_pred.set_data(time_sec, pred_values[: t + 1])
        line_gt.set_data(time_sec, gt_values[: t + 1])
        point_pred.set_data([time_sec[-1]], [pred_values[t]])
        point_gt.set_data([time_sec[-1]], [gt_values[t]])

        # Render curve to image via canvas buffer (faster than savefig+BytesIO)
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        curve_arr = np.asarray(buf)[:, :, :3]  # RGBA -> RGB
        curve_img = Image.fromarray(curve_arr).resize((curve_width, curve_h))

        # Combine
        combined = Image.new("RGB", (total_width, total_height))
        combined.paste(cam_with_status, (0, 0))
        combined.paste(curve_img, (cam_width, 0))

        frames.append(np.array(combined))

    plt.close(fig)

    if not frames:
        return None

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Write video
    try:
        import imageio.v3 as iio

        iio.imwrite(output_path, frames, fps=fps, codec="libx264", quality=8)
    except Exception:
        try:
            import cv2

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            h, w = frames[0].shape[:2]
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
        except Exception as e:
            raise RuntimeError(
                f"Failed to write video to {output_path}. " f"Install imageio-ffmpeg or opencv-python. Error: {e}"
            ) from e

    return output_path
