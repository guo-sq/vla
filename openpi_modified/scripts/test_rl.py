"""Minimal JAX-based evaluation script inspired by `scripts/train.py`.

Behaviors:
- Runs model parallel inference offline on user-provided dataset.

Usage (example):
  python scripts/test_rl.py \
    --ckpt_dir checkpoints/pi05_base_finetune_box_value_stage2/pi05_base_finetune_box_value_0311_good_bad_exp.0312_0000/5000 \
    --config_name src/openpi/configs/cfg_pi05_base_finetune_box_value_stage2.py \
    --dataset_root /mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/fold_box/ \
    --repo_id fold_box_scratch_infer.all.6000s.20260311.batch.1 \
    --vis_prefix test_value_model
"""

import dataclasses
import logging
import os
from pathlib import Path
import sys
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from openpi.shared import nnx_utils
import openpi.shared.normalize as _normalize
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader_rl as _data_loader
import openpi.training.sharding as sharding

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from scripts.train import init_train_state  # noqa: E402

ENABLE_BIMANUAL = True
try:
    import bimanual
except ImportError:
    ENABLE_BIMANUAL = False
    print("bimanual module not found; kinematic functions will be disabled.")


def init_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def visualize_pred_vs_gt(pred_seq, gt_seq, filename: str | None = None):
    """Visualize prediction vs GT for 14 dimensions over time.

    Args:
        pred_seq: array-like shape (T, action_dim) predicted sequence for one example.
        gt_seq: array-like shape (T, action_dim) ground-truth sequence for one example.
        filename: optional filename to save the figure. If None, uses
            `vis_pred_vs_gt_{timestamp}.png`.
    """
    pred = np.asarray(pred_seq)
    gt = np.asarray(gt_seq)
    if pred.ndim != 2 or gt.ndim != 2 or pred.shape[1] != gt.shape[1]:
        raise ValueError("pred_seq and gt_seq must have shape (T, action_dim)")
    if pred.shape[0] != gt.shape[0]:
        raise ValueError("pred_seq and gt_seq must have the same temporal length T")

    t_len = pred.shape[0]
    n = pred.shape[1]
    rows = 2
    cols = n // rows

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.flatten()

    x = list(range(t_len))
    for i in range(n):
        ax = axes[i]
        ax.plot(x, gt[:, i], label="GT", color="#1f77b4", linewidth=1.2)
        ax.plot(x, pred[:, i], label="Pred", color="#d62728", linestyle="--", linewidth=1.2)
        ax.set_title(f"Dim {i}")
        ax.set_xlabel("t")
        ax.grid(visible=True, linestyle="--", alpha=0.4)
        if i == 0:
            ax.legend()

    # Hide any unused axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    print(f"Saved visualization to {filename}")
    plt.close(fig)


def _to_pil_image(single_img):
    """Convert a single image array in [-1,1] to a PIL RGB image (uint8).

    Accepts JAX device arrays or numpy arrays. Expects HWC or HW.
    """
    import jax

    a = single_img
    a = jax.device_get(a) if hasattr(a, "device_buffer") or hasattr(a, "device") else np.asarray(a)
    # Convert from [-1,1] to [0,255]
    a = (a + 1.0) * 127.5
    a = np.clip(a, 0, 255).astype(np.uint8)

    # Single-channel -> convert to RGB
    if a.ndim == 2:
        return Image.fromarray(a).convert("RGB")
    if a.ndim == 3 and a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
        return Image.fromarray(a).convert("RGB")
    return Image.fromarray(a).convert("RGB")


def save_composite_image(img_left, img_head, img_right, pred_value, gt_value, vis_dir, batch_idx):
    """Stitch left|head|right images horizontally, add a title 'pred vs gt', and save to `vis_dir`.

    args:
      img_*: single-image arrays (H,W,C) or device arrays in range [-1,1]
      pred_value, gt_value: scalars or arrays (we'll squeeze to scalar for title)
      vis_dir: output directory
      batch_idx: integer batch index used in filename
    """
    os.makedirs(vis_dir, exist_ok=True)
    pil_left = _to_pil_image(img_left)
    pil_head = _to_pil_image(img_head)
    pil_right = _to_pil_image(img_right)

    # Make heights equal by resizing to the max height while preserving aspect ratio
    heights = [pil_left.height, pil_head.height, pil_right.height]
    max_h = max(heights)
    if any(h != max_h for h in heights):

        def _resize_to_h(img, new_h):
            w = int(img.width * (new_h / img.height))
            return img.resize((w, new_h))

        pil_left = _resize_to_h(pil_left, max_h)
        pil_head = _resize_to_h(pil_head, max_h)
        pil_right = _resize_to_h(pil_right, max_h)

    total_w = pil_left.width + pil_head.width + pil_right.width
    # Reserve a small top margin for the title
    title_h = 28
    composite = Image.new("RGB", (total_w, max_h + title_h), (255, 255, 255))

    x = 0
    composite.paste(pil_left, (x, title_h))
    x += pil_left.width
    composite.paste(pil_head, (x, title_h))
    x += pil_head.width
    composite.paste(pil_right, (x, title_h))

    # Draw the title text
    from PIL import ImageDraw

    draw = ImageDraw.Draw(composite)
    try:
        p = float(np.asarray(pred_value).squeeze())
        if gt_value is not None:
            g = float(np.asarray(gt_value).squeeze())
            title = f"pred {p:.4f} vs gt {g:.4f}"
        else:
            title = f"pred {p:.4f}"
    except Exception:
        title = f"pred {pred_value}" if gt_value is None else f"pred {pred_value} vs gt {gt_value}"

    draw.text((8, 6), title, fill=(0, 0, 0))

    out_path = os.path.join(vis_dir, f"batch{batch_idx}_composite.png")
    composite.save(out_path)
    print(f"Saved composite image to {out_path}")


def vis_batch(
    images,
    vis_left,
    vis_right,
    vis_dir: str = "./open_loop_vis_rl",
    preds_arr: np.ndarray | None = None,
    gts_arr: np.ndarray | None = None,
    filename: str | None = None,
    vis_pic_num: int | None = None,
    value_mask: np.ndarray | None = None,
    advantage_horizon: int = 50,
    episode_length: int = 3310,
):
    """Visualize a list/array of head images in a single horizontal row and save.

    If `preds_arr` and `gts_arr` are provided, a comparison plot is rendered
    and appended below the stitched images into a single output image.

    Args:
        images: iterable of single-image arrays (H,W,C) or device arrays in [-1,1].
        vis_left: left-camera images corresponding to each frame in *images*.
        vis_right: right-camera images corresponding to each frame in *images*.
        vis_dir: directory to save the output image into.
        preds_arr: (N, ...) numpy array of predictions to plot (flattened along time).
        gts_arr: (N, ...) numpy array of ground-truths matching preds_arr.
        filename: optional filename; if None uses timestamped name.
        vis_pic_num: number of evenly-spaced frames to display.
        value_mask: optional boolean mask indicating valid value entries.
        advantage_horizon: horizon window for advantage computation.
        episode_length: total episode length used to normalise the advantage baseline.
    """
    from time import time

    pil_images = []
    total_pic_num = len(images)
    vis_gap = (total_pic_num - 1) // (vis_pic_num - 1)
    images = images[::vis_gap]
    vis_left = vis_left[::vis_gap]
    vis_right = vis_right[::vis_gap]

    for img in images:
        pil = _to_pil_image(img)
        pil_images.append(pil)

    if not pil_images:
        print("No images to visualize in vis_batch.")
        return

    # Resize all images to the same height (max height) while preserving aspect ratio
    heights = [p.height for p in pil_images]
    max_h = max(heights)
    resized = []
    for p in pil_images:
        if p.height != max_h:
            w = int(p.width * (max_h / p.height))
            resized.append(p.resize((w, max_h)))
        else:
            resized.append(p)

    total_w = sum(p.width for p in resized)
    out = Image.new("RGB", (total_w, max_h), (255, 255, 255))
    x = 0
    for p in resized:
        out.paste(p, (x, 0))
        x += p.width

    # Build middle row from vis_left and vis_right if provided.
    mid_row = None
    if vis_left is not None or vis_right is not None:
        # Create stitched left row and right row (they may be lists or arrays)
        def _make_row(imgs):
            pil = [_to_pil_image(imgs[i]) for i in range(len(imgs))]
            # normalize heights to max of the row
            h_max = max(p.height for p in pil)
            resized_row = [
                (p.resize((int(p.width * (h_max / p.height)), h_max)) if p.height != h_max else p) for p in pil
            ]
            w_row = sum(p.width for p in resized_row)
            row_img = Image.new("RGB", (w_row, h_max), (255, 255, 255))
            xx = 0
            for p in resized_row:
                row_img.paste(p, (xx, 0))
                xx += p.width
            return row_img

        left_row = _make_row(vis_left) if vis_left is not None else None
        right_row = _make_row(vis_right) if vis_right is not None else None

        # If both sides are present, build an interleaved row: left0, right0, left1, right1, ...
        if left_row is not None and right_row is not None:
            # build interleaved list of images (preserve original order per side)
            left_list = list(vis_left)
            right_list = list(vis_right)
            inter = []
            n = max(len(left_list), len(right_list))
            for idx in range(n):
                if idx < len(left_list):
                    inter.append(left_list[idx])
                if idx < len(right_list):
                    inter.append(right_list[idx])

            inter_row = _make_row(inter)

            # target mid width equals out.width, target mid height = half of head height
            target_w = out.width
            target_h = max(1, out.height // 2)
            # Resize to exact layout size (may distort aspect ratio slightly)
            inter_row = inter_row.resize((target_w, target_h))
            mid_row = inter_row

    # If preds/gts provided, render a matplotlib figure and append vertically
    plot_img = None
    resid_img = None
    if preds_arr is not None:
        try:
            import io

            # Flatten to 1D time series if necessary
            p = np.asarray(preds_arr).reshape(-1)

            # Create a compact matplotlib figure
            fig, ax = plt.subplots(1, 1, figsize=(max(6, out.width / 100), 2.5))
            t = np.arange(len(p)) / 30.0
            if gts_arr is not None:
                g = np.asarray(gts_arr).reshape(-1)
                ax.plot(t, g, label="GT", color="#1f77b4", linewidth=1.0)
            ax.plot(t, p, label="Pred", color="#d62728", linestyle="--", linewidth=1.0)

            # plot decreasing part
            if value_mask is not None and value_mask.sum() != 0:
                value_mask = value_mask.reshape(-1)
                de_t = t[value_mask]
                de_pred = p[value_mask]
                ax.plot(
                    de_t,
                    de_pred,
                    label="Pred (laggy / decreasing)",
                    color="#d62728",
                    marker="o",
                    linestyle="",
                    markersize=4,
                )

            ax.set_xlabel("t")
            ax.set_ylabel("value")
            ax.grid(visible=True, linestyle="--", alpha=0.4)
            ax.legend()
            fig.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150)
            plt.close(fig)
            buf.seek(0)
            plot_img = Image.open(buf).convert("RGB")

            # advantage plot
            resid2 = -advantage_horizon / episode_length + p[advantage_horizon:] - p[:-advantage_horizon]
            fig2, ax2 = plt.subplots(1, 1, figsize=(max(6, out.width / 100), 2.0))
            ax2.plot(t[:-advantage_horizon], resid2, color="#e32817", linewidth=1.0)
            ax2.set_xlabel("t")
            ax2.set_ylabel("advantage")
            ax2.grid(visible=True, linestyle="--", alpha=0.4)
            ax2.legend()
            fig2.tight_layout()
            buf2 = io.BytesIO()
            fig2.savefig(buf2, format="png", dpi=150)
            plt.close(fig2)
            buf2.seek(0)
            resid_img = Image.open(buf2).convert("RGB")

        except (ValueError, RuntimeError) as e:
            logging.warning("Could not create preds/gts plot: %s", e)

    # If we have a plot image, resize it to match the stitched width and append
    if plot_img is not None:
        # Resize plot width to match top stitched width, keep aspect ratio
        pw = out.width
        ph = int(plot_img.height * (pw / plot_img.width))
        plot_img = plot_img.resize((pw, ph))
        # Advantage
        rh = int(resid_img.height * (pw / resid_img.width))
        resid_img = resid_img.resize((pw, rh))
        # Calculate total height: top out + optional mid_row + plot
        total_h = (
            out.height
            + (mid_row.height if mid_row is not None else 0)
            + plot_img.height
            + (resid_img.height if resid_img is not None else 0)
        )
        combined = Image.new("RGB", (out.width, total_h), (255, 255, 255))
        y = 0
        combined.paste(out, (0, y))
        y += out.height
        if mid_row is not None:
            combined.paste(mid_row, (0, y))
            y += mid_row.height
        combined.paste(plot_img, (0, y))
        y += plot_img.height
        combined.paste(resid_img, (0, y))
        final_img = combined
    elif mid_row is not None:
        # append only mid_row under out
        combined = Image.new("RGB", (out.width, out.height + mid_row.height), (255, 255, 255))
        combined.paste(out, (0, 0))
        combined.paste(mid_row, (0, out.height))
        final_img = combined
    else:
        final_img = out

    if filename is None:
        filename = f"vis_heads_{int(time())}.png"
    out_path = os.path.join(vis_dir, filename)
    os.makedirs(vis_dir, exist_ok=True)
    final_img.save(out_path)
    print(f"Saved vis batch image to {out_path}")


def vis_arm_states(
    left_states,
    right_states,
    vis_dir: str = "./open_loop_vis_rl",
    filename: str | None = None,
):
    """Visualize left and right arm states.

    Plots 3D trajectories and saves an image.
    Left arm is drawn in red, right arm in blue.

    Args:
        left_states: array-like shape (T, 3) or (T, >=3). XYZ in first 3 dims.
        right_states: array-like shape (T, 3) or (T, >=3). XYZ in first 3 dims.
        vis_dir: output directory to save the image.
        filename: optional filename; if None, uses timestamped name.
    """
    import os as _os
    import time as _time

    import matplotlib.pyplot as plt
    import numpy as np

    # Convert to numpy
    left_arr = np.asarray(left_states)
    right_arr = np.asarray(right_states)

    # Ensure two-dimensional arrays
    if left_arr.ndim == 1:
        left_arr = left_arr.reshape(-1, left_arr.shape[0]) if left_arr.size % 3 != 0 else left_arr.reshape(-1, 3)
    if right_arr.ndim == 1:
        right_arr = right_arr.reshape(-1, right_arr.shape[0]) if right_arr.size % 3 != 0 else right_arr.reshape(-1, 3)

    # If arrays have more than 3 dims per timestep, take first 3 as XYZ
    if left_arr.ndim == 2 and left_arr.shape[1] >= 3:
        l_xyz = left_arr[:, :3]
    else:
        raise ValueError("left_states must be shape (T,>=3) or (T,3)")
    if right_arr.ndim == 2 and right_arr.shape[1] >= 3:
        r_xyz = right_arr[:, :3]
    else:
        raise ValueError("right_states must be shape (T,>=3) or (T,3)")

    # Create a 3D plot of the trajectories and mark start/end
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Plot trajectories
    if l_xyz.shape[0] > 0:
        ax.plot(
            l_xyz[:, 0],
            l_xyz[:, 1],
            l_xyz[:, 2],
            color="red",
            label="left",
            linewidth=1.5,
        )
        # start/end markers
        ax.scatter(l_xyz[0, 0], l_xyz[0, 1], l_xyz[0, 2], color="red", marker="o", s=60)
        ax.scatter(l_xyz[-1, 0], l_xyz[-1, 1], l_xyz[-1, 2], color="red", marker="X", s=60)
        # annotate
        ax.text(l_xyz[0, 0], l_xyz[0, 1], l_xyz[0, 2], "L start", color="red")
        ax.text(l_xyz[-1, 0], l_xyz[-1, 1], l_xyz[-1, 2], "L end", color="red")

    if r_xyz.shape[0] > 0:
        ax.plot(
            r_xyz[:, 0],
            r_xyz[:, 1],
            r_xyz[:, 2],
            color="blue",
            label="right",
            linewidth=1.5,
        )
        ax.scatter(r_xyz[0, 0], r_xyz[0, 1], r_xyz[0, 2], color="blue", marker="o", s=60)
        ax.scatter(r_xyz[-1, 0], r_xyz[-1, 1], r_xyz[-1, 2], color="blue", marker="X", s=60)
        ax.text(r_xyz[0, 0], r_xyz[0, 1], r_xyz[0, 2], "R start", color="blue")
        ax.text(r_xyz[-1, 0], r_xyz[-1, 1], r_xyz[-1, 2], "R end", color="blue")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.grid(visible=True, linestyle="--", alpha=0.4)

    # Set equal aspect ratio for 3D plot
    def _set_axes_equal(ax3d):
        x_limits = ax3d.get_xlim3d()
        y_limits = ax3d.get_ylim3d()
        z_limits = ax3d.get_zlim3d()
        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])
        max_range = max(x_range, y_range, z_range)
        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)
        ax3d.set_xlim3d(x_middle - max_range / 2, x_middle + max_range / 2)
        ax3d.set_ylim3d(y_middle - max_range / 2, y_middle + max_range / 2)
        ax3d.set_zlim3d(z_middle - max_range / 2, z_middle + max_range / 2)

    _set_axes_equal(ax)

    fig.suptitle("3D Arm State Trajectories (left=red, right=blue)")

    if filename is None:
        filename = f"arm_states_3d_{int(_time.time())}.png"
    _os.makedirs(vis_dir, exist_ok=True)
    out_path = _os.path.join(vis_dir, filename)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved arm states visualization to {out_path}")


def save_episode_pred(
    repo_id,
    current_episode,
    frame_idx_i_np,
    pred_value_i_np,
    out_dir: Path | None = None,
):
    """Save per-frame predictions for an episode.

    Only saves the predicted value column, with the same length as the original parquet.
    """
    import os as _os

    import numpy as np
    import pandas as pd

    read_dir = out_dir / "data" / "chunk-000"
    chunk_out_dir = out_dir / "value_pred" / "chunk-000"

    _os.makedirs(chunk_out_dir, exist_ok=True)

    frame_idx = np.asarray(frame_idx_i_np).squeeze(1)
    pred_value = np.asarray(pred_value_i_np).squeeze(1)

    filename = f"episode_{str(current_episode).zfill(6)}.parquet"
    read_path = read_dir / filename
    out_path = chunk_out_dir / filename

    # Read original parquet to get the length
    df_orig = pd.read_parquet(read_path)
    total_len = len(df_orig)

    # Create pred_value_all array with same length as original parquet
    pred_value_all = np.zeros(total_len, dtype=np.float32)
    pred_value_is_valid = np.zeros(total_len, dtype=bool)
    pred_value_all[frame_idx] = pred_value
    pred_value_is_valid[frame_idx] = True

    # Build new DataFrame with only pred_value column
    df_out = pd.DataFrame({"pred_value": pred_value_all, "value_is_valid": pred_value_is_valid})

    # Save to parquet
    df_out.to_parquet(out_path, index=False)
    print(f"Saved episode predictions to {out_path} (length: {total_len})")


class FindGap:
    def __init__(self):
        self.last_frame_index_np = None
        self.threshold = 2

    def __call__(self, gt_value_np, pred_value_np, frame_index_np, *, debug=False):
        """Find the first index where the gap between consecutive elements exceeds a threshold."""
        assert len(gt_value_np) == len(pred_value_np)
        if self.last_frame_index_np is not None and abs(frame_index_np[0] - self.last_frame_index_np) > self.threshold:
            self.last_frame_index_np = frame_index_np[-1]
            return 0
        self.last_frame_index_np = frame_index_np[-1]
        for i in range(1, len(frame_index_np)):
            if abs(frame_index_np[i] - frame_index_np[i - 1]) > self.threshold:
                print(f"frame_gap = {frame_index_np[i]} - {frame_index_np[i - 1]}, ")
                return i
        return None


def find_idx_range(start_idx, end_idx, pred_value_i_np):
    cur_pred_value = pred_value_i_np[start_idx:end_idx]

    mask = (cur_pred_value >= -0.15) & (cur_pred_value <= -0.03)
    indices = np.where(mask.flatten())[0]
    if len(indices) == 0:
        return None, None
    diffs = np.diff(indices)
    breaks = np.where(diffs > 1)[0]
    last_section_indices = indices[breaks[-1] + 1 :] if len(breaks) > 0 else indices
    end_015_idx = last_section_indices[0]
    end_003_idx = last_section_indices[-1]

    return start_idx + end_015_idx, start_idx + end_003_idx


def calc_total_episode_num(dataset_root, repo_id):
    dataset_root_path = Path(dataset_root)
    repo_id_path = dataset_root_path / repo_id / "data" / "chunk-000"
    file_count = sum(1 for x in repo_id_path.iterdir() if x.is_file())
    print(f"Total episodes: {file_count}")
    return file_count


def main(
    checkpoint_dir: str,
    dataset_root: str,
    num_batches: int = 2,
    norm_stats_path: str | None = None,
    batch_size: int | None = None,
    vis_dir: str = "./open_loop_vis",
    repo_id: str | None = None,
    episode_fail: int = 0,
    config_name: str | None = None,
    total_episodes: int | None = None,
    *,
    enable_save_parquet: bool = False,
    enable_vis_arm_states: bool = False,
    segmented: bool = False,
):
    t0 = time.perf_counter()
    init_logging()

    checkpoint_dir = Path(checkpoint_dir)
    dataset_root = Path(dataset_root)
    episode_fail = [episode_fail]  # to list
    repo_path = dataset_root / repo_id

    checkpoint_base_dir = checkpoint_dir.parent.parent.parent
    config = _config.get_config(config_name)
    exp_name = checkpoint_dir.parent.name
    step = checkpoint_dir.name

    # Override checkpoint base dir / exp name and optionally batch_size
    replace_kwargs = {
        "checkpoint_base_dir": str(checkpoint_base_dir),
        "exp_name": exp_name,
    }
    if batch_size is not None:
        replace_kwargs["batch_size"] = batch_size
    config = dataclasses.replace(config, **replace_kwargs)
    print(f"Using config: {config.name}, exp: {config.exp_name}")

    t1 = time.perf_counter()
    print(f"Timing: config loading took {t1 - t0:.3f}s")

    t0 = time.perf_counter()
    try:
        data_cfg = config.data.create(config.assets_dirs, config.model)
        replace_data_kwargs = {
            "root_dir": str(dataset_root),
            "episode_fail": episode_fail,
        }
        if repo_id is not None:
            if isinstance(repo_id, str):
                repo_id = [repo_id]
            assert isinstance(repo_id, list)
            replace_data_kwargs["repo_id"] = repo_id

        data_cfg = dataclasses.replace(data_cfg, **replace_data_kwargs)

        # Test mode: don't require segment_values.json
        if hasattr(data_cfg, "base_config") and data_cfg.base_config is not None:
            vnc = getattr(data_cfg.base_config, "value_net_cfg", None)
        else:
            vnc = getattr(data_cfg, "value_net_cfg", None)
        if vnc is not None:
            vnc["require_segment_file"] = False

        norm_stats_path = checkpoint_dir / "assets" / data_cfg.asset_id
        if norm_stats_path is not None:
            norm_stats_file = norm_stats_path
            if norm_stats_file.exists():
                loaded = _normalize.load(norm_stats_file)
                data_cfg = dataclasses.replace(data_cfg, norm_stats=loaded)
                print(f"Loaded norm_stats from {norm_stats_file}")
            else:
                logging.warning(f"Provided norm_stats_path does not exist: {norm_stats_file}")
                raise FileNotFoundError(f"Provided norm_stats_path does not exist: {norm_stats_file}")

        class _SimpleFactory:
            def __init__(self, data_cfg):
                self._data_cfg = data_cfg
                self.episode_fail = data_cfg.episode_fail
                self.dataset_length = None

            def create(self, assets_dirs, model_config):
                return self._data_cfg

        config = dataclasses.replace(config, data=_SimpleFactory(data_cfg))
    except Exception:
        logging.warning("Could not patch data factory; proceeding and hope dataset paths are embedded in config.")

    # Create mesh and sharding same as train.py
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

    # Initialize checkpoint manager (resume mode)
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=False,
        resume=True,
    )
    # Create data loader. Use norm stats so inputs/GT align with training preprocessing.
    # Set drop_last=False for inference to ensure all frames are processed, including
    # the last incomplete batch. This fixes the issue where last frames were not labeled.
    data_loader = _data_loader.create_rl_data_loader(
        config,
        sharding=data_sharding,
        shuffle=False,
        num_batches=None,
        skip_norm_stats=False,
        drop_last=False,
    )

    t1 = time.perf_counter()
    print(f"Timing: data loader took {t1 - t0:.3f}s")

    t0 = time.perf_counter()

    rng = jax.random.key(config.seed)
    _, init_rng = jax.random.split(rng)

    # Request shapes for train state while indicating resume=True so init_train_state returns shape placeholders
    train_state_shape, state_sharding = init_train_state(config, init_rng, mesh, resume=True)

    train_state = _checkpoints.restore_state(checkpoint_manager, train_state_shape, data_loader, int(step))

    # Merge model def + params into an executable model
    model = nnx.merge(train_state.model_def, train_state.params)
    model.eval()
    score_observation_jit = nnx_utils.module_jit(model.score_observation)

    print("Model restored and set to eval mode.")

    if segmented:
        _repo_id = repo_id[0] if isinstance(repo_id, list) else repo_id
        seg_file = Path(dataset_root) / _repo_id / "meta" / "segment_values.json"
        if seg_file.exists():
            draw_gt = True
            print(f"--segmented: found {seg_file}, drawing GT from segment boundaries.")
        else:
            draw_gt = False
            print(
                f"--segmented: {seg_file} not found, skipping GT visualization and MSE "
                f"(GT would be fallback per_episode returns, which is misleading)."
            )
    else:
        draw_gt = True

    data_iter = iter(data_loader)

    all_preds = []
    all_gts = []
    all_frame_indices = []
    all_states = []
    batch_mses = []
    vis_batch_head = []
    vis_batch_left = []
    vis_batch_right = []
    current_episode = 0
    action_dim = 14

    t1 = time.perf_counter()
    print(f"Timing: prepare took {t1 - t0:.3f}s")
    find_gap_tool = FindGap()

    for i in range(num_batches):
        t0 = time.perf_counter()
        print(f"Current batch: {i}")

        batch = next(data_iter)
        observation, _, _ = batch
        images = observation.images

        img_head = images["base_0_rgb"]
        img_left_wrist = images["left_wrist_0_rgb"]
        img_right_wrist = images["right_wrist_0_rgb"]
        obs_state = observation.state

        t1 = time.perf_counter()
        print(f"- Timing: get batch data: {t1 - t0:.3f}s")

        t0 = time.perf_counter()
        # build RNG per batch
        rng, subkey = jax.random.split(rng)

        pred_value = score_observation_jit(subkey, observation)
        pred_value = pred_value.block_until_ready()
        pred_value_np = jax.device_get(pred_value)
        gt_value_np = jax.device_get(observation.returns)
        frame_index_np = jax.device_get(observation.frame_index)

        t1 = time.perf_counter()
        print(f"- Timing: sample_actions: {t1 - t0:.3f}s")

        t0 = time.perf_counter()

        # Compute per-batch MSE (only meaningful when GT is being drawn)
        mse_batch = float(np.mean((pred_value_np - gt_value_np) ** 2)) if draw_gt else 0.0

        # Save composite visualization (left|head|right) with title showing pred vs gt
        for j in range(0, len(img_head), 3):
            pv = pred_value_np[j] if hasattr(pred_value_np, "__len__") else pred_value_np
            gv = gt_value_np[j] if (draw_gt and hasattr(gt_value_np, "__len__")) else None
            save_composite_image(
                img_left_wrist[j],
                img_head[j],
                img_right_wrist[j],
                pv,
                gv,
                f"{vis_dir}/episode_{current_episode}",
                f"{i}_batch{j}",
            )

        debug = False
        gap = find_gap_tool(gt_value_np, pred_value_np, frame_index_np, debug=debug)
        if gap is not None:
            print(f"- Found gap in GT values at index {gap}, saving vis up to gap.")
            if len(vis_batch_head) > 0:
                vis_batch_head_i_np = np.concatenate(vis_batch_head, axis=0)
                vis_batch_left_i_np = np.concatenate(vis_batch_left, axis=0)
                vis_batch_right_i_np = np.concatenate(vis_batch_right, axis=0)
                pred_value_i_np = np.concatenate(all_preds, axis=0)
                gt_value_i_np = np.concatenate(all_gts, axis=0)
                frame_idx_i_np = np.concatenate(all_frame_indices, axis=0)
                all_states_i_np = np.concatenate(all_states, axis=0)
                vis_batch_head_i_np = np.concatenate([vis_batch_head_i_np, img_head[:gap]], axis=0)
                vis_batch_left_i_np = np.concatenate([vis_batch_left_i_np, img_left_wrist[:gap]], axis=0)
                vis_batch_right_i_np = np.concatenate([vis_batch_right_i_np, img_right_wrist[:gap]], axis=0)
                pred_value_i_np = np.concatenate([pred_value_i_np, pred_value_np[:gap]], axis=0)
                gt_value_i_np = np.concatenate([gt_value_i_np, gt_value_np[:gap]], axis=0)
                frame_idx_i_np = np.concatenate([frame_idx_i_np, frame_index_np[:gap]], axis=0)
                all_states_i_np = np.concatenate([all_states_i_np, obs_state[:gap]], axis=0)
            else:
                vis_batch_head_i_np = img_head[:gap]
                vis_batch_left_i_np = img_left_wrist[:gap]
                vis_batch_right_i_np = img_right_wrist[:gap]
                pred_value_i_np = pred_value_np[:gap]
                gt_value_i_np = gt_value_np[:gap]
                frame_idx_i_np = frame_index_np[:gap]
                all_states_i_np = obs_state[:gap]

            # Save episode predictions into the checkpoint's test_results directory
            if enable_save_parquet:
                save_episode_pred(
                    repo_id,
                    current_episode,
                    frame_idx_i_np,
                    pred_value_i_np,
                    out_dir=repo_path,
                )

            vis_batch(
                vis_batch_head_i_np,
                vis_batch_left_i_np,
                vis_batch_right_i_np,
                vis_dir=vis_dir if isinstance(vis_dir, str) else str(vis_dir),
                preds_arr=pred_value_i_np,
                gts_arr=gt_value_i_np if draw_gt else None,
                filename=f"vis_episode_{current_episode}_value.png",
                vis_pic_num=5,
                value_mask=None,  # value_mask is computed elsewhere and passed when available
            )

            current_episode += 1

            if current_episode >= total_episodes:
                print(f"Reached total episodes: {total_episodes}, stopping.")
                break

            vis_batch_head.clear()
            vis_batch_left.clear()
            vis_batch_right.clear()
            all_preds.clear()
            all_gts.clear()
            all_frame_indices.clear()
            all_states.clear()

            vis_batch_head.append(img_head[gap:])
            vis_batch_left.append(img_left_wrist[gap:])
            vis_batch_right.append(img_right_wrist[gap:])
            all_preds.append(pred_value_np[gap:])
            all_gts.append(gt_value_np[gap:])
            all_frame_indices.append(frame_index_np[gap:])
            all_states.append(obs_state[gap:])
        else:
            all_preds.append(pred_value_np)
            all_gts.append(gt_value_np)
            vis_batch_head.append(img_head)
            vis_batch_left.append(img_left_wrist)
            vis_batch_right.append(img_right_wrist)
            all_frame_indices.append(frame_index_np)
            all_states.append(obs_state)

        batch_mses.append(mse_batch)

        t1 = time.perf_counter()
        print(f"- Timing: Infer + post process: {t1 - t0:.3f}s")
        if draw_gt:
            print(f"- MSE: {mse_batch:.6e}")

    jax.clear_caches()
    # Summarize results
    if draw_gt:
        overall_mse = float(np.mean(batch_mses))
        print(f"Overall MSE across {len(batch_mses)} batches: {overall_mse:.6e}")
    preds_arr = np.concatenate(all_preds, axis=0)
    gts_arr = np.concatenate(all_gts, axis=0)

    save_path_root = checkpoint_base_dir / config_name / exp_name / step / "test_results"
    os.makedirs(save_path_root, exist_ok=True)
    np.save(save_path_root / "test_all_preds.npy", preds_arr)
    if draw_gt:
        np.save(save_path_root / "test_all_gts.npy", gts_arr)
        print(f"Saved all preds/gts to [{save_path_root}/test_all_preds.npy] and [{save_path_root}/test_all_gts.npy]")
    else:
        print(f"Saved all preds to [{save_path_root}/test_all_preds.npy]")

    # Visualization
    vis_batch_head_np = np.concatenate(vis_batch_head, axis=0)
    vis_batch_left_np = np.concatenate(vis_batch_left, axis=0)
    vis_batch_right_np = np.concatenate(vis_batch_right, axis=0)
    vis_batch(
        vis_batch_head_np,
        vis_batch_left_np,
        vis_batch_right_np,
        vis_dir,
        preds_arr,
        gts_arr if draw_gt else None,
        filename="vis_episode_final_value.png",
        vis_pic_num=5,
    )

    # Optional: visualize arm states using forward kinematics
    if enable_vis_arm_states and ENABLE_BIMANUAL:
        states_arr = jnp.concatenate(all_states, axis=0)[:, :action_dim]
        left_arm_states = states_arr[:, : action_dim // 2 - 1]  # [len, 6]
        right_arm_states = states_arr[:, action_dim // 2 : action_dim - 1]  # [len, 6]
        left_arm_states_np = np.asarray(left_arm_states)
        right_arm_states_np = np.asarray(right_arm_states)
        left_arm_pos = np.zeros(left_arm_states_np.shape)
        right_arm_pos = np.zeros(right_arm_states_np.shape)

        for i in range(left_arm_states_np.shape[0]):
            left_arm_pos[i] = bimanual.forward_kinematics(left_arm_states_np[i])
            right_arm_pos[i] = bimanual.forward_kinematics(right_arm_states_np[i])

        vis_arm_states(left_arm_pos, right_arm_pos, vis_dir)


if __name__ == "__main__":
    import argparse
    from datetime import datetime
    from datetime import timedelta
    from datetime import timezone

    beijing_tz = timezone(timedelta(hours=8))
    now = datetime.now(beijing_tz)
    time_str = now.strftime("%m%d_%H%M")

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--config_name", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--vis_prefix", type=str, required=True)
    parser.add_argument("--repo_id", type=str, default=None)
    parser.add_argument("--episode_fail", type=int, default=0)
    parser.add_argument("--num_batches", type=int, default=1000)
    parser.add_argument("--max_episode", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--enable_save_parquet", action="store_true", default=False)
    parser.add_argument("--enable_vis_arm_states", action="store_true", default=False)
    parser.add_argument(
        "--segmented",
        action="store_true",
        default=False,
        help="Indicate this checkpoint was trained with the segmented returns_norm_strategy. "
        "When set, the script checks for segment_values.json in the dataset. "
        "If present, draws GT based on the segment boundaries. "
        "If absent, skips GT visualization and MSE (because GT would be misleading "
        "per_episode fallback returns).",
    )

    args = parser.parse_args()

    total_episodes = calc_total_episode_num(args.dataset_root, args.repo_id)
    total_episodes = min(total_episodes, args.max_episode)
    print(f"Settings: enable_save_parquet = {args.enable_save_parquet}, " f"total_episodes = {total_episodes}")

    vis_dir = f"./{args.vis_prefix}_" + time_str + "_" + args.repo_id
    os.makedirs(vis_dir, exist_ok=True)

    main(
        args.ckpt_dir,
        args.dataset_root,
        args.num_batches,
        batch_size=args.batch_size,
        repo_id=args.repo_id,
        vis_dir=vis_dir,
        episode_fail=args.episode_fail,
        config_name=args.config_name,
        total_episodes=total_episodes,
        enable_save_parquet=args.enable_save_parquet,
        enable_vis_arm_states=args.enable_vis_arm_states,
        segmented=args.segmented,
    )
