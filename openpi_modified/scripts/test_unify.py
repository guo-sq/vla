"""Minimal JAX-based evaluation script inspired by `scripts/train.py`.

Behaviors:
- Runs model parallel inference offline on user-provided dataset.
- Saves predictions and GT as .npy files.
- Generates enhanced visualizations: multi-horizon, error heatmap, temporal metrics, smoothness.
- Supports comparing multiple evaluation runs (offline comparison mode).

Usage (example):

    === Inference Mode (default) ===
    Step1: Please modify REPO_ID_DICT in the script.
    Step2: Run the script.
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
        uv run scripts/test_unify.py \
            --ckpt_dir /mnt/oss_models/pretrained_models/pi05_anyverse/cfg_pi0.5_28_dim.all_public_datasets_anyverse_speed_up_20260324/40000 \
            --config_name /mnt/oss_models/pretrained_models/pi05_anyverse/cfg_pi0.5_28_dim.all_public_datasets_anyverse_speed_up_20260324/cfg_pi0.5_28_dim.all_public_datasets_anyverse_speed_up.py \
            --num_batches 10 \
            --batch_size 128 \
            --run_name v1.0.4_all_dataset

    === Compare Mode (offline comparison of multiple runs) ===
    Compare 2 model evaluation results:
        uv run scripts/test_unify.py \
            --compare_paths /mnt/oss_models/pretrained_models/pi05_anyverse/cpt/cfg_cfg_v1.0.0_28_dim.anyverse_20260320/v1.0.0_anyverse_vis,/mnt/oss_models/pretrained_models/pi05_anyverse/cpt/cfg_pi0.5_28_dim.anyverse_20260320/vis/pi05_anyverse \
            --compare_output outputs/compare_debug

    Compare 3+ models with custom labels:
        uv run scripts/test_unify.py \
            --compare_paths outputs/v1.0.0,outputs/v2.0.0,outputs/v3.0.0 \
            --compare_output outputs/compare_all \
            --compare_labels baseline,improved,latest

    Compare mode output structure:
        outputs/<compare_output>/
        ├── <task_type>/<repo_name>/
        │   ├── mse_bar_chart.png              # MSE bar chart comparison
        │   ├── error_heatmap_diff.png         # Error heatmap side-by-side
        │   ├── per_dim_mse_comparison.png     # Per-dimension MSE curves
        │   ├── horizon_trajectory_overlay.png # Trajectory overlay (pred + GT)
        │   └── smoothness_comparison.png      # Smoothness comparison
        ├── overall_mse_comparison.png         # Global overview
        ├── comparison_summary.json            # JSON summary
        ├── comparison_report.md               # Markdown report
        └── comparison_metrics.csv             # CSV for Excel/pandas
"""

import argparse
import csv
import dataclasses
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import json
import logging
import os
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from openpi.shared import nnx_utils
import openpi.shared.normalize as _normalize
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import openpi.transforms as transforms
from scripts.train import init_train_state

DATA_ROOT = "/mnt/"
REPO_ID_DICT = {
    # anyverse datasets
    "anyverse_fold_box": [
        "oss_data/anyverse_human_data_record/arxx5_bimanual/fold_box/fold_pre_breaking_box/record-arxx5_bimanual-fold-box-0204-fast_1",
    ],
    "anyverse_fold_clothes": [
        "oss_data/anyverse/bipiper/fold_clothes/record.clothes.bipiper.v1215.1",
    ],
    "anyverse_insert_tube": [
        "oss_data/anyverse_human_data_record/arxx5_bimanual/insert_tube/grab_and_attach_tube.continuous_5_tubes.150s.1217.batch.1",
    ],
    "anyverse_pack_socks": [
        "oss_data/anyverse_human_data_record/arxx5_bimanual/pack_socks/pack_socks.3_colors.M.continuous_pair_s0s1s2.panjinlong.20260224.batch.1",
    ],
    "anyverse_pick_place": [
        "oss_data/anyverse/bipiper/pick_place/anyverse_pickAndplace_record/record.pick.place.bipiper.v0106.1",
    ],
    "anyverse_pour_water": [
        "oss_data/anyverse_pour_water_record/record.pourwater.bipiper.0120.1",
    ],
    "anyverse_seatbelt": [
        "oss_data/anyverse_human_data_record/arxx5_bimanual/seatbelt/seatbelt.single.hang.baichenglong.20260205.batch.5",
    ],
    # public datasets
    "public_pick_and_place": [
        "oss_data/lerobot/aloha_static_candy",
        "oss_data/robotics-diffusion-transformer/rdt-ft-data/lerobot_data/pick_tomato_to_desk",
    ],
    "public_press_button": [
        "oss_data/IPEC-COMMUNITY/taco_play_lerobot",
        "oss_data/robotics-diffusion-transformer/rdt-ft-data/lerobot_data/press_socket_button",
    ],
    "public_pour": [
        "oss_data/robocoin/RoboCOIN/Cobot_Magic_pour_drink",
        "oss_data/robotics-diffusion-transformer/rdt-ft-data/lerobot_data/pour_water_cup_full",
    ],
    "public_fold": [
        "oss_data/robocoin/RoboCOIN/Cobot_Magic_fold_clothes",
        "oss_data/OpenGalaxea/Galaxea-Open-World-Dataset/lerobot_unzip/Fold_Clothes20250617_001",
    ],
}

ENABLE_BIMANUAL = True
try:
    import bimanual
except ImportError:
    ENABLE_BIMANUAL = False
    print("bimanual module not found; kinematic functions will be disabled.")


def init_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Dimension name utilities
# ---------------------------------------------------------------------------


def _get_action_dim_names(d: int) -> list[str]:
    """Return human-readable names for each action dimension.

    14-dim layout: 6 left joint angles + 1 left gripper + 6 right joint angles + 1 right gripper
    """
    if d == 14:
        return [
            "L_j1",
            "L_j2",
            "L_j3",
            "L_j4",
            "L_j5",
            "L_j6",
            "L_grip",
            "R_j1",
            "R_j2",
            "R_j3",
            "R_j4",
            "R_j5",
            "R_j6",
            "R_grip",
        ]
    if d == 7:
        return ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"]
    return [f"dim_{i}" for i in range(d)]


# ---------------------------------------------------------------------------
# Original visualization helpers (kept for bimanual arm viz)
# ---------------------------------------------------------------------------


def vis_arm_states(
    left_states,
    right_states,
    vis_dir: str = "./open_loop_vis_rl",
    filename: str | None = None,
    subtitle_suffix: str | None = None,
):
    """Visualize left and right arm states as 3D trajectories."""
    left_arr = np.asarray(left_states)
    right_arr = np.asarray(right_states)

    if left_arr.ndim == 1:
        left_arr = left_arr.reshape(-1, left_arr.shape[0]) if left_arr.size % 3 != 0 else left_arr.reshape(-1, 3)
    if right_arr.ndim == 1:
        right_arr = right_arr.reshape(-1, right_arr.shape[0]) if right_arr.size % 3 != 0 else right_arr.reshape(-1, 3)

    if left_arr.ndim == 2 and left_arr.shape[1] >= 3:
        left_xyz = left_arr[:, :3]
    else:
        raise ValueError("left_states must be shape (T,>=3) or (T,3)")
    if right_arr.ndim == 2 and right_arr.shape[1] >= 3:
        right_xyz = right_arr[:, :3]
    else:
        raise ValueError("right_states must be shape (T,>=3) or (T,3)")

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    if left_xyz.shape[0] > 0:
        ax.plot(left_xyz[:, 0], left_xyz[:, 1], left_xyz[:, 2], color="red", label="Pred", linewidth=1.5)
        ax.scatter(left_xyz[0, 0], left_xyz[0, 1], left_xyz[0, 2], color="red", marker="o", s=60)
        ax.scatter(left_xyz[-1, 0], left_xyz[-1, 1], left_xyz[-1, 2], color="red", marker="X", s=60)
        ax.text(left_xyz[0, 0], left_xyz[0, 1], left_xyz[0, 2], "L start", color="red")
        ax.text(left_xyz[-1, 0], left_xyz[-1, 1], left_xyz[-1, 2], "L end", color="red")

    if right_xyz.shape[0] > 0:
        ax.plot(right_xyz[:, 0], right_xyz[:, 1], right_xyz[:, 2], color="blue", label="GT", linewidth=1.5)
        ax.scatter(right_xyz[0, 0], right_xyz[0, 1], right_xyz[0, 2], color="blue", marker="o", s=60)
        ax.scatter(right_xyz[-1, 0], right_xyz[-1, 1], right_xyz[-1, 2], color="blue", marker="X", s=60)
        ax.text(right_xyz[0, 0], right_xyz[0, 1], right_xyz[0, 2], "R start", color="blue")
        ax.text(right_xyz[-1, 0], right_xyz[-1, 1], right_xyz[-1, 2], "R end", color="blue")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.grid(visible=True, linestyle="--", alpha=0.4)

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
    fig.suptitle(f"Arm State Trajectories ({subtitle_suffix})")

    assert filename is not None, "filename must be provided for vis_arm_states"
    os.makedirs(vis_dir, exist_ok=True)
    out_path = os.path.join(vis_dir, filename)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved arm states visualization to {out_path}")


def calculate_total_frames(data_loader: _data_loader.DataLoader) -> int:
    inner = getattr(data_loader, "_data_loader", None)
    torch_loader = inner.torch_loader
    ds = getattr(torch_loader, "dataset", None)
    return len(ds)


def visualize_pred_vs_gt(pred_seq, gt_seq, filename: str | None = None):
    """Visualize prediction vs GT for all dimensions over time (2D line plots)."""
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

    t = np.arange(t_len)  # / 30.0
    for i in range(n):
        ax = axes[i]
        ax.plot(t, gt[:, i], label="GT", color="#1f77b4", linewidth=1.2)
        ax.plot(t, pred[:, i], label="Pred", color="#d62728", linestyle="--", linewidth=1.2)
        ax.set_title(f"Dim {i}")
        ax.set_xlabel("frames")
        ax.grid(visible=True, linestyle="--", alpha=0.4)
        if i == 0:
            ax.legend()

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    print(f"Saved visualization to {filename}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Enhanced visualization functions
# ---------------------------------------------------------------------------


def visualize_multi_horizon(
    preds: np.ndarray,
    gts: np.ndarray,
    output_dir: Path,
    horizons: list[int] | None = None,
    action_dim_names: list[str] | None = None,
):
    """Visualize predictions vs ground truth at multiple horizon steps.

    Args:
        preds: shape (N, action_horizon, action_dim)
        gts: shape (N, action_horizon, action_dim)
    """
    if horizons is None:
        horizons = [0, 10, 20, 30, 40]

    _n, horizon_len, action_dim = preds.shape
    horizons = [h for h in horizons if h < horizon_len]

    multi_horizon_dir = output_dir / "multi_horizon"
    multi_horizon_dir.mkdir(parents=True, exist_ok=True)

    for h in horizons:
        pred_h = preds[:, h, :]
        gt_h = gts[:, h, :]

        fig, axes = plt.subplots(2, (action_dim + 1) // 2, figsize=(4 * ((action_dim + 1) // 2), 6))
        axes = axes.flatten()
        sample_indices = np.arange(len(pred_h))

        for dim in range(action_dim):
            ax = axes[dim]
            ax.scatter(sample_indices, gt_h[:, dim], label="GT", alpha=0.5, s=1, color="blue")
            ax.scatter(sample_indices, pred_h[:, dim], label="Pred", alpha=0.5, s=1, color="red")
            dim_name = action_dim_names[dim] if action_dim_names else f"Dim {dim}"
            ax.set_title(dim_name)
            ax.set_xlabel("Sample")
            ax.set_ylabel("Value")
            ax.grid(visible=True, linestyle="--", alpha=0.3)
            if dim == 0:
                ax.legend()

        for dim in range(action_dim, len(axes)):
            axes[dim].axis("off")

        fig.suptitle(f"Prediction vs GT at Horizon {h}")
        fig.tight_layout()
        fig.savefig(multi_horizon_dir / f"horizon_{h}.png", dpi=150)
        plt.close(fig)

    print(f"Saved multi-horizon visualizations to {multi_horizon_dir}")


def visualize_error_heatmap(
    preds: np.ndarray,
    gts: np.ndarray,
    output_dir: Path,
    action_dim_names: list[str] | None = None,
):
    """Visualize error heatmap (horizon x dimension).

    Args:
        preds: shape (N, action_horizon, action_dim)
        gts: shape (N, action_horizon, action_dim)
    """
    _n, horizon_len, action_dim = preds.shape
    error_map = np.mean((preds - gts) ** 2, axis=0)  # (H, D)

    fig, ax = plt.subplots(figsize=(max(12, horizon_len * 0.3), max(6, action_dim * 0.5)))
    y_labels = action_dim_names or [f"d{d}" for d in range(action_dim)]

    step = max(1, horizon_len // 10)
    x_labels = [f"h{h}" if h % step == 0 else "" for h in range(horizon_len)]

    sns.heatmap(
        error_map.T,
        ax=ax,
        cmap="YlOrRd",
        xticklabels=x_labels,
        yticklabels=y_labels,
        cbar_kws={"label": "MSE"},
    )
    ax.set_xlabel("Horizon Step")
    ax.set_ylabel("Action Dimension")
    ax.set_title("Error Heatmap: MSE per (Horizon, Dimension)")

    fig.tight_layout()
    fig.savefig(output_dir / "error_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"Saved error heatmap to {output_dir / 'error_heatmap.png'}")
    return error_map


def compute_temporal_metrics(
    preds: np.ndarray,
    gts: np.ndarray,
) -> dict:
    """Compute temporal metrics vs horizon (no figure; overlaps with multi-horizon / heatmap).

    14-dim layout: [L_j1..L_j6, L_grip, R_j1..R_j6, R_grip]

    Args:
        preds: shape (N, action_horizon, action_dim)
        gts: shape (N, action_horizon, action_dim)
    """
    _n, _horizon_len, action_dim = preds.shape

    mse_per_horizon = np.mean((preds - gts) ** 2, axis=(0, 2))  # (H,)

    # Arm-aware grouping for 14-dim dual arm (6 joints + 1 gripper per arm)
    if action_dim == 14:
        left_joint_err = np.mean(np.linalg.norm(preds[:, :, 0:6] - gts[:, :, 0:6], axis=-1), axis=0)
        right_joint_err = np.mean(np.linalg.norm(preds[:, :, 7:13] - gts[:, :, 7:13], axis=-1), axis=0)
        left_grip_err = np.mean((preds[:, :, 6] - gts[:, :, 6]) ** 2, axis=0)
        right_grip_err = np.mean((preds[:, :, 13] - gts[:, :, 13]) ** 2, axis=0)
    elif action_dim == 7:
        left_joint_err = np.mean(np.linalg.norm(preds[:, :, 0:6] - gts[:, :, 0:6], axis=-1), axis=0)
        right_joint_err = None
        left_grip_err = np.mean((preds[:, :, 6] - gts[:, :, 6]) ** 2, axis=0)
        right_grip_err = None
    else:
        left_joint_err = None
        right_joint_err = None
        left_grip_err = None
        right_grip_err = None

    result = {"mse_per_horizon": mse_per_horizon.tolist()}
    if left_joint_err is not None:
        result["left_joint_error"] = left_joint_err.tolist()
    if right_joint_err is not None:
        result["right_joint_error"] = right_joint_err.tolist()
    if left_grip_err is not None:
        result["left_gripper_error"] = left_grip_err.tolist()
    if right_grip_err is not None:
        result["right_gripper_error"] = right_grip_err.tolist()
    return result


def visualize_smoothness(
    preds: np.ndarray,
    gts: np.ndarray,
    output_dir: Path,
):
    """Visualize trajectory smoothness using Jerk (3rd derivative).

    Args:
        preds: shape (N, action_horizon, action_dim)
        gts: shape (N, action_horizon, action_dim)
    """
    _n, horizon_len, action_dim = preds.shape
    if horizon_len < 4:
        print("Warning: Horizon too short for smoothness analysis (need >= 4)")
        return None

    pred_jerk = np.diff(preds, n=3, axis=1)  # (N, H-3, D)
    gt_jerk = np.diff(gts, n=3, axis=1)

    pred_jerk_mag = np.mean(np.abs(pred_jerk), axis=1)  # (N, D)
    gt_jerk_mag = np.mean(np.abs(gt_jerk), axis=1)

    pred_smoothness = float(np.mean(pred_jerk_mag**2))
    gt_smoothness = float(np.mean(gt_jerk_mag**2))

    pred_smooth_per_dim = np.mean(pred_jerk_mag**2, axis=0)  # (D,)
    gt_smooth_per_dim = np.mean(gt_jerk_mag**2, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.hist(pred_jerk_mag.flatten(), bins=50, alpha=0.5, label="Pred", color="red", density=True)
    ax.hist(gt_jerk_mag.flatten(), bins=50, alpha=0.5, label="GT", color="blue", density=True)
    ax.set_xlabel("Jerk Magnitude")
    ax.set_ylabel("Density")
    ax.set_title("Jerk Distribution")
    ax.legend()
    ax.grid(visible=True, linestyle="--", alpha=0.3)

    ax = axes[1]
    x = np.arange(action_dim)
    width = 0.35
    ax.bar(x - width / 2, gt_smooth_per_dim, width, label="GT", color="blue", alpha=0.7)
    ax.bar(x + width / 2, pred_smooth_per_dim, width, label="Pred", color="red", alpha=0.7)
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Smoothness Score (Jerk^2)")
    ax.set_title("Smoothness per Dimension")
    ax.legend()
    ax.grid(visible=True, linestyle="--", alpha=0.3)

    ax = axes[2]
    categories = ["GT", "Pred"]
    values = [gt_smoothness, pred_smoothness]
    colors = ["blue", "red"]
    bars = ax.bar(categories, values, color=colors, alpha=0.7)
    ax.set_ylabel("Smoothness Score (Mean Jerk^2)")
    ax.set_title("Overall Smoothness Comparison")
    for bar, val in zip(bars, values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.4f}", ha="center", va="bottom", fontsize=10
        )
    ax.grid(visible=True, linestyle="--", alpha=0.3)

    fig.suptitle("Trajectory Smoothness Analysis (Jerk)")
    fig.tight_layout()
    fig.savefig(output_dir / "smoothness_analysis.png", dpi=150)
    plt.close(fig)
    print(f"Saved smoothness analysis to {output_dir / 'smoothness_analysis.png'}")

    return {
        "pred_smoothness": pred_smoothness,
        "gt_smoothness": gt_smoothness,
        "smoothness_ratio": pred_smoothness / gt_smoothness if gt_smoothness > 0 else None,
        "pred_smooth_per_dim": pred_smooth_per_dim.tolist(),
        "gt_smooth_per_dim": gt_smooth_per_dim.tolist(),
    }


def export_summary(
    output_dir: Path,
    overall_mse: float,
    error_map: np.ndarray | None,
    temporal_metrics: dict,
    smoothness_metrics: dict | None,
    action_horizon: int,
    action_dim: int,
    num_samples: int,
):
    """Export summary metrics to JSON file."""
    summary = {
        "metadata": {
            "num_samples": num_samples,
            "action_horizon": action_horizon,
            "action_dim": action_dim,
        },
        "overall_mse": float(overall_mse),
        "mse_per_horizon": temporal_metrics.get("mse_per_horizon", []),
        "smoothness": smoothness_metrics,
    }

    if error_map is not None:
        summary["mse_stats"] = {
            "mean": float(np.mean(error_map)),
            "std": float(np.std(error_map)),
            "max": float(np.max(error_map)),
            "min": float(np.min(error_map)),
            "per_horizon_mean": np.mean(error_map, axis=1).tolist(),
            "per_dim_mean": np.mean(error_map, axis=0).tolist(),
        }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")
    return summary


def sanitize_repo_id(repo_id: str, max_len: int = 120) -> str:
    """Convert repo_id into a stable filesystem-safe folder name."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", repo_id).strip("_")
    if not sanitized:
        sanitized = "repo"
    return sanitized[:max_len]


def export_task_type_aggregate(
    run_root: Path,
    task_type: str,
    per_repo_rows: list[dict],
):
    """Export task-level aggregate statistics for all repos under one task type."""
    task_dir = run_root / task_type
    task_dir.mkdir(parents=True, exist_ok=True)

    mse_values = [float(row["overall_mse"]) for row in per_repo_rows if "overall_mse" in row]
    summary = {
        "task_type": task_type,
        "num_repos": len(per_repo_rows),
        "repo_ids": [row["repo_id"] for row in per_repo_rows],
        "per_repo": per_repo_rows,
        "overall_mse_stats": {
            "mean": float(np.mean(mse_values)) if mse_values else None,
            "std": float(np.std(mse_values)) if mse_values else None,
            "min": float(np.min(mse_values)) if mse_values else None,
            "max": float(np.max(mse_values)) if mse_values else None,
        },
    }

    out_path = task_dir / "task_aggregate_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved task aggregate summary to {out_path}")
    return summary


def run_enhanced_visualization(
    preds_arr: np.ndarray,
    gts_arr: np.ndarray,
    vis_dir: Path,
    repo: str,
):
    """Run all enhanced visualizations for a single repo.

    Args:
        preds_arr: shape (N, action_horizon, action_dim)
        gts_arr: shape (N, action_horizon, action_dim)
        vis_dir: Path to the repo-level output directory
        repo: repo identifier string
    """
    if preds_arr.ndim == 2:
        num_samples, action_dim = preds_arr.shape
        preds_arr = preds_arr.reshape(num_samples, 1, action_dim)
        gts_arr = gts_arr.reshape(num_samples, 1, action_dim)

    num_samples, horizon_len, action_dim = preds_arr.shape
    action_dim_names = _get_action_dim_names(action_dim)
    overall_mse = float(np.mean((preds_arr - gts_arr) ** 2))

    print(f"\n--- Enhanced Visualization for {repo} ---")
    print(f"Samples: {num_samples}, Horizon: {horizon_len}, Action Dim: {action_dim}, Overall MSE: {overall_mse:.6f}")

    print("[1/4] Multi-horizon visualizations...")
    visualize_multi_horizon(preds_arr, gts_arr, vis_dir, action_dim_names=action_dim_names)

    print("[2/4] Error heatmap...")
    error_map = visualize_error_heatmap(preds_arr, gts_arr, vis_dir, action_dim_names=action_dim_names)

    print("[3/4] Temporal metrics (for summary.json only, no temporal_metrics.png)...")
    temporal_metrics = compute_temporal_metrics(preds_arr, gts_arr)

    print("[4/4] Smoothness analysis...")
    smoothness_metrics = visualize_smoothness(preds_arr, gts_arr, vis_dir)

    print("Exporting summary...")
    export_summary(
        vis_dir,
        overall_mse=overall_mse,
        error_map=error_map,
        temporal_metrics=temporal_metrics,
        smoothness_metrics=smoothness_metrics,
        action_horizon=horizon_len,
        action_dim=action_dim,
        num_samples=num_samples,
    )

    print(f"=== Enhanced visualization complete for {repo} ===\n")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def main(
    checkpoint_dir: str,
    config_name: str,
    task_types: list[str] | None = None,
    num_batches: int = 2,
    norm_stats_path: str | None = None,
    batch_size: int | None = None,
    vis_dir: str = "./outputs/open_loop_vis",
    sample_steps: int | None = None,
):
    t0 = time.perf_counter()
    init_logging()

    checkpoint_dir = Path(checkpoint_dir)
    dataset_root = Path(DATA_ROOT)
    selected_task_types = list(REPO_ID_DICT.keys()) if task_types is None else task_types
    unknown_task_types = [t for t in selected_task_types if t not in REPO_ID_DICT]
    if unknown_task_types:
        raise ValueError(f"Unknown task types: {unknown_task_types}, valid keys: {list(REPO_ID_DICT.keys())}")

    task_repo_pairs: list[tuple[str, str]] = []
    for task_type in selected_task_types:
        repos = REPO_ID_DICT.get(task_type, [])
        task_repo_pairs.extend((task_type, repo) for repo in repos)
    if not task_repo_pairs:
        raise ValueError("No repos found from REPO_ID_DICT for selected task types")

    checkpoint_base_dir = checkpoint_dir.parent.parent.parent
    config = _config.get_config(config_name)
    exp_name = checkpoint_dir.parent.name
    step = checkpoint_dir.name

    replace_kwargs = {"checkpoint_base_dir": str(checkpoint_base_dir), "exp_name": exp_name}
    if batch_size is not None:
        replace_kwargs["batch_size"] = batch_size
    config = dataclasses.replace(config, **replace_kwargs)
    print(f"Using config: {config.name}, exp: {config.exp_name}")

    t1 = time.perf_counter()
    print(f"Timing: config loading took {t1 - t0:.3f}s")

    t0 = time.perf_counter()

    first_repo = task_repo_pairs[0][1]
    assert len(first_repo) > 0, "No valid repo_id found for restore data loader"

    try:
        base_data_cfg = config.data.create(config.assets_dirs, config.model)
        replace_data_kwargs = {"root_dir": str(dataset_root), "repo_id": [first_repo]}

        base_data_cfg = dataclasses.replace(base_data_cfg, **replace_data_kwargs)

        # 直接从 assets 目录下的子文件夹获取 asset_id (子文件夹名是数字字符串)
        assets_dir = checkpoint_dir / "assets"
        asset_subdirs = [d for d in assets_dir.iterdir() if d.is_dir()] if assets_dir.exists() else []
        norm_stats_path = str(asset_subdirs[0]) if asset_subdirs else None
        if norm_stats_path is not None:
            norm_stats_file = norm_stats_path
            if Path(norm_stats_file).exists():
                loaded = _normalize.load(norm_stats_file)
                base_data_cfg = dataclasses.replace(base_data_cfg, norm_stats=loaded)
                print(f"Loaded norm_stats from {norm_stats_file}")
            else:
                logging.warning(f"Provided norm_stats_path does not exist: {norm_stats_file}")
                raise FileNotFoundError(f"Provided norm_stats_path does not exist: {norm_stats_file}")

        class _SimpleFactory:
            def __init__(self, data_cfg):
                self._data_cfg = data_cfg
                self.episode_fail = None
                self.dataset_length = None

            def create(self, assets_dirs, model_config):
                return self._data_cfg

        config = dataclasses.replace(config, data=_SimpleFactory(base_data_cfg))
    except Exception:
        logging.warning("Could not patch data factory; proceeding and hope dataset paths are embedded in config.")

    fsdp_devices = getattr(config, "fsdp_devices", None)
    if not fsdp_devices:
        try:
            fsdp_devices = list(range(jax.device_count()))
            print(f"Auto-configuring fsdp_devices to all local devices: {fsdp_devices}")
        except Exception:
            fsdp_devices = None

    mesh = sharding.make_mesh(fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

    checkpoint_manager, _resuming = _checkpoints.initialize_checkpoint_dir(
        checkpoint_dir.parent,
        keep_period=config.keep_period,
        overwrite=False,
        resume=True,
    )
    restore_data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=False,
        skip_norm_stats=False,
    )

    t1 = time.perf_counter()
    print(f"Timing: data loader took {t1 - t0:.3f}s")

    t0 = time.perf_counter()

    rng = jax.random.key(config.seed)
    _, init_rng = jax.random.split(rng)

    train_state_shape, _state_sharding = init_train_state(config, init_rng, mesh, resume=True)

    train_state = _checkpoints.restore_state(checkpoint_manager, train_state_shape, restore_data_loader, int(step))

    model = nnx.merge(train_state.model_def, train_state.params)
    model.eval()
    sample_fn_jit = nnx_utils.module_jit(model.sample_actions)

    logging.info("Model restored and set to eval mode.")

    output_fns = []
    output_fns.extend(base_data_cfg.model_transforms.outputs)
    output_fns.append(transforms.Unnormalize(base_data_cfg.norm_stats, use_quantiles=base_data_cfg.use_quantile_norm))
    output_fns.extend(base_data_cfg.data_transforms.outputs)
    output_fns.extend(base_data_cfg.repack_transforms.outputs)

    output_transform = transforms.compose(output_fns)

    t1 = time.perf_counter()
    print(f"Timing: prepare took {t1 - t0:.3f}s")

    vis_dir = Path(vis_dir)
    task_repo_results: dict[str, list[dict]] = {task_type: [] for task_type in selected_task_types}

    for task_type, repo in task_repo_pairs:
        print(f"\nEvaluating task_type={task_type}, repo={repo}")
        try:
            data_cfg = dataclasses.replace(base_data_cfg, root_dir=str(dataset_root), repo_id=[repo])

            class _SimpleFactoryLocal:
                def __init__(self, data_cfg):
                    self._data_cfg = data_cfg
                    self.episode_fail = None
                    self.dataset_length = None

                def create(self, assets_dirs, model_config):
                    return self._data_cfg

            cfg_for_loader = dataclasses.replace(config, data=_SimpleFactoryLocal(data_cfg))
        except Exception as e:
            logging.warning(f"Could not create data config for repo {repo}: {e}")
            continue
        meta = LeRobotDatasetMetadata(repo, Path(dataset_root) / repo)
        robot_type = meta.robot_type

        data_loader = _data_loader.create_data_loader(
            cfg_for_loader, sharding=data_sharding, shuffle=False, skip_norm_stats=False
        )

        data_iter = iter(data_loader)

        all_preds = []
        all_gts = []
        batch_mses = []

        delay = config.rtc_max_delay
        action_prefix = jnp.zeros((batch_size, config.model.action_horizon, config.model.action_dim))

        for i in range(num_batches):
            t0 = time.perf_counter()
            print(f"Current batch: {i}")

            batch = next(data_iter)
            observation, actions, _ = batch
            actions_np = jax.device_get(actions)
            t1 = time.perf_counter()
            print(f"- Timing: get batch data: {t1 - t0:.3f}s")

            t0 = time.perf_counter()
            rng, subkey = jax.random.split(rng)

            sampled = sample_fn_jit(subkey, observation, action_prefix=action_prefix, delay=delay)
            sampled = sampled.block_until_ready()
            sampled_np = jax.device_get(sampled)

            t1 = time.perf_counter()
            print(f"- Timing: sample_actions: {t1 - t0:.3f}s")

            t0 = time.perf_counter()

            obs_dict = observation.to_dict() if hasattr(observation, "to_dict") else {}
            obs_host = jax.tree_map(np.asarray, obs_dict)
            state_host = obs_host.get("state", None)

            pred_in = {
                "state": state_host,
                "actions": sampled_np,
                "robot_type": robot_type,
            }
            gt_in = {
                "state": state_host,
                "actions": actions_np,
                "robot_type": robot_type,
            }
            pred_trans = output_transform(pred_in)
            final_pred = pred_trans["actions"]
            gt_trans = output_transform(gt_in)
            final_gt = gt_trans["actions"]

            mse_batch = float(np.mean((final_pred - final_gt) ** 2))

            batch_mses.append(mse_batch)
            all_preds.append(final_pred)
            all_gts.append(final_gt)

            t1 = time.perf_counter()
            print(f"- Timing: Infer + post process: {t1 - t0:.3f}s")
            print(f"- MSE: {round(mse_batch, 4)}")

        jax.clear_caches()

        overall_mse = float(np.mean(batch_mses))
        print(f"Overall MSE across {len(batch_mses)} batches for repo {repo}: {round(overall_mse, 4)}")
        preds_arr = np.concatenate(all_preds, axis=0)
        gts_arr = np.concatenate(all_gts, axis=0)

        repo_name = sanitize_repo_id(repo)

        # Build output directory: vis_dir / task_type(from REPO_ID_DICT key) / repo_name
        repo_vis_dir = vis_dir / task_type / repo_name
        repo_vis_dir.mkdir(parents=True, exist_ok=True)

        # Save npy files
        npy_dir = repo_vis_dir / "npy"
        npy_dir.mkdir(parents=True, exist_ok=True)
        np.save(npy_dir / "test_all_preds.npy", preds_arr)
        np.save(npy_dir / "test_all_gts.npy", gts_arr)
        print(f"Saved npy files to {npy_dir}")

        # Legacy gap-50 visualization (first action of each input frame)
        vis_gap = 50
        action_dim = preds_arr.shape[-1]
        preds_arr_gap50 = preds_arr[::vis_gap].reshape(-1, action_dim)
        gts_arr_gap50 = gts_arr[::vis_gap].reshape(-1, action_dim)
        legacy_filename = str(repo_vis_dir / "pred_vs_gt_gap50.png")
        visualize_pred_vs_gt(preds_arr_gap50, gts_arr_gap50, filename=legacy_filename)

        # Enhanced visualizations
        run_enhanced_visualization(preds_arr, gts_arr, repo_vis_dir, repo)

        # Bimanual arm pos visualization
        if ENABLE_BIMANUAL and action_dim == 14:
            arm_vis_dir = str(repo_vis_dir / "arm_states")
            os.makedirs(arm_vis_dir, exist_ok=True)

            left_arm_pos_gap50 = np.zeros((preds_arr_gap50.shape[0], action_dim // 2 - 1))
            right_arm_pos_gap50 = np.zeros((preds_arr_gap50.shape[0], action_dim // 2 - 1))
            left_arm_pos_gt_gap50 = np.zeros((gts_arr_gap50.shape[0], action_dim // 2 - 1))
            right_arm_pos_gt_gap50 = np.zeros((gts_arr_gap50.shape[0], action_dim // 2 - 1))
            for i in range(preds_arr_gap50.shape[0]):
                left_arm_pos_gap50[i] = bimanual.forward_kinematics(preds_arr_gap50[i, : action_dim // 2 - 1])
                right_arm_pos_gap50[i] = bimanual.forward_kinematics(
                    preds_arr_gap50[i, action_dim // 2 : action_dim - 1]
                )
                left_arm_pos_gt_gap50[i] = bimanual.forward_kinematics(gts_arr_gap50[i, : action_dim // 2 - 1])
                right_arm_pos_gt_gap50[i] = bimanual.forward_kinematics(
                    gts_arr_gap50[i, action_dim // 2 : action_dim - 1]
                )
            arm_pred = np.concatenate([left_arm_pos_gap50, right_arm_pos_gap50], axis=1)
            arm_gt = np.concatenate([left_arm_pos_gt_gap50, right_arm_pos_gt_gap50], axis=1)

            visualize_pred_vs_gt(arm_pred, arm_gt, filename=f"{arm_vis_dir}/pred_vs_gt_arm_poses.png")
            vis_arm_states(
                left_arm_pos_gap50,
                left_arm_pos_gt_gap50,
                vis_dir=arm_vis_dir,
                filename="arm_states_left.png",
                subtitle_suffix="pred=red, gt=blue",
            )
            vis_arm_states(
                right_arm_pos_gap50,
                right_arm_pos_gt_gap50,
                vis_dir=arm_vis_dir,
                filename="arm_states_right.png",
                subtitle_suffix="pred=red, gt=blue",
            )
            vis_arm_states(
                left_arm_pos_gt_gap50,
                right_arm_pos_gt_gap50,
                vis_dir=arm_vis_dir,
                filename="arm_states_gt_both.png",
                subtitle_suffix="left=red, right=blue",
            )

        task_repo_results[task_type].append(
            {
                "repo_id": repo,
                "repo_output_dir": str(repo_vis_dir),
                "overall_mse": overall_mse,
                "summary_path": str(repo_vis_dir / "summary.json"),
            }
        )

    for task_type, rows in task_repo_results.items():
        export_task_type_aggregate(vis_dir, task_type, rows)


# ---------------------------------------------------------------------------
# Compare Mode Functions (for offline comparison of multiple runs)
# ---------------------------------------------------------------------------


def discover_run_structure(run_dirs: list[str]) -> dict[str, set[tuple[str, str]]]:
    """Scan each run directory and return inventory of (task_type, repo_name) pairs.

    Args:
        run_dirs: List of run directory paths

    Returns:
        dict mapping run label -> set of (task_type, repo_name) tuples
    """
    inventory = {}
    for run_dir in run_dirs:
        run_path = Path(run_dir)
        label = run_path.name
        pairs = set()

        if not run_path.exists():
            print(f"Warning: Run directory does not exist: {run_dir}")
            inventory[label] = pairs
            continue

        for task_type_dir in run_path.iterdir():
            if task_type_dir.is_dir():
                task_type = task_type_dir.name
                for repo_dir in task_type_dir.iterdir():
                    if repo_dir.is_dir():
                        # Check if it has npy files (valid result directory)
                        npy_dir = repo_dir / "npy"
                        if npy_dir.exists() and (npy_dir / "test_all_preds.npy").exists():
                            pairs.add((task_type, repo_dir.name))

        inventory[label] = pairs
        print(f"  {label}: {len({p[0] for p in pairs})} task_types, {len(pairs)} repos")

    return inventory


def compute_common_pairs(
    inventory: dict[str, set[tuple[str, str]]],
) -> tuple[set[tuple[str, str]], list[tuple[str, tuple[str, str]]]]:
    """Compute intersection of all runs and identify skipped pairs.

    Args:
        inventory: dict mapping run label -> set of (task_type, repo_name) tuples

    Returns:
        tuple of (common_pairs, skipped_list)
        - common_pairs: set of tuples present in ALL runs
        - skipped_list: list of (run_label, (task_type, repo_name)) that are not in all runs
    """
    if not inventory:
        return set(), []

    # Get intersection of all runs
    common_pairs = None
    for pairs in inventory.values():
        common_pairs = pairs.copy() if common_pairs is None else common_pairs & pairs

    if common_pairs is None:
        common_pairs = set()

    # Find skipped pairs (in some runs but not all)
    all_pairs = set()
    for pairs in inventory.values():
        all_pairs = all_pairs | pairs

    skipped = []
    for label, pairs in inventory.items():
        missing = pairs - common_pairs
        skipped.extend((label, pair) for pair in missing)

    return common_pairs, skipped


def load_run_data(run_dir: str, task_type: str, repo_name: str) -> dict | None:
    """Load npy files and summary.json for a single run/repo combination.

    Args:
        run_dir: Path to the run directory
        task_type: Task type name
        repo_name: Repository name (sanitized)

    Returns:
        dict with 'preds', 'gts', 'summary' keys, or None if loading fails
    """
    repo_path = Path(run_dir) / task_type / repo_name
    npy_dir = repo_path / "npy"
    summary_path = repo_path / "summary.json"

    try:
        preds = np.load(npy_dir / "test_all_preds.npy")
        gts = np.load(npy_dir / "test_all_gts.npy")

        summary = None
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)

        return {"preds": preds, "gts": gts, "summary": summary}
    except Exception as e:
        print(f"  Warning: Failed to load data from {repo_path}: {e}")
        return None


def compare_mse_bar(data_dict: dict[str, dict], labels: list[str], output_dir: Path):
    """Generate MSE bar chart comparing all models.

    Args:
        data_dict: dict mapping label -> {'preds', 'gts', 'summary'}
        labels: List of run labels (for consistent ordering)
        output_dir: Output directory path
    """
    mse_values = []
    for label in labels:
        data = data_dict[label]
        mse = float(np.mean((data["preds"] - data["gts"]) ** 2))
        mse_values.append(mse)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 6))
    colors = plt.cm.tab10(np.arange(len(labels)) % 10)

    x = np.arange(len(labels))
    bars = ax.bar(x, mse_values, color=colors, alpha=0.8)

    ax.set_xlabel("Model")
    ax.set_ylabel("Overall MSE")
    ax.set_title("Overall MSE Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(visible=True, linestyle="--", alpha=0.3, axis="y")

    for bar, val in zip(bars, mse_values, strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "mse_bar_chart.png", dpi=150)
    plt.close(fig)
    print("    Saved mse_bar_chart.png")

    return mse_values


def compare_error_heatmap(data_dict: dict[str, dict], labels: list[str], output_dir: Path):
    """Generate error heatmap comparison (side by side + difference for 2 models).

    Args:
        data_dict: dict mapping label -> {'preds', 'gts', 'summary'}
        labels: List of run labels
        output_dir: Output directory path
    """
    n_models = len(labels)

    # For 2 models, add a third subplot for difference
    n_plots = n_models + 1 if n_models == 2 else n_models
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    error_maps = []
    for idx, label in enumerate(labels):
        data = data_dict[label]
        preds, gts = data["preds"], data["gts"]
        _n, horizon_len, action_dim = preds.shape
        error_map = np.mean((preds - gts) ** 2, axis=0)  # (H, D)
        error_maps.append(error_map)

        action_dim_names = _get_action_dim_names(action_dim)
        step = max(1, horizon_len // 10)
        x_labels = [f"h{h}" if h % step == 0 else "" for h in range(horizon_len)]

        sns.heatmap(
            error_map.T,
            ax=axes[idx],
            cmap="YlOrRd",
            xticklabels=x_labels,
            yticklabels=action_dim_names,
            cbar_kws={"label": "MSE"},
        )
        axes[idx].set_xlabel("Horizon Step")
        axes[idx].set_ylabel("Action Dimension")
        axes[idx].set_title(f"{label}")

    # Add difference plot for 2 models
    if n_models == 2:
        diff_map = error_maps[0] - error_maps[1]  # (H, D)
        max_abs = np.max(np.abs(diff_map))

        sns.heatmap(
            diff_map.T,
            ax=axes[2],
            cmap="coolwarm",
            center=0,
            vmin=-max_abs,
            vmax=max_abs,
            xticklabels=x_labels,
            yticklabels=action_dim_names,
            cbar_kws={"label": "MSE Diff"},
        )
        axes[2].set_xlabel("Horizon Step")
        axes[2].set_ylabel("Action Dimension")
        axes[2].set_title(f"Diff: {labels[0]} - {labels[1]}")

    fig.suptitle("Error Heatmap Comparison")
    fig.tight_layout()
    fig.savefig(output_dir / "error_heatmap_diff.png", dpi=150)
    plt.close(fig)
    print("    Saved error_heatmap_diff.png")


def compare_per_dim_mse(data_dict: dict[str, dict], labels: list[str], output_dir: Path):
    """Generate per-dimension MSE line chart comparison.

    Args:
        data_dict: dict mapping label -> {'preds', 'gts', 'summary'}
        labels: List of run labels
        output_dir: Output directory path
    """
    # Compute per-dimension MSE for each model
    dim_mse_dict = {}
    action_dim = None

    for label in labels:
        data = data_dict[label]
        preds, gts = data["preds"], data["gts"]
        if action_dim is None:
            action_dim = preds.shape[-1]
        # MSE per dimension: mean over (N, H)
        dim_mse = np.mean((preds - gts) ** 2, axis=(0, 1))  # (D,)
        dim_mse_dict[label] = dim_mse

    if action_dim is None:
        return None

    action_dim_names = _get_action_dim_names(action_dim)
    colors = plt.cm.tab10(np.arange(len(labels)) % 10)

    fig, ax = plt.subplots(figsize=(max(12, action_dim * 0.8), 6))
    x = np.arange(action_dim)

    for idx, label in enumerate(labels):
        ax.plot(x, dim_mse_dict[label], marker="o", label=label, color=colors[idx], linewidth=2, markersize=6)

    ax.set_xlabel("Action Dimension")
    ax.set_ylabel("MSE")
    ax.set_title("Per-Dimension MSE Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(action_dim_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(visible=True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "per_dim_mse_comparison.png", dpi=150)
    plt.close(fig)
    print("    Saved per_dim_mse_comparison.png")

    return dim_mse_dict


def compare_horizon_overlay(data_dict: dict[str, dict], labels: list[str], output_dir: Path, max_samples: int = 100):
    """Generate horizon trajectory overlay comparison (multi-model preds + GT).

    Args:
        data_dict: dict mapping label -> {'preds', 'gts', 'summary'}
        labels: List of run labels
        output_dir: Output directory path
        max_samples: Maximum number of samples to visualize (for performance)
    """
    # Use first label's data for GT reference
    first_label = labels[0]
    first_data = data_dict[first_label]
    preds_ref, gts = first_data["preds"], first_data["gts"]
    num_samples, _horizon_len, action_dim = preds_ref.shape

    # Subsample if too many samples
    indices = (
        np.linspace(0, num_samples - 1, max_samples, dtype=int) if max_samples < num_samples else np.arange(num_samples)
    )

    action_dim_names = _get_action_dim_names(action_dim)
    colors = plt.cm.tab10(np.arange(len(labels)) % 10)

    # Show all dimensions, calculate grid layout
    dims_to_show = action_dim
    n_cols = min(dims_to_show, 7)  # Max 7 columns
    n_rows = (dims_to_show + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = axes.flatten() if dims_to_show > 1 else [axes]

    for dim_idx in range(dims_to_show):
        ax = axes[dim_idx]

        # Plot GT (same for all models)
        gt_samples = gts[indices, :, dim_idx]  # (n_samples, H)
        gt_mean = np.mean(gt_samples, axis=0)
        ax.plot(gt_mean, label="GT", color="black", linewidth=2.5, linestyle="--")

        # Plot each model's predictions
        for label_idx, label in enumerate(labels):
            data = data_dict[label]
            pred_samples = data["preds"][indices, :, dim_idx]
            pred_mean = np.mean(pred_samples, axis=0)
            ax.plot(pred_mean, label=label, color=colors[label_idx], linewidth=1.5, alpha=0.8)

        ax.set_title(action_dim_names[dim_idx])
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Value")
        ax.grid(visible=True, linestyle="--", alpha=0.3)
        if dim_idx == 0:
            ax.legend(loc="best", fontsize=8)

    for dim_idx in range(dims_to_show, len(axes)):
        axes[dim_idx].axis("off")

    fig.suptitle("Horizon Trajectory Overlay (Mean over samples)")
    fig.tight_layout()
    fig.savefig(output_dir / "horizon_trajectory_overlay.png", dpi=150)
    plt.close(fig)
    print("    Saved horizon_trajectory_overlay.png")


def compare_smoothness(data_dict: dict[str, dict], labels: list[str], output_dir: Path):
    """Generate smoothness comparison bar chart.

    Args:
        data_dict: dict mapping label -> {'preds', 'gts', 'summary'}
        labels: List of run labels
        output_dir: Output directory path
    """
    smoothness_values = []
    gt_smoothness = None

    for label in labels:
        data = data_dict[label]
        preds, gts = data["preds"], data["gts"]
        _, horizon_len, _ = preds.shape

        if horizon_len < 4:
            smoothness_values.append(None)
            continue

        pred_jerk = np.diff(preds, n=3, axis=1)
        pred_jerk_mag = np.mean(np.abs(pred_jerk), axis=1)
        pred_smoothness = float(np.mean(pred_jerk_mag**2))
        smoothness_values.append(pred_smoothness)

        # GT smoothness (same for all, compute once)
        if gt_smoothness is None:
            gt_jerk = np.diff(gts, n=3, axis=1)
            gt_jerk_mag = np.mean(np.abs(gt_jerk), axis=1)
            gt_smoothness = float(np.mean(gt_jerk_mag**2))

    # Filter out None values for plotting
    valid_labels = [label for label, v in zip(labels, smoothness_values, strict=False) if v is not None]
    valid_values = [v for v in smoothness_values if v is not None]

    if not valid_values:
        print("    Warning: Horizon too short for smoothness analysis")
        return None

    colors = plt.cm.tab10(np.arange(len(valid_labels)) % 10)

    fig, ax = plt.subplots(figsize=(max(8, len(valid_labels) * 1.5), 6))
    x = np.arange(len(valid_labels) + 1)  # +1 for GT
    all_values = valid_values + ([gt_smoothness] if gt_smoothness else [])
    all_labels = [*valid_labels, "GT"]
    bar_colors = [*list(colors[: len(valid_labels)]), "black"]

    bars = ax.bar(x, all_values, color=bar_colors, alpha=0.8)

    ax.set_xlabel("Model")
    ax.set_ylabel("Smoothness Score (Mean Jerk^2)")
    ax.set_title("Smoothness Comparison (Lower is smoother)")
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=45, ha="right")
    ax.grid(visible=True, linestyle="--", alpha=0.3, axis="y")

    for bar, val in zip(bars, all_values, strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "smoothness_comparison.png", dpi=150)
    plt.close(fig)
    print("    Saved smoothness_comparison.png")

    return {"smoothness_values": dict(zip(valid_labels, valid_values, strict=False)), "gt_smoothness": gt_smoothness}


def export_task_comparison(all_repo_results: dict, task_type: str, output_dir: Path):
    """Export task-level comparison summary.

    Args:
        all_repo_results: dict mapping repo_name -> dict of comparison results
        task_type: Task type name
        output_dir: Output directory path
    """
    summary = {
        "task_type": task_type,
        "num_repos": len(all_repo_results),
        "repos": list(all_repo_results.keys()),
        "per_repo_summary": {},
    }

    for repo_name, repo_data in all_repo_results.items():
        summary["per_repo_summary"][repo_name] = {
            "mse_values": repo_data.get("mse_values", {}),
        }

    out_path = output_dir / "task_comparison_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("  Saved task_comparison_summary.json")


def export_overall_comparison(all_results: dict, labels: list[str], output_dir: Path):
    """Generate overall MSE comparison bar chart across all repos.

    Args:
        all_results: dict with aggregated results
        labels: List of run labels
        output_dir: Output directory path
    """
    # Aggregate MSE per model across all repos
    model_mses = {label: [] for label in labels}

    for repo_data in all_results.values():
        mse_values = repo_data.get("mse_values", [])
        if mse_values:
            for label, mse in zip(labels, mse_values, strict=False):
                if label in model_mses:
                    model_mses[label].append(mse)

    # Compute mean MSE per model
    mean_mses = {}
    for label in labels:
        if model_mses[label]:
            mean_mses[label] = float(np.mean(model_mses[label]))

    if not mean_mses:
        return None

    colors = plt.cm.tab10(np.arange(len(labels)) % 10)

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.5), 7))
    x = np.arange(len(labels))
    values = [mean_mses[label] for label in labels]
    bars = ax.bar(x, values, color=colors, alpha=0.8)

    ax.set_xlabel("Model")
    ax.set_ylabel("Mean MSE (across all repos)")
    ax.set_title("Overall MSE Comparison (Aggregated)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(visible=True, linestyle="--", alpha=0.3, axis="y")

    for bar, val in zip(bars, values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.4f}", ha="center", va="bottom", fontsize=10
        )

    fig.tight_layout()
    fig.savefig(output_dir / "overall_mse_comparison.png", dpi=150)
    plt.close(fig)
    print("Saved overall_mse_comparison.png")

    return mean_mses


def export_comparison_json(all_results: dict, labels: list[str], output_dir: Path, run_dirs: list[str]):
    """Export comprehensive comparison summary as JSON.

    Args:
        all_results: dict with all comparison results
        labels: List of run labels
        output_dir: Output directory path
        run_dirs: Original run directory paths
    """
    summary = {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "run_directories": dict(zip(labels, run_dirs, strict=False)),
            "num_models": len(labels),
        },
        "models": labels,
        "per_repo_results": {},
    }

    for repo_key, repo_data in all_results.items():
        summary["per_repo_results"][repo_key] = {
            "mse_values": dict(zip(labels, repo_data.get("mse_values", []), strict=False)),
            "dim_mse": repo_data.get("dim_mse_dict", {}),
        }

    out_path = output_dir / "comparison_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved comparison_summary.json")


def export_comparison_csv(all_results: dict, labels: list[str], output_dir: Path):
    """Export comparison results as CSV for Excel/pandas analysis.

    Args:
        all_results: dict with all comparison results
        labels: List of run labels
        output_dir: Output directory path
    """
    out_path = output_dir / "comparison_metrics.csv"

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        header = ["task_type", "repo_name"] + [f"{label}_mse" for label in labels]
        writer.writerow(header)

        # Rows
        for repo_key, repo_data in all_results.items():
            # repo_key format: "task_type/repo_name"
            parts = repo_key.split("/", 1)
            task_type = parts[0] if len(parts) > 0 else ""
            repo_name = parts[1] if len(parts) > 1 else repo_key

            mse_values = repo_data.get("mse_values", [])
            row = [task_type, repo_name, *mse_values]
            writer.writerow(row)

    print("Saved comparison_metrics.csv")


def export_comparison_markdown(
    all_results: dict, labels: list[str], output_dir: Path, run_dirs: list[str], skipped: list
):
    """Export comparison results as Markdown report.

    Args:
        all_results: dict with all comparison results
        labels: List of run labels
        output_dir: Output directory path
        run_dirs: Original run directory paths
        skipped: List of skipped (label, (task_type, repo)) tuples
    """
    out_path = output_dir / "comparison_report.md"

    lines = []
    lines.append("# Model Comparison Report\n")
    lines.append(f"**Generated:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
    lines.append(f"**Models compared:** {', '.join(labels)}\n")

    lines.append("\n## Run Directories\n")
    for label, path in zip(labels, run_dirs, strict=False):
        lines.append(f"- **{label}**: `{path}`\n")

    if skipped:
        lines.append("\n## Skipped (not in all runs)\n")
        for label, (task_type, repo) in skipped:
            lines.append(f"- {label}: {task_type}/{repo}\n")

    lines.append("\n## MSE Comparison Table\n")
    lines.append("| Task Type | Repo | " + " | ".join([f"{label} MSE" for label in labels]) + " |\n")
    lines.append("|" + "|".join(["---"] * (2 + len(labels))) + "|\n")

    for repo_key, repo_data in all_results.items():
        parts = repo_key.split("/", 1)
        task_type = parts[0] if len(parts) > 0 else ""
        repo_name = parts[1] if len(parts) > 1 else repo_key

        mse_values = repo_data.get("mse_values", [])
        mse_strs = [f"{v:.6f}" if v is not None else "N/A" for v in mse_values]
        lines.append(f"| {task_type} | {repo_name} | " + " | ".join(mse_strs) + " |\n")

    lines.append("\n## Visualizations\n")
    lines.append("Per-repo visualizations are saved in respective subdirectories:\n")
    lines.append("- `mse_bar_chart.png`: MSE comparison bar chart\n")
    lines.append("- `error_heatmap_diff.png`: Error heatmap comparison\n")
    lines.append("- `per_dim_mse_comparison.png`: Per-dimension MSE curves\n")
    lines.append("- `horizon_trajectory_overlay.png`: Trajectory overlay comparison\n")
    lines.append("- `smoothness_comparison.png`: Smoothness comparison\n")

    with open(out_path, "w") as f:
        f.writelines(lines)

    print("Saved comparison_report.md")


def compare_runs(
    compare_paths: str,
    compare_output: str,
    compare_labels: str | None = None,
):
    """Main entry point for compare mode.

    Args:
        compare_paths: Comma-separated run directory paths
        compare_output: Output directory for comparison results
        compare_labels: Optional comma-separated labels (defaults to directory names)
    """
    print("\n=== Compare Mode ===\n")

    # Parse paths
    run_dirs = [p.strip() for p in compare_paths.split(",") if p.strip()]

    # Parse or generate labels
    if compare_labels:
        labels = [label.strip() for label in compare_labels.split(",") if label.strip()]
        if len(labels) != len(run_dirs):
            print(
                f"Warning: Number of labels ({len(labels)}) doesn't match number of paths ({len(run_dirs)}). Using directory names."
            )
            labels = [Path(p).name for p in run_dirs]
    else:
        labels = [Path(p).name for p in run_dirs]

    print(f"Run directories: {run_dirs}")
    print(f"Labels: {labels}")

    # Validate all directories exist
    for run_dir in run_dirs:
        if not Path(run_dir).exists():
            raise ValueError(f"Run directory does not exist: {run_dir}")

    output_dir = Path(compare_output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover structure
    print("\nDiscovering task_type/repo structure...")
    inventory = discover_run_structure(run_dirs)

    # Compute common pairs
    common_pairs, skipped = compute_common_pairs(inventory)
    print(f"\nIntersection: {len({p[0] for p in common_pairs})} task_types, {len(common_pairs)} repos")

    if skipped:
        print(f"Skipped (not in all runs): {len(skipped)} pairs")
        for label, (task_type, repo) in skipped[:5]:  # Show first 5
            print(f"  - {label}: {task_type}/{repo}")
        if len(skipped) > 5:
            print(f"  ... and {len(skipped) - 5} more")

    if not common_pairs:
        print("Error: No common (task_type, repo) pairs found across all runs!")
        return

    # Group by task_type for organized output
    pairs_by_task: dict[str, list[str]] = {}
    for task_type, repo_name in common_pairs:
        if task_type not in pairs_by_task:
            pairs_by_task[task_type] = []
        pairs_by_task[task_type].append(repo_name)

    # Process each (task_type, repo) pair
    all_results: dict[str, dict] = {}  # key: "task_type/repo_name"
    task_results: dict[str, dict] = {}  # key: task_type, value: {repo_name: results}

    for task_type, repo_names in pairs_by_task.items():
        print(f"\n--- Processing task_type: {task_type} ---")
        task_output_dir = output_dir / task_type
        task_output_dir.mkdir(parents=True, exist_ok=True)
        task_results[task_type] = {}

        for repo_name in repo_names:
            print(f"\nComparing {task_type} / {repo_name}")
            repo_output_dir = task_output_dir / repo_name
            repo_output_dir.mkdir(parents=True, exist_ok=True)

            # Load data from all runs
            data_dict = {}
            for run_dir, label in zip(run_dirs, labels, strict=False):
                data = load_run_data(run_dir, task_type, repo_name)
                if data is not None:
                    data_dict[label] = data
                else:
                    print(f"  Warning: Could not load data for {label}")

            if len(data_dict) != len(labels):
                print(f"  Skipping {task_type}/{repo_name}: not all runs have valid data")
                continue

            # [1/5] MSE comparison
            print("  [1/5] MSE comparison table")
            mse_values = compare_mse_bar(data_dict, labels, repo_output_dir)

            # [2/5] Error heatmap difference
            print("  [2/5] Error heatmap difference")
            compare_error_heatmap(data_dict, labels, repo_output_dir)

            # [3/5] Per-dimension MSE curves
            print("  [3/5] Per-dimension MSE curves")
            dim_mse_dict = compare_per_dim_mse(data_dict, labels, repo_output_dir)

            # [4/5] Horizon trajectory overlay
            print("  [4/5] Horizon trajectory overlay")
            compare_horizon_overlay(data_dict, labels, repo_output_dir)

            # [5/5] Smoothness comparison
            print("  [5/5] Smoothness comparison")
            smoothness_result = compare_smoothness(data_dict, labels, repo_output_dir)

            # Store results
            repo_key = f"{task_type}/{repo_name}"
            all_results[repo_key] = {
                "mse_values": mse_values,
                "dim_mse_dict": (
                    {k: v.tolist() if hasattr(v, "tolist") else v for k, v in dim_mse_dict.items()}
                    if dim_mse_dict
                    else {}
                ),
                "smoothness": smoothness_result,
            }
            task_results[task_type][repo_name] = all_results[repo_key]

            # Release memory
            del data_dict

        # Export task-level summary
        export_task_comparison(task_results[task_type], task_type, task_output_dir)

    # Generate overall reports
    print("\nGenerating overall reports...")

    export_overall_comparison(all_results, labels, output_dir)
    export_comparison_json(all_results, labels, output_dir, run_dirs)
    export_comparison_csv(all_results, labels, output_dir)
    export_comparison_markdown(all_results, labels, output_dir, run_dirs, skipped)

    print(f"\n=== Compare complete. Results in {compare_output} ===\n")


if __name__ == "__main__":
    beijing_tz = timezone(timedelta(hours=8))
    now = datetime.now(beijing_tz)
    time_str = now.strftime("%m%d_%H%M")

    parser = argparse.ArgumentParser()
    # Original inference mode arguments (no longer required globally)
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--config_name", type=str, default=None)
    parser.add_argument(
        "--task_types",
        type=str,
        default=None,
        help="Comma-separated task type keys from REPO_ID_DICT; default uses all keys.",
    )
    parser.add_argument("--num_batches", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--sample_steps", type=int, default=10)
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Sub-folder name under output_root. Defaults to MMdd_HHMM timestamp.",
    )

    # New compare mode arguments
    parser.add_argument(
        "--compare_paths",
        type=str,
        default=None,
        help="Comma-separated run directory paths to compare (activates compare mode).",
    )
    parser.add_argument(
        "--compare_output",
        type=str,
        default=None,
        help="Output directory for comparison results (required in compare mode).",
    )
    parser.add_argument(
        "--compare_labels",
        type=str,
        default=None,
        help="Comma-separated labels for each run (defaults to directory names).",
    )

    args = parser.parse_args()

    # Branch: compare mode vs inference mode
    if args.compare_paths:
        # Compare mode
        if not args.compare_output:
            parser.error("--compare_output is required when using --compare_paths")

        compare_runs(
            compare_paths=args.compare_paths,
            compare_output=args.compare_output,
            compare_labels=args.compare_labels,
        )
    else:
        # Original inference mode
        if not args.ckpt_dir or not args.config_name:
            parser.error(
                "--ckpt_dir and --config_name are required in inference mode (or use --compare_paths for compare mode)"
            )

        task_types = None
        if args.task_types:
            task_types = [t.strip() for t in args.task_types.split(",") if t.strip()]

        run_name = args.run_name or time_str
        vis_dir = os.path.join(args.output_root, run_name)
        os.makedirs(vis_dir, exist_ok=True)

        main(
            checkpoint_dir=args.ckpt_dir,
            config_name=args.config_name,
            task_types=task_types,
            num_batches=args.num_batches,
            batch_size=args.batch_size,
            vis_dir=vis_dir,
            sample_steps=args.sample_steps,
        )
