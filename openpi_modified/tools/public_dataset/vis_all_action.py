import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import pandas as pd


def visualize_observation_field(data, field_name, filename):
    """Visualize an observation field per-segment (every 100 frames)."""
    arr = np.asarray(data)
    T, n = arr.shape

    segment_len = 100
    num_segments = (T + segment_len - 1) // segment_len

    rows = 2
    cols = max(n // rows, 1)

    base, ext = os.path.splitext(filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    for seg in range(num_segments):
        s = seg * segment_len
        e = min(s + segment_len, T)
        seg_arr = arr[s:e]
        seg_x = list(range(s, e))

        fig, axes_plot = plt.subplots(rows, cols, figsize=(3 * cols, 2.5 * rows))
        axes_flat = axes_plot.flatten()

        for i in range(n):
            ax = axes_flat[i]
            ax.plot(seg_x, seg_arr[:, i], linewidth=1.0)
            ax.set_title(f"{field_name} dim {i}")
            ax.grid(True, linestyle="--", alpha=0.4)

        for j in range(n, len(axes_flat)):
            axes_flat[j].axis("off")

        fig.suptitle(f"frames {s}-{e-1}", fontsize=10)
        fig.tight_layout()
        seg_filename = f"{base}_seg{seg:03d}{ext}"
        fig.savefig(seg_filename, dpi=300)
        plt.close(fig)

    print(f"  Saved {num_segments} segment plots to {base}_seg*{ext}")


def visualize_state_and_action(pred_seq, gt_seq, filename: str | None = None):
    """Visualize prediction vs GT for each dimension over time."""
    pred = np.asarray(pred_seq)
    gt = np.asarray(gt_seq)
    if pred.ndim != 2 or gt.ndim != 2 or pred.shape[1] != gt.shape[1]:
        raise ValueError("pred_seq and gt_seq must have shape (T, action_dim)")
    if pred.shape[0] != gt.shape[0]:
        raise ValueError("pred_seq and gt_seq must have the same temporal length T")

    T = pred.shape[0]
    n = pred.shape[1]
    rows = 2
    cols = n // rows

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.5 * rows))
    axes = axes.flatten()

    x = list(range(T))

    for i in range(n):
        ax = axes[i]
        ax.plot(x, gt[:, i], label="State", color="#1f77b4", linewidth=1.2)
        ax.plot(
            x, pred[:, i], label="Action", color="#d62728", linestyle="--", linewidth=1.2
        )
        ax.set_title(f"Dim {i}")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle="--", alpha=0.4)
        if i == 0:
            ax.legend()

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig.savefig(filename, dpi=300)
    print(f"Saved visualization to {filename}")
    plt.close(fig)


def read_parquet_and_vis(path: Path, repo_id, episode_idx,
                 output_dir="./vis_action",
                 vis_velocity=False, vis_current=False,
                 vis_ee_pose=False):
    """Read parquet and visualize.

    Always visualizes: current/effort, velocity, state, action.
    Optional (off by default): ee_pose.
    """
    df = pd.read_parquet(path)
    print(f"  Columns in parquet: {list(df.columns)}")

    state_np = np.stack(df["observation.state"].to_numpy())
    action_np = np.stack(df["action"].to_numpy())

    prefix = f"{output_dir}/{repo_id}/episode_{episode_idx:06d}/episode_{episode_idx:06d}"

    visualize_state_and_action(state_np, action_np, f"{prefix}.png")

    # Velocity (optional)
    if vis_velocity and "observation.velocity" in df.columns:
        vel_np = np.stack(df["observation.velocity"].to_numpy())
        visualize_observation_field(vel_np, "velocity", f"{prefix}_velocity.png")

    # ee_pose (optional)
    if vis_ee_pose and "observation.ee_pose" in df.columns:
        ee_np = np.stack(df["observation.ee_pose"].to_numpy())
        visualize_observation_field(ee_np, "ee_pose", f"{prefix}_ee_pose.png")
        
    if vis_current and "observation.current" in df.columns:
        current_np = np.stack(df["observation.current"].to_numpy())
        visualize_observation_field(current_np, "current", f"{prefix}_current.png")

    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize actions from parquet datasets.")
    parser.add_argument("--dataset_root", type=str,
                        default="/mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/fold_box/",
                        help="Root directory of the dataset.")
    parser.add_argument("--repo_ids", type=str, nargs="+", required=True,
                        help="List of repo IDs to visualize.")
    parser.add_argument("--output_dir", type=str, default="./vis_action",
                        help="Output directory for visualizations.")
    parser.add_argument("--max_episodes", type=int, default=100,
                        help="Max number of episodes to visualize per repo.")
    parser.add_argument("--vis_velocity", action="store_true", default=False,
                        help="Visualize velocity (default: off).")
    parser.add_argument("--vis_current", action="store_true", default=False,
                        help="Visualize current (default: off).")
    parser.add_argument("--vis_ee_pose", action="store_true", default=False,
                        help="Visualize ee_pose (default: off).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    for repo_id in args.repo_ids:
        for episode_idx in range(args.max_episodes):
            parquet_path = (
                Path(args.dataset_root)
                / repo_id
                / "data"
                / "chunk-000"
                / f"episode_{episode_idx:06d}.parquet"
            )
            if os.path.exists(parquet_path):
                print(f"Processing {parquet_path}")
                read_parquet_and_vis(parquet_path, repo_id, episode_idx,
                             output_dir=args.output_dir,
                             vis_velocity=args.vis_velocity,
                             vis_current=args.vis_current,
                             vis_ee_pose=args.vis_ee_pose)
            else:
                print(f"File {parquet_path} does not exist, skipping.")
                break

'''
python tools/public_dataset/vis_all_action.py \                                                                                                                                          
--dataset_root /path/to/data \                                                                                                                                                           
--repo_ids repo1 repo2 \                                                                                                                                                               
--output_dir ./my_output \                                                                                                                                                               
--max_episodes 50 \
--vis_velocity \                                                                                                                                                                                        
--vis_current \                                                                                                                                                                      
--vis_ee_pose                                                                                                                                                                          
                                                                                                                                                                                            
参数说明：
- --dataset_root：数据根目录（有默认值）                                                                                                                                                   
- --repo_ids：repo ID 列表（必填）                                                                                                                                                         
- --output_dir：输出目录，默认 ./vis_action0123
- --max_episodes：每个 repo 最多可视化多少 episode，默认 100                                                                                                                               
- --vis_state / --no_vis_state：state 可视化，默认开启                                                                                                                                     
- --vis_action / --no_vis_action：action 可视化，默认开启                                                                                                                                  
- --vis_ee_pose：ee_pose 可视化，默认关闭

举例：
python tools/public_dataset/vis_all_action.py \
--dataset_root /mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/fold_box/fold_box_from_scratch/infer/ \
--repo_ids fold_box_scratch_infer_0322_nc_4w.all.6000s.20260323.batch.1 \
--vis_velocity \
--vis_current \
--vis_ee_pose 
'''
