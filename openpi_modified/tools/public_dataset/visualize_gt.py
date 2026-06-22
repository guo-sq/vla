"""Visualize ground truth (GT) data for each repo id.

Usage (example):
    uv run tools/public_dataset/visualize_gt.py \
        --dataset_root /mnt/ \
        --config_name src/openpi/configs/cfg_opensource/cfg_pi0.5_28_dim.all_public_datasets.py \
        --repo_id oss_data/anyverse_pour_water_record/record.pourwater.bipiper.0128.3 \
        --output_dir outputs/gt_dev_anyverse \
        --num_samples 1000

    uv run tools/public_dataset/visualize_gt.py \
        --dataset_root /mnt/ \
        --config_name src/openpi/configs/cfg_pi0.5_pour_water_14_dim.py \
        --repo_id "oss_data/anyverse_pour_water_record/record.pourwater.bipiper.0304.8" \
        --output_dir outputs/gt_bipiper.0304.8 \
        --num_samples 100

"""

import dataclasses
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.shared.normalize as _normalize
import openpi.transforms as transforms

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata


def init_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )


def log_step(step_name: str, print_func=print):
    """Log a step with timestamp."""
    import time

    timestamp = time.strftime("%H:%M:%S")
    print_func(f"[{timestamp}] >>> {step_name}")


def visualize_gt(gt_seq, filename: str, title: str = "Ground Truth"):
    """Visualize ground truth for all dimensions over time.

    Args:
        gt_seq: array-like shape (T, action_dim) ground-truth sequence.
        filename: filename to save the figure.
        title: title for the figure.
    """
    gt = np.asarray(gt_seq)
    if gt.ndim != 2:
        raise ValueError("gt_seq must have shape (T, action_dim)")

    T = gt.shape[0]
    n = gt.shape[1]

    # Determine layout based on number of dimensions
    if n <= 7:
        rows = 1
        cols = n
    elif n <= 14:
        rows = 2
        cols = (n + 1) // 2
    else:
        rows = (n + 6) // 7
        cols = 7

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    x = list(range(T))
    for i in range(n):
        ax = axes[i]
        ax.plot(x, gt[:, i], label="GT", color="#1f77b4", linewidth=1.2)
        ax.set_title(f"Dim {i}")
        ax.set_xlabel("frames")
        ax.grid(True, linestyle="--", alpha=0.4)
        if i == 0:
            ax.legend()

    # Hide any unused axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    print(f"Saved visualization to {filename}")
    plt.close(fig)


def visualize_gt_single_dim(gt_seq, dim_idx: int, filename: str, title: str = None):
    """Visualize a single dimension of ground truth.

    Args:
        gt_seq: array-like shape (T, action_dim) ground-truth sequence.
        dim_idx: dimension index to visualize.
        filename: filename to save the figure.
        title: optional title for the figure.
    """
    gt = np.asarray(gt_seq)
    if gt.ndim != 2:
        raise ValueError("gt_seq must have shape (T, action_dim)")

    if dim_idx >= gt.shape[1]:
        raise ValueError(f"dim_idx {dim_idx} out of range (action_dim={gt.shape[1]})")

    T = gt.shape[0]

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    x = list(range(T))
    ax.plot(x, gt[:, dim_idx], label=f"Dim {dim_idx}", color="#1f77b4", linewidth=1.5)
    ax.set_xlabel("frames")
    ax.set_ylabel("value")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    if title:
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    print(f"Saved single-dim visualization to {filename}")
    plt.close(fig)


def main(
    dataset_root: str,
    config_name: str,
    repo_id: str,
    output_dir: str = "outputs/gt",
    vis_gap: int = 50,
    num_samples: int | None = None,
):
    """Visualize ground truth data for specified repo ids.

    Args:
        dataset_root: Root directory of the dataset.
        config_name: Config file path.
        repo_id: Comma-separated repo ids to visualize.
        output_dir: Output directory for visualizations.
        vis_gap: Gap between samples for visualization (to reduce plot density).
        num_samples: Number of samples to load. If None, load all data.
    """
    init_logging()

    log_step("Starting GT visualization script")
    log_step(f"Parameters: dataset_root={dataset_root}, config_name={config_name}")
    log_step(f"Parameters: repo_id={repo_id}, output_dir={output_dir}")
    log_step(f"Parameters: vis_gap={vis_gap}, num_samples={num_samples}")

    dataset_root = Path(dataset_root)

    log_step("Loading config...")
    config = _config.get_config(config_name)
    log_step(f"Config loaded: {config.name}")

    # Parse repo ids
    log_step("Parsing repo ids...")
    all_repo_ids = [r.strip() for r in repo_id.split(",") if r.strip()]
    if not all_repo_ids:
        raise ValueError("No valid repo_id provided")

    print(f"Processing repo ids: {all_repo_ids}")

    # Process each repo
    for repo_idx, repo in enumerate(all_repo_ids):
        print(f"\n{'='*60}")
        print(f"[{repo_idx+1}/{len(all_repo_ids)}] Processing repo: {repo}")
        print(f"{'='*60}")
        log_step(f"Creating data config for repo: {repo}")

        # Create data config for this repo
        try:
            log_step("  -> Calling config.data.create()...")
            base_data_cfg = config.data.create(config.assets_dirs, config.model)
            log_step(f"  -> base_data_cfg created, asset_id={base_data_cfg.asset_id}")

            replace_data_kwargs = dict(root_dir=str(dataset_root), repo_id=[repo])
            base_data_cfg = dataclasses.replace(base_data_cfg, **replace_data_kwargs)
            log_step(f"  -> base_data_cfg updated with repo_id=[{repo}]")

            # Try to load norm stats if available (optional for GT visualization)
            norm_stats_path = Path(output_dir) / "norm_stats" / base_data_cfg.asset_id
            log_step(f"  -> Checking norm_stats at: {norm_stats_path}")
            if norm_stats_path.exists():
                loaded = _normalize.load(norm_stats_path)
                base_data_cfg = dataclasses.replace(base_data_cfg, norm_stats=loaded)
                print(f"  -> Loaded norm_stats from {norm_stats_path}")
            else:
                print(
                    f"  -> Note: No norm_stats found at {norm_stats_path}, using raw data"
                )

        except Exception as e:
            logging.error(f"Could not create data config for repo {repo}: {e}")
            import traceback

            traceback.print_exc()
            continue

        # Get metadata
        log_step(f"Loading metadata for repo: {repo}")
        try:
            meta = LeRobotDatasetMetadata(repo, Path(dataset_root) / repo)
            robot_type = meta.robot_type
            print(f"  -> Robot type: {robot_type}")
        except Exception as e:
            logging.warning(f"Could not load metadata for repo {repo}: {e}")
            import traceback

            traceback.print_exc()
            robot_type = "unknown"

        # Create dataset directly (not using dataloader)
        log_step(f"Creating dataset for repo: {repo}")
        log_step("  -> This may take a while for large datasets...")

        dataset = _data_loader.create_anyverse_dataset(base_data_cfg, config.model)
        log_step(f"  -> Dataset created successfully, length: {len(dataset)}")

        # Build output transforms (for unnormalization)
        log_step("Building output transforms...")
        output_fns = []
        output_fns.extend(base_data_cfg.model_transforms.outputs)
        log_step(
            f"  -> Added {len(base_data_cfg.model_transforms.outputs)} model_transforms.outputs"
        )

        if base_data_cfg.norm_stats is not None:
            output_fns.append(
                transforms.Unnormalize(
                    base_data_cfg.norm_stats,
                    use_quantiles=base_data_cfg.use_quantile_norm,
                )
            )
            log_step("  -> Added Unnormalize transform")

        output_fns.extend(base_data_cfg.data_transforms.outputs)
        log_step(
            f"  -> Added {len(base_data_cfg.data_transforms.outputs)} data_transforms.outputs"
        )

        output_fns.extend(base_data_cfg.repack_transforms.outputs)
        log_step(
            f"  -> Added {len(base_data_cfg.repack_transforms.outputs)} repack_transforms.outputs"
        )

        output_transform = transforms.compose(output_fns)
        log_step("  -> Output transform composed successfully")

        # Load data directly from dataset
        log_step(f"Starting data loading (num_samples={num_samples})...")
        all_gts = []

        total_samples = len(dataset)
        samples_to_load = (
            min(total_samples, num_samples) if num_samples else total_samples
        )
        log_step(
            f"  -> Total dataset size: {total_samples}, will load: {samples_to_load}"
        )

        # Build input transforms (same as transform_dataset does)
        input_fns = [
            *base_data_cfg.public_dataset_map_transform.inputs,
            *base_data_cfg.repack_transforms.inputs,
            *base_data_cfg.data_transforms.inputs,
        ]
        if base_data_cfg.norm_stats is not None:
            input_fns.append(
                transforms.Normalize(
                    base_data_cfg.norm_stats,
                    use_quantiles=base_data_cfg.use_quantile_norm,
                )
            )
        input_fns.extend(base_data_cfg.model_transforms.inputs)
        input_transform = transforms.compose(input_fns)

        for i in range(samples_to_load):
            if i % 1000 == 0:
                log_step(f"  -> Processing sample {i+1}/{samples_to_load}...")

            try:
                # Get raw sample from dataset
                raw_sample = dataset[i]

                # Apply input transform (normalization, etc.)
                transformed_sample = input_transform(raw_sample)

                # Extract state and actions
                if isinstance(transformed_sample, dict):
                    state_host = transformed_sample.get("state", None)
                    actions_np = np.asarray(transformed_sample.get("actions", None))
                else:
                    # If it's a tuple or other structure
                    state_host = None
                    actions_np = None
                    if hasattr(transformed_sample, "__getitem__"):
                        # Try to extract actions
                        if len(transformed_sample) >= 2:
                            actions_np = np.asarray(transformed_sample[1])
                        if len(transformed_sample) >= 1:
                            obs = transformed_sample[0]
                            if hasattr(obs, "to_dict"):
                                obs_dict = obs.to_dict()
                                state_host = obs_dict.get("state", None)
                            elif isinstance(obs, dict):
                                state_host = obs.get("state", None)

                if actions_np is None:
                    logging.warning(f"  -> Could not extract actions from sample {i}")
                    continue

                # Apply output transform to get unnormalized actions
                gt_in = {
                    "state": state_host,
                    "actions": actions_np,
                    "robot_type": robot_type,
                }
                gt_trans = output_transform(gt_in)
                final_gt = gt_trans["actions"]

                all_gts.append(final_gt)

            except Exception as e:
                logging.error(f"Error loading sample {i}: {e}")
                import traceback

                traceback.print_exc()
                continue

        log_step(f"Total samples loaded: {len(all_gts)}")

        if not all_gts:
            logging.warning(f"No data loaded for repo {repo}")
            continue

        # Concatenate all GT data
        log_step("Concatenating all GT data...")
        gts_arr = np.concatenate(all_gts, axis=0)
        print(f"Total GT samples: {gts_arr.shape[0]}, action_dim: {gts_arr.shape[-1]}")

        # Create output directory for this repo
        log_step("Creating output directory...")
        repo_output_dir = Path(output_dir) / repo.replace("/", "_")
        os.makedirs(repo_output_dir, exist_ok=True)
        print(f"Output directory: {repo_output_dir}")

        # Save GT data
        log_step("Saving GT data to npy file...")
        np.save(repo_output_dir / "gt_data.npy", gts_arr)
        print(f"Saved GT data to {repo_output_dir / 'gt_data.npy'}")

        # Visualize GT (with gap to reduce plot density)
        log_step(f"Creating GT visualization (vis_gap={vis_gap})...")
        action_dim = gts_arr.shape[-1]
        gts_arr_sampled = gts_arr[::vis_gap].reshape(-1, action_dim)
        log_step(
            f"  -> Sampled {gts_arr_sampled.shape[0]} points from {gts_arr.shape[0]} total"
        )

        vis_filename = repo_output_dir / "gt_visualization.png"
        log_step(f"  -> Generating plot with {action_dim} dimensions...")
        visualize_gt(gts_arr_sampled, str(vis_filename), title=f"Ground Truth - {repo}")

        # Also create per-dimension visualizations for first few dimensions
        log_step("Creating per-dimension visualizations...")
        dim_output_dir = repo_output_dir / "per_dim"
        os.makedirs(dim_output_dir, exist_ok=True)

        for dim_idx in range(min(action_dim, 7)):  # Visualize first 7 dimensions
            dim_filename = dim_output_dir / f"gt_dim_{dim_idx}.png"
            visualize_gt_single_dim(
                gts_arr_sampled,
                dim_idx,
                str(dim_filename),
                title=f"GT Dim {dim_idx} - {repo}",
            )
        log_step(f"  -> Created {min(action_dim, 7)} per-dimension plots")

        # Save statistics
        log_step("Computing and saving statistics...")
        stats = {
            "repo_id": repo,
            "total_samples": gts_arr.shape[0],
            "action_dim": action_dim,
            "robot_type": robot_type,
            "mean": gts_arr.mean(axis=0).tolist(),
            "std": gts_arr.std(axis=0).tolist(),
            "min": gts_arr.min(axis=0).tolist(),
            "max": gts_arr.max(axis=0).tolist(),
        }

        import json

        stats_file = repo_output_dir / "gt_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved statistics to {stats_file}")

        log_step(f"Finished processing repo: {repo}")

    print(f"\n{'='*60}")
    log_step("All repos processed successfully!")
    print(f"All done! Output saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize ground truth data for each repo id"
    )
    parser.add_argument(
        "--dataset_root", type=str, required=True, help="Root directory of the dataset"
    )
    parser.add_argument(
        "--config_name", type=str, required=True, help="Config file path"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Comma-separated repo ids to visualize",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/gt",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--vis_gap", type=int, default=50, help="Gap between samples for visualization"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to load (None = all)",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    main(
        args.dataset_root,
        args.config_name,
        args.repo_id,
        args.output_dir,
        args.vis_gap,
        args.num_samples,
    )
