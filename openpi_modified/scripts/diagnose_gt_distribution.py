"""Diagnose GT returns distribution for a value model config.

Usage:
    python scripts/diagnose_gt_distribution.py --config src/openpi/configs/cfg_pi06_seatbelt_value_selfplay_fixed.py

Outputs:
    - test_results/gt_distribution/<task_name>/stats.json
    - test_results/gt_distribution/<task_name>/histogram.png
    - test_results/gt_distribution/<task_name>/per_episode_stats.csv

Or compare two configs (e.g. fixed vs ablation):
    python scripts/diagnose_gt_distribution.py \\
        --config src/openpi/configs/cfg_pi06_seatbelt_value_selfplay_fixed.py \\
        --compare src/openpi/configs/cfg_pi06_seatbelt_value_selfplay_fixed_ablation_no_exclude_failures.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from openpi.training import data_loader_rl as _data_loader_rl
from openpi.training.config import Config
from openpi.training.frame_attributes_preprocessors.base import EpisodeBoundary

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def collect_returns(cfg) -> dict:
    """Build dataset (no model, no dataloader, no training) and extract precomputed returns.

    Returns dict with:
        - all_returns: np.ndarray of all GT returns across all datasets
        - per_dataset: list of {repo_id, returns, n_episodes, n_failures, n_negatives}
        - task_to_norm_length: dict
    """
    # Build DataConfig directly (skip dataloader, model, sharding, transforms)
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    data_config = dataclasses.replace(
        data_config,
        episode_fail=cfg.data.episode_fail,
        dataset_length=cfg.data.dataset_length,
    )

    # Create the multi-dataset directly — runs preprocessor pipeline + computes returns
    multi_ds = _data_loader_rl.create_anyverse_dataset(data_config, cfg.model)

    sub_datasets = multi_ds._datasets  # noqa: SLF001
    repo_ids = multi_ds.repo_ids
    task_to_norm = getattr(multi_ds, "task_to_norm_length", {})

    all_returns: list[np.ndarray] = []
    per_dataset: list[dict] = []

    for repo_id, ds in zip(repo_ids, sub_datasets, strict=True):
        returns = ds._precomputed_returns.numpy()  # noqa: SLF001
        n_eps = len(ds.episode_mapping)

        # Per-episode stats: count negatives + failure subclasses via the boundary tensor
        n_negs = 0
        n_failures = 0
        if ds.episode_boundary_tensor is not None and ds.is_negative_episode_tensor is not None:
            for valid_start, _valid_end in ds.episode_mapping.values():
                eb = int(ds.episode_boundary_tensor[valid_start])
                isneg = bool(ds.is_negative_episode_tensor[valid_start])
                if eb in (
                    EpisodeBoundary.UNCONFIRMED_NEGATIVE_END,
                    EpisodeBoundary.UNCONFIRMED_POSITIVE_END,
                ):
                    n_failures += 1
                if isneg:
                    n_negs += 1

        all_returns.append(returns)
        per_dataset.append(
            {
                "repo_id": repo_id,
                "n_frames": int(len(returns)),
                "n_episodes": int(n_eps),
                "n_failures": int(n_failures),
                "n_negatives": int(n_negs),
                "returns_mean": float(returns.mean()) if len(returns) else 0.0,
                "returns_min": float(returns.min()) if len(returns) else 0.0,
                "returns_max": float(returns.max()) if len(returns) else 0.0,
            }
        )
        logger.info(
            "[%s] frames=%d eps=%d neg=%d fail=%d returns: min=%.3f mean=%.3f max=%.3f",
            repo_id,
            len(returns),
            n_eps,
            n_negs,
            n_failures,
            float(returns.min()) if len(returns) else 0.0,
            float(returns.mean()) if len(returns) else 0.0,
            float(returns.max()) if len(returns) else 0.0,
        )

    return {
        "config_name": cfg.name,
        "all_returns": np.concatenate(all_returns) if all_returns else np.array([]),
        "per_dataset": per_dataset,
        "task_to_norm_length": task_to_norm,
    }


def compute_stats(returns: np.ndarray) -> dict:
    """Compute summary statistics for a returns array."""
    if len(returns) == 0:
        return {}

    return {
        "n_frames": int(len(returns)),
        "min": float(returns.min()),
        "max": float(returns.max()),
        "mean": float(returns.mean()),
        "median": float(np.median(returns)),
        "std": float(returns.std()),
        "p1": float(np.percentile(returns, 1)),
        "p5": float(np.percentile(returns, 5)),
        "p10": float(np.percentile(returns, 10)),
        "p25": float(np.percentile(returns, 25)),
        "p75": float(np.percentile(returns, 75)),
        "p90": float(np.percentile(returns, 90)),
        "p95": float(np.percentile(returns, 95)),
        "p99": float(np.percentile(returns, 99)),
        "frac_below_minus95": float((returns < -0.95).mean()),
        "frac_below_minus80": float((returns < -0.80).mean()),
        "frac_below_minus50": float((returns < -0.50).mean()),
        "frac_above_minus10": float((returns > -0.10).mean()),
        "frac_zero_or_positive": float((returns >= 0.0).mean()),
    }


def plot_histogram(results: list[dict], out_path: Path) -> None:
    """Plot histogram(s) of GT returns. Supports 1 or 2 configs side-by-side."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)

    for ax, result in zip(axes[0], results, strict=True):
        returns = result["all_returns"]
        if len(returns) == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            continue

        stats = compute_stats(returns)
        ax.hist(returns, bins=50, range=(-1.0, 0.05), edgecolor="black", alpha=0.7)
        ax.axvline(stats["median"], color="red", linestyle="--", label=f"median={stats['median']:.3f}")
        ax.axvline(stats["mean"], color="green", linestyle="--", label=f"mean={stats['mean']:.3f}")
        ax.axvline(-0.95, color="orange", linestyle=":", label="-0.95 threshold")
        ax.set_title(
            f"{result['config_name']}\n"
            f"n={stats['n_frames']:,} | "
            f"frac<-0.95={stats['frac_below_minus95']*100:.1f}% | "
            f"frac>=0={stats['frac_zero_or_positive']*100:.1f}%"
        )
        ax.set_xlabel("GT return")
        ax.set_ylabel("frame count")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(visible=True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    logger.info(f"Saved histogram to {out_path}")


def diagnose(config_path: str) -> dict:
    logger.info(f"Loading config: {config_path}")
    cfg = Config.fromfile(config_path).cfg
    logger.info(f"Collecting returns for {cfg.name} ...")
    result = collect_returns(cfg)
    result["stats"] = compute_stats(result["all_returns"])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to v3-style or v4-style config")
    parser.add_argument("--compare", default=None, help="Optional second config to compare against")
    parser.add_argument("--out-dir", default="test_results/gt_distribution", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = [diagnose(args.config)]
    if args.compare:
        results.append(diagnose(args.compare))

    # Print stats summary
    print("\n" + "=" * 80)
    for r in results:
        print(f"\n>>> {r['config_name']}")
        for k, v in r["stats"].items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    print("=" * 80 + "\n")

    # Save JSON
    suffix = "_compare" if args.compare else ""
    json_path = out_dir / f"stats{suffix}.json"
    with json_path.open("w") as f:
        json.dump(
            [
                {
                    "config_name": r["config_name"],
                    "stats": r["stats"],
                    "task_to_norm_length": {k: int(v) for k, v in r["task_to_norm_length"].items()},
                    "per_dataset": r["per_dataset"],
                }
                for r in results
            ],
            f,
            indent=2,
        )
    logger.info(f"Saved JSON stats to {json_path}")

    # Save histogram
    png_path = out_dir / f"histogram{suffix}.png"
    plot_histogram(results, png_path)


if __name__ == "__main__":
    main()
