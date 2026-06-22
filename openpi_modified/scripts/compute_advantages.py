#!/usr/bin/env python3
"""Compute advantages and indicators from pre-computed value parquets.

This script reads per-episode value parquets, computes advantages using
N-step or GAE methods, then computes percentile thresholds and binary
indicators for advantage-conditioned policy training (RECAP).

Supports three threshold modes for multi-dataset scenarios:
- per_dataset (default): Each dataset computes its own threshold independently.
- global: A single threshold is computed across all datasets.
- per_task: Thresholds are computed per task, aggregating same-name tasks across datasets.

Directory structure (input):
    <dataset_root>/
    ├── value_pred/chunk-000/episode_000000.parquet  # columns: pred_value, value_is_valid
    ├── meta/values_config.json
    └── meta/episodes.jsonl                          # (required for per_task mode)

Directory structure (output):
    <dataset_root>/
    ├── advantages/chunk-000/episode_000000.parquet  # columns: frame_index, advantage
    ├── indicators/chunk-000/episode_000000.parquet  # columns: frame_index, indicator
    └── meta/
        ├── advantages_config.json
        └── indicators_config.json

Usage:
    # Single dataset (backward compatible):
    python scripts/compute_advantages.py \\
        --dataset-dir <dataset_root> \\
        [--method n_step|gae] \\
        [--n-step 50] \\
        [--gamma 1.0] \\
        [--lambda_ 0.95] \\
        [--percentile 30.0] \\
        [--clip-percentile 1.0] \\
        [--num-bins 1]

    # Multi-dataset via config:
    python scripts/compute_advantages.py \\
        --config-name <config_name> \\
        --threshold-mode global|per_task \\
        [--method n_step|gae] ...
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Literal

import numpy as np
import pandas as pd
from tqdm import tqdm

from openpi.training.advantage import compute_n_step_advantage_numpy
from openpi.training.advantage_utils import clip_advantages
from openpi.training.advantage_utils import compute_indicators
from openpi.training.advantage_utils import compute_percentile_threshold

logger = logging.getLogger(__name__)


def _suffixed(base: str, suffix: str) -> str:
    """Append suffix with underscore separator, or return base unchanged."""
    return f"{base}_{suffix}" if suffix else base


# ---------------------------------------------------------------------------
# GAE numpy implementation
# ---------------------------------------------------------------------------


def compute_gae_advantage_numpy(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lambda_: float,
) -> np.ndarray:
    """Compute Generalized Advantage Estimation (GAE) using pure NumPy.

    Args:
        rewards: Shape [T] reward array.
        values: Shape [T+1] value array (includes bootstrap value).
        dones: Shape [T] done flag array (boolean).
        gamma: Discount factor.
        lambda_: GAE lambda parameter.

    Returns:
        Shape [T] advantage array.
    """
    seq_len = len(rewards)
    if seq_len == 0:
        return np.array([], dtype=np.float32)

    advantages = np.zeros(seq_len, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(seq_len)):
        not_done = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * values[t + 1] * not_done - values[t]
        gae = delta + gamma * lambda_ * not_done * gae
        advantages[t] = gae
    return advantages


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EpisodeInfo:
    """Metadata for a single episode parquet file."""

    chunk_idx: int
    episode_idx: int
    values_path: Path

    @property
    def chunk_str(self) -> str:
        return f"chunk-{self.chunk_idx:03d}"

    @property
    def episode_str(self) -> str:
        return f"episode_{self.episode_idx:06d}"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


_EPISODE_RE = re.compile(r"chunk-(?P<chunk>\d+)/episode_(?P<ep>\d+)\.parquet$")


def discover_episodes(value_pred_dir: Path) -> list[EpisodeInfo]:
    """Discover all episode parquet files under the value_pred directory.

    Args:
        value_pred_dir: Path to the ``value_pred/`` directory.

    Returns:
        Sorted list of ``EpisodeInfo`` instances.

    Raises:
        FileNotFoundError: If the value_pred directory does not exist.
        RuntimeError: If no episode parquets are found.
    """
    if not value_pred_dir.is_dir():
        raise FileNotFoundError(f"Value prediction directory not found: {value_pred_dir}")

    episodes: list[EpisodeInfo] = []
    for root, _dirs, files in os.walk(value_pred_dir):
        for fname in files:
            full_path = Path(root) / fname
            rel = full_path.relative_to(value_pred_dir).as_posix()
            m = _EPISODE_RE.search(rel)
            if m:
                episodes.append(
                    EpisodeInfo(
                        chunk_idx=int(m.group("chunk")),
                        episode_idx=int(m.group("ep")),
                        values_path=full_path,
                    )
                )

    if not episodes:
        raise RuntimeError(
            f"No episode parquets found under {value_pred_dir}. " "Expected pattern: chunk-*/episode_*.parquet"
        )

    episodes.sort(key=lambda e: (e.chunk_idx, e.episode_idx))
    logger.info("Discovered %d episodes across %d chunks", len(episodes), len({e.chunk_idx for e in episodes}))
    return episodes


# ---------------------------------------------------------------------------
# Per-episode advantage computation
# ---------------------------------------------------------------------------


def compute_episode_advantages(
    values: np.ndarray,
    method: str,
    n_step: int,
    gamma: float,
    lambda_: float,
) -> np.ndarray:
    """Compute advantages for a single episode.

    Args:
        values: Shape [T] value array from the parquet.
        method: Either ``"n_step"`` or ``"gae"``.
        n_step: N-step lookahead (used by n_step method).
        gamma: Discount factor.
        lambda_: GAE lambda (used by gae method).

    Returns:
        Shape [T] advantage array.
    """
    seq_len = len(values)
    if seq_len == 0:
        return np.array([], dtype=np.float32)

    # Rewards are all zeros for success trajectories
    rewards = np.zeros(seq_len, dtype=np.float32)
    dones = np.zeros(seq_len, dtype=bool)

    # Bootstrap value: append last value for T+1 element
    values_with_bootstrap = np.concatenate([values.astype(np.float32), values[-1:].astype(np.float32)])

    if method == "n_step":
        advantages = compute_n_step_advantage_numpy(
            rewards=rewards,
            values=values_with_bootstrap,
            dones=dones,
            n_step=n_step,
        )
    elif method == "gae":
        advantages = compute_gae_advantage_numpy(
            rewards=rewards,
            values=values_with_bootstrap,
            dones=dones,
            gamma=gamma,
            lambda_=lambda_,
        )
    else:
        raise ValueError(f"Unknown method: {method!r}. Expected 'n_step' or 'gae'.")

    return advantages


# ---------------------------------------------------------------------------
# Atomic file writing helpers
# ---------------------------------------------------------------------------


def _atomic_write_parquet(df: pd.DataFrame, target_path: Path) -> None:
    """Write a DataFrame to parquet atomically (write .tmp then rename).

    Args:
        df: DataFrame to write.
        target_path: Final destination path.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        suffix=".tmp",
        dir=target_path.parent,
    )
    os.close(tmp_fd)
    try:
        df.to_parquet(tmp_path, index=False, engine="pyarrow")
        os.replace(tmp_path, target_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _atomic_write_json(data: dict, target_path: Path) -> None:
    """Write JSON data atomically (write .tmp then rename).

    Args:
        data: Dictionary to serialize.
        target_path: Final destination path.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        suffix=".tmp",
        dir=target_path.parent,
    )
    os.close(tmp_fd)
    try:
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp_path, target_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


# ---------------------------------------------------------------------------
# Episode task loading (for per_task mode)
# ---------------------------------------------------------------------------


def load_episode_tasks(dataset_dir: Path) -> dict[int, str]:
    """Load episode → task mapping from LeRobot meta/episodes.jsonl.

    Args:
        dataset_dir: Root directory of the dataset.

    Returns:
        Dict mapping episode_index to task name.

    Raises:
        FileNotFoundError: If episodes.jsonl does not exist.
    """
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(
            f"episodes.jsonl not found at {episodes_path}. " "Required for per_task threshold mode."
        )

    episode_tasks: dict[int, str] = {}
    with open(episodes_path) as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped:
                continue
            entry = json.loads(stripped)
            ep_idx = entry["episode_index"]
            tasks = entry.get("tasks", [])
            if tasks:
                episode_tasks[ep_idx] = tasks[0]
            else:
                logger.warning("Episode %d has no tasks, skipping for per_task mode", ep_idx)

    logger.info("Loaded %d episode-task mappings from %s", len(episode_tasks), episodes_path)
    return episode_tasks


# ---------------------------------------------------------------------------
# Quantile-based cross-dataset utilities
# ---------------------------------------------------------------------------

_QUANTILE_POINTS = np.linspace(0, 1, 101)  # 0%, 1%, ..., 100%


def collect_per_task_quantiles(
    episodes: list[EpisodeInfo],
    episode_advantages: list[np.ndarray],
    episode_tasks: dict[int, str],
    clip_percentile: float,
) -> dict[str, dict]:
    """Collect per-task quantile summaries (memory-efficient).

    Args:
        episodes: List of episode info.
        episode_advantages: Corresponding advantage arrays.
        episode_tasks: Mapping from episode_index to task name.
        clip_percentile: Clipping percentile for variance reduction.

    Returns:
        Dict mapping task name to {"quantiles": list[float], "count": int}.
    """
    task_advs: dict[str, list[np.ndarray]] = {}
    for ep, adv in zip(episodes, episode_advantages, strict=True):
        if len(adv) == 0:
            continue
        task = episode_tasks.get(ep.episode_idx)
        if task is None:
            continue
        if task not in task_advs:
            task_advs[task] = []
        task_advs[task].append(adv)

    result: dict[str, dict] = {}
    for task, adv_list in task_advs.items():
        all_adv = np.concatenate(adv_list)
        if clip_percentile > 0:
            all_adv = clip_advantages(all_adv, clip_percentile)
        result[task] = {
            "quantiles": np.quantile(all_adv, _QUANTILE_POINTS).tolist(),
            "count": len(all_adv),
        }

    return result


def merge_task_quantiles(
    all_datasets_quantiles: list[dict[str, dict]],
) -> dict[str, dict]:
    """Merge per-task quantiles across datasets using weighted averaging.

    Args:
        all_datasets_quantiles: List of per-dataset task quantile dicts.

    Returns:
        Merged dict mapping task name to {"quantiles": list[float], "count": int}.
    """
    task_entries: dict[str, list[dict]] = {}
    for ds_quantiles in all_datasets_quantiles:
        for task, entry in ds_quantiles.items():
            if task not in task_entries:
                task_entries[task] = []
            task_entries[task].append(entry)

    merged: dict[str, dict] = {}
    for task, entries in task_entries.items():
        if len(entries) == 1:
            merged[task] = entries[0]
            continue

        total_count = sum(e["count"] for e in entries)
        if total_count == 0:
            continue

        merged_q = np.zeros(101, dtype=np.float64)
        for e in entries:
            w = e["count"] / total_count
            merged_q += w * np.array(e["quantiles"])

        merged[task] = {
            "quantiles": merged_q.tolist(),
            "count": total_count,
        }

    return merged


def compute_task_thresholds(
    merged_quantiles: dict[str, dict],
    percentile: float,
) -> dict[str, float]:
    """Compute per-task thresholds from merged quantiles.

    The threshold is interpolated from the 101-point quantile summary
    at the (100 - percentile) position. E.g., percentile=30 → index 70.

    Args:
        merged_quantiles: Merged task quantile dicts.
        percentile: Percentile for threshold (top p% are positive).

    Returns:
        Dict mapping task name to threshold value.
    """
    thresholds: dict[str, float] = {}
    target_idx = 100.0 - percentile  # e.g., percentile=30 → target_idx=70

    for task, entry in merged_quantiles.items():
        q = entry["quantiles"]
        # Interpolate from the 101-point grid (indices 0..100 correspond to 0%..100%)
        lower_idx = int(target_idx)
        upper_idx = min(lower_idx + 1, 100)
        frac = target_idx - lower_idx
        threshold = q[lower_idx] * (1 - frac) + q[upper_idx] * frac
        thresholds[task] = float(threshold)

    return thresholds


# ---------------------------------------------------------------------------
# Pipeline: Phase 1 - Compute and save advantages
# ---------------------------------------------------------------------------


def compute_and_save_advantages(
    dataset_dir: Path,
    method: str = "n_step",
    n_step: int = 50,
    gamma: float = 1.0,
    lambda_: float = 0.95,
    suffix: str = "",
    value_suffix: str = "",
) -> tuple[list[EpisodeInfo], list[np.ndarray], list[np.ndarray]]:
    """Discover episodes, compute per-episode advantages, and save parquets.

    This is the first phase of the pipeline (Steps 1-3).

    Args:
        dataset_dir: Root directory of the dataset.
        method: Advantage method ("n_step" or "gae").
        n_step: N-step lookahead window size.
        gamma: Discount factor.
        lambda_: GAE lambda parameter.
        suffix: Version suffix for output dirs (e.g. "v2" → "advantages_v2/").
        value_suffix: Version suffix for input value_pred dir (e.g. "v2" → "value_pred_v2/").

    Returns:
        Tuple of (episodes, episode_advantages, episode_frame_indices).
    """
    value_pred_dir = dataset_dir / _suffixed("value_pred", value_suffix)
    advantages_dir = dataset_dir / _suffixed("advantages", suffix)
    meta_dir = dataset_dir / "meta"

    # Step 1: Discover episodes
    logger.info("Discovering episodes in %s", value_pred_dir)
    episodes = discover_episodes(value_pred_dir)

    # Step 2: Compute per-episode advantages
    logger.info(
        "Computing advantages (method=%s, n_step=%d, gamma=%.4f, lambda=%.4f)",
        method,
        n_step,
        gamma,
        lambda_,
    )

    episode_advantages: list[np.ndarray] = []
    episode_frame_indices: list[np.ndarray] = []

    for ep in tqdm(episodes, desc="Computing advantages"):
        values_df = pd.read_parquet(ep.values_path)

        # Support both new format (pred_value + value_is_valid) and legacy (value + frame_index)
        if "pred_value" in values_df.columns:
            # New format from compute_values.py (aligned with test_rl.py)
            all_values = values_df["pred_value"].to_numpy(dtype=np.float32)
            if "value_is_valid" in values_df.columns:
                valid_mask = values_df["value_is_valid"].to_numpy(dtype=bool)
                if not valid_mask.all():
                    n_invalid = (~valid_mask).sum()
                    logger.warning(
                        "Episode %d has %d/%d invalid frames, using only valid frames",
                        ep.episode_idx,
                        n_invalid,
                        len(valid_mask),
                    )
                values = all_values[valid_mask]
                frame_indices = np.where(valid_mask)[0].astype(np.int64)
            else:
                values = all_values
                frame_indices = np.arange(len(values), dtype=np.int64)
        elif "value" in values_df.columns:
            # Legacy format
            values = values_df["value"].to_numpy(dtype=np.float32)
            if "frame_index" in values_df.columns:
                frame_indices = values_df["frame_index"].to_numpy()
            else:
                frame_indices = np.arange(len(values), dtype=np.int64)
        else:
            raise ValueError(
                f"Missing 'pred_value' or 'value' column in {ep.values_path}. "
                f"Found columns: {list(values_df.columns)}"
            )

        if len(values) == 0:
            logger.warning("Empty episode: %s, skipping", ep.values_path)
            episode_advantages.append(np.array([], dtype=np.float32))
            episode_frame_indices.append(np.array([], dtype=np.int64))
            continue

        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite values in {ep.values_path}: " f"NaN={np.isnan(values).sum()}, Inf={np.isinf(values).sum()}"
            )

        advantages = compute_episode_advantages(
            values=values,
            method=method,
            n_step=n_step,
            gamma=gamma,
            lambda_=lambda_,
        )
        episode_advantages.append(advantages)
        episode_frame_indices.append(frame_indices)

    # Step 3: Save per-episode advantage parquets
    logger.info("Saving advantage parquets to %s", advantages_dir)
    for ep, adv, fidx in tqdm(
        zip(episodes, episode_advantages, episode_frame_indices, strict=True),
        total=len(episodes),
        desc="Saving advantages",
    ):
        out_path = advantages_dir / ep.chunk_str / f"{ep.episode_str}.parquet"
        df_adv = pd.DataFrame(
            {
                "frame_index": fidx,
                "advantage": adv.astype(np.float32),
            }
        )
        _atomic_write_parquet(df_adv, out_path)

    # Save advantages metadata
    timestamp = datetime.now(UTC).isoformat()
    all_advantages = np.concatenate([a for a in episode_advantages if len(a) > 0])
    advantage_stats = (
        {
            "mean": float(np.mean(all_advantages)),
            "std": float(np.std(all_advantages)),
            "min": float(np.min(all_advantages)),
            "max": float(np.max(all_advantages)),
            "median": float(np.median(all_advantages)),
        }
        if len(all_advantages) > 0
        else {}
    )

    advantages_config = {
        "method": method,
        "n_step": n_step,
        "gamma": gamma,
        "lambda_": lambda_,
        "num_episodes": len(episodes),
        "num_samples": len(all_advantages),
        "advantage_stats": advantage_stats,
        "timestamp": timestamp,
    }
    if suffix:
        advantages_config["suffix"] = suffix
    if value_suffix:
        advantages_config["value_suffix"] = value_suffix
    meta_dir.mkdir(parents=True, exist_ok=True)
    config_filename = f"{_suffixed('advantages_config', suffix)}.json"
    _atomic_write_json(advantages_config, meta_dir / config_filename)
    logger.info("Saved advantages config: %s", meta_dir / config_filename)

    return episodes, episode_advantages, episode_frame_indices


# ---------------------------------------------------------------------------
# Pipeline: Phase 2 - Compute and save indicators
# ---------------------------------------------------------------------------


def compute_and_save_indicators(
    dataset_dir: Path,
    episodes: list[EpisodeInfo],
    episode_advantages: list[np.ndarray],
    episode_frame_indices: list[np.ndarray],
    percentile: float = 30.0,
    clip_percentile: float = 1.0,
    num_bins: int = 1,
    threshold_mode: Literal["per_dataset", "global", "per_task"] = "per_dataset",
    global_threshold: float | None = None,
    task_thresholds: dict[str, float] | None = None,
    episode_tasks: dict[int, str] | None = None,
    extra_metadata: dict | None = None,
    suffix: str = "",
) -> dict:
    """Compute indicators and save parquets + metadata.

    This is the second phase of the pipeline (Steps 4-6).

    Args:
        dataset_dir: Root directory of the dataset.
        episodes: List of episode info from Phase 1.
        episode_advantages: Advantage arrays from Phase 1.
        episode_frame_indices: Frame index arrays from Phase 1.
        percentile: Percentile for indicator threshold.
        clip_percentile: Clipping percentile for variance reduction.
        num_bins: Number of value-based bins.
        threshold_mode: Threshold computation mode.
        global_threshold: Pre-computed global threshold (for global mode).
        task_thresholds: Pre-computed per-task thresholds (for per_task mode).
        episode_tasks: Episode → task mapping (for per_task mode).
        extra_metadata: Additional metadata to include in indicators_config.json.
        suffix: Version suffix for output dirs (e.g. "v2" → "indicators_v2/").

    Returns:
        indicators_config dict.
    """
    indicators_dir = dataset_dir / _suffixed("indicators", suffix)
    meta_dir = dataset_dir / "meta"

    all_advantages = np.concatenate([a for a in episode_advantages if len(a) > 0])
    num_samples = len(all_advantages)

    if num_samples == 0:
        raise RuntimeError("No advantage samples computed. Check input data.")

    clipped_advantages = clip_advantages(all_advantages, clip_percentile) if clip_percentile > 0 else all_advantages

    # --- Compute indicators based on threshold_mode ---
    if threshold_mode == "per_dataset":
        threshold = compute_percentile_threshold(clipped_advantages, percentile, clip_percentile=None)
        all_indicators = compute_indicators(
            advantages=clipped_advantages,
            percentile=percentile,
            clip_percentile=None,
            num_bins=num_bins,
        )
        logger.info("Threshold (per_dataset): %.6f (percentile=%.1f)", threshold, percentile)

    elif threshold_mode == "global":
        if global_threshold is None:
            raise ValueError("global_threshold must be provided for global mode")
        threshold = global_threshold
        all_indicators = clipped_advantages > threshold
        logger.info("Threshold (global): %.6f", threshold)

    elif threshold_mode == "per_task":
        if task_thresholds is None or episode_tasks is None:
            raise ValueError("task_thresholds and episode_tasks must be provided for per_task mode")
        threshold = float(np.mean(list(task_thresholds.values())))  # for metadata summary

        # Build per-frame indicators using task-specific thresholds
        all_indicators = np.zeros(num_samples, dtype=bool)
        offset = 0
        for ep, adv in zip(episodes, episode_advantages, strict=True):
            n = len(adv)
            if n == 0:
                continue
            task = episode_tasks.get(ep.episode_idx)
            if task is not None and task in task_thresholds:
                ep_clipped = clip_advantages(adv, clip_percentile) if clip_percentile > 0 else adv
                all_indicators[offset : offset + n] = ep_clipped > task_thresholds[task]
            else:
                # Fallback: use dataset-level threshold for episodes without task info
                ep_clipped = clip_advantages(adv, clip_percentile) if clip_percentile > 0 else adv
                fallback_th = compute_percentile_threshold(clipped_advantages, percentile, clip_percentile=None)
                all_indicators[offset : offset + n] = ep_clipped > fallback_th
                logger.warning(
                    "Episode %d: task=%r not in task_thresholds, using fallback threshold %.6f",
                    ep.episode_idx,
                    task,
                    fallback_th,
                )
            offset += n

        logger.info(
            "Threshold (per_task): %d tasks, mean=%.6f",
            len(task_thresholds),
            threshold,
        )
    else:
        raise ValueError(f"Unknown threshold_mode: {threshold_mode!r}")

    positive_ratio = float(np.mean(all_indicators))
    logger.info("Positive ratio: %.4f", positive_ratio)

    # Split indicators back into per-episode arrays and save
    offset = 0
    episode_indicators: list[np.ndarray] = []
    for adv in episode_advantages:
        n = len(adv)
        if n > 0:
            episode_indicators.append(all_indicators[offset : offset + n])
            offset += n
        else:
            episode_indicators.append(np.array([], dtype=bool))

    logger.info("Saving indicator parquets to %s", indicators_dir)
    for ep, ind, fidx in tqdm(
        zip(episodes, episode_indicators, episode_frame_indices, strict=True),
        total=len(episodes),
        desc="Saving indicators",
    ):
        out_path = indicators_dir / ep.chunk_str / f"{ep.episode_str}.parquet"
        df_ind = pd.DataFrame(
            {
                "frame_index": fidx,
                "indicator": ind.astype(bool),
            }
        )
        _atomic_write_parquet(df_ind, out_path)

    # Save metadata
    timestamp = datetime.now(UTC).isoformat()
    advantage_stats = {
        "mean": float(np.mean(all_advantages)),
        "std": float(np.std(all_advantages)),
        "min": float(np.min(all_advantages)),
        "max": float(np.max(all_advantages)),
        "median": float(np.median(all_advantages)),
    }

    # Quantiles for cross-dataset merge (always save, useful for future merges)
    quantiles_100 = np.quantile(all_advantages, _QUANTILE_POINTS).tolist()

    indicators_config: dict = {
        "percentile": percentile,
        "threshold": float(threshold),
        "threshold_mode": threshold_mode,
        "clip_percentile": clip_percentile,
        "num_bins": num_bins,
        "positive_ratio": positive_ratio,
        "num_samples": num_samples,
        "num_episodes": len(episodes),
        "advantage_stats": advantage_stats,
        "per_task_stats": {
            "quantiles_100": quantiles_100,
            "count": num_samples,
        },
        "timestamp": timestamp,
    }

    if threshold_mode == "global" and global_threshold is not None:
        indicators_config["global_threshold"] = float(global_threshold)

    if threshold_mode == "per_task" and task_thresholds is not None:
        indicators_config["task_thresholds"] = {k: float(v) for k, v in task_thresholds.items()}

    if suffix:
        indicators_config["suffix"] = suffix

    if extra_metadata:
        indicators_config.update(extra_metadata)

    meta_dir.mkdir(parents=True, exist_ok=True)
    config_filename = f"{_suffixed('indicators_config', suffix)}.json"
    _atomic_write_json(indicators_config, meta_dir / config_filename)
    logger.info("Saved indicators config: %s", meta_dir / config_filename)

    # Summary
    logger.info(
        "Done. Episodes=%d, Samples=%d, Threshold=%.6f, Positive=%.2f%%",
        len(episodes),
        num_samples,
        threshold,
        positive_ratio * 100,
    )
    logger.info(
        "  Advantage stats: mean=%.6f std=%.6f min=%.6f max=%.6f",
        advantage_stats["mean"],
        advantage_stats["std"],
        advantage_stats["min"],
        advantage_stats["max"],
    )

    return indicators_config


# ---------------------------------------------------------------------------
# Backward-compatible run()
# ---------------------------------------------------------------------------


def run(
    dataset_dir: Path,
    method: str = "n_step",
    n_step: int = 50,
    gamma: float = 1.0,
    lambda_: float = 0.95,
    percentile: float = 30.0,
    clip_percentile: float = 1.0,
    num_bins: int = 1,
    suffix: str = "",
    value_suffix: str = "",
) -> None:
    """Run the full advantage and indicator computation pipeline.

    Backward-compatible entry point that processes a single dataset with
    per_dataset threshold mode.

    Args:
        dataset_dir: Root directory of the dataset.
        method: Advantage method ("n_step" or "gae").
        n_step: N-step lookahead window size.
        gamma: Discount factor.
        lambda_: GAE lambda parameter.
        percentile: Percentile for indicator threshold.
        clip_percentile: Clipping percentile for variance reduction.
        num_bins: Number of value-based bins for indicator computation.
        suffix: Version suffix for output dirs.
        value_suffix: Version suffix for input value_pred dir.
    """
    episodes, episode_advantages, episode_frame_indices = compute_and_save_advantages(
        dataset_dir=dataset_dir,
        method=method,
        n_step=n_step,
        gamma=gamma,
        lambda_=lambda_,
        suffix=suffix,
        value_suffix=value_suffix,
    )
    compute_and_save_indicators(
        dataset_dir=dataset_dir,
        episodes=episodes,
        episode_advantages=episode_advantages,
        episode_frame_indices=episode_frame_indices,
        percentile=percentile,
        clip_percentile=clip_percentile,
        num_bins=num_bins,
        threshold_mode="per_dataset",
        suffix=suffix,
    )


# ---------------------------------------------------------------------------
# Multi-dataset orchestration
# ---------------------------------------------------------------------------


def run_multi_dataset(
    dataset_dirs: list[Path],
    method: str = "n_step",
    n_step: int = 50,
    gamma: float = 1.0,
    lambda_: float = 0.95,
    percentile: float = 30.0,
    clip_percentile: float = 1.0,
    num_bins: int = 1,
    threshold_mode: Literal["per_dataset", "global", "per_task"] = "per_dataset",
    repo_ids: list[str] | None = None,
    config_name: str | None = None,
    suffix: str = "",
    value_suffix: str = "",
) -> None:
    """Run the pipeline across multiple datasets with cross-dataset threshold modes.

    Args:
        dataset_dirs: List of dataset root directories (one per repo_id).
        method: Advantage method ("n_step" or "gae").
        n_step: N-step lookahead window size.
        gamma: Discount factor.
        lambda_: GAE lambda parameter.
        percentile: Percentile for indicator threshold.
        clip_percentile: Clipping percentile for variance reduction.
        num_bins: Number of value-based bins.
        threshold_mode: Threshold computation mode.
        repo_ids: List of repo_id strings (for metadata).
        config_name: Config name used (for metadata).
        suffix: Version suffix for output dirs.
        value_suffix: Version suffix for input value_pred dir.
    """
    extra_metadata: dict = {}
    if repo_ids:
        extra_metadata["repo_ids"] = repo_ids
    if config_name:
        extra_metadata["config_name"] = config_name

    # Phase 1: Compute advantages for each dataset independently
    logger.info("=" * 60)
    logger.info("Phase 1: Computing advantages for %d datasets", len(dataset_dirs))
    logger.info("=" * 60)

    results: dict[Path, tuple[list[EpisodeInfo], list[np.ndarray], list[np.ndarray]]] = {}
    for d in dataset_dirs:
        logger.info("Processing dataset: %s", d)
        results[d] = compute_and_save_advantages(
            dataset_dir=d,
            method=method,
            n_step=n_step,
            gamma=gamma,
            lambda_=lambda_,
            suffix=suffix,
            value_suffix=value_suffix,
        )

    # Phase 2: Compute indicators based on threshold_mode
    logger.info("=" * 60)
    logger.info("Phase 2: Computing indicators (threshold_mode=%s)", threshold_mode)
    logger.info("=" * 60)

    if threshold_mode == "per_dataset":
        for d in dataset_dirs:
            episodes, ep_advs, ep_fidxs = results[d]
            compute_and_save_indicators(
                dataset_dir=d,
                episodes=episodes,
                episode_advantages=ep_advs,
                episode_frame_indices=ep_fidxs,
                percentile=percentile,
                clip_percentile=clip_percentile,
                num_bins=num_bins,
                threshold_mode="per_dataset",
                extra_metadata=extra_metadata,
                suffix=suffix,
            )

    elif threshold_mode == "global":
        # Aggregate all advantages across datasets
        all_advs_list = []
        for d in dataset_dirs:
            _, ep_advs, _ = results[d]
            all_advs_list.extend(a for a in ep_advs if len(a) > 0)

        all_advs = np.concatenate(all_advs_list)
        clipped = clip_advantages(all_advs, clip_percentile) if clip_percentile > 0 else all_advs
        global_threshold = compute_percentile_threshold(clipped, percentile, clip_percentile=None)
        logger.info(
            "Global threshold: %.6f (from %d samples across %d datasets)",
            global_threshold,
            len(all_advs),
            len(dataset_dirs),
        )

        for d in dataset_dirs:
            episodes, ep_advs, ep_fidxs = results[d]
            compute_and_save_indicators(
                dataset_dir=d,
                episodes=episodes,
                episode_advantages=ep_advs,
                episode_frame_indices=ep_fidxs,
                percentile=percentile,
                clip_percentile=clip_percentile,
                num_bins=num_bins,
                threshold_mode="global",
                global_threshold=global_threshold,
                extra_metadata=extra_metadata,
                suffix=suffix,
            )

    elif threshold_mode == "per_task":
        # Phase 2a: Collect per-task quantiles from each dataset
        all_quantiles: list[dict[str, dict]] = []
        all_episode_tasks: dict[Path, dict[int, str]] = {}

        for d in dataset_dirs:
            ep_tasks = load_episode_tasks(d)
            all_episode_tasks[d] = ep_tasks
            episodes, ep_advs, _ = results[d]
            q = collect_per_task_quantiles(episodes, ep_advs, ep_tasks, clip_percentile)
            all_quantiles.append(q)
            logger.info("  %s: %d tasks, %d episodes with task info", d, len(q), len(ep_tasks))

        # Phase 2b: Merge quantiles across datasets
        merged = merge_task_quantiles(all_quantiles)
        logger.info("Merged %d unique tasks across %d datasets", len(merged), len(dataset_dirs))

        # Phase 2c: Compute per-task thresholds
        task_thresholds = compute_task_thresholds(merged, percentile)
        for task, th in sorted(task_thresholds.items()):
            logger.info("  Task %r: threshold=%.6f (count=%d)", task, th, merged[task]["count"])

        # Phase 2d: Generate indicators for each dataset
        extra_metadata["per_task_quantiles"] = {
            task: {"quantiles": entry["quantiles"], "count": entry["count"]} for task, entry in merged.items()
        }
        for d in dataset_dirs:
            episodes, ep_advs, ep_fidxs = results[d]
            compute_and_save_indicators(
                dataset_dir=d,
                episodes=episodes,
                episode_advantages=ep_advs,
                episode_frame_indices=ep_fidxs,
                percentile=percentile,
                clip_percentile=clip_percentile,
                num_bins=num_bins,
                threshold_mode="per_task",
                task_thresholds=task_thresholds,
                episode_tasks=all_episode_tasks[d],
                extra_metadata=extra_metadata,
                suffix=suffix,
            )

    logger.info("=" * 60)
    logger.info("All %d datasets processed successfully.", len(dataset_dirs))
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Compute advantages and indicators from pre-computed values.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--dataset-dir",
        type=Path,
        help="Root directory of a single dataset containing value_pred/ and meta/.",
    )
    input_group.add_argument(
        "--config-name",
        type=str,
        help="Training config name to auto-read repo_ids, root_dir, and advantage params.",
    )

    # Threshold mode
    parser.add_argument(
        "--threshold-mode",
        type=str,
        choices=["per_dataset", "global", "per_task"],
        default="per_dataset",
        help="Threshold computation mode for multi-dataset scenarios.",
    )

    # Advantage parameters (override config values when specified)
    parser.add_argument(
        "--method",
        type=str,
        choices=["n_step", "gae"],
        default=None,
        help="Advantage computation method.",
    )
    parser.add_argument(
        "--n-step",
        type=int,
        default=None,
        help="N-step lookahead for n_step method.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Discount factor.",
    )
    parser.add_argument(
        "--lambda_",
        type=float,
        default=None,
        help="GAE lambda parameter.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=None,
        help="Percentile for indicator threshold (top p%% are positive).",
    )
    parser.add_argument(
        "--clip-percentile",
        type=float,
        default=None,
        help="Clipping percentile for variance reduction (0 to disable).",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=None,
        help="Number of value-based bins for indicator computation.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Version suffix for output dirs/files (e.g. 'v2' → advantages_v2/, indicators_v2/).",
    )
    parser.add_argument(
        "--value-suffix",
        type=str,
        default="",
        help="Version suffix for input value_pred dir (e.g. 'v2' → read from value_pred_v2/).",
    )

    return parser.parse_args(argv)


# Default parameter values (used when neither CLI nor config specifies a value)
_DEFAULTS = {
    "method": "n_step",
    "n_step": 50,
    "gamma": 1.0,
    "lambda_": 0.95,
    "percentile": 30.0,
    "clip_percentile": 1.0,
    "num_bins": 1,
}


def _resolve_param(args_val, config_val, default_val):
    """Resolve parameter: CLI > config > default."""
    if args_val is not None:
        return args_val
    if config_val is not None:
        return config_val
    return default_val


def main(argv: list[str] | None = None) -> None:
    """Entry point for the compute_advantages script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args(argv)
    threshold_mode = args.threshold_mode

    if args.config_name:
        # Multi-dataset mode via config
        import openpi.training.config as _config

        config = _config.get_config(args.config_name)
        data_factory = config.data
        data_config = data_factory.create(config.assets_dirs, config.model)

        # Extract repo_ids and root_dir
        repo_id_raw = data_factory.repo_id
        if isinstance(repo_id_raw, list):
            repo_ids = repo_id_raw
        elif repo_id_raw is not None:
            repo_ids = [repo_id_raw]
        else:
            # Fallback to data_config
            repo_id_raw = data_config.repo_id
            if isinstance(repo_id_raw, list):
                repo_ids = repo_id_raw
            elif repo_id_raw is not None:
                repo_ids = [repo_id_raw]
            else:
                logger.error("Could not determine repo_ids from config '%s'", args.config_name)
                sys.exit(1)

        root_dir = data_factory.root_dir or data_config.root_dir
        if root_dir is None:
            logger.error("Could not determine root_dir from config '%s'", args.config_name)
            sys.exit(1)

        dataset_dirs = [Path(root_dir) / rid for rid in repo_ids]

        # Validate dataset dirs exist
        value_pred_dirname = _suffixed("value_pred", args.value_suffix)
        for d in dataset_dirs:
            if not d.is_dir():
                logger.error("Dataset directory not found: %s", d)
                sys.exit(1)
            if not (d / value_pred_dirname).is_dir():
                logger.error("Value prediction directory not found: %s", d / value_pred_dirname)
                sys.exit(1)

        # Resolve parameters: CLI > config > defaults
        method = _resolve_param(args.method, getattr(data_config, "advantage_method", None), _DEFAULTS["method"])
        n_step = _resolve_param(args.n_step, getattr(data_config, "advantage_n_step", None), _DEFAULTS["n_step"])
        gamma = _resolve_param(args.gamma, getattr(data_config, "advantage_gamma", None), _DEFAULTS["gamma"])
        lambda_ = _resolve_param(args.lambda_, getattr(data_config, "advantage_lambda", None), _DEFAULTS["lambda_"])
        percentile = _resolve_param(
            args.percentile, getattr(data_config, "advantage_percentile", None), _DEFAULTS["percentile"]
        )
        clip_percentile = _resolve_param(
            args.clip_percentile, getattr(data_config, "advantage_clip_percentile", None), _DEFAULTS["clip_percentile"]
        )
        num_bins = _resolve_param(
            args.num_bins, getattr(data_config, "advantage_num_bins", None), _DEFAULTS["num_bins"]
        )

        logger.info("Config: %s", args.config_name)
        logger.info("Repo IDs: %s", repo_ids)
        logger.info("Root dir: %s", root_dir)
        logger.info("Threshold mode: %s", threshold_mode)
        logger.info(
            "Params: method=%s n_step=%d gamma=%.4f lambda=%.4f " "percentile=%.1f clip_percentile=%.1f num_bins=%d",
            method,
            n_step,
            gamma,
            lambda_,
            percentile,
            clip_percentile,
            num_bins,
        )

        if len(dataset_dirs) == 1 and threshold_mode == "global":
            logger.warning("Only 1 dataset with global mode — equivalent to per_dataset.")

        run_multi_dataset(
            dataset_dirs=dataset_dirs,
            method=method,
            n_step=n_step,
            gamma=gamma,
            lambda_=lambda_,
            percentile=percentile,
            clip_percentile=clip_percentile,
            num_bins=num_bins,
            threshold_mode=threshold_mode,
            repo_ids=repo_ids,
            config_name=args.config_name,
            suffix=args.suffix,
            value_suffix=args.value_suffix,
        )

    else:
        # Single dataset mode (backward compatible)
        dataset_dir: Path = args.dataset_dir.resolve()
        if not dataset_dir.is_dir():
            logger.error("Dataset directory not found: %s", dataset_dir)
            sys.exit(1)

        value_pred_dir = dataset_dir / _suffixed("value_pred", args.value_suffix)
        if not value_pred_dir.is_dir():
            logger.error("Value prediction directory not found: %s", value_pred_dir)
            sys.exit(1)

        # Resolve parameters: CLI > defaults
        method = _resolve_param(args.method, None, _DEFAULTS["method"])
        n_step = _resolve_param(args.n_step, None, _DEFAULTS["n_step"])
        gamma = _resolve_param(args.gamma, None, _DEFAULTS["gamma"])
        lambda_ = _resolve_param(args.lambda_, None, _DEFAULTS["lambda_"])
        percentile = _resolve_param(args.percentile, None, _DEFAULTS["percentile"])
        clip_percentile = _resolve_param(args.clip_percentile, None, _DEFAULTS["clip_percentile"])
        num_bins = _resolve_param(args.num_bins, None, _DEFAULTS["num_bins"])

        logger.info("Dataset directory: %s", dataset_dir)
        logger.info(
            "Config: method=%s n_step=%d gamma=%.4f lambda=%.4f " "percentile=%.1f clip_percentile=%.1f num_bins=%d",
            method,
            n_step,
            gamma,
            lambda_,
            percentile,
            clip_percentile,
            num_bins,
        )

        if threshold_mode != "per_dataset":
            logger.warning(
                "threshold_mode=%s with single --dataset-dir: "
                "using per_dataset mode (use --config-name for multi-dataset).",
                threshold_mode,
            )
            threshold_mode = "per_dataset"

        run(
            dataset_dir=dataset_dir,
            method=method,
            n_step=n_step,
            gamma=gamma,
            lambda_=lambda_,
            percentile=percentile,
            clip_percentile=clip_percentile,
            num_bins=num_bins,
            suffix=args.suffix,
            value_suffix=args.value_suffix,
        )


if __name__ == "__main__":
    main()
