#!/usr/bin/env python3
"""Compute value predictions and save as per-episode parquet files.

This script:
1. Loads a pre-trained value function model
2. Iterates over each repo_id independently (no MultiRLAnyverseDataset)
3. Saves per-episode parquet files to value_pred/chunk-XXX/episode_XXXXXX.parquet
   Format: pred_value (float32) + value_is_valid (bool), aligned with test_rl.py
4. Saves metadata to meta/values_config.json

Supports multi-GPU data-parallel inference via --num_gpus. Model params are
replicated across all GPUs, and each batch is sharded along the batch dimension.

Usage:
    # Single repo_id:
    python scripts/compute_values.py \\
        --config_name <config> \\
        --ckpt_dir <path> \\
        --repo_id <repo_id>

    # Multiple repo_ids with 4 GPUs:
    python scripts/compute_values.py \\
        --config_name <value_model_config> \\
        --ckpt_dir <path> \\
        --data_config_name <data_config> \\
        --num_gpus 4

    # All repo_ids from config:
    python scripts/compute_values.py \\
        --config_name <config> \\
        --ckpt_dir <path>
"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import UTC
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from tqdm import tqdm

# Delay JAX imports until after CUDA_VISIBLE_DEVICES is set.
_jax_imported = False


def _ensure_jax():
    global _jax_imported, jax, jnp  # noqa: PLW0603, PLW0602
    if not _jax_imported:
        import jax as _jax
        import jax.numpy as _jnp

        globals()["jax"] = _jax
        globals()["jnp"] = _jnp
        _jax_imported = True
    return jax, jnp


logger = logging.getLogger(__name__)


def _suffixed(base: str, suffix: str) -> str:
    """Append suffix with underscore separator, or return base unchanged."""
    return f"{base}_{suffix}" if suffix else base


def _resume_marker_path(dataset_dir: Path, suffix: str) -> Path:
    """Path to the completion marker written last by save_values_config().

    meta/values_config[_suffix].json is the final file written during a successful
    run; checking its existence (not the output dir's existence) is what prevents
    --resume from silently skipping half-written runs.
    """
    output_dirname = f"value_pred_{suffix}" if suffix else "value_pred"
    return dataset_dir / output_dirname / "meta" / f"{_suffixed('values_config', suffix)}.json"


def _should_skip_repo_for_resume(dataset_dir: Path, suffix: str) -> bool:
    """Return True iff --resume should skip this repo_id because a completion marker exists.

    Intentionally checks the marker file, NOT `dataset_dir / 'value_pred*' .is_dir()`:
    a crashed run leaves the directory (and partial parquet shards) behind without
    ever writing the marker, and must be re-run.
    """
    return _resume_marker_path(dataset_dir, suffix).exists()


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


class RenameKeysTransform:
    """Rename keys from LeRobot format to Gr00t format.

    LeRobot uses "." separator; Gr00t expects "/".
    """

    def __init__(self, key_map: dict[str, str]):
        self.key_map = key_map

    def __call__(self, data: dict) -> dict:
        result = {}
        for old_key, value in data.items():
            new_key = self.key_map.get(old_key)
            if new_key is None:
                new_key = old_key.replace(".", "/") if old_key.startswith("observation.") else old_key
            result[new_key] = value

        if "action_mask" not in result and "action" in result:
            action_shape = result["action"].shape
            result["action_mask"] = np.ones(action_shape[:-1] + (1,), dtype=bool)

        return result


class TorchResizeTransform:
    """PyTorch-based image resize for performance.

    Much faster than JAX resize_with_pad called in Observation.from_dict.
    """

    def __init__(self, height: int = 224, width: int = 224):
        self.height = height
        self.width = width

    def __call__(self, data: dict) -> dict:
        if "observation" not in data:
            return data

        import torch

        obs = data["observation"]
        for key, value in list(obs.items()):
            if "image" not in key.lower():
                continue

            if isinstance(value, np.ndarray):
                value_torch = torch.from_numpy(value)
            elif isinstance(value, torch.Tensor):
                value_torch = value
            else:
                continue

            shape = value_torch.shape
            if len(shape) != 3:
                continue

            if shape[-1] <= 4:  # (H, W, C)
                h, w = shape[0], shape[1]
            else:  # (C, H, W)
                h, w = shape[1], shape[2]

            if h != self.height or w != self.width:
                from openpi.shared import image_tools

                value_resized = image_tools.resize_with_pad_torch(value_torch, self.height, self.width)
                obs[key] = value_resized.numpy() if isinstance(value_resized, torch.Tensor) else value_resized

        return data


def _simple_collate_fn(batch):
    """Pickle-safe numpy collate for DataLoader workers."""
    if not batch:
        return {}

    def _tree_stack(items):
        if not items:
            return items
        first = items[0]

        if isinstance(first, dict):
            result = {}
            all_keys = set()
            for item in items:
                if not isinstance(item, dict):
                    raise TypeError(f"Expected dict, got {type(item).__name__}")
                all_keys.update(item.keys())
            # sorted() ensures deterministic dict insertion order across workers;
            # downstream _pad_batch_to_align and JIT tracing rely on a stable pytree layout.
            for k in sorted(all_keys):
                values = [item[k] for item in items if k in item]
                if values:
                    result[k] = _tree_stack(values)
            return result
        if isinstance(first, (list, tuple)):  # noqa: UP038
            return [_tree_stack([item[i] for item in items]) for i in range(len(first))]
        return np.stack([np.asarray(item) for item in items], axis=0)

    return _tree_stack(batch)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def assign_datasets_balanced(dataset_sizes: list[int], num_gpus: int) -> list[list[int]]:
    """Greedy load-balanced GPU assignment."""
    if num_gpus <= 0:
        return []
    if not dataset_sizes:
        return [[] for _ in range(num_gpus)]

    datasets = [(size, idx) for idx, size in enumerate(dataset_sizes)]
    gpu_assignments: list[list[int]] = [[] for _ in range(num_gpus)]
    gpu_loads = [0] * num_gpus

    datasets.sort(reverse=True, key=lambda x: x[0])
    for size, idx in datasets:
        min_gpu = min(range(num_gpus), key=lambda g: gpu_loads[g])
        gpu_assignments[min_gpu].append(idx)
        gpu_loads[min_gpu] += size

    return gpu_assignments


def create_dataset_from_config(config, repo_id_list_override=None):
    """Create dataset using the training config.

    .. deprecated::
        Use :func:`create_single_dataset` with per-repo_id iteration instead.
        Kept temporarily for backward compatibility with external callers.
    """
    from openpi.training import rl_dataset

    data_config_factory = config.data
    data_config = data_config_factory.create(config.assets_dirs, config.model)

    data_config = dataclasses.replace(
        data_config,
        episode_fail=data_config_factory.episode_fail,
        dataset_length=data_config_factory.dataset_length,
    )

    if repo_id_list_override is not None:
        data_config = dataclasses.replace(data_config, repo_id=repo_id_list_override)

    repo_id = data_config.repo_id
    if isinstance(repo_id, list):
        dataset = rl_dataset.MultiRLAnyverseDataset(data_config, config.model.action_horizon)
        logger.info(f"Multi-dataset loaded: {len(dataset)} total samples")
        return dataset

    dataset_root = Path(data_config.root_dir) / repo_id
    dataset = rl_dataset.LeRobotRLDataset(
        repo_id=repo_id,
        root=str(dataset_root),
    )
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    return dataset


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_value_model(model_config, checkpoint_path: str):
    """Load pre-trained value function model."""
    import flax.nnx as nnx

    from openpi.models.model import restore_params

    _jax, _jnp = _ensure_jax()

    params_path = checkpoint_path
    if not os.path.exists(os.path.join(checkpoint_path, "_METADATA")):
        params_subdir = os.path.join(checkpoint_path, "params")
        if os.path.exists(os.path.join(params_subdir, "_METADATA")):
            params_path = params_subdir

    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Value function checkpoint not found at {params_path}.")

    logger.info(f"Loading value function from {params_path}")
    params = restore_params(params_path, restore_type=np.ndarray)

    cpu_device = _jax.devices("cpu")[0]
    logger.info("Initializing model on CPU")
    with _jax.default_device(cpu_device):
        rng = _jax.random.key(0)
        model = model_config.create(rng)

        graphdef, state = nnx.split(model)
        state.replace_by_pure_dict(params)
        model = nnx.merge(graphdef, state)

    model.eval()
    logger.info("Value function loaded successfully")
    return model


def replicate_model_params(model, mesh):
    """Replicate model parameters across all devices in the mesh for data-parallel inference."""
    import flax.nnx as nnx

    _jax, _ = _ensure_jax()

    replicated_sharding = _jax.sharding.NamedSharding(mesh, _jax.sharding.PartitionSpec())
    graphdef, state = nnx.split(model)
    state = _jax.device_put(state, replicated_sharding)
    model = nnx.merge(graphdef, state)
    model.eval()
    logger.info(f"Model params replicated across {len(mesh.devices.flat)} devices")
    return model


# ---------------------------------------------------------------------------
# Value computation
# ---------------------------------------------------------------------------


def _pad_batch_to_align(batch_dict: dict, aligned_size: int) -> tuple[dict, int]:
    """Pad a batch dict so the batch dimension is divisible by num_gpus.

    Returns:
        (padded_batch, original_size): padded batch dict and original batch size for truncation.
    """
    first_val = next(v for v in batch_dict.values() if isinstance(v, np.ndarray))
    original_size = first_val.shape[0]

    if original_size >= aligned_size:
        return batch_dict, original_size

    pad_size = aligned_size - original_size

    def _pad_value(val):
        if isinstance(val, dict):
            return {k: _pad_value(v) for k, v in val.items()}
        if not isinstance(val, np.ndarray):
            return val
        pad_widths = [(0, pad_size)] + [(0, 0)] * (val.ndim - 1)
        return np.pad(val, pad_widths, mode="edge")

    padded = {k: _pad_value(v) for k, v in batch_dict.items()}
    return padded, original_size


def compute_values_for_dataset(
    model,
    dataset,
    config,
    value_checkpoint: str,
    batch_size: int = 64,
    num_gpus: int = 1,
    score_observation_jit=None,
    data_sharding=None,
    num_workers: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute value predictions for all samples in a single-repo dataset.

    When num_gpus > 1, uses JAX data-parallel inference: model params are
    replicated across devices, and each batch is sharded along the batch
    dimension across all GPUs.

    Returns:
        values: [num_samples] value predictions
        episode_indices: [num_samples] episode index per sample
        frame_indices: [num_samples] frame index per sample
    """
    from torch.utils.data import DataLoader

    from openpi.shared import nnx_utils
    from openpi.training import data_loader_rl as _data_loader

    _jax, _jnp = _ensure_jax()

    # Setup multi-GPU sharding if requested (reuse passed-in sharding to avoid JIT re-trace)
    if data_sharding is None and num_gpus > 1:
        from openpi.training import sharding as _sharding

        mesh = _sharding.make_mesh(num_fsdp_devices=1)
        data_sharding = _jax.sharding.NamedSharding(mesh, _jax.sharding.PartitionSpec(_sharding.DATA_AXIS))

    if num_gpus > 1:
        # Align batch_size to num_gpus
        batch_size = ((batch_size + num_gpus - 1) // num_gpus) * num_gpus
        logger.info(f"Multi-GPU: batch_size aligned to {batch_size} for {num_gpus} GPUs")

    if score_observation_jit is None:
        score_observation_jit = nnx_utils.module_jit(model.score_observation)

    num_samples = len(dataset)
    num_batches = (num_samples + batch_size - 1) // batch_size
    logger.info(f"Computing values for {num_samples} samples in {num_batches} batches")

    # Build transforms — reuse the same pipeline as training/test_rl.py
    data_cfg = config.data.create(config.assets_dirs, config.model)

    # Load norm stats from checkpoint
    checkpoint_dir = value_checkpoint
    while checkpoint_dir and not os.path.exists(os.path.join(checkpoint_dir, "_METADATA")):
        parent = os.path.dirname(checkpoint_dir)
        if parent == checkpoint_dir:
            break
        checkpoint_dir = parent

    if checkpoint_dir:
        for asset_id in [data_cfg.asset_id, "norm_stats"]:
            candidate_path = os.path.join(checkpoint_dir, "assets", asset_id)
            if os.path.exists(candidate_path):
                import openpi.shared.normalize as _normalize

                norm_stats = _normalize.load(candidate_path)
                data_cfg = dataclasses.replace(data_cfg, norm_stats=norm_stats)
                logger.info(f"Loaded norm_stats from {candidate_path}")
                break

    transformed_dataset = _data_loader.transform_rl_dataset(
        dataset, data_cfg, skip_norm_stats=(data_cfg.norm_stats is None)
    )

    torch_loader = DataLoader(
        transformed_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        # "fork" is intentional: runs on Linux GPU servers only; avoids
        # re-serializing CUDA/JAX contexts that "spawn" would require.
        multiprocessing_context="fork" if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        collate_fn=_simple_collate_fn,
        drop_last=False,
    )
    logger.info(f"Created DataLoader with num_workers={num_workers}")

    all_episode_indices = []
    all_frame_indices = []
    all_values = []

    rng = _jax.random.key(0)

    logger.info("Running value inference...")
    for _batch_idx, batch in enumerate(tqdm(torch_loader, total=num_batches, desc="Computing values")):
        from openpi.models.model import Observation

        rng, subkey = _jax.random.split(rng)

        # Pad last batch to fixed batch_size — avoids JAX re-trace on different shapes
        current_batch, original_batch_size = _pad_batch_to_align(batch, batch_size)

        obs = Observation.from_dict(current_batch)

        # Shard observation across GPUs for data-parallel inference
        if data_sharding is not None:
            obs = _jax.tree.map(
                lambda x: _jax.device_put(x, data_sharding) if hasattr(x, "shape") else x,
                obs,
            )

        score = score_observation_jit(subkey, obs)

        # score_observation returns a JAX array [b, value_bins]
        # value_bins=1: already sigmoid-transformed scalar in [-1, 0]
        # value_bins>1: distributional logits, need conversion
        score_np = np.asarray(score)
        if score_np.shape[-1] == 1:
            batch_values = score_np.flatten()
        elif hasattr(model, "value_distribution_to_scalar"):
            batch_values = np.asarray(model.value_distribution_to_scalar(score)).flatten()
        else:
            batch_values = score_np.mean(axis=-1).flatten()

        # Truncate padded results
        if original_batch_size < len(batch_values):
            batch_values = batch_values[:original_batch_size]

        all_values.append(batch_values)

        # Collect native keys from batch (truncate if padded)
        if "episode_index" in batch:
            ep_idx = np.asarray(batch["episode_index"]).flatten()
            if original_batch_size < len(ep_idx):
                ep_idx = ep_idx[:original_batch_size]
            all_episode_indices.append(ep_idx)
        if "frame_index" in batch:
            fr_idx = np.asarray(batch["frame_index"]).flatten()
            if original_batch_size < len(fr_idx):
                fr_idx = fr_idx[:original_batch_size]
            all_frame_indices.append(fr_idx)
    values = np.concatenate(all_values)
    episode_indices = (
        np.concatenate(all_episode_indices) if all_episode_indices else np.zeros(len(values), dtype=np.int32)
    )
    frame_indices = np.concatenate(all_frame_indices) if all_frame_indices else np.arange(len(values), dtype=np.int32)

    logger.info(f"Computed {len(values)} values, range [{values.min():.4f}, {values.max():.4f}]")

    return values, episode_indices, frame_indices


# ---------------------------------------------------------------------------
# Saving per-episode parquets
# ---------------------------------------------------------------------------


def _find_episode_data_parquet(output_dir: Path, ep_idx: int) -> tuple[int, Path] | None:
    """Find the original data parquet for an episode, searching all chunk dirs.

    Returns:
        (chunk_id, path) if found, None otherwise.
    """
    data_dir = output_dir / "data"
    if not data_dir.exists():
        return None
    filename = f"episode_{ep_idx:06d}.parquet"
    for chunk_dir in sorted(data_dir.glob("chunk-*")):
        candidate = chunk_dir / filename
        if candidate.exists():
            chunk_str = chunk_dir.name  # "chunk-000"
            chunk_id = int(chunk_str.split("-")[1])
            return chunk_id, candidate
    return None


def _get_episode_total_frames(data_parquet_path: Path) -> int:
    """Read original data parquet to get the total frame count for an episode.

    Same approach as test_rl.py:536-538.
    """
    df_orig = pd.read_parquet(data_parquet_path)
    return len(df_orig)


def save_values_per_episode(
    values: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    output_dir: Path,
    dataset_meta=None,
    suffix: str = "",
) -> int:
    """Save values as per-episode parquet files (test_rl.py-compatible format).

    Output format (aligned with test_rl.py save_episode_pred):
        value_pred/chunk-XXX/episode_XXXXXX.parquet
        Columns: pred_value (float32), value_is_valid (bool)
        Row count = full episode length (uninferred frames are 0.0/False)

    Args:
        values: [N] value predictions
        episode_indices: [N] episode index per sample
        frame_indices: [N] frame index per sample
        output_dir: Root directory for output (value_pred/ will be created here)
        dataset_meta: Optional dataset metadata for chunk computation
        suffix: Optional version suffix (e.g. "v2" → "value_pred_v2/")

    Returns:
        Number of episodes saved
    """
    value_pred_dir = output_dir / _suffixed("value_pred", suffix)

    unique_episodes = np.unique(episode_indices)
    logger.info(f"Saving values for {len(unique_episodes)} episodes to {value_pred_dir}")

    for ep_idx in tqdm(unique_episodes, desc="Saving value parquets"):
        ep_mask = episode_indices == ep_idx

        ep_values = values[ep_mask]
        ep_frames = frame_indices[ep_mask]

        # Find chunk_id and total_frames from original data parquet
        found = _find_episode_data_parquet(output_dir, int(ep_idx))
        if found is not None:
            chunk_id, data_parquet_path = found
            total_frames = _get_episode_total_frames(data_parquet_path)
        else:
            # Fallback when data parquet not found
            chunk_size = 1000  # LeRobot default
            if dataset_meta is not None and hasattr(dataset_meta, "get_episode_chunk"):
                chunk_id = dataset_meta.get_episode_chunk(int(ep_idx))
            else:
                chunk_id = int(ep_idx) // chunk_size
            total_frames = int(ep_frames.max()) + 1 if len(ep_frames) > 0 else 0
            logger.warning(
                "Could not find original data parquet for episode %d, " "using chunk_id=%d, total_frames=%d",
                int(ep_idx),
                chunk_id,
                total_frames,
            )

        chunk_dir = value_pred_dir / f"chunk-{chunk_id:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        # Fill to full episode length (same as test_rl.py:540-544)
        pred_value_all = np.zeros(total_frames, dtype=np.float32)
        value_is_valid = np.zeros(total_frames, dtype=bool)
        pred_value_all[ep_frames.astype(int)] = ep_values.astype(np.float32)
        value_is_valid[ep_frames.astype(int)] = True

        values_df = pd.DataFrame(
            {
                "pred_value": pred_value_all,
                "value_is_valid": value_is_valid,
            }
        )

        # Atomic write
        output_path = chunk_dir / f"episode_{int(ep_idx):06d}.parquet"
        tmp_path = output_path.with_suffix(".parquet.tmp")
        values_df.to_parquet(tmp_path, index=False)
        tmp_path.rename(output_path)

    return len(unique_episodes)


def save_values_config(
    output_dir: Path,
    config_name: str,
    value_checkpoint: str,
    num_episodes: int,
    num_frames: int,
    values: np.ndarray,
    suffix: str = "",
):
    """Save values_config.json metadata."""
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    config_data = {
        "value_checkpoint": value_checkpoint,
        "config_name": config_name,
        "timestamp": datetime.now(UTC).isoformat(),
        "num_episodes": num_episodes,
        "num_frames": num_frames,
        "value_stats": {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        },
    }
    if suffix:
        config_data["suffix"] = suffix

    config_path = meta_dir / f"{_suffixed('values_config', suffix)}.json"
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)

    logger.info(f"Saved values config to {config_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def create_single_dataset(repo_id: str, dataset_dir: Path, data_config=None, action_horizon: int = 1):
    """Create a single-repo LeRobotRLDataset (no MultiRLAnyverseDataset).

    Extracts required parameters from data_config, mirroring how
    MultiRLAnyverseDataset creates individual datasets.
    """
    from openpi.training import rl_dataset

    # Extract params from data_config (same logic as MultiRLAnyverseDataset.__init__)
    robot_align_info = getattr(data_config, "robot_align_info", None) if data_config else None
    align_dim = getattr(data_config, "align_dim", 28) if data_config else 28
    unify_action_space = getattr(data_config, "unify_action_space", False) if data_config else False
    frame_attributes_preprocessors = (
        getattr(data_config, "frame_attributes_preprocessors", None) if data_config else None
    )

    if data_config and hasattr(data_config, "base_config") and data_config.base_config is not None:
        base = data_config.base_config
        action_keys = getattr(base, "action_sequence_keys", ["action"])
        value_net_cfg = getattr(base, "value_net_cfg", None)
    elif data_config:
        action_keys = getattr(data_config, "action_sequence_keys", ["action"])
        value_net_cfg = getattr(data_config, "value_net_cfg", None)
    else:
        action_keys = ["action"]
        value_net_cfg = None

    if not isinstance(action_keys, list | tuple):
        action_keys = ["action"]

    delta_indices = {key: list(range(action_horizon)) for key in action_keys}

    dataset = rl_dataset.LeRobotRLDataset(
        repo_id=repo_id,
        root=str(dataset_dir),
        delta_indices=delta_indices,
        robot_align_info=robot_align_info,
        align_dim=align_dim,
        unify_action_space=unify_action_space,
        value_net_cfg=value_net_cfg,
        frame_attributes_preprocessors=frame_attributes_preprocessors,
    )
    logger.info(f"Dataset '{repo_id}' loaded: {len(dataset)} samples")
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Compute value predictions and save as per-episode parquets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Parameter names aligned with test_rl.py
    parser.add_argument(
        "--config_name",
        required=True,
        help="Value model training config name.",
    )
    parser.add_argument(
        "--ckpt_dir",
        required=True,
        help="Path to value model checkpoint.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Dataset root directory. Defaults to config's root_dir.",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="Single repo_id to process. Overrides config.",
    )
    parser.add_argument(
        "--data_config_name",
        type=str,
        default=None,
        help="Separate config for repo_id list (overrides --config_name's repo_ids).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs for data-parallel inference. Clamped to jax.device_count().",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Version suffix for output dirs/files (e.g. 'v2' → value_pred_v2/).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=12,
        help="Number of DataLoader workers for parallel data loading.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Skip repo_ids whose output directory (value_pred/ or value_pred_{suffix}/) already exists.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # DataLoader uses multiprocessing_context="fork" which is Linux-only; workers that
    # import CUDA/JAX after a fork() crash with obscure errors on macOS/WSL. Fail loudly.
    if args.num_workers > 0 and platform.system() != "Linux":
        raise RuntimeError(
            f"--num_workers > 0 requires Linux (fork multiprocessing context); "
            f"detected {platform.system()}. Use --num_workers 0 instead."
        )

    from openpi.models import pi0_config
    from openpi.training import config as _config

    # Load value model config
    config = _config.get_config(args.config_name)
    model_config = config.model

    # Ensure value head is enabled
    if not model_config.enable_rl_value_head:
        logger.warning("Value head not enabled in config, enabling it.")
        model_config = pi0_config.Pi0Config(
            **dict(model_config.__dict__.items()),
            enable_rl_value_head=True,
        )
        config = dataclasses.replace(config, model=model_config)

    # Load value model's data_config (for value_net_cfg, robot_align_info, etc.)
    value_data_config = config.data.create(config.assets_dirs, config.model)
    action_horizon = config.model.action_horizon

    # Determine repo_id list: --repo_id > --data_config_name > --config_name
    if args.repo_id:
        repo_ids = [args.repo_id]
        data_config = value_data_config
    elif args.data_config_name:
        data_cfg = _config.get_config(args.data_config_name)
        data_config = data_cfg.data.create(data_cfg.assets_dirs, data_cfg.model)
        repo_ids = data_config.repo_id
        if isinstance(repo_ids, str):
            repo_ids = [repo_ids]
    else:
        data_config = value_data_config
        repo_ids = data_config.repo_id
        if isinstance(repo_ids, str):
            repo_ids = [repo_ids]

    # dataset_root: explicit > config
    root_dir = Path(args.dataset_root) if args.dataset_root else Path(data_config.root_dir)

    logger.info(f"Processing {len(repo_ids)} repo_ids under {root_dir}")

    _ensure_jax()
    model = load_value_model(model_config, args.ckpt_dir)

    # Multi-GPU: replicate model params across all devices
    num_gpus = min(args.num_gpus, jax.device_count())
    data_sharding = None
    if num_gpus > 1:
        from openpi.training import sharding as _sharding

        mesh = _sharding.make_mesh(num_fsdp_devices=1)
        model = replicate_model_params(model, mesh)
        # Create data sharding ONCE — reusing the same sharding object across datasets
        # prevents JAX from re-tracing the JIT function (different sharding objects
        # cause different trace cache keys even with identical logical shardings)
        data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(_sharding.DATA_AXIS))
        logger.info(f"Using {num_gpus} GPUs for data-parallel inference")
    else:
        logger.info("Using single GPU for inference")

    # Create JIT-compiled inference function ONCE (avoid per-dataset recompilation)
    from openpi.shared import nnx_utils

    score_observation_jit = nnx_utils.module_jit(model.score_observation)
    logger.info("Created score_observation_jit (will be reused across all datasets)")

    # Process each repo_id independently (like test_rl.py)
    total_episodes = 0
    total_frames = 0

    for i, repo_id in enumerate(repo_ids):
        logger.info(f"[{i + 1}/{len(repo_ids)}] Processing {repo_id}")
        dataset_dir = root_dir / repo_id

        if not dataset_dir.exists():
            logger.warning(f"Dataset directory not found: {dataset_dir}, skipping")
            continue

        if args.resume and _should_skip_repo_for_resume(dataset_dir, args.suffix):
            logger.info(f"[{i + 1}/{len(repo_ids)}] Resuming: skip {repo_id} (marker exists)")
            continue

        # Create single-repo dataset
        dataset = create_single_dataset(
            repo_id, dataset_dir, data_config=value_data_config, action_horizon=action_horizon
        )

        # Compute values (multi-GPU sharded inference when num_gpus > 1)
        values, episode_indices, frame_indices = compute_values_for_dataset(
            model,
            dataset,
            config,
            args.ckpt_dir,
            batch_size=args.batch_size,
            num_gpus=num_gpus,
            score_observation_jit=score_observation_jit,
            data_sharding=data_sharding,
            num_workers=min(args.num_workers, os.cpu_count() or 1),
        )

        # Save per-episode parquets (test_rl.py-compatible format)
        num_episodes = save_values_per_episode(
            values,
            episode_indices,
            frame_indices,
            dataset_dir,
            suffix=args.suffix,
        )

        # Save metadata
        save_values_config(
            dataset_dir,
            config_name=args.config_name,
            value_checkpoint=args.ckpt_dir,
            num_episodes=num_episodes,
            num_frames=len(values),
            values=values,
            suffix=args.suffix,
        )

        total_episodes += num_episodes
        total_frames += len(values)

    logger.info(f"Done! Saved {total_episodes} episodes, {total_frames} frames " f"across {len(repo_ids)} datasets")


if __name__ == "__main__":
    main()
