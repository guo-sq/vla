"""Dataset loading for visual-proprio action diffusion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from diffusion_proxy.utils import episode_parquet_path
from diffusion_proxy.utils import load_json
from diffusion_proxy.utils import load_jsonl
from diffusion_proxy.vision import load_episode_embeddings


PROPRIO_COLUMNS = ("observation.state", "observation.velocity", "observation.current")
ACTION_COLUMN = "action"
ACTION_DIM = 14
PROPRIO_PHASE_DIM = 43


@dataclass(frozen=True)
class EpisodeData:
    repo_id: str
    episode_index: int
    task: list[str]
    fps: int
    condition: np.ndarray
    actions: np.ndarray

    @property
    def key(self) -> tuple[str, int]:
        return (self.repo_id, self.episode_index)

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])


@dataclass(frozen=True)
class DatasetArrays:
    condition: np.ndarray
    targets: np.ndarray
    target_mask: np.ndarray
    actions: np.ndarray
    episode_keys: list[tuple[str, int]]
    episode_slices: dict[tuple[str, int], slice]
    episodes: list[EpisodeData]


@dataclass(frozen=True)
class NormalizationStats:
    condition_mean: np.ndarray
    condition_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray


def _list_column_to_array(table, name: str, *, expected_dim: int = ACTION_DIM) -> np.ndarray:
    values = np.asarray(table.column(name).to_pylist(), dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"{name}: expected 2D list column, got shape={values.shape}")
    if values.shape[1] != expected_dim:
        raise ValueError(f"{name}: expected dim {expected_dim}, got {values.shape[1]}")
    return values


def load_episode(
    *,
    root_dir: Path,
    repo_id: str,
    info: dict[str, Any],
    episode_meta: dict[str, Any],
    vision_cache: Path,
    camera: str,
    encoder: str,
) -> EpisodeData:
    repo_root = root_dir / repo_id
    ep_idx = int(episode_meta["episode_index"])
    path = episode_parquet_path(repo_root, info, ep_idx)
    table = pq.read_table(str(path), columns=[*PROPRIO_COLUMNS, ACTION_COLUMN])
    proprio = [_list_column_to_array(table, name) for name in PROPRIO_COLUMNS]
    actions = _list_column_to_array(table, ACTION_COLUMN)
    expected_length = int(episode_meta.get("length", len(actions)))
    if len(actions) != expected_length:
        raise ValueError(f"{repo_id} ep {ep_idx}: parquet rows {len(actions)} != length {expected_length}")
    phase = np.arange(len(actions), dtype=np.float32).reshape(-1, 1) / max(len(actions) - 1, 1)
    visual = load_episode_embeddings(vision_cache, repo_id, ep_idx, camera, expected_length=expected_length, encoder=encoder)
    condition = np.concatenate([*proprio, phase, visual], axis=1).astype(np.float32)
    if condition.shape[1] <= PROPRIO_PHASE_DIM:
        raise ValueError(f"{repo_id} ep {ep_idx}: invalid condition dim {condition.shape[1]}")
    return EpisodeData(
        repo_id=repo_id,
        episode_index=ep_idx,
        task=list(episode_meta.get("tasks", [])),
        fps=int(info.get("fps", 30)),
        condition=condition,
        actions=actions.astype(np.float32),
    )


def load_episodes(
    root_dir: Path,
    repo_ids: list[str],
    *,
    vision_cache: Path,
    camera: str,
    encoder: str = "resnet18",
    max_episodes_per_repo: int | None = None,
) -> list[EpisodeData]:
    episodes: list[EpisodeData] = []
    for repo_id in repo_ids:
        repo_root = root_dir / repo_id
        info = load_json(repo_root / "meta" / "info.json")
        episode_meta = load_jsonl(repo_root / "meta" / "episodes.jsonl")
        if max_episodes_per_repo is not None:
            episode_meta = episode_meta[:max_episodes_per_repo]
        for ep in episode_meta:
            episodes.append(
                load_episode(
                    root_dir=root_dir,
                    repo_id=repo_id,
                    info=info,
                    episode_meta=ep,
                    vision_cache=vision_cache,
                    camera=camera,
                    encoder=encoder,
                )
            )
    return episodes


def build_action_chunk_targets(actions: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"actions must be 2D, got {actions.shape}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    n = len(actions)
    offsets = np.arange(horizon, dtype=np.int64)
    starts = np.arange(n, dtype=np.int64)[:, None]
    raw_indices = starts + offsets[None, :]
    mask = raw_indices < n
    clipped = np.minimum(raw_indices, max(n - 1, 0))
    return actions[clipped].astype(np.float32), mask.astype(bool)


def build_dataset_arrays(episodes: list[EpisodeData], horizon: int) -> DatasetArrays:
    if not episodes:
        raise ValueError("No episodes loaded")
    conditions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    episode_keys: list[tuple[str, int]] = []
    episode_slices: dict[tuple[str, int], slice] = {}
    offset = 0
    for episode in episodes:
        chunks, chunk_mask = build_action_chunk_targets(episode.actions, horizon)
        episode_keys.append(episode.key)
        episode_slices[episode.key] = slice(offset, offset + episode.length)
        offset += episode.length
        conditions.append(episode.condition)
        targets.append(chunks)
        masks.append(chunk_mask)
        actions.append(episode.actions)
    return DatasetArrays(
        condition=np.concatenate(conditions, axis=0).astype(np.float32),
        targets=np.concatenate(targets, axis=0).astype(np.float32),
        target_mask=np.concatenate(masks, axis=0).astype(bool),
        actions=np.concatenate(actions, axis=0).astype(np.float32),
        episode_keys=episode_keys,
        episode_slices=episode_slices,
        episodes=episodes,
    )


def subset_arrays(arrays: DatasetArrays, episode_keys: set[tuple[str, int]]) -> DatasetArrays:
    episodes = [ep for ep in arrays.episodes if ep.key in episode_keys]
    if not episodes:
        raise ValueError("subset has no episodes")
    indices: list[np.ndarray] = []
    for ep in episodes:
        sl = arrays.episode_slices[ep.key]
        indices.append(np.arange(sl.start, sl.stop, dtype=np.int64))
    flat = np.concatenate(indices, axis=0)
    offset = 0
    episode_slices: dict[tuple[str, int], slice] = {}
    for ep in episodes:
        episode_slices[ep.key] = slice(offset, offset + ep.length)
        offset += ep.length
    return DatasetArrays(
        condition=arrays.condition[flat].astype(np.float32),
        targets=arrays.targets[flat].astype(np.float32),
        target_mask=arrays.target_mask[flat].astype(bool),
        actions=arrays.actions[flat].astype(np.float32),
        episode_keys=[ep.key for ep in episodes],
        episode_slices=episode_slices,
        episodes=episodes,
    )


def fit_normalization(arrays: DatasetArrays) -> NormalizationStats:
    condition_mean = arrays.condition.mean(axis=0).astype(np.float32)
    condition_std = arrays.condition.std(axis=0).astype(np.float32)
    condition_std = np.where(condition_std < 1e-6, 1.0, condition_std).astype(np.float32)
    action_mean = arrays.actions.mean(axis=0).astype(np.float32)
    action_std = arrays.actions.std(axis=0).astype(np.float32)
    action_std = np.where(action_std < 1e-6, 1.0, action_std).astype(np.float32)
    return NormalizationStats(condition_mean, condition_std, action_mean, action_std)


def normalize_condition(condition: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return ((condition - stats.condition_mean) / stats.condition_std).astype(np.float32)


def normalize_targets(targets: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return ((targets - stats.action_mean.reshape(1, 1, -1)) / stats.action_std.reshape(1, 1, -1)).astype(np.float32)


def denormalize_targets(targets: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return (targets * stats.action_std.reshape(1, 1, -1) + stats.action_mean.reshape(1, 1, -1)).astype(np.float32)


def stats_to_tensors(stats: NormalizationStats):
    import torch

    return {
        "condition_mean": torch.as_tensor(stats.condition_mean, dtype=torch.float32),
        "condition_std": torch.as_tensor(stats.condition_std, dtype=torch.float32),
        "action_mean": torch.as_tensor(stats.action_mean, dtype=torch.float32),
        "action_std": torch.as_tensor(stats.action_std, dtype=torch.float32),
    }


def stats_from_checkpoint(data_stats: dict) -> NormalizationStats:
    def to_numpy(value):
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy().astype(np.float32)
        return np.asarray(value, dtype=np.float32)

    return NormalizationStats(
        condition_mean=to_numpy(data_stats["condition_mean"]),
        condition_std=to_numpy(data_stats["condition_std"]),
        action_mean=to_numpy(data_stats["action_mean"]),
        action_std=to_numpy(data_stats["action_std"]),
    )


def episode_folds(episode_keys: list[tuple[str, int]], folds: int, *, seed: int = 0) -> list[list[tuple[str, int]]]:
    if folds < 2:
        raise ValueError(f"folds must be >= 2, got {folds}")
    if folds > len(episode_keys):
        raise ValueError(f"folds {folds} > episode count {len(episode_keys)}")
    rng = np.random.default_rng(seed)
    keys = list(episode_keys)
    rng.shuffle(keys)
    out: list[list[tuple[str, int]]] = [[] for _ in range(folds)]
    for idx, key in enumerate(keys):
        out[idx % folds].append(key)
    return out
