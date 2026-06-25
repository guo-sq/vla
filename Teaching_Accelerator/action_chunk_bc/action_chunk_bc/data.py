"""LeRobot parquet loading and action-chunk target construction."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


FEATURE_COLUMNS = ("observation.state", "observation.velocity", "observation.current")
ACTION_COLUMN = "action"
ACTION_DIM = 14
FEATURE_DIM = 43


@dataclass(frozen=True)
class EpisodeData:
    repo_id: str
    episode_index: int
    task: list[str]
    fps: int
    features: np.ndarray
    actions: np.ndarray

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])


@dataclass(frozen=True)
class DatasetArrays:
    features: np.ndarray
    targets: np.ndarray
    target_mask: np.ndarray
    actions: np.ndarray
    episode_keys: list[tuple[str, int]]
    episode_slices: dict[tuple[str, int], slice]
    episodes: list[EpisodeData]


@dataclass(frozen=True)
class NormalizationStats:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def repo_ids_from_file(path: Path) -> list[str]:
    repo_ids: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if stripped and not stripped.startswith("#"):
                repo_ids.append(stripped)
    return repo_ids


def episode_chunk(info: dict[str, Any], episode_index: int) -> int:
    chunk_size = int(info.get("chunks_size", 1000))
    return int(episode_index) // max(chunk_size, 1)


def episode_parquet_path(repo_root: Path, info: dict[str, Any], episode_index: int) -> Path:
    pattern = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    rel = pattern.format(
        episode_chunk=episode_chunk(info, episode_index),
        episode_index=int(episode_index),
    )
    return repo_root / rel


def _list_column_to_array(table: pq.Table, name: str, *, expected_dim: int = ACTION_DIM) -> np.ndarray:
    values = np.asarray(table.column(name).to_pylist(), dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"{name}: expected 2D list column, got shape={values.shape}")
    if values.shape[1] != expected_dim:
        raise ValueError(f"{name}: expected dim {expected_dim}, got {values.shape[1]}")
    return values


def load_episode(repo_root: Path, info: dict[str, Any], episode_meta: dict[str, Any]) -> EpisodeData:
    ep_idx = int(episode_meta["episode_index"])
    path = episode_parquet_path(repo_root, info, ep_idx)
    table = pq.read_table(str(path), columns=[*FEATURE_COLUMNS, ACTION_COLUMN])
    features = [_list_column_to_array(table, name) for name in FEATURE_COLUMNS]
    actions = _list_column_to_array(table, ACTION_COLUMN)
    expected_length = int(episode_meta.get("length", len(actions)))
    if len(actions) != expected_length:
        raise ValueError(f"{repo_root.name} ep {ep_idx}: parquet rows {len(actions)} != length {expected_length}")
    phase = np.arange(len(actions), dtype=np.float32).reshape(-1, 1) / max(len(actions) - 1, 1)
    feature_array = np.concatenate([*features, phase], axis=1).astype(np.float32)
    if feature_array.shape[1] != FEATURE_DIM:
        raise ValueError(f"{repo_root.name} ep {ep_idx}: expected feature dim {FEATURE_DIM}, got {feature_array.shape[1]}")
    return EpisodeData(
        repo_id=repo_root.name,
        episode_index=ep_idx,
        task=list(episode_meta.get("tasks", [])),
        fps=int(info.get("fps", 30)),
        features=feature_array,
        actions=actions.astype(np.float32),
    )


def load_episodes(
    root_dir: Path,
    repo_ids: list[str],
    *,
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
            episodes.append(load_episode(repo_root, info, ep))
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
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    episode_keys: list[tuple[str, int]] = []
    episode_slices: dict[tuple[str, int], slice] = {}
    offset = 0
    for episode in episodes:
        chunks, chunk_mask = build_action_chunk_targets(episode.actions, horizon)
        key = (episode.repo_id, episode.episode_index)
        episode_keys.append(key)
        episode_slices[key] = slice(offset, offset + episode.length)
        offset += episode.length
        features.append(episode.features)
        targets.append(chunks)
        masks.append(chunk_mask)
        actions.append(episode.actions)
    return DatasetArrays(
        features=np.concatenate(features, axis=0).astype(np.float32),
        targets=np.concatenate(targets, axis=0).astype(np.float32),
        target_mask=np.concatenate(masks, axis=0).astype(bool),
        actions=np.concatenate(actions, axis=0).astype(np.float32),
        episode_keys=episode_keys,
        episode_slices=episode_slices,
        episodes=episodes,
    )


def fit_normalization(arrays: DatasetArrays) -> NormalizationStats:
    feature_mean = arrays.features.mean(axis=0).astype(np.float32)
    feature_std = arrays.features.std(axis=0).astype(np.float32)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std).astype(np.float32)
    action_mean = arrays.actions.mean(axis=0).astype(np.float32)
    action_std = arrays.actions.std(axis=0).astype(np.float32)
    action_std = np.where(action_std < 1e-6, 1.0, action_std).astype(np.float32)
    return NormalizationStats(feature_mean, feature_std, action_mean, action_std)


def normalize_features(features: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return ((features - stats.feature_mean) / stats.feature_std).astype(np.float32)


def normalize_targets(targets: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return ((targets - stats.action_mean.reshape(1, 1, -1)) / stats.action_std.reshape(1, 1, -1)).astype(np.float32)


def stats_to_tensors(stats: NormalizationStats):
    import torch

    return {
        "feature_mean": torch.as_tensor(stats.feature_mean, dtype=torch.float32),
        "feature_std": torch.as_tensor(stats.feature_std, dtype=torch.float32),
        "action_mean": torch.as_tensor(stats.action_mean, dtype=torch.float32),
        "action_std": torch.as_tensor(stats.action_std, dtype=torch.float32),
    }


def stats_from_checkpoint(data_stats: dict) -> NormalizationStats:
    def to_numpy(value):
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy().astype(np.float32)
        return np.asarray(value, dtype=np.float32)

    return NormalizationStats(
        feature_mean=to_numpy(data_stats["feature_mean"]),
        feature_std=to_numpy(data_stats["feature_std"]),
        action_mean=to_numpy(data_stats["action_mean"]),
        action_std=to_numpy(data_stats["action_std"]),
    )

