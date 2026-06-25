"""Utilities for applying offline per-frame difficulty labels to sampling."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_DIFFICULTY_LABEL_FILE = "meta/difficulty_labels.jsonl"


def resolve_difficulty_label_path(root: str | Path | None, label_file: str | Path) -> Path:
    """Resolve a difficulty label path relative to one LeRobot repo root."""
    path = Path(label_file)
    if path.is_absolute():
        return path
    if root is None:
        return path
    return Path(root) / path


def _as_numpy_index_array(values: Any) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.int64)


def build_episode_slice_map(
    episode_data_index: dict,
    *,
    meta_episodes: Any | None = None,
    episodes: list[int] | None = None,
) -> dict[int, tuple[int, int]]:
    """Build real episode_index -> global [start, end) frame slice."""
    starts = _as_numpy_index_array(episode_data_index["from"])
    ends = _as_numpy_index_array(episode_data_index["to"])
    if len(starts) != len(ends):
        raise ValueError(f"episode_data_index from/to length mismatch: {len(starts)} != {len(ends)}")

    if episodes is not None:
        episode_keys = [int(ep) for ep in episodes]
    elif isinstance(meta_episodes, dict):
        episode_keys = [int(ep) for ep in sorted(meta_episodes.keys())]
    else:
        episode_keys = list(range(len(starts)))

    if len(episode_keys) != len(starts):
        raise ValueError(
            f"Episode key count {len(episode_keys)} does not match episode_data_index length {len(starts)}"
        )

    return {ep: (int(s), int(e)) for ep, s, e in zip(episode_keys, starts, ends, strict=True)}


def load_difficulty_sample_weights(
    *,
    root: str | Path | None,
    total_frames: int,
    episode_data_index: dict,
    meta_episodes: Any | None = None,
    episodes: list[int] | None = None,
    label_file: str | Path = DEFAULT_DIFFICULTY_LABEL_FILE,
    strict: bool = False,
) -> np.ndarray | None:
    """Load per-frame sample weights from an offline difficulty label jsonl file.

    Expected jsonl format, one record per episode:
        {"episode_index": 0, "sample_weight": [1, 0, 1, ...]}

    Missing files return None unless strict=True. Length mismatches are truncated
    to the overlapping range unless strict=True.
    """
    label_path = resolve_difficulty_label_path(root, label_file)
    if not label_path.exists():
        msg = f"Difficulty label file not found: {label_path}"
        if strict:
            raise FileNotFoundError(msg)
        logger.warning(msg)
        return None

    episode_slices = build_episode_slice_map(
        episode_data_index,
        meta_episodes=meta_episodes,
        episodes=episodes,
    )
    weights = np.ones(int(total_frames), dtype=np.int32)
    seen_episodes: set[int] = set()

    with label_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if "episode_index" not in record:
                raise ValueError(f"{label_path}:{line_no}: missing episode_index")
            ep_idx = int(record["episode_index"])
            if ep_idx not in episode_slices:
                if strict:
                    raise ValueError(f"{label_path}:{line_no}: episode_index {ep_idx} not in dataset")
                continue
            raw = record.get("sample_weight", record.get("weights"))
            if raw is None:
                raise ValueError(f"{label_path}:{line_no}: missing sample_weight")

            ep_weights = np.asarray(raw, dtype=np.int32)
            if ep_weights.ndim != 1:
                raise ValueError(f"{label_path}:{line_no}: sample_weight must be 1D")
            if np.any(ep_weights < 0):
                raise ValueError(f"{label_path}:{line_no}: sample_weight contains negative values")

            start, end = episode_slices[ep_idx]
            expected = end - start
            if len(ep_weights) != expected:
                msg = (
                    f"{label_path}:{line_no}: episode {ep_idx} sample_weight length "
                    f"{len(ep_weights)} != expected {expected}"
                )
                if strict:
                    raise ValueError(msg)
                logger.warning("%s; truncating to overlap", msg)
            n = min(len(ep_weights), expected)
            weights[start : start + n] = ep_weights[:n]
            if n < expected and not strict:
                weights[start + n : end] = 1
            seen_episodes.add(ep_idx)

    if not seen_episodes:
        msg = f"No matching episodes found in difficulty label file: {label_path}"
        if strict:
            raise ValueError(msg)
        logger.warning(msg)
        return None

    kept = int(np.count_nonzero(weights))
    logger.info(
        "Loaded difficulty sampling weights from %s: kept=%d/%d (%.2f%%), max_weight=%d",
        label_path,
        kept,
        total_frames,
        kept / max(total_frames, 1) * 100,
        int(weights.max()) if len(weights) else 0,
    )
    return weights
