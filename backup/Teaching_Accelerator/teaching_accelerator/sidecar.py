"""Load teaching-acceleration label sidecars."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_LABEL_FILE = "meta/teaching_acceleration_labels.jsonl"


@dataclass(frozen=True)
class EpisodeAccelerationLabels:
    episode_index: int
    label: tuple[str, ...]
    acceleration_stride: np.ndarray
    precision_score: np.ndarray | None = None
    casualness_score: np.ndarray | None = None

    @property
    def length(self) -> int:
        return len(self.label)


def resolve_label_path(repo_root: str | Path, label_file: str | Path = DEFAULT_LABEL_FILE) -> Path:
    path = Path(label_file)
    if path.is_absolute():
        return path
    return Path(repo_root) / path


def load_episode_labels(
    repo_root: str | Path,
    *,
    label_file: str | Path = DEFAULT_LABEL_FILE,
    strict: bool = True,
) -> dict[int, EpisodeAccelerationLabels]:
    """Load one repo's teaching-acceleration sidecar."""

    path = resolve_label_path(repo_root, label_file)
    if not path.exists():
        if strict:
            raise FileNotFoundError(path)
        return {}

    out: dict[int, EpisodeAccelerationLabels] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            record: dict[str, Any] = json.loads(stripped)
            if "episode_index" not in record:
                raise ValueError(f"{path}:{line_no}: missing episode_index")
            if "label" not in record:
                raise ValueError(f"{path}:{line_no}: missing label")
            if "acceleration_stride" not in record:
                raise ValueError(f"{path}:{line_no}: missing acceleration_stride")

            labels = tuple(str(x) for x in record["label"])
            strides = np.asarray(record["acceleration_stride"], dtype=np.int32)
            if strides.ndim != 1:
                raise ValueError(f"{path}:{line_no}: acceleration_stride must be 1D")
            if len(strides) != len(labels):
                raise ValueError(
                    f"{path}:{line_no}: label length {len(labels)} != stride length {len(strides)}"
                )
            if np.any(strides < 1):
                raise ValueError(f"{path}:{line_no}: acceleration_stride must be >= 1")

            precision = record.get("precision_score")
            casualness = record.get("casualness_score")
            out[int(record["episode_index"])] = EpisodeAccelerationLabels(
                episode_index=int(record["episode_index"]),
                label=labels,
                acceleration_stride=strides,
                precision_score=np.asarray(precision, dtype=np.float32) if precision is not None else None,
                casualness_score=np.asarray(casualness, dtype=np.float32) if casualness is not None else None,
            )
    return out


def build_frame_to_episode_map(episode_data_index: dict) -> dict[int, tuple[int, int, int]]:
    """Return global frame index -> (episode_index, episode_start, local_index).

    ``episode_data_index`` follows LeRobot's ``{"from": ..., "to": ...}``.
    Episode keys are assumed to be positional unless the caller remaps them.
    """

    starts = np.asarray(episode_data_index["from"], dtype=np.int64)
    ends = np.asarray(episode_data_index["to"], dtype=np.int64)
    if len(starts) != len(ends):
        raise ValueError(f"from/to length mismatch: {len(starts)} != {len(ends)}")
    out: dict[int, tuple[int, int, int]] = {}
    for ep_idx, (start, end) in enumerate(zip(starts, ends, strict=True)):
        for global_idx in range(int(start), int(end)):
            out[global_idx] = (ep_idx, int(start), int(global_idx - start))
    return out
