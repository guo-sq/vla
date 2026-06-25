"""Build accelerated action targets from per-frame precision labels."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


_PRECISION_LABELS = {"precision", "hard"}
_CASUAL_LABELS = {"casual", "easy"}


def _stride_for_label(label: str, *, precision_stride: int, neutral_stride: int, casual_stride: int) -> int:
    if label in _PRECISION_LABELS:
        return precision_stride
    if label in _CASUAL_LABELS:
        return casual_stride
    return neutral_stride


def build_accelerated_action_indices(
    labels: Sequence[str],
    *,
    start: int,
    horizon: int,
    precision_stride: int = 2,
    neutral_stride: int = 2,
    casual_stride: int = 4,
    include_start: bool = True,
) -> np.ndarray:
    """Return source action indices for one accelerated action chunk.

    The policy still observes frame ``start``. Its target action chunk is built
    by walking forward with label-dependent strides, matching the DemoSpeedup
    idea: precision frames advance slowly, casual frames advance faster.
    """

    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if min(precision_stride, neutral_stride, casual_stride) < 1:
        raise ValueError("all strides must be >= 1")
    n = len(labels)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if start < 0 or start >= n:
        raise IndexError(f"start={start} outside [0, {n})")

    indices: list[int] = []
    cursor = int(start)
    if include_start:
        indices.append(cursor)

    while len(indices) < horizon:
        stride = _stride_for_label(
            str(labels[cursor]),
            precision_stride=precision_stride,
            neutral_stride=neutral_stride,
            casual_stride=casual_stride,
        )
        cursor = min(cursor + stride, n - 1)
        indices.append(cursor)
        if cursor == n - 1:
            break

    return np.asarray(indices, dtype=np.int64)


def gather_accelerated_actions(
    actions: np.ndarray,
    labels: Sequence[str],
    *,
    start: int,
    horizon: int,
    precision_stride: int = 2,
    neutral_stride: int = 2,
    casual_stride: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(chunk, is_pad, indices)`` for one accelerated action target."""

    actions = np.asarray(actions)
    if actions.ndim != 2:
        raise ValueError(f"actions must be 2D, got shape={actions.shape}")

    indices = build_accelerated_action_indices(
        labels,
        start=start,
        horizon=horizon,
        precision_stride=precision_stride,
        neutral_stride=neutral_stride,
        casual_stride=casual_stride,
    )
    chunk = actions[indices]
    is_pad = np.zeros(horizon, dtype=bool)
    if len(chunk) < horizon:
        pad_count = horizon - len(chunk)
        last = chunk[-1:] if len(chunk) else actions[start : start + 1]
        chunk = np.concatenate([chunk, np.repeat(last, pad_count, axis=0)], axis=0)
        is_pad[-pad_count:] = True
    return chunk.astype(actions.dtype, copy=False), is_pad, indices
