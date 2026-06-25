"""Shared numeric helpers."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def robust_unit_scale(values: np.ndarray, lower: float = 5.0, upper: float = 95.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    lo, hi = np.percentile(finite, [lower, upper])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def merge_boolean_spans(
    mask: np.ndarray,
    *,
    fps: int,
    min_span_frames: int = 15,
    merge_gap_frames: int = 10,
    padding_frames: int = 8,
    score: np.ndarray | None = None,
) -> list[dict[str, float | int]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            spans.append((start, idx))
            start = None
    if start is not None:
        spans.append((start, len(mask)))

    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if not merged or start - merged[-1][1] > merge_gap_frames:
            merged.append((start, end))
        else:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))

    out: list[dict[str, float | int]] = []
    for start, end in merged:
        if end - start < min_span_frames:
            continue
        padded_start = max(0, start - padding_frames)
        padded_end = min(len(mask), end + padding_frames)
        record: dict[str, Any] = {
            "start_frame": int(padded_start),
            "end_frame": int(padded_end),
            "start_s": round(float(padded_start) / float(fps), 4),
            "end_s": round(float(padded_end) / float(fps), 4),
            "duration_s": round(float(padded_end - padded_start) / float(fps), 4),
        }
        if score is not None:
            record["mean_bc_precision_score"] = round(float(np.mean(score[padded_start:padded_end])), 6)
        out.append(record)
    return out

