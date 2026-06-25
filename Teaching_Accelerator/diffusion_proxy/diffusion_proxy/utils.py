"""Shared helpers for diffusion proxy labels."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


LABEL_PRECISION = "precision"
LABEL_NEUTRAL = "neutral"
LABEL_CASUAL = "casual"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def moving_average(values: np.ndarray, half_window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if half_window <= 0 or len(values) <= 2:
        return values.astype(np.float32, copy=False)
    window = half_window * 2 + 1
    kernel = np.ones(window, dtype=np.float32) / float(window)
    padded = np.pad(values, (half_window, half_window), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def merge_boolean_spans(
    mask: np.ndarray,
    *,
    fps: int,
    min_span_frames: int = 15,
    merge_gap_frames: int = 10,
    padding_frames: int = 8,
    score: np.ndarray | None = None,
    score_name: str = "mean_diffusion_precision_score",
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
        span: dict[str, float | int] = {
            "start_frame": int(padded_start),
            "end_frame": int(padded_end),
            "start_s": round(float(padded_start) / float(fps), 4),
            "end_s": round(float(padded_end) / float(fps), 4),
            "duration_s": round(float(padded_end - padded_start) / float(fps), 4),
        }
        if score is not None:
            span[score_name] = round(float(np.mean(score[padded_start:padded_end])), 6)
        out.append(span)
    return out


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


def episode_video_path(repo_root: Path, info: dict[str, Any], episode_index: int, camera: str) -> Path:
    pattern = info.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
    rel = pattern.format(
        episode_chunk=episode_chunk(info, episode_index),
        episode_index=int(episode_index),
        video_key=camera,
    )
    return repo_root / rel
