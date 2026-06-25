"""Rule-based precision/casualness labels for teaching acceleration.

This is the migrated replacement for the earlier demo-difficulty prototype.
The important semantic correction is that phase-conditioned action dispersion
is treated as a casualness signal, not a hard/difficulty signal. High dispersion
means many actions appear plausible in the same normalized task phase, which is
closer to DemoSpeedup's high-entropy/casual interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ScoreWeights:
    consistency: float = 0.75
    turn: float = 0.20
    speed: float = 0.0
    acceleration: float = 0.05


def phase_bins(length: int, num_bins: int) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.int32)
    phases = np.arange(length, dtype=np.float32) / max(length - 1, 1)
    return np.minimum((phases * num_bins).astype(np.int32), num_bins - 1)


def robust_unit_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    lo, hi = np.percentile(values, [5, 95])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def moving_average(values: np.ndarray, half_window: int) -> np.ndarray:
    if half_window <= 0 or len(values) <= 2:
        return values.astype(np.float32, copy=False)
    window = half_window * 2 + 1
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values, kernel, mode="same").astype(np.float32)


def direction_change(velocity: np.ndarray) -> np.ndarray:
    """Return per-frame turning amount from consecutive action velocity vectors."""

    n = len(velocity)
    out = np.zeros(n, dtype=np.float32)
    if n <= 2:
        return out

    prev = velocity[:-1]
    curr = velocity[1:]
    denom = np.linalg.norm(prev, axis=1) * np.linalg.norm(curr, axis=1)
    valid = denom > 1e-6
    cos = np.ones(n - 1, dtype=np.float32)
    cos[valid] = np.sum(prev[valid] * curr[valid], axis=1) / denom[valid]
    cos = np.clip(cos, -1.0, 1.0)
    out[1:] = (1.0 - cos) * 0.5
    return out


def compute_scores_from_actions(
    actions_by_episode: dict[int, np.ndarray],
    *,
    phase_bin_count: int = 24,
    smoothing_half_window: int = 2,
    weights: ScoreWeights = ScoreWeights(),
) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, np.ndarray], dict[str, Any]]:
    """Compute precision and casualness scores.

    ``precision_score`` is high when the action is locally consistent with its
    phase peers and/or has sharp local curvature. ``casualness_score`` is high
    when phase-conditioned action dispersion is high.
    """

    if not actions_by_episode:
        raise ValueError("No episodes to score")

    all_actions = np.concatenate(list(actions_by_episode.values()), axis=0)
    action_mean = all_actions.mean(axis=0)
    action_std = all_actions.std(axis=0)
    action_std = np.where(action_std < 1e-6, 1.0, action_std)

    norm_actions = {
        ep: ((actions - action_mean) / action_std).astype(np.float32)
        for ep, actions in actions_by_episode.items()
    }
    bins_by_episode = {ep: phase_bins(len(actions), phase_bin_count) for ep, actions in norm_actions.items()}

    phase_dispersion = np.zeros(phase_bin_count, dtype=np.float32)
    phase_counts = np.zeros(phase_bin_count, dtype=np.int64)
    for bin_idx in range(phase_bin_count):
        parts = [
            actions[bins == bin_idx]
            for ep, actions in norm_actions.items()
            for bins in [bins_by_episode[ep]]
            if np.any(bins == bin_idx)
        ]
        if not parts:
            continue
        values = np.concatenate(parts, axis=0)
        phase_counts[bin_idx] = len(values)
        phase_dispersion[bin_idx] = float(np.mean(np.var(values, axis=0)))

    dispersion_values: list[np.ndarray] = []
    turn_values: list[np.ndarray] = []
    speed_values: list[np.ndarray] = []
    accel_values: list[np.ndarray] = []
    for ep, actions in norm_actions.items():
        bins = bins_by_episode[ep]
        dispersion_values.append(phase_dispersion[bins])

        velocity = np.diff(actions, axis=0, prepend=actions[:1])
        acceleration = np.diff(velocity, axis=0, prepend=velocity[:1])
        speed = np.linalg.norm(velocity, axis=1)
        accel = np.linalg.norm(acceleration, axis=1)
        turn = direction_change(velocity)
        turn_values.append(moving_average(turn, smoothing_half_window))
        speed_values.append(moving_average(speed, smoothing_half_window))
        accel_values.append(moving_average(accel, smoothing_half_window))

    dispersion_flat = robust_unit_scale(np.concatenate(dispersion_values, axis=0))
    consistency_flat = 1.0 - dispersion_flat
    turn_flat = robust_unit_scale(np.concatenate(turn_values, axis=0))
    speed_flat = robust_unit_scale(np.concatenate(speed_values, axis=0))
    accel_flat = robust_unit_scale(np.concatenate(accel_values, axis=0))

    denom = max(weights.consistency + weights.turn + weights.speed + weights.acceleration, 1e-6)
    scored_by_episode: dict[int, dict[str, np.ndarray]] = {}
    offset = 0
    for ep, actions in norm_actions.items():
        n = len(actions)
        precision_score = (
            weights.consistency * consistency_flat[offset : offset + n]
            + weights.turn * turn_flat[offset : offset + n]
            + weights.speed * speed_flat[offset : offset + n]
            + weights.acceleration * accel_flat[offset : offset + n]
        ) / denom
        scored_by_episode[ep] = {
            "precision_score": precision_score.astype(np.float32),
            "casualness_score": dispersion_flat[offset : offset + n].astype(np.float32),
            "turn_score": turn_flat[offset : offset + n].astype(np.float32),
            "speed_score": speed_flat[offset : offset + n].astype(np.float32),
            "accel_score": accel_flat[offset : offset + n].astype(np.float32),
        }
        offset += n

    stats = {
        "phase_bin_count": int(phase_bin_count),
        "phase_counts": phase_counts.tolist(),
        "phase_dispersion": phase_dispersion.round(6).tolist(),
        "action_dim": int(all_actions.shape[1]),
        "component_weights": {
            "consistency_from_inverse_phase_dispersion": float(weights.consistency),
            "turn": float(weights.turn),
            "speed": float(weights.speed),
            "acceleration": float(weights.acceleration),
        },
        "semantics": {
            "phase_dispersion": "casualness_positive",
            "inverse_phase_dispersion": "precision_positive",
        },
    }
    return scored_by_episode, bins_by_episode, stats


def labels_and_strides_from_scores(
    scored_by_episode: dict[int, dict[str, np.ndarray]],
    *,
    precision_quantile: float = 0.75,
    casual_quantile: float = 0.65,
    precision_stride: int = 2,
    neutral_stride: int = 2,
    casual_stride: int = 4,
    always_precision_head_tail: int = 1,
) -> tuple[dict[int, list[str]], dict[int, np.ndarray], dict[str, Any]]:
    """Map scores to precision/neutral/casual labels and acceleration strides."""

    precision_all = np.concatenate([x["precision_score"] for x in scored_by_episode.values()], axis=0)
    casual_all = np.concatenate([x["casualness_score"] for x in scored_by_episode.values()], axis=0)
    precision_threshold = float(np.quantile(precision_all, precision_quantile))
    casual_threshold = float(np.quantile(casual_all, casual_quantile))

    labels_by_episode: dict[int, list[str]] = {}
    strides_by_episode: dict[int, np.ndarray] = {}
    counts = {"precision": 0, "neutral": 0, "casual": 0}

    for ep, scores in scored_by_episode.items():
        precision_score = scores["precision_score"]
        casual_score = scores["casualness_score"]
        labels = np.full(len(precision_score), "neutral", dtype=object)
        labels[casual_score >= casual_threshold] = "casual"
        labels[precision_score >= precision_threshold] = "precision"

        if always_precision_head_tail > 0 and len(labels):
            margin = min(always_precision_head_tail, len(labels))
            labels[:margin] = "precision"
            labels[-margin:] = "precision"

        strides = np.full(len(labels), neutral_stride, dtype=np.int32)
        strides[labels == "precision"] = precision_stride
        strides[labels == "casual"] = casual_stride
        for label in counts:
            counts[label] += int(np.count_nonzero(labels == label))

        labels_by_episode[ep] = [str(x) for x in labels.tolist()]
        strides_by_episode[ep] = strides

    summary = {
        "thresholds": {
            "precision_min": precision_threshold,
            "casual_min": casual_threshold,
            "precision_quantile": precision_quantile,
            "casual_quantile": casual_quantile,
        },
        "label_counts": counts,
        "strides": {
            "precision": precision_stride,
            "neutral": neutral_stride,
            "casual": casual_stride,
        },
    }
    return labels_by_episode, strides_by_episode, summary
