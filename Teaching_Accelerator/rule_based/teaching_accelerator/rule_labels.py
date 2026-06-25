"""Action-only rule labels for seatbelt teaching acceleration.

The score semantics intentionally follow DemoSpeedup's precision/casual split:
high phase dispersion is a casualness signal, while low phase dispersion is a
precision signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


LABEL_PRECISION = "precision"
LABEL_NEUTRAL = "neutral"
LABEL_CASUAL = "casual"


@dataclass(frozen=True)
class RuleConfig:
    phase_bins: int = 24
    smoothing_half_window: int = 2
    gripper_event_window_frames: int = 30
    static_speed_quantile: float = 0.10
    precision_quantile: float = 0.75
    casual_quantile: float = 0.65
    precision_stride: int = 2
    neutral_stride: int = 2
    casual_stride: int = 4
    min_span_frames: int = 15
    merge_gap_frames: int = 10
    span_padding_frames: int = 8
    weight_phase_consistency: float = 0.35
    weight_gripper_event: float = 0.30
    weight_turn: float = 0.15
    weight_jerk: float = 0.15
    weight_coordination: float = 0.05


@dataclass(frozen=True)
class RuleResult:
    scores_by_episode: dict[tuple[str, int], dict[str, np.ndarray]]
    labels_by_episode: dict[tuple[str, int], list[str]]
    strides_by_episode: dict[tuple[str, int], np.ndarray]
    hard_spans_by_episode: dict[tuple[str, int], list[dict[str, float | int]]]
    summary: dict[str, Any]


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
    return np.convolve(values, kernel, mode="same").astype(np.float32)


def dilate_1d(values: np.ndarray, radius: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if radius <= 0 or values.size == 0:
        return values.astype(np.float32, copy=False)
    out = values.copy()
    n = len(values)
    for idx in np.flatnonzero(values > 0):
        start = max(0, int(idx) - radius)
        end = min(n, int(idx) + radius + 1)
        out[start:end] = np.maximum(out[start:end], values[idx])
    return out.astype(np.float32)


def phase_bins(length: int, num_bins: int) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.int32)
    if num_bins < 2:
        raise ValueError(f"phase_bins must be >= 2, got {num_bins}")
    phase = np.arange(length, dtype=np.float32) / max(length - 1, 1)
    return np.minimum((phase * num_bins).astype(np.int32), num_bins - 1)


def direction_change(velocity: np.ndarray) -> np.ndarray:
    velocity = np.asarray(velocity, dtype=np.float32)
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
    out[1:] = (1.0 - np.clip(cos, -1.0, 1.0)) * 0.5
    return out


def merge_boolean_spans(
    mask: np.ndarray,
    *,
    fps: int,
    min_span_frames: int,
    merge_gap_frames: int,
    padding_frames: int,
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
        out.append(
            {
                "start_frame": int(padded_start),
                "end_frame": int(padded_end),
                "start_s": round(float(padded_start) / float(fps), 4),
                "end_s": round(float(padded_end) / float(fps), 4),
                "duration_s": round(float(padded_end - padded_start) / float(fps), 4),
            }
        )
    return out


def _normalize_actions(actions_by_key: dict[tuple[str, int], np.ndarray]) -> dict[tuple[str, int], np.ndarray]:
    all_actions = np.concatenate(list(actions_by_key.values()), axis=0).astype(np.float32)
    mean = all_actions.mean(axis=0)
    std = all_actions.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return {key: ((actions - mean) / std).astype(np.float32) for key, actions in actions_by_key.items()}


def _phase_dispersion_scores(
    norm_actions: dict[tuple[str, int], np.ndarray],
    *,
    num_bins: int,
) -> tuple[dict[tuple[str, int], np.ndarray], dict[str, Any]]:
    bins_by_key = {key: phase_bins(len(actions), num_bins) for key, actions in norm_actions.items()}
    dispersion = np.zeros(num_bins, dtype=np.float32)
    counts = np.zeros(num_bins, dtype=np.int64)
    for bin_idx in range(num_bins):
        parts = [actions[bins_by_key[key] == bin_idx] for key, actions in norm_actions.items()]
        parts = [part for part in parts if len(part)]
        if not parts:
            continue
        values = np.concatenate(parts, axis=0)
        counts[bin_idx] = len(values)
        dispersion[bin_idx] = float(np.mean(np.var(values, axis=0)))

    per_frame = {key: dispersion[bins] for key, bins in bins_by_key.items()}
    flat = robust_unit_scale(np.concatenate(list(per_frame.values()), axis=0))
    out: dict[tuple[str, int], np.ndarray] = {}
    offset = 0
    for key, values in per_frame.items():
        n = len(values)
        out[key] = flat[offset : offset + n].astype(np.float32)
        offset += n

    stats = {
        "phase_bins": int(num_bins),
        "phase_counts": counts.astype(int).tolist(),
        "phase_dispersion": np.round(dispersion, 6).tolist(),
        "semantics": {
            "phase_dispersion": "casualness_positive",
            "phase_consistency": "precision_positive",
        },
    }
    return out, stats


def _component_scores(
    norm_actions: dict[tuple[str, int], np.ndarray],
    casualness_by_key: dict[tuple[str, int], np.ndarray],
    config: RuleConfig,
) -> dict[tuple[str, int], dict[str, np.ndarray]]:
    raw_turn: dict[tuple[str, int], np.ndarray] = {}
    raw_jerk: dict[tuple[str, int], np.ndarray] = {}
    raw_speed: dict[tuple[str, int], np.ndarray] = {}
    raw_gripper: dict[tuple[str, int], np.ndarray] = {}
    raw_coordination: dict[tuple[str, int], np.ndarray] = {}

    for key, actions in norm_actions.items():
        velocity = np.diff(actions, axis=0, prepend=actions[:1])
        acceleration = np.diff(velocity, axis=0, prepend=velocity[:1])
        jerk = np.diff(acceleration, axis=0, prepend=acceleration[:1])

        raw_speed[key] = moving_average(np.linalg.norm(velocity, axis=1), config.smoothing_half_window)
        raw_turn[key] = moving_average(direction_change(velocity), config.smoothing_half_window)
        raw_jerk[key] = moving_average(np.linalg.norm(jerk, axis=1), config.smoothing_half_window)

        gripper_velocity = np.abs(velocity[:, [6, 13]])
        gripper_change = np.max(gripper_velocity, axis=1)
        gripper_score = robust_unit_scale(gripper_change, 85.0, 99.0)
        raw_gripper[key] = moving_average(
            dilate_1d(gripper_score, config.gripper_event_window_frames),
            config.smoothing_half_window,
        )

        left_motion = np.linalg.norm(velocity[:, :7], axis=1)
        right_motion = np.linalg.norm(velocity[:, 7:], axis=1)
        raw_coordination[key] = moving_average(np.minimum(left_motion, right_motion), config.smoothing_half_window)

    turn_flat = robust_unit_scale(np.concatenate(list(raw_turn.values()), axis=0))
    jerk_flat = robust_unit_scale(np.concatenate(list(raw_jerk.values()), axis=0))
    speed_flat = robust_unit_scale(np.concatenate(list(raw_speed.values()), axis=0))
    gripper_flat = robust_unit_scale(np.concatenate(list(raw_gripper.values()), axis=0))
    coordination_flat = robust_unit_scale(np.concatenate(list(raw_coordination.values()), axis=0))

    out: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    offset = 0
    for key, actions in norm_actions.items():
        n = len(actions)
        phase_consistency = 1.0 - casualness_by_key[key]
        denom = max(
            config.weight_phase_consistency
            + config.weight_gripper_event
            + config.weight_turn
            + config.weight_jerk
            + config.weight_coordination,
            1e-6,
        )
        hard_score = (
            config.weight_phase_consistency * phase_consistency
            + config.weight_gripper_event * gripper_flat[offset : offset + n]
            + config.weight_turn * turn_flat[offset : offset + n]
            + config.weight_jerk * jerk_flat[offset : offset + n]
            + config.weight_coordination * coordination_flat[offset : offset + n]
        ) / denom
        out[key] = {
            "hard_score": hard_score.astype(np.float32),
            "casualness_score": casualness_by_key[key].astype(np.float32),
            "phase_consistency_score": phase_consistency.astype(np.float32),
            "gripper_event_score": gripper_flat[offset : offset + n].astype(np.float32),
            "turn_score": turn_flat[offset : offset + n].astype(np.float32),
            "jerk_score": jerk_flat[offset : offset + n].astype(np.float32),
            "coordination_score": coordination_flat[offset : offset + n].astype(np.float32),
            "speed_score": speed_flat[offset : offset + n].astype(np.float32),
        }
        offset += n
    return out


def compute_rule_labels(
    actions_by_key: dict[tuple[str, int], np.ndarray],
    *,
    fps_by_key: dict[tuple[str, int], int],
    config: RuleConfig = RuleConfig(),
) -> RuleResult:
    if not actions_by_key:
        raise ValueError("No actions provided")
    for key, actions in actions_by_key.items():
        if actions.ndim != 2:
            raise ValueError(f"{key}: actions must be 2D, got {actions.shape}")
        if actions.shape[1] != 14:
            raise ValueError(f"{key}: expected 14-dim action, got {actions.shape[1]}")

    norm_actions = _normalize_actions(actions_by_key)
    casualness, phase_stats = _phase_dispersion_scores(norm_actions, num_bins=config.phase_bins)
    scores = _component_scores(norm_actions, casualness, config)

    hard_all = np.concatenate([episode_scores["hard_score"] for episode_scores in scores.values()], axis=0)
    casual_all = np.concatenate([episode_scores["casualness_score"] for episode_scores in scores.values()], axis=0)
    speed_all = np.concatenate([episode_scores["speed_score"] for episode_scores in scores.values()], axis=0)
    hard_threshold = float(np.quantile(hard_all, config.precision_quantile))
    casual_threshold = float(np.quantile(casual_all, config.casual_quantile))
    static_threshold = float(np.quantile(speed_all, config.static_speed_quantile))

    labels_by_episode: dict[tuple[str, int], list[str]] = {}
    strides_by_episode: dict[tuple[str, int], np.ndarray] = {}
    spans_by_episode: dict[tuple[str, int], list[dict[str, float | int]]] = {}
    label_counts = {LABEL_PRECISION: 0, LABEL_NEUTRAL: 0, LABEL_CASUAL: 0}
    total_spans = 0

    for key, episode_scores in scores.items():
        n = len(episode_scores["hard_score"])
        labels = np.full(n, LABEL_NEUTRAL, dtype=object)
        labels[episode_scores["casualness_score"] >= casual_threshold] = LABEL_CASUAL

        non_static = episode_scores["speed_score"] > static_threshold
        precision_mask = (episode_scores["hard_score"] >= hard_threshold) & non_static
        labels[precision_mask] = LABEL_PRECISION

        strides = np.full(n, config.neutral_stride, dtype=np.int32)
        strides[labels == LABEL_PRECISION] = config.precision_stride
        strides[labels == LABEL_CASUAL] = config.casual_stride

        fps = int(fps_by_key[key])
        spans = merge_boolean_spans(
            labels == LABEL_PRECISION,
            fps=fps,
            min_span_frames=config.min_span_frames,
            merge_gap_frames=config.merge_gap_frames,
            padding_frames=config.span_padding_frames,
        )
        for span in spans:
            span["mean_hard_score"] = round(
                float(np.mean(episode_scores["hard_score"][int(span["start_frame"]) : int(span["end_frame"])])),
                6,
            )

        for label_name in label_counts:
            label_counts[label_name] += int(np.count_nonzero(labels == label_name))
        total_spans += len(spans)
        labels_by_episode[key] = [str(x) for x in labels.tolist()]
        strides_by_episode[key] = strides
        spans_by_episode[key] = spans

    summary = {
        "method": "rule_precision_segments_v1",
        "config": {
            name: value for name, value in config.__dict__.items()
        },
        "thresholds": {
            "hard_min": hard_threshold,
            "casual_min": casual_threshold,
            "static_speed_max": static_threshold,
            "precision_quantile": config.precision_quantile,
            "casual_quantile": config.casual_quantile,
            "static_speed_quantile": config.static_speed_quantile,
        },
        "label_counts": label_counts,
        "num_hard_spans": total_spans,
        "num_episodes": len(actions_by_key),
        "num_frames": int(sum(len(actions) for actions in actions_by_key.values())),
        "action_dim": 14,
        "phase": phase_stats,
    }
    return RuleResult(
        scores_by_episode=scores,
        labels_by_episode=labels_by_episode,
        strides_by_episode=strides_by_episode,
        hard_spans_by_episode=spans_by_episode,
        summary=summary,
    )

