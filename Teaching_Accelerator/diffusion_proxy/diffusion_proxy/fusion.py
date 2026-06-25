"""Fuse rule-based and diffusion proxy labels."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from diffusion_proxy.sidecar import round_array
from diffusion_proxy.utils import LABEL_CASUAL
from diffusion_proxy.utils import LABEL_NEUTRAL
from diffusion_proxy.utils import LABEL_PRECISION
from diffusion_proxy.utils import merge_boolean_spans
from diffusion_proxy.utils import robust_unit_scale


def _reason(rule_label: str, diffusion_label: str, final_label: str) -> str:
    if final_label == LABEL_PRECISION and rule_label == LABEL_PRECISION and diffusion_label == LABEL_PRECISION:
        return "both_precision"
    if final_label == LABEL_PRECISION and rule_label == LABEL_PRECISION:
        return "rule_precision"
    if final_label == LABEL_PRECISION and diffusion_label == LABEL_PRECISION:
        return "diffusion_precision"
    if final_label == LABEL_CASUAL and rule_label == LABEL_CASUAL and diffusion_label == LABEL_CASUAL:
        return "both_casual"
    if final_label == LABEL_CASUAL and rule_label == LABEL_CASUAL:
        return "rule_casual"
    if final_label == LABEL_CASUAL and diffusion_label == LABEL_CASUAL:
        return "diffusion_casual"
    if rule_label != diffusion_label:
        return "source_conflict"
    return "neutral"


def fuse_records(
    rule_records: list[dict[str, Any]],
    diffusion_records: list[dict[str, Any]],
    *,
    rule_weight: float = 0.50,
    diffusion_weight: float = 0.30,
    event_weight: float = 0.20,
    precision_quantile: float = 0.75,
    casual_quantile: float = 0.65,
    precision_stride: int = 2,
    neutral_stride: int = 2,
    casual_stride: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rule_by_key = {(r["repo_id"], int(r["episode_index"])): r for r in rule_records}
    diffusion_by_key = {(r["repo_id"], int(r["episode_index"])): r for r in diffusion_records}
    keys = sorted(set(rule_by_key) & set(diffusion_by_key))
    if len(keys) != len(rule_by_key) or len(keys) != len(diffusion_by_key):
        raise ValueError("rule and diffusion records do not have identical keys")

    raw_precision_parts: list[np.ndarray] = []
    raw_casual_parts: list[np.ndarray] = []
    for key in keys:
        rule = rule_by_key[key]
        diffusion = diffusion_by_key[key]
        if int(rule["length"]) != int(diffusion["length"]):
            raise ValueError(f"{key}: length mismatch")
        event_score = np.maximum.reduce(
            [
                np.asarray(rule["gripper_event_score"], dtype=np.float32),
                np.asarray(rule["jerk_score"], dtype=np.float32),
                np.asarray(rule["turn_score"], dtype=np.float32),
            ]
        )
        raw_precision = (
            rule_weight * np.asarray(rule["hard_score"], dtype=np.float32)
            + diffusion_weight * np.asarray(diffusion["diffusion_precision_score"], dtype=np.float32)
            + event_weight * event_score
        )
        raw_casual = 0.5 * np.asarray(rule["casualness_score"], dtype=np.float32) + 0.5 * np.asarray(
            diffusion["diffusion_entropy_score"], dtype=np.float32
        )
        raw_precision_parts.append(raw_precision)
        raw_casual_parts.append(raw_casual)

    precision_flat = robust_unit_scale(np.concatenate(raw_precision_parts, axis=0))
    casual_flat = robust_unit_scale(np.concatenate(raw_casual_parts, axis=0))
    precision_threshold = float(np.quantile(precision_flat, precision_quantile))
    casual_threshold = float(np.quantile(casual_flat, casual_quantile))

    records: list[dict[str, Any]] = []
    label_counts = Counter()
    reason_counts = Counter()
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    spans_by_repo: dict[str, int] = defaultdict(int)
    frames_by_repo: dict[str, int] = defaultdict(int)
    offset = 0
    for raw_index, key in enumerate(keys):
        rule = rule_by_key[key]
        diffusion = diffusion_by_key[key]
        n = int(rule["length"])
        ep_precision = precision_flat[offset : offset + n]
        ep_casual = casual_flat[offset : offset + n]
        labels = np.full(n, LABEL_NEUTRAL, dtype=object)
        labels[ep_casual >= casual_threshold] = LABEL_CASUAL
        rule_event = np.maximum.reduce(
            [
                np.asarray(rule["gripper_event_score"], dtype=np.float32),
                np.asarray(rule["jerk_score"], dtype=np.float32),
                np.asarray(rule["turn_score"], dtype=np.float32),
            ]
        )
        event_keep = rule_event >= float(np.quantile(rule_event, 0.80))
        precision_mask = (ep_precision >= precision_threshold) | (
            (np.asarray(rule["label"], dtype=object) == LABEL_PRECISION) & event_keep
        )
        labels[precision_mask] = LABEL_PRECISION
        strides = np.full(n, neutral_stride, dtype=np.int32)
        strides[labels == LABEL_PRECISION] = precision_stride
        strides[labels == LABEL_CASUAL] = casual_stride
        reasons = [
            _reason(str(rule["label"][idx]), str(diffusion["label"][idx]), str(labels[idx]))
            for idx in range(n)
        ]
        spans = merge_boolean_spans(
            labels == LABEL_PRECISION,
            fps=int(rule["fps"]),
            min_span_frames=15,
            merge_gap_frames=10,
            padding_frames=8,
            score=ep_precision,
            score_name="mean_fusion_precision_score",
        )
        for label in labels.tolist():
            label_counts[str(label)] += 1
        for reason in reasons:
            reason_counts[reason] += 1
        for rule_label, diffusion_label in zip(rule["label"], diffusion["label"]):
            confusion[str(rule_label)][str(diffusion_label)] += 1
        repo_id = str(rule["repo_id"])
        spans_by_repo[repo_id] += len(spans)
        frames_by_repo[repo_id] += n
        records.append(
            {
                "repo_id": repo_id,
                "episode_index": int(rule["episode_index"]),
                "task": rule.get("task", []),
                "length": n,
                "fps": int(rule["fps"]),
                "fusion_precision_score": round_array(ep_precision),
                "fusion_casualness_score": round_array(ep_casual),
                "fusion_reason": reasons,
                "rule_hard_score": rule["hard_score"],
                "rule_gripper_event_score": rule["gripper_event_score"],
                "rule_turn_score": rule["turn_score"],
                "rule_jerk_score": rule["jerk_score"],
                "rule_label": rule["label"],
                "diffusion_precision_score": diffusion["diffusion_precision_score"],
                "diffusion_entropy_score": diffusion["diffusion_entropy_score"],
                "diffusion_label": diffusion["label"],
                "label": [str(x) for x in labels.tolist()],
                "acceleration_stride": strides.astype(int).tolist(),
                "hard_spans": spans,
                "rule_hard_spans": rule.get("hard_spans", []),
                "diffusion_hard_spans": diffusion.get("hard_spans", []),
            }
        )
        offset += n
    summary = {
        "method": "rule_diffusion_fusion_v1",
        "weights": {
            "rule_weight": rule_weight,
            "diffusion_weight": diffusion_weight,
            "event_weight": event_weight,
        },
        "thresholds": {
            "fusion_precision_min": precision_threshold,
            "fusion_casualness_min": casual_threshold,
            "precision_quantile": precision_quantile,
            "casual_quantile": casual_quantile,
        },
        "label_counts": dict(label_counts),
        "reason_counts": dict(reason_counts),
        "source_confusion_rule_to_diffusion": {row: dict(cols) for row, cols in confusion.items()},
        "num_episodes": len(records),
        "num_frames": int(sum(r["length"] for r in records)),
        "num_hard_spans": int(sum(len(r["hard_spans"]) for r in records)),
        "frames_by_repo": dict(frames_by_repo),
        "hard_spans_by_repo": dict(spans_by_repo),
    }
    return records, summary
