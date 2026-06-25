"""Sidecar schema helpers for rule-based teaching labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


ARRAY_FIELDS = (
    "hard_score",
    "casualness_score",
    "phase_consistency_score",
    "gripper_event_score",
    "turn_score",
    "jerk_score",
    "coordination_score",
    "label",
    "acceleration_stride",
)


def round_array(values: np.ndarray, digits: int = 6) -> list[float]:
    return np.round(np.asarray(values, dtype=np.float32), digits).tolist()


def validate_record(record: dict[str, Any]) -> None:
    required = {
        "repo_id",
        "episode_index",
        "task",
        "length",
        "fps",
        "hard_score",
        "casualness_score",
        "phase_consistency_score",
        "gripper_event_score",
        "turn_score",
        "jerk_score",
        "coordination_score",
        "label",
        "acceleration_stride",
        "hard_spans",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"record missing required fields: {missing}")

    length = int(record["length"])
    if length < 0:
        raise ValueError("length must be non-negative")
    for field in ARRAY_FIELDS:
        if len(record[field]) != length:
            raise ValueError(f"{record['repo_id']} ep {record['episode_index']}: {field} length mismatch")
    for label in record["label"]:
        if label not in {"precision", "neutral", "casual"}:
            raise ValueError(f"invalid label: {label}")
    for stride in record["acceleration_stride"]:
        if int(stride) < 1:
            raise ValueError(f"invalid stride: {stride}")
    for span in record["hard_spans"]:
        start = int(span["start_frame"])
        end = int(span["end_frame"])
        if start < 0 or end > length or start >= end:
            raise ValueError(f"invalid span for {record['repo_id']} ep {record['episode_index']}: {span}")


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            validate_record(record)
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            try:
                validate_record(record)
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            records.append(record)
    return records


def write_summary(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

