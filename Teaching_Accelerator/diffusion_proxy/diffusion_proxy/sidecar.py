"""Sidecar helpers for diffusion proxy labels and fusion labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


DIFFUSION_ARRAY_FIELDS = (
    "diffusion_precision_score",
    "diffusion_entropy_score",
    "diffusion_reconstruction_error",
    "label",
    "acceleration_stride",
)

FUSION_ARRAY_FIELDS = (
    "fusion_precision_score",
    "fusion_casualness_score",
    "fusion_reason",
    "label",
    "acceleration_stride",
)


def round_array(values: np.ndarray, digits: int = 6) -> list[float]:
    return np.round(np.asarray(values, dtype=np.float32), digits).tolist()


def validate_common(record: dict[str, Any]) -> int:
    required = {"repo_id", "episode_index", "task", "length", "fps", "label", "acceleration_stride", "hard_spans"}
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"record missing required fields: {missing}")
    length = int(record["length"])
    if length < 0:
        raise ValueError("length must be non-negative")
    if len(record["label"]) != length:
        raise ValueError(f"{record['repo_id']} ep {record['episode_index']}: label length mismatch")
    if len(record["acceleration_stride"]) != length:
        raise ValueError(f"{record['repo_id']} ep {record['episode_index']}: acceleration_stride length mismatch")
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
            raise ValueError(f"invalid span: {span}")
    return length


def validate_diffusion_record(record: dict[str, Any]) -> None:
    length = validate_common(record)
    required = {"diffusion_precision_score", "diffusion_entropy_score", "diffusion_reconstruction_error"}
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"diffusion record missing required fields: {missing}")
    for field in DIFFUSION_ARRAY_FIELDS:
        if len(record[field]) != length:
            raise ValueError(f"{record['repo_id']} ep {record['episode_index']}: {field} length mismatch")


def validate_fusion_record(record: dict[str, Any]) -> None:
    length = validate_common(record)
    required = {
        "fusion_precision_score",
        "fusion_casualness_score",
        "fusion_reason",
        "rule_label",
        "diffusion_label",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"fusion record missing required fields: {missing}")
    for field in FUSION_ARRAY_FIELDS:
        if len(record[field]) != length:
            raise ValueError(f"{record['repo_id']} ep {record['episode_index']}: {field} length mismatch")
    for field in ("rule_label", "diffusion_label"):
        if len(record[field]) != length:
            raise ValueError(f"{record['repo_id']} ep {record['episode_index']}: {field} length mismatch")


def write_jsonl(records: list[dict[str, Any]], path: Path, *, kind: str = "diffusion") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    validator = validate_fusion_record if kind == "fusion" else validate_diffusion_record
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            validator(record)
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def write_summary(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

