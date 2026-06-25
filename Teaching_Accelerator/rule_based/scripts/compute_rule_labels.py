#!/usr/bin/env python3
"""Compute action-only rule labels for the seatbelt both-hang pilot."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from teaching_accelerator.rule_labels import RuleConfig
from teaching_accelerator.rule_labels import compute_rule_labels
from teaching_accelerator.sidecar import round_array
from teaching_accelerator.sidecar import write_jsonl
from teaching_accelerator.sidecar import write_summary


LOGGER = logging.getLogger("compute_rule_labels")


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


def read_actions(repo_root: Path, info: dict[str, Any], episode_index: int) -> np.ndarray:
    path = episode_parquet_path(repo_root, info, episode_index)
    table = pq.read_table(str(path), columns=["action"])
    actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"{path}: expected 2D action, got {actions.shape}")
    if actions.shape[1] != 14:
        raise ValueError(f"{path}: expected 14-dim action, got {actions.shape[1]}")
    return actions


def load_inputs(
    root_dir: Path,
    repo_ids: list[str],
    *,
    max_episodes_per_repo: int | None,
) -> tuple[
    dict[tuple[str, int], np.ndarray],
    dict[tuple[str, int], int],
    dict[tuple[str, int], dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    actions_by_key: dict[tuple[str, int], np.ndarray] = {}
    fps_by_key: dict[tuple[str, int], int] = {}
    episode_meta_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    repo_info_by_id: dict[str, dict[str, Any]] = {}

    for repo_id in repo_ids:
        repo_root = root_dir / repo_id
        info_path = repo_root / "meta" / "info.json"
        episodes_path = repo_root / "meta" / "episodes.jsonl"
        if not info_path.exists() or not episodes_path.exists():
            raise FileNotFoundError(f"{repo_id}: missing meta/info.json or meta/episodes.jsonl")
        info = load_json(info_path)
        episodes = load_jsonl(episodes_path)
        if max_episodes_per_repo is not None:
            episodes = episodes[:max_episodes_per_repo]
        fps = int(info.get("fps", 30))
        repo_info_by_id[repo_id] = info

        for ep in episodes:
            ep_idx = int(ep["episode_index"])
            key = (repo_id, ep_idx)
            actions = read_actions(repo_root, info, ep_idx)
            expected_length = int(ep.get("length", len(actions)))
            if len(actions) != expected_length:
                raise ValueError(f"{repo_id} ep {ep_idx}: parquet rows {len(actions)} != episode length {expected_length}")
            actions_by_key[key] = actions
            fps_by_key[key] = fps
            episode_meta_by_key[key] = ep
    return actions_by_key, fps_by_key, episode_meta_by_key, repo_info_by_id


def build_records(
    result,
    *,
    episode_meta_by_key: dict[tuple[str, int], dict[str, Any]],
    fps_by_key: dict[tuple[str, int], int],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for key in sorted(result.labels_by_episode):
        repo_id, ep_idx = key
        scores = result.scores_by_episode[key]
        labels = result.labels_by_episode[key]
        strides = result.strides_by_episode[key]
        ep_meta = episode_meta_by_key[key]
        record = {
            "repo_id": repo_id,
            "episode_index": int(ep_idx),
            "task": ep_meta.get("tasks", []),
            "length": int(ep_meta.get("length", len(labels))),
            "fps": int(fps_by_key[key]),
            "hard_score": round_array(scores["hard_score"]),
            "casualness_score": round_array(scores["casualness_score"]),
            "phase_consistency_score": round_array(scores["phase_consistency_score"]),
            "gripper_event_score": round_array(scores["gripper_event_score"]),
            "turn_score": round_array(scores["turn_score"]),
            "jerk_score": round_array(scores["jerk_score"]),
            "coordination_score": round_array(scores["coordination_score"]),
            "label": labels,
            "acceleration_stride": strides.astype(int).tolist(),
            "hard_spans": result.hard_spans_by_episode[key],
        }
        records.append(record)
    return records


def build_summary(
    result,
    records: list[dict[str, Any]],
    *,
    root_dir: Path,
    repo_ids: list[str],
    output_path: Path,
) -> dict[str, Any]:
    label_counts = Counter(label for record in records for label in record["label"])
    spans_by_repo: dict[str, int] = defaultdict(int)
    episodes_by_repo: dict[str, int] = defaultdict(int)
    frames_by_repo: dict[str, int] = defaultdict(int)
    for record in records:
        repo_id = str(record["repo_id"])
        spans_by_repo[repo_id] += len(record["hard_spans"])
        episodes_by_repo[repo_id] += 1
        frames_by_repo[repo_id] += int(record["length"])

    return {
        **result.summary,
        "root_dir": str(root_dir),
        "repo_ids": repo_ids,
        "output": str(output_path),
        "label_counts": dict(label_counts),
        "episodes_by_repo": dict(episodes_by_repo),
        "frames_by_repo": dict(frames_by_repo),
        "hard_spans_by_repo": dict(spans_by_repo),
    }


def write_report(records: list[dict[str, Any]], summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    label_counts = summary["label_counts"]
    lines = [
        "# Rule Label Report",
        "",
        f"- Episodes: {summary['num_episodes']}",
        f"- Frames: {summary['num_frames']}",
        f"- Hard spans: {summary['num_hard_spans']}",
        f"- Label counts: precision={label_counts.get('precision', 0)}, neutral={label_counts.get('neutral', 0)}, casual={label_counts.get('casual', 0)}",
        f"- Output: `{summary['output']}`",
        "",
        "## Top Hard Spans",
        "",
        "| repo_id | episode | start_s | end_s | duration_s | mean_hard_score |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    spans: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for record in records:
        for span in record["hard_spans"]:
            spans.append((float(span.get("mean_hard_score", 0.0)), record, span))
    for _, record, span in sorted(spans, key=lambda x: x[0], reverse=True)[:30]:
        lines.append(
            "| {repo} | {ep} | {start:.3f} | {end:.3f} | {dur:.3f} | {score:.6f} |".format(
                repo=record["repo_id"],
                ep=record["episode_index"],
                start=float(span["start_s"]),
                end=float(span["end_s"]),
                dur=float(span["duration_s"]),
                score=float(span.get("mean_hard_score", 0.0)),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def cleanup_old_difficulty_labels(root_dir: Path, repo_ids: list[str]) -> None:
    for repo_id in repo_ids:
        meta = root_dir / repo_id / "meta"
        for name in ("difficulty_labels.jsonl", "difficulty_labels_summary.json"):
            path = meta / name
            if path.exists():
                LOGGER.info("Removing old label file: %s", path)
                path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--repo-list-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--max-episodes-per-repo", type=int, default=None)
    parser.add_argument("--cleanup-old-difficulty-labels", action="store_true")
    parser.add_argument("--phase-bins", type=int, default=24)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s:%(name)s:%(message)s")

    repo_ids = repo_ids_from_file(args.repo_list_file)
    if not repo_ids:
        raise SystemExit(f"No repo ids found in {args.repo_list_file}")
    if args.cleanup_old_difficulty_labels:
        cleanup_old_difficulty_labels(args.root_dir, repo_ids)

    actions_by_key, fps_by_key, episode_meta_by_key, _ = load_inputs(
        args.root_dir,
        repo_ids,
        max_episodes_per_repo=args.max_episodes_per_repo,
    )
    config = RuleConfig(phase_bins=args.phase_bins)
    result = compute_rule_labels(actions_by_key, fps_by_key=fps_by_key, config=config)
    records = build_records(result, episode_meta_by_key=episode_meta_by_key, fps_by_key=fps_by_key)

    output_path = args.output_dir / "rule_labels.jsonl"
    summary_path = args.output_dir / "summary.json"
    report_path = args.report_dir / "rule_label_report.md"
    summary = build_summary(result, records, root_dir=args.root_dir, repo_ids=repo_ids, output_path=output_path)

    write_jsonl(records, output_path)
    write_summary(summary, summary_path)
    write_report(records, summary, report_path)

    LOGGER.info("Wrote %d episode records to %s", len(records), output_path)
    LOGGER.info("Wrote summary to %s", summary_path)
    LOGGER.info("Wrote report to %s", report_path)


if __name__ == "__main__":
    main()

