#!/usr/bin/env python3
"""Compute rule-based labels for teaching acceleration.

The output is intentionally not ``difficulty_labels.jsonl``. It is an
acceleration-oriented sidecar:

``label == "precision"`` means use a small stride for action targets.
``label == "casual"`` means use a larger stride for action targets.
``label == "neutral"`` is the middle ground.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from teaching_accelerator.labels import ScoreWeights
from teaching_accelerator.labels import compute_scores_from_actions
from teaching_accelerator.labels import labels_and_strides_from_scores


LOGGER = logging.getLogger("compute_teaching_labels")
EXCLUDE_NAME_PARTS = ("self_play", "raw_self_play", ".cpt")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _episode_chunk(info: dict[str, Any], episode_index: int) -> int:
    chunks_size = int(info.get("chunks_size", 1000))
    return int(episode_index) // max(chunks_size, 1)


def _episode_parquet_path(repo_root: Path, info: dict[str, Any], episode_index: int) -> Path:
    data_path = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    rel = data_path.format(
        episode_chunk=_episode_chunk(info, episode_index),
        episode_index=int(episode_index),
    )
    return repo_root / rel


def _read_actions(repo_root: Path, info: dict[str, Any], episode_index: int) -> np.ndarray:
    parquet_path = _episode_parquet_path(repo_root, info, episode_index)
    table = pq.read_table(str(parquet_path), columns=["action"])
    actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"{parquet_path}: action must be 2D, got shape={actions.shape}")
    return actions


def _repo_ids_from_file(path: Path) -> list[str]:
    repo_ids = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if stripped and not stripped.startswith("#"):
                repo_ids.append(stripped)
    return repo_ids


def _discover_repos(
    root_dir: Path,
    *,
    repo_ids: list[str],
    repo_list_file: Path | None,
    discover_glob: str | None,
    include_self_play: bool,
    limit_repos: int | None,
) -> list[str]:
    resolved: list[str] = []
    resolved.extend(repo_ids)
    if repo_list_file is not None:
        resolved.extend(_repo_ids_from_file(repo_list_file))
    if discover_glob:
        resolved.extend(path.name for path in sorted(root_dir.glob(discover_glob)) if path.is_dir())

    deduped: list[str] = []
    seen = set()
    for repo_id in resolved:
        if repo_id in seen:
            continue
        seen.add(repo_id)
        repo_root = root_dir / repo_id
        lowered = repo_id.lower()
        if not include_self_play and any(part in lowered for part in EXCLUDE_NAME_PARTS):
            continue
        if not (repo_root / "meta" / "info.json").exists():
            LOGGER.warning("Skipping %s: missing meta/info.json", repo_root)
            continue
        deduped.append(repo_id)
        if limit_repos is not None and len(deduped) >= limit_repos:
            break
    return deduped


def _write_repo_labels(
    repo_root: Path,
    episodes: list[dict[str, Any]],
    scored_by_episode: dict[int, dict[str, np.ndarray]],
    bins_by_episode: dict[int, np.ndarray],
    labels_by_episode: dict[int, list[str]],
    strides_by_episode: dict[int, np.ndarray],
    summary: dict[str, Any],
    *,
    output_name: str,
    dry_run: bool,
) -> None:
    output_path = repo_root / "meta" / output_name
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    episode_by_index = {int(ep["episode_index"]): ep for ep in episodes}
    label_counts = Counter(label for labels in labels_by_episode.values() for label in labels)
    total_frames = int(sum(len(labels) for labels in labels_by_episode.values()))

    summary = {
        **summary,
        "repo_id": repo_root.name,
        "num_episodes": len(labels_by_episode),
        "num_frames": total_frames,
        "label_counts": dict(label_counts),
        "output": str(output_path),
        "method": "rule_precision_casualness_v1",
        "schema": {
            "label": "precision|neutral|casual",
            "acceleration_stride": "per-frame stride for action-target acceleration",
            "precision_score": "higher means preserve precision with smaller stride",
            "casualness_score": "higher means safe-to-accelerate candidate",
        },
    }

    LOGGER.info("%s: frames=%d labels=%s", repo_root.name, total_frames, dict(label_counts))
    if dry_run:
        return

    with output_path.open("w", encoding="utf-8") as f:
        for ep_idx in sorted(labels_by_episode.keys()):
            scores = scored_by_episode[ep_idx]
            record = {
                "episode_index": int(ep_idx),
                "task": episode_by_index.get(ep_idx, {}).get("tasks", []),
                "length": len(labels_by_episode[ep_idx]),
                "phase_bin": bins_by_episode[ep_idx].astype(int).tolist(),
                "precision_score": scores["precision_score"].round(6).tolist(),
                "casualness_score": scores["casualness_score"].round(6).tolist(),
                "turn_score": scores["turn_score"].round(6).tolist(),
                "accel_score": scores["accel_score"].round(6).tolist(),
                "label": labels_by_episode[ep_idx],
                "acceleration_stride": strides_by_episode[ep_idx].astype(int).tolist(),
            }
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")


def compute_for_repo(repo_root: Path, args: argparse.Namespace) -> None:
    info = _load_json(repo_root / "meta" / "info.json")
    episodes = _load_jsonl(repo_root / "meta" / "episodes.jsonl")
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    actions_by_episode: dict[int, np.ndarray] = {}
    for ep in episodes:
        ep_idx = int(ep["episode_index"])
        try:
            actions_by_episode[ep_idx] = _read_actions(repo_root, info, ep_idx)
        except Exception as exc:
            if args.strict:
                raise
            LOGGER.warning("Skipping %s episode %s: %s", repo_root.name, ep_idx, exc)

    if not actions_by_episode:
        raise ValueError(f"No readable actions found in {repo_root}")

    scores, bins, score_summary = compute_scores_from_actions(
        actions_by_episode,
        phase_bin_count=args.phase_bins,
        smoothing_half_window=args.smoothing_half_window,
        weights=ScoreWeights(
            consistency=args.consistency_weight,
            turn=args.turn_weight,
            speed=args.speed_weight,
            acceleration=args.accel_weight,
        ),
    )
    labels, strides, label_summary = labels_and_strides_from_scores(
        scores,
        precision_quantile=args.precision_quantile,
        casual_quantile=args.casual_quantile,
        precision_stride=args.precision_stride,
        neutral_stride=args.neutral_stride,
        casual_stride=args.casual_stride,
        always_precision_head_tail=args.always_precision_head_tail,
    )
    _write_repo_labels(
        repo_root,
        episodes,
        scores,
        bins,
        labels,
        strides,
        {**score_summary, **label_summary},
        output_name=args.output_name,
        dry_run=args.dry_run,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, required=True, help="Parent directory containing repo_id dirs.")
    parser.add_argument("--repo-id", action="append", default=[], help="Repo id to process. Can be repeated.")
    parser.add_argument("--repo-list-file", type=Path, default=None, help="Text file with one repo id per line.")
    parser.add_argument("--discover-glob", type=str, default=None, help="Optional glob under root-dir.")
    parser.add_argument("--include-self-play", action="store_true")
    parser.add_argument("--limit-repos", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--output-name", type=str, default="teaching_acceleration_labels.jsonl")
    parser.add_argument("--phase-bins", type=int, default=24)
    parser.add_argument("--smoothing-half-window", type=int, default=2)
    parser.add_argument("--consistency-weight", type=float, default=0.75)
    parser.add_argument("--turn-weight", type=float, default=0.20)
    parser.add_argument("--speed-weight", type=float, default=0.0)
    parser.add_argument("--accel-weight", type=float, default=0.05)
    parser.add_argument("--precision-quantile", type=float, default=0.75)
    parser.add_argument("--casual-quantile", type=float, default=0.65)
    parser.add_argument("--precision-stride", type=int, default=2)
    parser.add_argument("--neutral-stride", type=int, default=2)
    parser.add_argument("--casual-stride", type=int, default=4)
    parser.add_argument("--always-precision-head-tail", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    if args.phase_bins < 2:
        parser.error("--phase-bins must be >= 2")
    if not 0.0 <= args.precision_quantile <= 1.0:
        parser.error("--precision-quantile must be in [0, 1]")
    if not 0.0 <= args.casual_quantile <= 1.0:
        parser.error("--casual-quantile must be in [0, 1]")
    for name in ("precision_stride", "neutral_stride", "casual_stride"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    for name in ("consistency_weight", "turn_weight", "speed_weight", "accel_weight"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be >= 0")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s:%(name)s:%(message)s")
    repo_ids = _discover_repos(
        args.root_dir,
        repo_ids=args.repo_id,
        repo_list_file=args.repo_list_file,
        discover_glob=args.discover_glob,
        include_self_play=args.include_self_play,
        limit_repos=args.limit_repos,
    )
    if not repo_ids:
        raise SystemExit("No repos selected. Use --repo-id, --repo-list-file, or --discover-glob.")
    LOGGER.info("Processing %d repos under %s", len(repo_ids), args.root_dir)
    for repo_id in repo_ids:
        compute_for_repo(args.root_dir / repo_id, args)


if __name__ == "__main__":
    main()
