"""Compute offline action-difficulty labels for LeRobot/AnyVerse demos.

The estimator is intentionally small and transparent. It compares actions
within the same normalized task phase bin across successful demonstrations,
then mixes that phase-conditioned dispersion with local direction changes and
small acceleration/jerk guards. Raw speed defaults to weight 0 because fast,
smooth, consistent motion is often easy rather than hard.

Example:
    python src/openpi/training/demo_difficulty/compute_demo_difficulty_labels.py \
        --root-dir /mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/seatbelt \
        --repo-id seatbelt.single.take_off_move.panjinlong.20260302.batch.1 \
        --repo-id seatbelt.single.take_off_move.panjinlong.20260302.batch.2
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

LOGGER = logging.getLogger("compute_demo_difficulty_labels")
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
    values = table.column("action").to_pylist()
    actions = np.asarray(values, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"{parquet_path}: action must be 2D, got shape={actions.shape}")
    return actions


def _phase_bins(length: int, num_bins: int) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.int32)
    phases = np.arange(length, dtype=np.float32) / max(length - 1, 1)
    return np.minimum((phases * num_bins).astype(np.int32), num_bins - 1)


def _robust_unit_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    lo, hi = np.percentile(values, [5, 95])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _moving_average(values: np.ndarray, half_window: int) -> np.ndarray:
    if half_window <= 0 or len(values) <= 2:
        return values.astype(np.float32, copy=False)
    window = half_window * 2 + 1
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values, kernel, mode="same").astype(np.float32)


def _direction_change(velocity: np.ndarray) -> np.ndarray:
    """Return per-frame turning amount from consecutive action velocity vectors.

    This measures direction/curvature instead of speed. A fast but straight
    segment gets low values, while a sharp change in action direction gets high
    values even if the absolute speed is moderate.
    """
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


def _compute_repo_scores(
    actions_by_episode: dict[int, np.ndarray],
    *,
    phase_bin_count: int,
    smoothing_half_window: int,
    entropy_weight: float,
    turn_weight: float,
    speed_weight: float,
    accel_weight: float,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[str, Any]]:
    if not actions_by_episode:
        raise ValueError("No episodes to score")

    all_actions = np.concatenate(list(actions_by_episode.values()), axis=0)
    action_mean = all_actions.mean(axis=0)
    action_std = all_actions.std(axis=0)
    action_std = np.where(action_std < 1e-6, 1.0, action_std)

    norm_actions = {
        ep: ((actions - action_mean) / action_std).astype(np.float32) for ep, actions in actions_by_episode.items()
    }
    bins_by_episode = {ep: _phase_bins(len(actions), phase_bin_count) for ep, actions in norm_actions.items()}

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

    entropy_values: list[np.ndarray] = []
    turn_values: list[np.ndarray] = []
    speed_values: list[np.ndarray] = []
    accel_values: list[np.ndarray] = []
    for ep, actions in norm_actions.items():
        bins = bins_by_episode[ep]
        entropy_values.append(phase_dispersion[bins])

        velocity = np.diff(actions, axis=0, prepend=actions[:1])
        acceleration = np.diff(velocity, axis=0, prepend=velocity[:1])
        speed = np.linalg.norm(velocity, axis=1)
        accel = np.linalg.norm(acceleration, axis=1)
        turn = _direction_change(velocity)
        turn_values.append(_moving_average(turn, smoothing_half_window))
        speed_values.append(_moving_average(speed, smoothing_half_window))
        accel_values.append(_moving_average(accel, smoothing_half_window))

    entropy_flat = _robust_unit_scale(np.concatenate(entropy_values, axis=0))
    turn_flat = _robust_unit_scale(np.concatenate(turn_values, axis=0))
    speed_flat = _robust_unit_scale(np.concatenate(speed_values, axis=0))
    accel_flat = _robust_unit_scale(np.concatenate(accel_values, axis=0))

    scores_by_episode: dict[int, np.ndarray] = {}
    offset = 0
    for ep, actions in norm_actions.items():
        n = len(actions)
        score = (
            entropy_weight * entropy_flat[offset : offset + n]
            + turn_weight * turn_flat[offset : offset + n]
            + speed_weight * speed_flat[offset : offset + n]
            + accel_weight * accel_flat[offset : offset + n]
        )
        denom = max(entropy_weight + turn_weight + speed_weight + accel_weight, 1e-6)
        scores_by_episode[ep] = (score / denom).astype(np.float32)
        offset += n

    stats = {
        "phase_bin_count": int(phase_bin_count),
        "phase_counts": phase_counts.tolist(),
        "phase_dispersion": phase_dispersion.round(6).tolist(),
        "action_dim": int(all_actions.shape[1]),
        "component_weights": {
            "phase_dispersion": float(entropy_weight),
            "turn": float(turn_weight),
            "speed": float(speed_weight),
            "acceleration": float(accel_weight),
        },
    }
    return scores_by_episode, bins_by_episode, stats


def _labels_and_weights(
    scores_by_episode: dict[int, np.ndarray],
    *,
    easy_quantile: float,
    hard_quantile: float,
    easy_stride: int,
    medium_stride: int,
    hard_stride: int,
    easy_weight: int,
    medium_weight: int,
    hard_weight: int,
    always_keep_head_tail: int,
) -> tuple[dict[int, list[str]], dict[int, np.ndarray], dict[str, Any]]:
    all_scores = np.concatenate(list(scores_by_episode.values()), axis=0)
    easy_threshold = float(np.quantile(all_scores, easy_quantile))
    hard_threshold = float(np.quantile(all_scores, hard_quantile))
    if hard_threshold < easy_threshold:
        raise ValueError("hard_threshold < easy_threshold; check quantiles")

    label_by_episode: dict[int, list[str]] = {}
    weights_by_episode: dict[int, np.ndarray] = {}
    label_counts: Counter[str] = Counter()
    kept_counts: Counter[str] = Counter()

    stride_by_label = {"easy": easy_stride, "medium": medium_stride, "hard": hard_stride}
    weight_by_label = {"easy": easy_weight, "medium": medium_weight, "hard": hard_weight}

    for ep, scores in scores_by_episode.items():
        labels_arr = np.full(len(scores), "medium", dtype=object)
        labels_arr[scores <= easy_threshold] = "easy"
        labels_arr[scores >= hard_threshold] = "hard"

        weights = np.zeros(len(scores), dtype=np.int32)
        for label in ("easy", "medium", "hard"):
            idx = np.where(labels_arr == label)[0]
            if len(idx) == 0:
                continue
            stride = max(int(stride_by_label[label]), 1)
            keep_idx = idx[np.arange(len(idx)) % stride == 0]
            weights[keep_idx] = int(weight_by_label[label])
            label_counts[label] += len(idx)
            kept_counts[label] += len(keep_idx)

        if always_keep_head_tail > 0 and len(weights) > 0:
            margin = min(always_keep_head_tail, len(weights))
            for keep_idx in list(range(margin)) + list(range(len(weights) - margin, len(weights))):
                label = str(labels_arr[keep_idx])
                weights[keep_idx] = max(weights[keep_idx], int(weight_by_label[label]))

        label_by_episode[ep] = [str(x) for x in labels_arr.tolist()]
        weights_by_episode[ep] = weights

    summary = {
        "thresholds": {
            "easy_max": easy_threshold,
            "hard_min": hard_threshold,
            "easy_quantile": easy_quantile,
            "hard_quantile": hard_quantile,
        },
        "label_counts": dict(label_counts),
        "kept_counts": dict(kept_counts),
        "strides": {
            "easy": easy_stride,
            "medium": medium_stride,
            "hard": hard_stride,
        },
        "label_weights": {
            "easy": easy_weight,
            "medium": medium_weight,
            "hard": hard_weight,
        },
    }
    return label_by_episode, weights_by_episode, summary


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
    scores_by_episode: dict[int, np.ndarray],
    bins_by_episode: dict[int, np.ndarray],
    labels_by_episode: dict[int, list[str]],
    weights_by_episode: dict[int, np.ndarray],
    summary: dict[str, Any],
    *,
    output_name: str,
    dry_run: bool,
) -> None:
    output_path = repo_root / "meta" / output_name
    summary_path = output_path.with_name(output_path.stem + "_summary.json")

    total_frames = int(sum(len(w) for w in weights_by_episode.values()))
    kept_frames = int(sum(np.count_nonzero(w) for w in weights_by_episode.values()))
    summary = {
        **summary,
        "repo_id": repo_root.name,
        "num_episodes": len(weights_by_episode),
        "num_frames": total_frames,
        "kept_frames": kept_frames,
        "keep_ratio": kept_frames / max(total_frames, 1),
        "output": str(output_path),
        "method": "phase_conditioned_action_dispersion_v2",
    }

    LOGGER.info(
        "%s: frames=%d kept=%d keep_ratio=%.2f%%",
        repo_root.name,
        total_frames,
        kept_frames,
        summary["keep_ratio"] * 100,
    )
    if dry_run:
        return

    episode_by_index = {int(ep["episode_index"]): ep for ep in episodes}
    with output_path.open("w", encoding="utf-8") as f:
        for ep_idx in sorted(weights_by_episode.keys()):
            record = {
                "episode_index": int(ep_idx),
                "task": episode_by_index.get(ep_idx, {}).get("tasks", []),
                "length": len(weights_by_episode[ep_idx]),
                "score": scores_by_episode[ep_idx].round(6).tolist(),
                "phase_bin": bins_by_episode[ep_idx].astype(int).tolist(),
                "label": labels_by_episode[ep_idx],
                "sample_weight": weights_by_episode[ep_idx].astype(int).tolist(),
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
            actions = _read_actions(repo_root, info, ep_idx)
        except Exception as exc:
            if args.strict:
                raise
            LOGGER.warning("Skipping %s episode %s: %s", repo_root.name, ep_idx, exc)
            continue
        actions_by_episode[ep_idx] = actions

    if not actions_by_episode:
        raise ValueError(f"No readable actions found in {repo_root}")

    scores_by_episode, bins_by_episode, score_summary = _compute_repo_scores(
        actions_by_episode,
        phase_bin_count=args.phase_bins,
        smoothing_half_window=args.smoothing_half_window,
        entropy_weight=args.entropy_weight,
        turn_weight=args.turn_weight,
        speed_weight=args.speed_weight,
        accel_weight=args.accel_weight,
    )
    labels_by_episode, weights_by_episode, label_summary = _labels_and_weights(
        scores_by_episode,
        easy_quantile=args.easy_quantile,
        hard_quantile=args.hard_quantile,
        easy_stride=args.easy_stride,
        medium_stride=args.medium_stride,
        hard_stride=args.hard_stride,
        easy_weight=args.easy_weight,
        medium_weight=args.medium_weight,
        hard_weight=args.hard_weight,
        always_keep_head_tail=args.always_keep_head_tail,
    )
    _write_repo_labels(
        repo_root,
        episodes,
        scores_by_episode,
        bins_by_episode,
        labels_by_episode,
        weights_by_episode,
        {**score_summary, **label_summary},
        output_name=args.output_name,
        dry_run=args.dry_run,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, required=True, help="Parent directory containing repo_id dirs.")
    parser.add_argument("--repo-id", action="append", default=[], help="Repo id to process. Can be repeated.")
    parser.add_argument("--repo-list-file", type=Path, default=None, help="Text file with one repo id per line.")
    parser.add_argument(
        "--discover-glob", type=str, default=None, help="Optional glob under root-dir, e.g. 'seatbelt.single.*'."
    )
    parser.add_argument(
        "--include-self-play", action="store_true", help="Do not filter self_play/raw_self_play/cpt names."
    )
    parser.add_argument("--limit-repos", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--output-name", type=str, default="difficulty_labels.jsonl")
    parser.add_argument("--phase-bins", type=int, default=24)
    parser.add_argument("--smoothing-half-window", type=int, default=2)
    parser.add_argument("--entropy-weight", type=float, default=0.75)
    parser.add_argument("--turn-weight", type=float, default=0.20)
    parser.add_argument(
        "--speed-weight",
        type=float,
        default=0.0,
        help="Optional anti-aliasing guard for fast motion. Defaults to 0 because speed alone is not difficulty.",
    )
    parser.add_argument("--accel-weight", type=float, default=0.05)
    parser.add_argument("--easy-quantile", type=float, default=0.35)
    parser.add_argument("--hard-quantile", type=float, default=0.75)
    parser.add_argument("--easy-stride", type=int, default=3)
    parser.add_argument("--medium-stride", type=int, default=2)
    parser.add_argument("--hard-stride", type=int, default=1)
    parser.add_argument("--easy-weight", type=int, default=1)
    parser.add_argument("--medium-weight", type=int, default=1)
    parser.add_argument("--hard-weight", type=int, default=1)
    parser.add_argument("--always-keep-head-tail", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    if args.phase_bins < 2:
        parser.error("--phase-bins must be >= 2")
    if not 0.0 <= args.easy_quantile <= 1.0:
        parser.error("--easy-quantile must be in [0, 1]")
    if not 0.0 <= args.hard_quantile <= 1.0:
        parser.error("--hard-quantile must be in [0, 1]")
    if args.easy_quantile >= args.hard_quantile:
        parser.error("--easy-quantile must be < --hard-quantile")
    for name in ("easy_stride", "medium_stride", "hard_stride", "easy_weight", "medium_weight", "hard_weight"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    for name in ("entropy_weight", "turn_weight", "speed_weight", "accel_weight"):
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
