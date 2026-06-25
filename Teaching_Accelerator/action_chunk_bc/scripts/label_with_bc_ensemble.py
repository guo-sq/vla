#!/usr/bin/env python3
"""Generate precision/casual sidecar labels with a BC ensemble."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from action_chunk_bc.data import build_dataset_arrays
from action_chunk_bc.data import load_episodes
from action_chunk_bc.data import repo_ids_from_file
from action_chunk_bc.inference import ensemble_disagreement
from action_chunk_bc.inference import labels_from_disagreement
from action_chunk_bc.sidecar import round_array
from action_chunk_bc.sidecar import write_jsonl
from action_chunk_bc.sidecar import write_summary


LOGGER = logging.getLogger("label_with_bc_ensemble")


def _seed_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"seed_(\d+)\.pt$", path.name)
    return (int(match.group(1)) if match else 10**9, path.name)


def checkpoint_paths(checkpoint_dir: Path, limit: int | None = None) -> list[Path]:
    paths = sorted(checkpoint_dir.glob("seed_*.pt"), key=_seed_sort_key)
    if limit is not None:
        paths = paths[:limit]
    if len(paths) < 2:
        raise FileNotFoundError(f"Need at least two seed_*.pt checkpoints in {checkpoint_dir}")
    return paths


def checkpoint_horizon(path: Path) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return int(ckpt.get("model_config", {}).get("horizon", 16))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--repo-list-file", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--max-episodes-per-repo", type=int, default=None)
    parser.add_argument("--ensemble-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision-quantile", type=float, default=0.75)
    parser.add_argument("--casual-quantile", type=float, default=0.65)
    parser.add_argument("--static-speed-quantile", type=float, default=0.10)
    parser.add_argument("--plot-repo-suffix", default="batch.1")
    parser.add_argument("--plot-episodes", type=int, default=5)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def build_records(arrays, label_result: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    records_by_key = label_result["records_by_key"]
    for episode in arrays.episodes:
        key = (episode.repo_id, episode.episode_index)
        values = records_by_key[key]
        record = {
            "repo_id": episode.repo_id,
            "episode_index": int(episode.episode_index),
            "task": episode.task,
            "length": episode.length,
            "fps": int(episode.fps),
            "bc_precision_score": round_array(values["bc_precision_score"]),
            "ensemble_disagreement_score": round_array(values["ensemble_disagreement_score"]),
            "label": values["label"],
            "acceleration_stride": np.asarray(values["acceleration_stride"], dtype=np.int32).tolist(),
            "hard_spans": values["hard_spans"],
        }
        records.append(record)
    return records


def build_summary(
    label_result: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    root_dir: Path,
    repo_ids: list[str],
    checkpoint_dir: Path,
    checkpoints: list[Path],
    output_path: Path,
) -> dict[str, Any]:
    spans_by_repo: dict[str, int] = defaultdict(int)
    episodes_by_repo: dict[str, int] = defaultdict(int)
    frames_by_repo: dict[str, int] = defaultdict(int)
    fps_values = sorted({int(record["fps"]) for record in records})
    for record in records:
        repo_id = str(record["repo_id"])
        spans_by_repo[repo_id] += len(record["hard_spans"])
        episodes_by_repo[repo_id] += 1
        frames_by_repo[repo_id] += int(record["length"])
    return {
        **label_result["summary"],
        "records": len(records),
        "fps": fps_values,
        "root_dir": str(root_dir),
        "repo_ids": repo_ids,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoints": [str(path) for path in checkpoints],
        "output": str(output_path),
        "episodes_by_repo": dict(episodes_by_repo),
        "frames_by_repo": dict(frames_by_repo),
        "hard_spans_by_repo": dict(spans_by_repo),
    }


def write_report(records: list[dict[str, Any]], summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    label_counts = Counter(label for record in records for label in record["label"])
    lines = [
        "# BC Ensemble Label Report",
        "",
        f"- Episodes: {summary['num_episodes']}",
        f"- Frames: {summary['num_frames']}",
        f"- FPS: {summary.get('fps', [])}",
        f"- Hard spans: {summary['num_hard_spans']}",
        f"- Label counts: precision={label_counts.get('precision', 0)}, neutral={label_counts.get('neutral', 0)}, casual={label_counts.get('casual', 0)}",
        f"- Checkpoints: `{summary['checkpoint_dir']}`",
        f"- Output: `{summary['output']}`",
        "",
        "## Thresholds",
        "",
        "```json",
        json.dumps(summary["thresholds"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Top Precision Spans",
        "",
        "| repo_id | episode | start_s | end_s | duration_s | mean_bc_precision_score |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    spans: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for record in records:
        for span in record["hard_spans"]:
            spans.append((float(span.get("mean_bc_precision_score", 0.0)), record, span))
    for _, record, span in sorted(spans, key=lambda x: x[0], reverse=True)[:30]:
        lines.append(
            "| {repo} | {ep} | {start:.3f} | {end:.3f} | {dur:.3f} | {score:.6f} |".format(
                repo=record["repo_id"],
                ep=record["episode_index"],
                start=float(span["start_s"]),
                end=float(span["end_s"]),
                dur=float(span["duration_s"]),
                score=float(span.get("mean_bc_precision_score", 0.0)),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _polyline(values: list[float], *, width: int, height: int, margin: int) -> str:
    if len(values) == 1:
        values = values * 2
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    points = []
    for idx, value in enumerate(values):
        x = margin + plot_w * idx / max(len(values) - 1, 1)
        y = margin + plot_h * (1.0 - max(0.0, min(1.0, float(value))))
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def write_episode_plot(record: dict[str, Any], threshold: float, path: Path) -> None:
    width, height, margin = 960, 320, 44
    length = max(int(record["length"]), 1)
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    rects = []
    for span in record["hard_spans"]:
        x = margin + plot_w * int(span["start_frame"]) / length
        w = plot_w * max(int(span["end_frame"]) - int(span["start_frame"]), 1) / length
        rects.append(f'<rect x="{x:.2f}" y="{margin}" width="{w:.2f}" height="{plot_h}" fill="#f3c565" opacity="0.24"/>')
    threshold_y = margin + plot_h * (1.0 - max(0.0, min(1.0, threshold)))
    poly = _polyline(record["bc_precision_score"], width=width, height=height, margin=margin)
    title = f"{record['repo_id']} ep {int(record['episode_index']):06d}"
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fffaf0"/>
  {' '.join(rects)}
  <line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#333" stroke-width="1"/>
  <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#333" stroke-width="1"/>
  <line x1="{margin}" y1="{threshold_y:.2f}" x2="{width - margin}" y2="{threshold_y:.2f}" stroke="#a23b3b" stroke-width="1.5" stroke-dasharray="6 6"/>
  <polyline points="{poly}" fill="none" stroke="#145c9e" stroke-width="2.2"/>
  <text x="{margin}" y="24" font-family="monospace" font-size="15" fill="#222">{title}</text>
  <text x="{width - margin - 220}" y="24" font-family="monospace" font-size="13" fill="#145c9e">bc_precision_score</text>
  <text x="{width - margin - 220}" y="44" font-family="monospace" font-size="12" fill="#a23b3b">precision threshold</text>
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def write_combined_plot(records: list[dict[str, Any]], path: Path) -> None:
    width, height, margin = 960, 360, 52
    colors = ["#145c9e", "#cc5a43", "#2d7d4f", "#7a4ca0", "#8a6d18"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffaf0"/>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#333" stroke-width="1"/>',
        f'<text x="{margin}" y="28" font-family="monospace" font-size="15" fill="#222">BC precision score curves</text>',
    ]
    for idx, record in enumerate(records):
        color = colors[idx % len(colors)]
        poly = _polyline(record["bc_precision_score"], width=width, height=height, margin=margin)
        lines.append(f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="1.8" opacity="0.86"/>')
        lines.append(
            f'<text x="{width - margin - 180}" y="{28 + idx * 18}" font-family="monospace" font-size="12" fill="{color}">ep {int(record["episode_index"]):06d}</text>'
        )
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plots(records: list[dict[str, Any]], summary: dict[str, Any], report_dir: Path, repo_suffix: str, count: int) -> None:
    if count <= 0:
        return
    selected = [record for record in records if str(record["repo_id"]).endswith(repo_suffix)]
    selected = sorted(selected, key=lambda r: int(r["episode_index"]))[:count]
    if not selected:
        return
    threshold = float(summary["thresholds"]["bc_precision_min"])
    plot_dir = report_dir / "plots"
    for record in selected:
        path = plot_dir / f"batch1_episode_{int(record['episode_index']):06d}_bc_precision_score.svg"
        write_episode_plot(record, threshold, path)
    combined = plot_dir / "batch1_episodes_000000_000004_bc_precision_score_curves.svg"
    write_combined_plot(selected, combined)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s:%(name)s:%(message)s")

    repo_ids = repo_ids_from_file(args.repo_list_file)
    if not repo_ids:
        raise SystemExit(f"No repo ids found in {args.repo_list_file}")
    checkpoints = checkpoint_paths(args.checkpoint_dir, args.ensemble_size)
    horizon = checkpoint_horizon(checkpoints[0])

    LOGGER.info("Loading episodes with horizon=%d", horizon)
    episodes = load_episodes(args.root_dir, repo_ids, max_episodes_per_repo=args.max_episodes_per_repo)
    arrays = build_dataset_arrays(episodes, horizon)
    LOGGER.info("Loaded %d episodes, %d frames", len(episodes), len(arrays.features))
    LOGGER.info("Running ensemble inference with %d checkpoints", len(checkpoints))
    disagreement_raw, _ = ensemble_disagreement(checkpoints, arrays, batch_size=args.batch_size, device=args.device)

    label_result = labels_from_disagreement(
        arrays,
        disagreement_raw,
        precision_quantile=args.precision_quantile,
        casual_quantile=args.casual_quantile,
        static_speed_quantile=args.static_speed_quantile,
    )
    records = build_records(arrays, label_result)
    output_path = args.output_dir / "bc_ensemble_labels.jsonl"
    summary_path = args.output_dir / "summary.json"
    report_path = args.report_dir / "bc_ensemble_report.md"
    summary = build_summary(
        label_result,
        records,
        root_dir=args.root_dir,
        repo_ids=repo_ids,
        checkpoint_dir=args.checkpoint_dir,
        checkpoints=checkpoints,
        output_path=output_path,
    )

    write_jsonl(records, output_path)
    write_summary(summary, summary_path)
    write_report(records, summary, report_path)
    write_plots(records, summary, args.report_dir, args.plot_repo_suffix, args.plot_episodes)

    LOGGER.info("Wrote %d episode records to %s", len(records), output_path)
    LOGGER.info("Wrote summary to %s", summary_path)
    LOGGER.info("Wrote report to %s", report_path)


if __name__ == "__main__":
    main()
