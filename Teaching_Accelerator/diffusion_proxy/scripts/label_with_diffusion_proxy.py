#!/usr/bin/env python3
"""Generate sidecar labels from out-of-fold diffusion proxy checkpoints."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import logging
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffusion_proxy.data import build_dataset_arrays
from diffusion_proxy.data import load_episodes
from diffusion_proxy.inference import infer_checkpoint_heldout
from diffusion_proxy.inference import labels_from_entropy
from diffusion_proxy.sidecar import round_array
from diffusion_proxy.sidecar import write_jsonl
from diffusion_proxy.sidecar import write_summary
from diffusion_proxy.utils import repo_ids_from_file
from diffusion_proxy.vision import DEFAULT_ENCODER


LOGGER = logging.getLogger("label_with_diffusion_proxy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--repo-list-file", type=Path, required=True)
    parser.add_argument("--vision-cache", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--camera", default="observation.images.head")
    parser.add_argument("--encoder", choices=["resnet18", "grid"], default=DEFAULT_ENCODER)
    parser.add_argument("--max-episodes-per-repo", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--samples-per-frame", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision-quantile", type=float, default=0.75)
    parser.add_argument("--casual-quantile", type=float, default=0.65)
    parser.add_argument("--static-speed-quantile", type=float, default=0.10)
    parser.add_argument("--smoothing-half-window", type=int, default=8)
    parser.add_argument("--min-span-frames", type=int, default=15)
    parser.add_argument("--merge-gap-frames", type=int, default=10)
    parser.add_argument("--span-padding-frames", type=int, default=8)
    parser.add_argument("--plot-repo-suffix", default="batch.1")
    parser.add_argument("--plot-episodes", type=int, default=5)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def checkpoint_paths(checkpoint_dir: Path) -> list[Path]:
    paths = sorted(checkpoint_dir.glob("fold_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No fold_*.pt checkpoints in {checkpoint_dir}")
    return paths


def build_records(arrays, label_result: dict) -> list[dict]:
    records = []
    records_by_key = label_result["records_by_key"]
    for episode in arrays.episodes:
        values = records_by_key[episode.key]
        records.append(
            {
                "repo_id": episode.repo_id,
                "episode_index": int(episode.episode_index),
                "task": episode.task,
                "length": episode.length,
                "fps": int(episode.fps),
                "diffusion_precision_score": round_array(values["diffusion_precision_score"]),
                "diffusion_entropy_score": round_array(values["diffusion_entropy_score"]),
                "diffusion_reconstruction_error": round_array(values["diffusion_reconstruction_error"]),
                "label": values["label"],
                "acceleration_stride": np.asarray(values["acceleration_stride"], dtype=np.int32).tolist(),
                "hard_spans": values["hard_spans"],
            }
        )
    return records


def write_report(records: list[dict], summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(label for record in records for label in record["label"])
    lines = [
        "# Diffusion Proxy Label Report",
        "",
        f"- Episodes: {summary['num_episodes']}",
        f"- Frames: {summary['num_frames']}",
        f"- Hard spans: {summary['num_hard_spans']}",
        f"- Label counts: precision={counts.get('precision', 0)}, neutral={counts.get('neutral', 0)}, casual={counts.get('casual', 0)}",
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
        "| repo_id | episode | start_s | end_s | duration_s | mean_diffusion_precision_score |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    spans = []
    for record in records:
        for span in record["hard_spans"]:
            spans.append((float(span.get("mean_diffusion_precision_score", 0.0)), record, span))
    for _, record, span in sorted(spans, key=lambda x: x[0], reverse=True)[:30]:
        lines.append(
            "| {repo} | {ep} | {start:.3f} | {end:.3f} | {dur:.3f} | {score:.6f} |".format(
                repo=record["repo_id"],
                ep=record["episode_index"],
                start=float(span["start_s"]),
                end=float(span["end_s"]),
                dur=float(span["duration_s"]),
                score=float(span.get("mean_diffusion_precision_score", 0.0)),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def write_episode_plot(record: dict, threshold: float, path: Path) -> None:
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
    poly = _polyline(record["diffusion_precision_score"], width=width, height=height, margin=margin)
    title = f"{record['repo_id']} ep {int(record['episode_index']):06d}"
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fffaf0"/>
  {' '.join(rects)}
  <line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#333" stroke-width="1"/>
  <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#333" stroke-width="1"/>
  <line x1="{margin}" y1="{threshold_y:.2f}" x2="{width - margin}" y2="{threshold_y:.2f}" stroke="#a23b3b" stroke-width="1.5" stroke-dasharray="6 6"/>
  <polyline points="{poly}" fill="none" stroke="#145c9e" stroke-width="2.2"/>
  <text x="{margin}" y="24" font-family="monospace" font-size="15" fill="#222">{title}</text>
  <text x="{width - margin - 260}" y="24" font-family="monospace" font-size="13" fill="#145c9e">diffusion_precision_score</text>
  <text x="{width - margin - 260}" y="44" font-family="monospace" font-size="12" fill="#a23b3b">precision threshold</text>
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def write_plots(records: list[dict], summary: dict, report_dir: Path, repo_suffix: str, count: int) -> None:
    if count <= 0:
        return
    selected = [record for record in records if str(record["repo_id"]).endswith(repo_suffix)]
    selected = sorted(selected, key=lambda r: int(r["episode_index"]))[:count]
    if not selected:
        return
    threshold = float(summary["thresholds"]["diffusion_precision_min"])
    plot_dir = report_dir / "plots"
    for record in selected:
        path = plot_dir / f"batch1_episode_{int(record['episode_index']):06d}_diffusion_precision_score.svg"
        write_episode_plot(record, threshold, path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s:%(name)s:%(message)s")
    repo_ids = repo_ids_from_file(args.repo_list_file)
    episodes = load_episodes(
        args.root_dir,
        repo_ids,
        vision_cache=args.vision_cache,
        camera=args.camera,
        encoder=args.encoder,
        max_episodes_per_repo=args.max_episodes_per_repo,
    )
    arrays = build_dataset_arrays(episodes, horizon=16)
    entropy_by_key = {}
    recon_by_key = {}
    checkpoints = checkpoint_paths(args.checkpoint_dir)
    for path in checkpoints:
        LOGGER.info("Inferring heldout fold from %s", path)
        outputs, _ = infer_checkpoint_heldout(
            path,
            arrays,
            batch_size=args.batch_size,
            sampling_steps=args.sampling_steps,
            samples_per_frame=args.samples_per_frame,
            device=args.device,
        )
        for key, values in outputs.items():
            if key in entropy_by_key:
                raise ValueError(f"Duplicate heldout prediction for {key}")
            entropy_by_key[key] = values["diffusion_entropy_raw"]
            recon_by_key[key] = values["diffusion_reconstruction_error_raw"]
    missing = [ep.key for ep in arrays.episodes if ep.key not in entropy_by_key]
    if missing:
        raise ValueError(f"Missing diffusion outputs for {len(missing)} episodes, first={missing[0]}")
    label_result = labels_from_entropy(
        arrays,
        entropy_by_key,
        recon_by_key,
        precision_quantile=args.precision_quantile,
        casual_quantile=args.casual_quantile,
        static_speed_quantile=args.static_speed_quantile,
        smoothing_half_window=args.smoothing_half_window,
        min_span_frames=args.min_span_frames,
        merge_gap_frames=args.merge_gap_frames,
        span_padding_frames=args.span_padding_frames,
    )
    records = build_records(arrays, label_result)
    output_path = args.output_dir / "diffusion_labels.jsonl"
    summary_path = args.output_dir / "summary.json"
    report_path = args.report_dir / "diffusion_proxy_report.md"
    spans_by_repo = defaultdict(int)
    frames_by_repo = defaultdict(int)
    for record in records:
        spans_by_repo[record["repo_id"]] += len(record["hard_spans"])
        frames_by_repo[record["repo_id"]] += int(record["length"])
    summary = {
        **label_result["summary"],
        "root_dir": str(args.root_dir),
        "repo_ids": repo_ids,
        "vision_cache": str(args.vision_cache),
        "camera": args.camera,
        "encoder": args.encoder,
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoints": [str(path) for path in checkpoints],
        "output": str(output_path),
        "records": len(records),
        "frames_by_repo": dict(frames_by_repo),
        "hard_spans_by_repo": dict(spans_by_repo),
    }
    write_jsonl(records, output_path, kind="diffusion")
    write_summary(summary, summary_path)
    write_report(records, summary, report_path)
    write_plots(records, summary, args.report_dir, args.plot_repo_suffix, args.plot_episodes)
    LOGGER.info("Wrote diffusion labels to %s", output_path)


if __name__ == "__main__":
    main()
