"""Compute Separation Score (Cohen's d) for value model benchmarks.

For each model x task, measures how well the model's tail_pred separates
benchmark quadrants (e.g. fold TP vs TN, TP vs FP). Produces a 4-model
comparison table for the v0409 multi-task split.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Contrast definitions (which quadrant pairs to compare per task)
# ---------------------------------------------------------------------------

TASK_CONTRASTS: dict[str, list[tuple[str, str]]] = {
    "fold": [("TN", "TP"), ("FP", "TP"), ("edge", "TP")],
    "flatten": [("TN", "TP")],
}


# ---------------------------------------------------------------------------
# Cohen's d
# ---------------------------------------------------------------------------


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d = (mean_b - mean_a) / pooled_sd.

    Returns NaN for degenerate inputs (empty, zero variance).
    """
    if len(a) == 0 or len(b) == 0:
        return float("nan")

    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))

    # Pooled standard deviation with ddof=1 (sample sd)
    var_a = float(np.var(a, ddof=1)) if len(a) > 1 else 0.0
    var_b = float(np.var(b, ddof=1)) if len(b) > 1 else 0.0
    n_a, n_b = len(a), len(b)

    if n_a + n_b <= 2:
        return float("nan")

    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    if pooled_var <= 0:
        return float("nan")

    return (mean_b - mean_a) / math.sqrt(pooled_var)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_episode_scores(path: Path) -> dict[str, dict[str, float]]:
    """Load episode_details.json -> {episode_key: {tail_pred, head_pred, ...}}."""
    data = json.loads(Path(path).read_text())
    scores: dict[str, dict[str, float]] = {}
    for ep in data:
        key = ep.get("episode_key")
        tail = ep.get("tail_pred")
        if key is None or tail is None:
            continue
        scores[key] = {
            "tail_pred": float(tail),
            "head_pred": float(ep.get("head_pred")) if ep.get("head_pred") is not None else float("nan"),
        }
    return scores


def load_manifest_quadrants(path: Path) -> dict[str, list[str]]:
    """Load benchmark manifest.json -> {quadrant: [episode_keys]}."""
    manifest = json.loads(Path(path).read_text())
    return manifest["episodes"]


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def _gather_tail_preds(
    episode_keys: list[str],
    scores: dict[str, dict[str, float]],
    missing: list[str],
) -> np.ndarray:
    values = []
    for key in episode_keys:
        if key in scores:
            values.append(scores[key]["tail_pred"])
        else:
            missing.append(key)
    return np.array(values, dtype=float)


def compute_task_separation_scores(
    model_dir: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Compute Separation Scores for one model on one task.

    Args:
        model_dir: Path to model output root (contains metrics/episode_details.json).
        manifest_path: Path to task manifest.json.

    Returns:
        dict with keys: model_name, task_name, contrasts, missing_episodes.
    """
    model_dir = Path(model_dir)
    manifest_path = Path(manifest_path)

    episode_details_path = model_dir / "metrics" / "episode_details.json"
    scores = load_episode_scores(episode_details_path)
    quadrants = load_manifest_quadrants(manifest_path)

    manifest = json.loads(manifest_path.read_text())
    task_name = manifest["task_name"]
    contrasts_to_run = TASK_CONTRASTS.get(task_name, [])

    missing: list[str] = []
    contrasts: dict[str, dict[str, Any]] = {}

    for q_a, q_b in contrasts_to_run:
        keys_a = quadrants.get(q_a, [])
        keys_b = quadrants.get(q_b, [])
        arr_a = _gather_tail_preds(keys_a, scores, missing)
        arr_b = _gather_tail_preds(keys_b, scores, missing)
        d = cohen_d(arr_a, arr_b)
        contrasts[f"{q_b}_vs_{q_a}"] = {
            "cohen_d": d,
            "n_a": len(arr_a),
            "n_b": len(arr_b),
            "mean_a": float(np.mean(arr_a)) if len(arr_a) else float("nan"),
            "mean_b": float(np.mean(arr_b)) if len(arr_b) else float("nan"),
            "std_a": float(np.std(arr_a, ddof=1)) if len(arr_a) > 1 else float("nan"),
            "std_b": float(np.std(arr_b, ddof=1)) if len(arr_b) > 1 else float("nan"),
        }

    return {
        "model_name": model_dir.name,
        "task_name": task_name,
        "contrasts": contrasts,
        "missing_episodes": sorted(set(missing)),
    }


# ---------------------------------------------------------------------------
# CLI: multi-model report
# ---------------------------------------------------------------------------


@dataclass
class ModelSource:
    name: str
    model_dir: Path


def _format_contrast_table(per_model: list[dict[str, Any]]) -> str:
    """Render a markdown table: rows = models, columns = contrasts."""
    lines = []
    if not per_model:
        return "(no results)"

    task_name = per_model[0]["task_name"]
    contrast_names = list(per_model[0]["contrasts"].keys())

    header = ["Model", *contrast_names]
    lines.append(f"### Task: {task_name}")
    lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for res in per_model:
        row = [res["model_name"]]
        for cname in contrast_names:
            c = res["contrasts"].get(cname, {})
            d = c.get("cohen_d", float("nan"))
            n_a = c.get("n_a", 0)
            n_b = c.get("n_b", 0)
            row.append(f"d={d:.2f} (n={n_a}/{n_b})")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="model=path/to/model_dir (repeat for each model)",
    )
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        help="path/to/task/manifest.json (repeat for each task, e.g. fold + flatten)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON output path",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Optional markdown output path",
    )
    args = parser.parse_args()

    # Parse --model name=path args
    sources: list[ModelSource] = []
    for spec in args.model:
        if "=" not in spec:
            raise SystemExit(f"--model expects name=path, got: {spec}")
        name, path = spec.split("=", 1)
        sources.append(ModelSource(name=name.strip(), model_dir=Path(path.strip())))

    manifests = [Path(p) for p in args.manifest]

    report: dict[str, Any] = {"tasks": {}}
    md_chunks: list[str] = ["# v0409 Value Model Separation Score Comparison", ""]

    for manifest_path in manifests:
        per_model: list[dict[str, Any]] = []
        for src in sources:
            result = compute_task_separation_scores(src.model_dir, manifest_path)
            result["model_name"] = src.name  # override dir name with CLI label
            per_model.append(result)
        task_name = per_model[0]["task_name"]
        report["tasks"][task_name] = per_model
        md_chunks.append(_format_contrast_table(per_model))
        md_chunks.append("")

    md = "\n".join(md_chunks)
    print(md)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2))
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(md)


if __name__ == "__main__":
    main()
