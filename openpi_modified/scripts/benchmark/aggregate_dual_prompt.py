"""Aggregate hang/takeoff run_benchmark outputs into a dual-prompt report.

Consumes two directory trees of per-model `quadrant_summaries.json` files (one
from a hang-prompt run, one from a takeoff-prompt run), classifies each
(model, quadrant) pair into a qualitative dual-prompt bucket, and emits a
markdown comparison table.

The classification is the load-bearing bit — hang-only benchmarks cannot
distinguish a prompt-conditioned value model from a prompt-agnostic visual
regressor. Dual-prompt reveals four archetypal failure modes:

    aligned    - both prompts give high positive Pearson (true prompt-conditioned)
    reversed   - one positive, one strongly negative (model ignores prompt, GT flips)
    collapsed  - one high, the other near zero (training prompt vs vision gap)
    partial    - one high, the other mid-range (multitask-style leakage)
    degenerate - nan / both negative (no usable signal)

Priority order: degenerate → reversed → aligned → collapsed → partial.

CLI::

    python -m scripts.benchmark.aggregate_dual_prompt --out dual_prompt_report.md

Defaults are pinned to `test_results/benchmark/cleaned_test_split{,_takeoff}`.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)


_QUADRANTS_IN_REPORT: tuple[str, ...] = ("true_positive", "true_negative", "false_positive")

_DEFAULT_HANG_DIR = Path("test_results/benchmark/cleaned_test_split")
_DEFAULT_TAKEOFF_DIR = Path("test_results/benchmark/cleaned_test_split_takeoff")


def classify_dual_prompt_status(hang: float, takeoff: float) -> str:
    """Classify a (hang_pearson, takeoff_pearson) pair into a qualitative bucket.

    Priority (evaluated top-down, first match wins):

    1. ``degenerate``: either value is nan, or both are negative
    2. ``reversed``: hang > 0.5 and takeoff < -0.5 (pred unchanged, GT flipped)
    3. ``aligned``: both > 0.5 (prompt-conditioned)
    4. ``collapsed``: one side high (> 0.5), other near zero (|.| < 0.3)
    5. ``partial``: fallback for everything else
    """
    if math.isnan(hang) or math.isnan(takeoff):
        return "degenerate"
    if hang < 0 and takeoff < 0:
        return "degenerate"
    if hang > 0.5 and takeoff < -0.5:
        return "reversed"
    if hang > 0.5 and takeoff > 0.5:
        return "aligned"
    if (hang > 0.5 and abs(takeoff) < 0.3) or (takeoff > 0.5 and abs(hang) < 0.3):
        return "collapsed"
    return "partial"


def load_run_outputs(base_dir: Path) -> dict[str, dict]:
    """Load `quadrant_summaries.json` for each model under ``base_dir``.

    Returns ``{model_name: raw_summary_dict}``. Models whose
    ``metrics/quadrant_summaries.json`` is missing are silently skipped — the
    caller treats them as "no data for this prompt".
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Run output dir not found: {base_dir}")

    result: dict[str, dict] = {}
    for model_dir in sorted(base_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        summary_file = model_dir / "metrics" / "quadrant_summaries.json"
        if not summary_file.exists():
            continue
        with open(summary_file) as f:
            result[model_dir.name] = json.load(f)
    return result


def _extract_quadrant_metric(
    data: dict[str, dict],
    model: str,
    quadrant: str,
    metric: str,
) -> float | None:
    """Look up ``data[model][quadrant][metric]``; return None if any step missing."""
    if model not in data:
        return None
    model_data = data[model]
    if quadrant not in model_data:
        return None
    quadrant_data = model_data[quadrant]
    if metric not in quadrant_data:
        return None
    return quadrant_data[metric]


def diff_hang_vs_takeoff(
    hang_data: dict[str, dict],
    takeoff_data: dict[str, dict],
) -> list[dict]:
    """Join hang and takeoff runs into per-(model, quadrant) rows.

    For each (model, quadrant) pair in TP/TN/FP, compute Pearson / tail_mse for both
    prompts and classify the status. Models missing entirely from one side
    produce rows with ``status = "TBD"`` and None values on that side. FN is
    skipped because it's typically empty or degenerate in seatbelt data.
    """
    rows: list[dict] = []
    all_models = sorted(set(hang_data.keys()) | set(takeoff_data.keys()))

    for model in all_models:
        for quadrant in _QUADRANTS_IN_REPORT:
            hang_pearson = _extract_quadrant_metric(hang_data, model, quadrant, "median_pearson")
            takeoff_pearson = _extract_quadrant_metric(takeoff_data, model, quadrant, "median_pearson")
            hang_mse = _extract_quadrant_metric(hang_data, model, quadrant, "median_tail_mse")
            takeoff_mse = _extract_quadrant_metric(takeoff_data, model, quadrant, "median_tail_mse")

            if hang_pearson is None or takeoff_pearson is None:
                status = "TBD"
            else:
                status = classify_dual_prompt_status(hang_pearson, takeoff_pearson)

            rows.append(
                {
                    "quadrant": quadrant,
                    "model": model,
                    "hang_pearson": hang_pearson,
                    "takeoff_pearson": takeoff_pearson,
                    "hang_tail_mse": hang_mse,
                    "takeoff_tail_mse": takeoff_mse,
                    "status": status,
                }
            )
    return rows


def _fmt_pearson(value: float | None) -> str:
    if value is None:
        return "TBD"
    if math.isnan(value):
        return "nan"
    return f"{round(value, 4):+.4f}"


def _fmt_mse(value: float | None) -> str:
    if value is None:
        return "TBD"
    if math.isnan(value):
        return "nan"
    return f"{round(value, 4):.4f}"


def _quadrant_label(quadrant: str) -> str:
    return {
        "true_positive": "TP",
        "true_negative": "TN",
        "false_positive": "FP",
        "false_negative": "FN",
    }.get(quadrant, quadrant)


def render_report_markdown(rows: list[dict]) -> str:
    """Render the diff rows as a markdown report.

    The output has two tables: Median Pearson and Median Tail MSE, both with
    columns ``Quadrant | Model | Hang | Takeoff | Status``. Values are rounded
    to 4 decimal places to match the 飞书 section 5.2 table byte-for-byte.
    """
    lines: list[str] = []
    lines.append("# Dual-Prompt Benchmark Report")
    lines.append("")
    lines.append(
        "Generated by `scripts/benchmark/aggregate_dual_prompt.py`. "
        "Compares hang-prompt vs takeoff-prompt `run_benchmark.py` outputs."
    )
    lines.append("")
    lines.append(
        "**Status legend**: `aligned` = prompt-conditioned; "
        "`reversed` = ignores prompt; `collapsed` = one side collapsed to 0; "
        "`partial` = partial coverage; `degenerate` = nan / no signal; "
        "`TBD` = missing data on one side."
    )
    lines.append("")

    lines.append("## Median Pearson")
    lines.append("")
    lines.append("| Quadrant | Model | Hang | Takeoff | Status |")
    lines.append("|---|---|---|---|---|")
    lines.extend(
        f"| {_quadrant_label(row['quadrant'])} "
        f"| {row['model']} "
        f"| {_fmt_pearson(row['hang_pearson'])} "
        f"| {_fmt_pearson(row['takeoff_pearson'])} "
        f"| {row['status']} |"
        for row in rows
    )
    lines.append("")

    lines.append("## Median Tail MSE")
    lines.append("")
    lines.append("| Quadrant | Model | Hang | Takeoff |")
    lines.append("|---|---|---|---|")
    lines.extend(
        f"| {_quadrant_label(row['quadrant'])} "
        f"| {row['model']} "
        f"| {_fmt_mse(row['hang_tail_mse'])} "
        f"| {_fmt_mse(row['takeoff_tail_mse'])} |"
        for row in rows
    )
    lines.append("")

    return "\n".join(lines)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument(
        "--hang-dir",
        type=Path,
        default=_DEFAULT_HANG_DIR,
        help=f"Directory with hang-prompt per-model outputs (default: {_DEFAULT_HANG_DIR}).",
    )
    parser.add_argument(
        "--takeoff-dir",
        type=Path,
        default=_DEFAULT_TAKEOFF_DIR,
        help=f"Directory with takeoff-prompt per-model outputs (default: {_DEFAULT_TAKEOFF_DIR}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output markdown path.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    hang_data = load_run_outputs(args.hang_dir)
    takeoff_data = load_run_outputs(args.takeoff_dir)

    logger.info("hang models: %s", sorted(hang_data.keys()))
    logger.info("takeoff models: %s", sorted(takeoff_data.keys()))

    rows = diff_hang_vs_takeoff(hang_data, takeoff_data)
    md = render_report_markdown(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md)
    logger.info("Wrote %d rows to %s", len(rows), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
