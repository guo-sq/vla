"""Compute HEAD_PRED_RANGES prior for the 2D episode classifier.

Given a sparse benchmark's ``episode_details.json`` and the merged
``ground_truth_labels.json`` from :mod:`merge_ground_truth_labels`, group
episodes by their ground-truth category and compute the p5/p25/p50/p75/p95
percentiles of ``head_pred``. The output can be dropped into
``openpi.training.episode_classifier_2d.HEAD_PRED_RANGES`` to promote the
classifier from 1D (tail only) to full 2D dispatch.

Categories with fewer than ``min_samples`` labeled episodes return ``None`` so
the downstream classifier keeps them on the 1D tail_pred fallback rather than
using an unreliable percentile range.

Usage:

    python scripts/benchmark/fill_head_pred_ranges.py \\
      --episode-details test_results/benchmark/clothes_v0409_sparse/fast_mode_max3600/batch_000/metrics/episode_details.json \\
      --ground-truth-labels test_results/data_audit/ground_truth_labels.json \\
      --output test_results/data_audit/head_pred_ranges.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

# Allow ``python scripts/benchmark/fill_head_pred_ranges.py`` direct invocation
# to resolve the ``scripts.benchmark.episode_classifier_2d`` lazy import.
_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)


ALL_CATEGORIES = (
    "fold_success",
    "flatten_success",
    "shuffle_success",
    "fold_failure",
    "intervention_recovery",
)


def lookup_category(
    episode_key: str,
    episode_labels: dict[str, dict[str, Any]],
    repo_labels: dict[str, dict[str, Any]],
) -> str | None:
    """Resolve an episode's ground-truth category with episode->repo fallback."""
    if episode_key in episode_labels:
        return episode_labels[episode_key].get("category")
    repo_id = episode_key.rsplit(":", 1)[0]
    if repo_id in repo_labels:
        return repo_labels[repo_id].get("category")
    return None


def _classify_by_tail_pred_1d(
    episode: dict[str, Any],
    regions_1d: Any,
) -> str | None:
    """Return the 1D-classified category for an episode, or None if we can't.

    Only the ``tail_pred`` column is consulted, because the 1D classifier's
    strength is precisely that it ignores head_pred. Accepts all confidence
    levels because the purpose of bootstrapping is to **maximize head_pred
    sample coverage** - tight confidence filtering would defeat the point.
    Note that 1D mode cannot distinguish ``fold_success`` from
    ``intervention_recovery`` (both have tail_pred ≈ 0 with overlapping
    distributions); which of the two a sample gets routed to does not
    affect downstream 2D classification because head_pred is the axis that
    separates them.
    """
    tail_pred = episode.get("tail_pred")
    if tail_pred is None or (isinstance(tail_pred, float) and math.isnan(tail_pred)):
        return None

    from scripts.benchmark.episode_classifier_2d import classify_episode

    category_enum, _conf, _dist, _mode = classify_episode(
        head_pred=None,
        tail_pred=float(tail_pred),
        regions=regions_1d,
    )
    return category_enum.value


def compute_head_pred_ranges(
    episodes: list[dict[str, Any]],
    episode_labels: dict[str, dict[str, Any]],
    repo_labels: dict[str, dict[str, Any]],
    *,
    min_samples: int = 20,
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0,
    bootstrap_from_1d: bool = False,
) -> dict[str, dict[str, float] | None]:
    """Group episodes by category and compute head_pred percentile ranges.

    The downstream classifier uses ``head_min``/``head_max`` as its region
    bounds (the "active" range), while ``p5`` through ``p95`` are reported as
    summary stats for human inspection regardless of the bound choice.

    Args:
        episodes: Rows from a sparse ``episode_details.json``. Each must have
            ``episode_key`` and ``head_pred`` (numeric or None/NaN).
        episode_labels: Per-episode ground-truth from ``merge_ground_truth_labels``.
        repo_labels: Per-repo ground-truth fallback from the same source.
        min_samples: Minimum episodes per category before we emit a range.
            Below this, the category's entry is ``None`` so the downstream
            classifier can fall back to 1D.
        lower_percentile: Percentile used for ``head_min`` (0.0-100.0).
            Default 5.0. Use 1.0 for wider coverage, 0.0 for raw minimum.
        upper_percentile: Percentile used for ``head_max`` (0.0-100.0).
            Default 95.0. Use 99.0 for wider coverage, 100.0 for raw maximum.
        bootstrap_from_1d: When ``True``, episodes missing from both label
            dicts fall back to classification by 1D tail_pred (via
            ``episode_classifier_2d.classify_episode`` in 1D mode). This
            trades strict ground-truth fidelity for dramatically better
            sample coverage of the real head_pred distribution - the 1D
            tail_pred classifier was shown to be 100%-separating on the
            fast_mode model in Step 4c-1, so the shuffle_success /
            fold_success categories it produces are reliable enough for
            the prior. Explicit labels always take precedence; only
            unlabeled episodes get bootstrapped.

    Returns:
        ``{category: {"head_min", "head_max", "p5", "p25", "p50", "p75",
        "p95", "n_samples", "lower_percentile", "upper_percentile"} | None}``.
        All five categories from :data:`ALL_CATEGORIES` appear as keys.

    Raises:
        ValueError: When the percentile bounds are invalid.
    """
    if not (0.0 <= lower_percentile <= 100.0):
        raise ValueError(f"lower_percentile must be in [0, 100], got {lower_percentile}")
    if not (0.0 <= upper_percentile <= 100.0):
        raise ValueError(f"upper_percentile must be in [0, 100], got {upper_percentile}")
    if lower_percentile >= upper_percentile:
        raise ValueError(f"lower_percentile ({lower_percentile}) must be < upper_percentile " f"({upper_percentile})")

    regions_1d = None
    if bootstrap_from_1d:
        # Import lazily to avoid a circular dependency at module load time.
        from scripts.benchmark.episode_classifier_2d import build_default_regions

        # Default regions have all head_*=None -> classify_episode runs in 1D mode.
        regions_1d = build_default_regions()

    grouped: dict[str, list[float]] = {cat: [] for cat in ALL_CATEGORIES}

    for episode in episodes:
        episode_key = episode.get("episode_key")
        if not episode_key:
            continue
        head_pred = episode.get("head_pred")
        if head_pred is None or (isinstance(head_pred, float) and math.isnan(head_pred)):
            continue
        category = lookup_category(episode_key, episode_labels, repo_labels)
        if category is None and bootstrap_from_1d:
            category = _classify_by_tail_pred_1d(episode, regions_1d)
        if category is None or category not in grouped:
            continue
        grouped[category].append(float(head_pred))

    result: dict[str, dict[str, float] | None] = {}
    for category, values in grouped.items():
        if len(values) < min_samples:
            result[category] = None
            continue
        p5, p25, p50, p75, p95 = np.percentile(values, [5, 25, 50, 75, 95])
        head_min = float(np.percentile(values, lower_percentile))
        head_max = float(np.percentile(values, upper_percentile))
        result[category] = {
            "head_min": head_min,
            "head_max": head_max,
            "p5": float(p5),
            "p25": float(p25),
            "p50": float(p50),
            "p75": float(p75),
            "p95": float(p95),
            "n_samples": len(values),
            "lower_percentile": float(lower_percentile),
            "upper_percentile": float(upper_percentile),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episode-details", type=Path, required=True, help="Path to episode_details.json from a sparse benchmark run"
    )
    parser.add_argument(
        "--ground-truth-labels", type=Path, default=Path("test_results/data_audit/ground_truth_labels.json")
    )
    parser.add_argument("--output", type=Path, default=Path("test_results/data_audit/head_pred_ranges.json"))
    parser.add_argument(
        "--min-samples", type=int, default=20, help="Minimum labeled episodes per category before emitting a range"
    )
    parser.add_argument(
        "--lower-percentile",
        type=float,
        default=5.0,
        help="Percentile for head_min. Default 5.0. Use 1.0 for wider " "coverage or 0.0 for raw minimum.",
    )
    parser.add_argument(
        "--upper-percentile",
        type=float,
        default=95.0,
        help="Percentile for head_max. Default 95.0. Use 99.0 for wider " "coverage or 100.0 for raw maximum.",
    )
    parser.add_argument(
        "--bootstrap-from-1d",
        action="store_true",
        help="When set, episodes without an explicit ground-truth "
        "label fall back to 1D tail_pred classification to "
        "greatly expand head_pred sample coverage. Safe because "
        "1D is 100%% separating on fast_mode for shuffle vs "
        "fold, and head_pred itself is what then distinguishes "
        "fold_success from intervention_recovery within the "
        "tail-overlapping region.",
    )
    args = parser.parse_args()

    with open(args.episode_details) as f:
        episodes = json.load(f)
    with open(args.ground_truth_labels) as f:
        labels = json.load(f)

    result = compute_head_pred_ranges(
        episodes,
        labels.get("episode_labels", {}),
        labels.get("repo_labels", {}),
        min_samples=args.min_samples,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
        bootstrap_from_1d=args.bootstrap_from_1d,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(
        f"[fill] Head-pred ranges by category "
        f"(min_samples={args.min_samples}, "
        f"lower={args.lower_percentile}, upper={args.upper_percentile}):"
    )
    for cat in ALL_CATEGORIES:
        r = result[cat]
        if r is None:
            print(f" {cat:22s} None (insufficient samples)")
        else:
            print(
                f" {cat:22s} n={r['n_samples']:5d} "
                f"head_min={r['head_min']:+.3f} head_max={r['head_max']:+.3f} "
                f"(p50={r['p50']:+.3f})"
            )
    print(f"[fill] Written to {args.output}")


if __name__ == "__main__":
    main()
