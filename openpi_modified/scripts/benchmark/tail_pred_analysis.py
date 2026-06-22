#!/usr/bin/env python3
"""1D tail_pred distribution analysis for clothes-folding value model benchmark.

Loads pre-computed benchmark results, maps episodes to semantic categories,
computes per-category statistics, pairwise separation metrics, and generates
distribution visualizations.

Usage:
    PYTHONPATH=src:. python scripts/benchmark/tail_pred_analysis.py \
        --expanded_dir /path/to/clothes_v0401_expanded \
        --error_dir /path/to/clothes_v0401_error \
        --qc_path /path/to/self_play_label_qc.json \
        --output_dir /path/to/output \
        --primary_model fast_mode_max3600
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.ndimage import gaussian_filter1d

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

plt.switch_backend("Agg")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODELS = [
    "fast_mode_max3600",
    "1215_0227_max3600",
    "per_task_p90",
    "stage2_all_0322",
]

CATEGORY_COLORS: dict[str, str] = {
    "fold_success": "#2ecc71",
    "shuffle_success": "#3498db",
    "fold_failure": "#e74c3c",
    "intervention_recovery": "#f39c12",
    "other": "#9b59b6",
}

CATEGORY_ORDER = ["fold_success", "shuffle_success", "fold_failure", "intervention_recovery", "other"]

SMALL_SAMPLE_THRESHOLD = 10


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CategoryStats:
    n: int = 0
    mean: float = float("nan")
    std: float = float("nan")
    min: float = float("nan")
    max: float = float("nan")
    p5: float = float("nan")
    p25: float = float("nan")
    p50: float = float("nan")
    p75: float = float("nan")
    p95: float = float("nan")
    warning: str | None = None


@dataclass
class PairSeparation:
    mean_distance: float = 0.0
    overlap_pct: float = 0.0
    midpoint_threshold: float = 0.0


# ---------------------------------------------------------------------------
# Semantic label assignment
# ---------------------------------------------------------------------------


def assign_semantic_label(quadrant: str, intervention_count: int) -> str:
    """Map an episode to its semantic category.

    Priority:
      1. intervention_count > 0 -> "intervention_recovery"
      2. true_positive -> "fold_success"
      3. true_negative -> "shuffle_success"
      4. false_positive -> "fold_failure"
      5. anything else -> "other"
    """
    if intervention_count > 0:
        return "intervention_recovery"
    mapping = {
        "true_positive": "fold_success",
        "true_negative": "shuffle_success",
        "false_positive": "fold_failure",
        "false_negative": "other",
    }
    return mapping.get(quadrant, "other")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_benchmark_episodes(
    base_dir: Path,
    models: list[str],
) -> dict[str, list[dict]]:
    """Load episode_details.json for each model under *base_dir*.

    Returns:
        {model_name: [episode_dict, ...]}
    """
    result: dict[str, list[dict]] = {}
    for model in models:
        path = base_dir / model / "metrics" / "episode_details.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing episode details: {path}")
        with open(path) as f:
            result[model] = json.load(f)
    return result


def load_qc_data(qc_path: Path) -> dict[str, dict]:
    """Load self-play QC JSON, returning only entries with a non-null tail_pred.

    Returns:
        {episode_key: qc_entry}
    """
    with open(qc_path) as f:
        raw = json.load(f)
    return {entry["episode_key"]: entry for entry in raw if entry.get("tail_pred") is not None}


# ---------------------------------------------------------------------------
# Merge and label
# ---------------------------------------------------------------------------


def merge_and_label(
    expanded_episodes: list[dict],
    error_episodes: list[dict],
    qc_lookup: dict[str, dict],
) -> list[dict]:
    """Merge expanded + error episodes, deduplicate, assign semantic labels.

    QC data is used *only* to detect intervention (intervention_count > 0).
    """
    seen: dict[str, dict] = {}
    for ep in expanded_episodes:
        seen[ep["episode_key"]] = ep
    for ep in error_episodes:
        if ep["episode_key"] not in seen:
            seen[ep["episode_key"]] = ep

    labeled: list[dict] = []
    for key, ep in seen.items():
        intervention_count = 0
        qc_entry = qc_lookup.get(key)
        if qc_entry is not None:
            intervention_count = qc_entry.get("intervention_count", 0)

        ep_copy = {**ep}
        ep_copy["semantic_label"] = assign_semantic_label(
            quadrant=ep["quadrant"],
            intervention_count=intervention_count,
        )
        labeled.append(ep_copy)
    return labeled


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def compute_category_stats(values: list[float] | np.ndarray) -> CategoryStats:
    """Compute descriptive statistics for a set of tail_pred values."""
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return CategoryStats()

    warning = "small_sample" if n < SMALL_SAMPLE_THRESHOLD else None

    return CategoryStats(
        n=n,
        mean=float(np.nanmean(arr)),
        std=float(np.nanstd(arr, ddof=1)) if n > 1 else 0.0,
        min=float(np.nanmin(arr)),
        max=float(np.nanmax(arr)),
        p5=float(np.nanpercentile(arr, 5)),
        p25=float(np.nanpercentile(arr, 25)),
        p50=float(np.nanpercentile(arr, 50)),
        p75=float(np.nanpercentile(arr, 75)),
        p95=float(np.nanpercentile(arr, 95)),
        warning=warning,
    )


def compute_overlap_pct(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    """Fraction of combined samples in the overlap region [max(min_A, min_B), min(max_A, max_B)]."""
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    if len(arr_a) == 0 or len(arr_b) == 0:
        return 0.0

    lo = max(float(np.nanmin(arr_a)), float(np.nanmin(arr_b)))
    hi = min(float(np.nanmax(arr_a)), float(np.nanmax(arr_b)))
    if lo >= hi:
        return 0.0

    combined = np.concatenate([arr_a, arr_b])
    n_in_overlap = int(np.sum((combined >= lo) & (combined <= hi)))
    return n_in_overlap / len(combined)


def compute_separation(
    a: list[float] | np.ndarray,
    b: list[float] | np.ndarray,
) -> PairSeparation:
    """Compute separation metrics between two value distributions."""
    mean_a = float(np.nanmean(a))
    mean_b = float(np.nanmean(b))
    return PairSeparation(
        mean_distance=abs(mean_a - mean_b),
        overlap_pct=compute_overlap_pct(a, b),
        midpoint_threshold=(mean_a + mean_b) / 2,
    )


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def _smooth_histogram(
    values: np.ndarray, bins: int = 200, x_range: tuple[float, float] = (-1.1, 0.1)
) -> tuple[np.ndarray, np.ndarray]:
    """Build a smoothed density curve from histogram counts."""
    counts, edges = np.histogram(values, bins=bins, range=x_range, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    if _HAS_SCIPY:
        counts = gaussian_filter1d(counts, sigma=3.0)
    return centers, counts


def plot_main_distribution(
    category_values: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Create 3-panel distribution plot: histograms, box plots, KDE."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), dpi=150)

    # --- Panel 1: Overlapping histograms ---
    ax = axes[0]
    for cat in CATEGORY_ORDER:
        vals = category_values.get(cat)
        if vals is None or len(vals) == 0:
            continue
        ax.hist(
            vals,
            bins=30,
            alpha=0.5,
            range=(-1.1, 0.1),
            label=f"{cat} (n={len(vals)})",
            color=CATEGORY_COLORS[cat],
        )
        ax.axvline(
            np.nanmean(vals),
            color=CATEGORY_COLORS[cat],
            linestyle="--",
            linewidth=1.5,
        )
    ax.set_xlabel("tail_pred")
    ax.set_ylabel("Count")
    ax.set_title("Tail Prediction Distribution by Category")
    ax.legend(fontsize=8)

    # --- Panel 2: Horizontal box plots ---
    ax = axes[1]
    box_data = []
    box_labels = []
    box_colors = []
    for cat in CATEGORY_ORDER:
        vals = category_values.get(cat)
        if vals is None or len(vals) == 0:
            continue
        box_data.append(vals)
        box_labels.append(cat)
        box_colors.append(CATEGORY_COLORS[cat])
    if box_data:
        bp = ax.boxplot(box_data, orientation="horizontal", patch_artist=True, tick_labels=box_labels)
        for patch, color in zip(bp["boxes"], box_colors, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    ax.set_xlabel("tail_pred")
    ax.set_title("Box Plot Comparison")

    # --- Panel 3: KDE density curves ---
    ax = axes[2]
    for cat in CATEGORY_ORDER:
        vals = category_values.get(cat)
        if vals is None or len(vals) == 0:
            continue
        centers, density = _smooth_histogram(vals)
        ax.plot(centers, density, label=cat, color=CATEGORY_COLORS[cat], linewidth=1.5)
        ax.fill_between(centers, density, alpha=0.15, color=CATEGORY_COLORS[cat])
    ax.set_xlabel("tail_pred")
    ax.set_ylabel("Density")
    ax.set_title("Smoothed Density Estimate")
    ax.legend(fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_model_comparison(
    model_category_values: dict[str, dict[str, np.ndarray]],
    models: list[str],
    output_path: Path,
) -> None:
    """Create 2x2 model comparison plot."""
    n_models = min(len(models), 4)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=150)
    axes_flat = axes.flatten()

    for idx in range(4):
        ax = axes_flat[idx]
        if idx >= n_models:
            ax.set_visible(False)
            continue
        model = models[idx]
        cat_vals = model_category_values.get(model, {})

        for cat in CATEGORY_ORDER:
            vals = cat_vals.get(cat)
            if vals is None or len(vals) == 0:
                continue
            ax.hist(
                vals,
                bins=30,
                alpha=0.5,
                range=(-1.1, 0.1),
                label=cat,
                color=CATEGORY_COLORS[cat],
            )

        # Compute separation for subtitle
        fs = cat_vals.get("fold_success", np.array([]))
        ff = cat_vals.get("fold_failure", np.array([]))
        sep = abs(float(np.nanmean(fs)) - float(np.nanmean(ff))) if len(fs) > 0 and len(ff) > 0 else 0.0
        ax.set_title(f"{model}\nseparation(success-failure)={sep:.3f}", fontsize=10)
        ax.set_xlabel("tail_pred")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------


def run_analysis(
    expanded_dir: Path,
    error_dir: Path,
    qc_path: Path,
    output_dir: Path,
    primary_model: str,
    models: list[str] | None = None,
) -> dict:
    """Execute the full tail_pred analysis pipeline.

    Returns the analysis dict (also saved to JSON).
    """
    if models is None:
        models = list(DEFAULT_MODELS)

    # --- Load data ---
    expanded_data = load_benchmark_episodes(expanded_dir, models)
    error_data = load_benchmark_episodes(error_dir, models)
    qc_lookup = load_qc_data(qc_path)

    # --- Merge and label per model ---
    model_labeled: dict[str, list[dict]] = {}
    for model in models:
        model_labeled[model] = merge_and_label(
            expanded_data[model],
            error_data[model],
            qc_lookup,
        )

    # --- Primary model analysis ---
    primary_episodes = model_labeled[primary_model]

    # Group by category
    category_values: dict[str, np.ndarray] = {}
    category_lists: dict[str, list[float]] = defaultdict(list)
    for ep in primary_episodes:
        tp = ep.get("tail_pred")
        if tp is not None:
            category_lists[ep["semantic_label"]].append(tp)
    for cat, vals in category_lists.items():
        category_values[cat] = np.array(vals, dtype=np.float64)

    # Per-category stats
    per_category: dict[str, dict] = {}
    for cat in CATEGORY_ORDER:
        vals = category_values.get(cat, np.array([]))
        if len(vals) > 0:
            stats = compute_category_stats(vals)
            stats_dict = asdict(stats)
            if stats_dict["warning"] is None:
                del stats_dict["warning"]
            per_category[cat] = stats_dict

    # Pairwise separation
    pairwise: dict[str, dict] = {}
    cats_present = [c for c in CATEGORY_ORDER if c in category_values and len(category_values[c]) > 0]
    for i, cat_a in enumerate(cats_present):
        for cat_b in cats_present[i + 1 :]:
            pair_key = f"{cat_a}_vs_{cat_b}"
            sep = compute_separation(category_values[cat_a], category_values[cat_b])
            pairwise[pair_key] = asdict(sep)

    # Key thresholds
    key_thresholds: dict[str, dict] = {}
    for pair_name, cat_a, cat_b in [
        ("fold_success_vs_fold_failure", "fold_success", "fold_failure"),
        ("fold_success_vs_shuffle_success", "fold_success", "shuffle_success"),
    ]:
        if cat_a in category_values and cat_b in category_values:
            sep = compute_separation(category_values[cat_a], category_values[cat_b])
            key_thresholds[pair_name] = asdict(sep)

    # Model comparison
    model_comparison: dict[str, dict] = {}
    model_category_values: dict[str, dict[str, np.ndarray]] = {}
    for model in models:
        mc_lists: dict[str, list[float]] = defaultdict(list)
        for ep in model_labeled[model]:
            tp = ep.get("tail_pred")
            if tp is not None:
                mc_lists[ep["semantic_label"]].append(tp)
        mc_arrays = {cat: np.array(vals, dtype=np.float64) for cat, vals in mc_lists.items()}
        model_category_values[model] = mc_arrays

        fs_mean = float(np.nanmean(mc_arrays["fold_success"])) if "fold_success" in mc_arrays else float("nan")
        ff_mean = float(np.nanmean(mc_arrays["fold_failure"])) if "fold_failure" in mc_arrays else float("nan")
        separation = abs(fs_mean - ff_mean) if not (np.isnan(fs_mean) or np.isnan(ff_mean)) else float("nan")
        model_comparison[model] = {
            "fold_success_mean": fs_mean,
            "fold_failure_mean": ff_mean,
            "separation": separation,
        }

    # --- Assemble result ---
    analysis = {
        "metadata": {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "n_episodes_total": len(primary_episodes),
            "primary_model": primary_model,
        },
        "per_category": per_category,
        "separation": {
            "pairwise": pairwise,
            "key_thresholds": key_thresholds,
        },
        "model_comparison": model_comparison,
    }

    # --- Write JSON ---
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "tail_pred_1d_analysis.json"
    with open(json_path, "w") as f:
        json.dump(analysis, f, indent=2, default=_json_default)

    # --- Generate plots ---
    plots_dir = output_dir / "plots"
    plot_main_distribution(category_values, plots_dir / "tail_pred_distributions.png")
    plot_model_comparison(model_category_values, models, plots_dir / "tail_pred_by_model.png")

    # --- Print summary ---
    print(f"Analysis written to {json_path}")
    print(f"Primary model: {primary_model}, {len(primary_episodes)} episodes")
    for cat in CATEGORY_ORDER:
        if cat in per_category:
            s = per_category[cat]
            print(f"  {cat}: n={s['n']}, mean={s['mean']:.4f}, std={s['std']:.4f}")
    if "fold_success_vs_fold_failure" in key_thresholds:
        t = key_thresholds["fold_success_vs_fold_failure"]
        print(
            f"Key threshold (success vs failure): midpoint={t['midpoint_threshold']:.4f}, "
            f"separation={t['mean_distance']:.4f}, overlap={t['overlap_pct']:.2%}"
        )

    return analysis


def _json_default(obj: object) -> object:
    """JSON serializer fallback for numpy types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="1D tail_pred distribution analysis for clothes-folding benchmark",
    )
    parser.add_argument(
        "--expanded_dir",
        type=Path,
        required=True,
        help="Root directory of expanded benchmark results",
    )
    parser.add_argument(
        "--error_dir",
        type=Path,
        required=True,
        help="Root directory of error benchmark results",
    )
    parser.add_argument(
        "--qc_path",
        type=Path,
        default=Path("test_results/data_audit/self_play_label_qc.json"),
        help="Path to self-play QC JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("test_results/data_audit"),
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--primary_model",
        type=str,
        default="fast_mode_max3600",
        help="Model to use for primary distribution analysis",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_analysis(
        expanded_dir=args.expanded_dir,
        error_dir=args.error_dir,
        qc_path=args.qc_path,
        output_dir=args.output_dir,
        primary_model=args.primary_model,
    )


if __name__ == "__main__":
    main()
