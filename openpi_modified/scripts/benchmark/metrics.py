"""Metrics for value model benchmark — per-episode and per-quadrant computation.

Computes head/tail MSE, MAE, RMSE, Pearson, R² with priority levels:
- High (tail MSE for TP/TN): GT endpoint is known with certainty
- Medium (MAE, RMSE, head MSE): GT trend is approximately known
- Low (Pearson, R²): trend correlation, reference value
"""

from __future__ import annotations

import numpy as np


def compute_episode_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Compute all metrics for a single episode.

    Supports dense (len >= 3) and sparse (len 0/1/2) episodes. Sparse mode is
    used by ``run_inference_on_repo(sparse_mode=True)`` in MR !9's sparse
    inference pipeline, which retains only head + tail frames. For len < 3 the
    dense metrics (``mae``, ``rmse``, ``pearson``, ``r_squared``) are returned
    as ``NaN`` so ``compute_quadrant_summary``'s ``nanmean/nanstd`` aggregation
    can mix sparse and dense episodes transparently.

    Args:
        pred: predicted values, shape (T,)
        gt: ground truth values, shape (T,)

    Returns:
        Dict with keys:
            - head_mse, tail_mse, mae, rmse, pearson, r_squared (all always present)
            - head_pred, tail_pred, head_gt, tail_gt (always present; NaN when the
              corresponding frame does not exist)

        ``NaN`` semantics:
            - len == 0 -> every metric and head/tail pred/gt are NaN
            - len == 1 -> head_mse / head_pred / head_gt computed; tail_* NaN;
              dense metrics NaN
            - len == 2 -> head_mse / tail_mse / head_pred / tail_pred / head_gt /
              tail_gt all computed; dense metrics NaN
            - len >= 3 -> everything computed (unchanged behavior)
    """
    pred = np.asarray(pred).ravel()
    gt = np.asarray(gt).ravel()

    if len(pred) != len(gt):
        raise ValueError(f"pred ({len(pred)}) and gt ({len(gt)}) must have same length")

    n = len(pred)
    nan = float("nan")

    # Empty input: defensive guard. Every metric NaN; no IndexError.
    if n == 0:
        return {
            "head_mse": nan,
            "tail_mse": nan,
            "mae": nan,
            "rmse": nan,
            "pearson": nan,
            "r_squared": nan,
            "head_pred": nan,
            "tail_pred": nan,
            "head_gt": nan,
            "tail_gt": nan,
        }

    errors = pred - gt
    sq_errors = errors**2

    head_pred = float(pred[0])
    head_gt = float(gt[0])
    head_mse = float(sq_errors[0])
    tail_pred = float(pred[-1]) if n >= 2 else nan
    tail_gt = float(gt[-1]) if n >= 2 else nan
    tail_mse = float(sq_errors[-1]) if n >= 2 else nan

    # Sparse mode (len < 3): dense metrics are not meaningful.
    if n < 3:
        return {
            "head_mse": head_mse,
            "tail_mse": tail_mse,
            "mae": nan,
            "rmse": nan,
            "pearson": nan,
            "r_squared": nan,
            "head_pred": head_pred,
            "tail_pred": tail_pred,
            "head_gt": head_gt,
            "tail_gt": tail_gt,
        }

    # Dense mode (len >= 3): original behavior.
    abs_errors = np.abs(errors)
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(sq_errors)))

    # Pearson correlation
    std_pred = np.std(pred)
    std_gt = np.std(gt)
    pearson = nan if std_pred == 0 or std_gt == 0 else float(np.corrcoef(pred, gt)[0, 1])

    # R²
    ss_res = np.sum(sq_errors)
    ss_tot = np.sum((gt - np.mean(gt)) ** 2)
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else nan

    return {
        "head_mse": head_mse,
        "tail_mse": tail_mse,
        "mae": mae,
        "rmse": rmse,
        "pearson": pearson,
        "r_squared": r_squared,
        "head_pred": head_pred,
        "tail_pred": tail_pred,
        "head_gt": head_gt,
        "tail_gt": tail_gt,
    }


def compute_quadrant_summary(episode_metrics: list[dict]) -> dict:
    """Aggregate episode metrics into a quadrant summary.

    Args:
        episode_metrics: list of dicts from compute_episode_metrics

    Returns:
        Dict with mean/std for each metric, plus n_episodes.
    """
    n = len(episode_metrics)
    if n == 0:
        return {
            "n_episodes": 0,
            "mean_head_mse": float("nan"),
            "mean_tail_mse": float("nan"),
            "mean_mae": float("nan"),
            "mean_rmse": float("nan"),
            "mean_pearson": float("nan"),
            "mean_r_squared": float("nan"),
            "std_head_mse": float("nan"),
            "std_tail_mse": float("nan"),
            "std_mae": float("nan"),
            "std_rmse": float("nan"),
            "std_pearson": float("nan"),
            "std_r_squared": float("nan"),
        }

    keys = ["head_mse", "tail_mse", "mae", "rmse", "pearson", "r_squared"]
    result: dict = {"n_episodes": n}
    for key in keys:
        values = [m[key] for m in episode_metrics]
        values_arr = np.array(values)
        result[f"mean_{key}"] = float(np.nanmean(values_arr))
        result[f"std_{key}"] = float(np.nanstd(values_arr))
        result[f"median_{key}"] = float(np.nanmedian(values_arr))

    # Length-bucketed Pearson: short(<500), medium(500-1000), long(>1000)
    buckets = {"short": [], "medium": [], "long": []}
    for m in episode_metrics:
        nf = m.get("n_frames", 0)
        p = m.get("pearson", float("nan"))
        if nf < 500:
            buckets["short"].append(p)
        elif nf <= 1000:
            buckets["medium"].append(p)
        else:
            buckets["long"].append(p)
    for bname, bvals in buckets.items():
        arr = np.array(bvals) if bvals else np.array([])
        result[f"pearson_{bname}_n"] = len(bvals)
        result[f"pearson_{bname}_mean"] = float(np.nanmean(arr)) if bvals else float("nan")
        result[f"pearson_{bname}_median"] = float(np.nanmedian(arr)) if bvals else float("nan")

    return result


class MetricPriority:
    """Metric priority levels per quadrant.

    Priority reflects GT certainty:
    - high: GT endpoint known (tail MSE for TP/TN)
    - medium: GT trend approximately known (MAE, RMSE, head MSE)
    - low: correlation reference (Pearson, R²)
    """

    _PRIORITIES = {
        "true_positive": {
            "tail_mse": "high",
            "head_mse": "medium",
            "mae": "medium",
            "rmse": "medium",
            "pearson": "low",
            "r_squared": "low",
        },
        "true_negative": {
            "tail_mse": "high",
            "head_mse": "medium",
            "mae": "medium",
            "rmse": "medium",
            "pearson": "low",
            "r_squared": "low",
        },
        "false_positive": {
            "tail_mse": "medium",
            "head_mse": "medium",
            "mae": "medium",
            "rmse": "medium",
            "pearson": "low",
            "r_squared": "low",
        },
        "false_negative": {
            "tail_mse": "medium",
            "head_mse": "medium",
            "mae": "medium",
            "rmse": "medium",
            "pearson": "low",
            "r_squared": "low",
        },
    }

    @classmethod
    def for_quadrant(cls, quadrant: str) -> dict[str, str]:
        """Return metric priority dict for a given quadrant name."""
        return dict(cls._PRIORITIES[quadrant])


# ---------------------------------------------------------------------------
# Extended metrics (adapted from openpi_reward branch)
# ---------------------------------------------------------------------------


def compute_binned_mse(
    pred: np.ndarray,
    gt: np.ndarray,
    value_range: tuple[float, float] = (-1.0, 0.0),
    num_bins: int = 10,
) -> dict:
    """Compute MSE statistics for different value bins.

    Divides the value range into bins based on ground truth values and computes
    per-bin MSE, count, and standard deviation.

    Args:
        pred: Predicted values.
        gt: Ground truth values.
        value_range: (min, max) range for binning.
        num_bins: Number of bins.

    Returns:
        Dict with bin_edges, bin_centers, mse_per_bin, count_per_bin,
        std_per_bin, bin_labels.
    """
    pred = np.asarray(pred).ravel()
    gt = np.asarray(gt).ravel()

    bin_edges = np.linspace(value_range[0], value_range[1], num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    mse_per_bin = np.full(num_bins, np.nan)
    count_per_bin = np.zeros(num_bins, dtype=int)
    std_per_bin = np.full(num_bins, np.nan)

    squared_errors = (pred - gt) ** 2
    gt_clipped = np.clip(gt, value_range[0], value_range[1])
    bin_indices = np.clip(np.digitize(gt_clipped, bin_edges) - 1, 0, num_bins - 1)

    for i in range(num_bins):
        mask = bin_indices == i
        count_per_bin[i] = int(np.sum(mask))
        if count_per_bin[i] > 0:
            mse_per_bin[i] = float(np.mean(squared_errors[mask]))
            std_per_bin[i] = float(np.std(squared_errors[mask]))

    bin_labels = [f"[{bin_edges[i]:.2f}, {bin_edges[i + 1]:.2f})" for i in range(num_bins)]

    return {
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "mse_per_bin": mse_per_bin,
        "count_per_bin": count_per_bin,
        "std_per_bin": std_per_bin,
        "bin_labels": bin_labels,
    }


def compute_comprehensive_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    value_range: tuple[float, float] = (-1.0, 0.0),
    num_bins: int = 10,
    tolerances: list[float] | tuple[float, ...] = (0.01, 0.05, 0.1),
) -> dict:
    """Compute comprehensive value prediction metrics.

    Includes overall error metrics, correlation metrics, tolerance percentages,
    and binned statistics.

    Args:
        pred: Predicted values.
        gt: Ground truth values.
        value_range: (min, max) for binning.
        num_bins: Number of bins for binned stats.
        tolerances: Tolerance values for within-tolerance percentages.

    Returns:
        Dict with overall_mse, overall_mae, overall_rmse, max_error,
        median_abs_error, bias, r_squared, pearson_corr, spearman_corr,
        mape, within_tolerance, binned_stats, n_samples.

    Raises:
        ValueError: If pred and gt have different lengths.
    """
    pred = np.asarray(pred).ravel()
    gt = np.asarray(gt).ravel()

    if len(pred) != len(gt):
        raise ValueError(f"pred ({len(pred)}) and gt ({len(gt)}) must have same length")

    n_samples = len(pred)
    errors = pred - gt
    abs_errors = np.abs(errors)
    squared_errors = errors**2

    overall_mse = float(np.mean(squared_errors))
    overall_mae = float(np.mean(abs_errors))
    overall_rmse = float(np.sqrt(overall_mse))
    max_error = float(np.max(abs_errors)) if n_samples > 0 else float("nan")
    median_abs_error = float(np.median(abs_errors)) if n_samples > 0 else float("nan")
    bias = float(np.mean(errors))

    # R-squared
    ss_res = np.sum(squared_errors)
    ss_tot = np.sum((gt - np.mean(gt)) ** 2)
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    # Pearson
    pearson_corr = float(np.corrcoef(pred, gt)[0, 1]) if n_samples > 1 else float("nan")

    # Spearman
    try:
        from scipy.stats import spearmanr

        spearman_corr = float(spearmanr(pred, gt).correlation)
    except ImportError:
        # Fallback: rank correlation via numpy
        def _rankdata(x):
            order = np.argsort(x)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(1, len(x) + 1, dtype=float)
            return ranks

        spearman_corr = float(np.corrcoef(_rankdata(pred), _rankdata(gt))[0, 1]) if n_samples > 1 else float("nan")

    # MAPE
    mape_mask = np.abs(gt) > 1e-6
    mape = (
        float(np.mean(np.abs((gt[mape_mask] - pred[mape_mask]) / gt[mape_mask])) * 100)
        if np.sum(mape_mask) > 0
        else float("nan")
    )

    # Within tolerance
    within_tolerance = {}
    for tol in tolerances:
        count = int(np.sum(abs_errors <= tol))
        within_tolerance[f"within_{tol}"] = {
            "tolerance": tol,
            "count": count,
            "percentage": float(count / n_samples * 100) if n_samples > 0 else 0.0,
        }

    # Binned stats
    binned_stats = compute_binned_mse(pred, gt, value_range=value_range, num_bins=num_bins)

    return {
        "overall_mse": overall_mse,
        "overall_mae": overall_mae,
        "overall_rmse": overall_rmse,
        "max_error": max_error,
        "median_abs_error": median_abs_error,
        "bias": bias,
        "r_squared": r_squared,
        "pearson_corr": pearson_corr,
        "spearman_corr": spearman_corr,
        "mape": mape,
        "within_tolerance": within_tolerance,
        "binned_stats": binned_stats,
        "n_samples": n_samples,
    }
