"""NumPy-based advantage computation utilities.

This module provides shared NumPy implementations for advantage computation,
used by offline scripts (compute_values.py, compute_advantages.py).

For JAX-based training computation, see advantage.py.

Key functions:
- compute_percentile_threshold: Percentile-based threshold
- compute_bin_edges: Value-based binning edges
- compute_indicators: Binary indicators (supports binning)
- clip_advantages: Variance reduction via clipping
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_percentile_threshold(
    advantages: np.ndarray,
    percentile: float = 30.0,
    clip_percentile: float | None = None,
) -> float:
    """Compute advantage threshold from percentile.

    Args:
        advantages: Advantage values, shape [num_samples]
        percentile: Percentile for threshold (default 30% as per paper)
        clip_percentile: Optional clipping to reduce variance

    Returns:
        Threshold value such that approximately percentile% of samples
        have advantage above threshold.

    Raises:
        ValueError: If advantages is empty or contains NaN/Inf values
    """
    if len(advantages) == 0:
        raise ValueError("Cannot compute threshold from empty advantages array")

    if not np.isfinite(advantages).all():
        raise ValueError("Advantages array contains NaN or Inf values")

    if len(advantages) < 100:
        logger.warning(f"Small dataset size ({len(advantages)}) may produce unreliable percentiles")

    work_advantages = advantages
    if clip_percentile is not None:
        lower = np.percentile(work_advantages, clip_percentile)
        upper = np.percentile(work_advantages, 100.0 - clip_percentile)
        work_advantages = np.clip(work_advantages, lower, upper)

    threshold = np.percentile(work_advantages, 100 - percentile)

    return float(threshold)


def compute_bin_edges(
    values: np.ndarray,
    num_bins: int = 5,
) -> np.ndarray:
    """Compute quantile-based bin edges from values.

    Args:
        values: Value estimates, shape [num_samples]
        num_bins: Number of bins (1 returns [min, max])

    Returns:
        Bin edges array of shape [num_bins + 1]

    Raises:
        ValueError: If values is empty, contains NaN/Inf, or num_bins < 1
    """
    if len(values) == 0:
        raise ValueError("Cannot compute bin edges from empty array")

    if not np.isfinite(values).all():
        raise ValueError("Values array contains NaN or Inf values")

    if num_bins < 1:
        raise ValueError(f"num_bins must be >= 1, got {num_bins}")

    if num_bins == 1:
        return np.array([np.min(values), np.max(values)])

    return np.quantile(values, np.linspace(0, 1, num_bins + 1))


def compute_indicators(
    advantages: np.ndarray,
    percentile: float = 30.0,
    clip_percentile: float | None = None,
    values: np.ndarray | None = None,
    num_bins: int = 1,
    min_samples_per_bin: int = 100,
    bin_edges: np.ndarray | None = None,
) -> np.ndarray:
    """Compute binary advantage indicators with optional value-based binning.

    Args:
        advantages: Advantage estimates, shape [num_samples]
        percentile: Percentile for threshold (default 30%)
        clip_percentile: Optional clipping to reduce variance
        values: Value estimates for binning, shape [num_samples].
            Required if num_bins > 1.
        num_bins: Number of value bins (1 = global mode, default)
        min_samples_per_bin: Minimum samples per bin for bin-specific threshold
        bin_edges: Pre-computed bin edges for multi-GPU consistency

    Returns:
        Binary indicators, shape [num_samples]. True indicates positive advantage.

    Raises:
        ValueError: If array shapes don't match, values not provided for binning,
            or arrays contain NaN/Inf values
    """
    if len(advantages) == 0:
        raise ValueError("Cannot compute indicators from empty advantages array")

    if not np.isfinite(advantages).all():
        raise ValueError("Advantages array contains NaN or Inf values")

    if values is not None and not np.isfinite(values).all():
        raise ValueError("Values array contains NaN or Inf values")

    # Global mode: no binning
    if num_bins == 1:
        threshold = compute_percentile_threshold(advantages, percentile, clip_percentile)
        return advantages > threshold

    # Binning mode: validate inputs
    if values is None:
        raise ValueError("values must be provided when num_bins > 1")

    if len(advantages) != len(values):
        raise ValueError(f"advantages length {len(advantages)} != values length {len(values)}") from None

    # Compute or use provided bin edges
    if bin_edges is None:
        bin_edges = compute_bin_edges(values, num_bins)

    # Assign samples to bins
    bin_indices = np.digitize(values, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    # Apply clipping to advantages if specified
    if clip_percentile is not None:
        advantages = clip_advantages(advantages, clip_percentile).astype(advantages.dtype)

    # Compute indicators for each bin
    indicators = np.zeros_like(advantages, dtype=bool)

    for bin_idx in range(num_bins):
        mask = bin_indices == bin_idx
        n_samples = np.sum(mask)

        if n_samples == 0:
            continue

        if n_samples < min_samples_per_bin:
            threshold = compute_percentile_threshold(advantages, percentile, None)
        else:
            bin_advantages = advantages[mask]
            threshold = np.percentile(bin_advantages, 100 - percentile)

        indicators[mask] = advantages[mask] > threshold

    return indicators


def clip_advantages(
    advantages: np.ndarray,
    clip_percentile: float = 1.0,
) -> np.ndarray:
    """Clip extreme advantages to reduce variance.

    Args:
        advantages: Advantage values
        clip_percentile: Percentile for clipping (e.g., 1.0 clips bottom/top 1%)

    Returns:
        Clipped advantages

    Raises:
        ValueError: If array is empty, contains NaN/Inf, or clip_percentile invalid
    """
    if len(advantages) == 0:
        raise ValueError("Cannot clip empty advantages array")

    if not np.isfinite(advantages).all():
        raise ValueError("Advantages array contains NaN or Inf values")

    if len(advantages) == 1:
        return advantages

    if clip_percentile != 0.0 and not (0.0 < clip_percentile < 50.0):
        raise ValueError(f"clip_percentile must be 0.0 or in (0, 50), got {clip_percentile}")

    if clip_percentile == 0.0:
        return advantages

    lower = np.percentile(advantages, clip_percentile)
    upper = np.percentile(advantages, 100.0 - clip_percentile)

    return np.clip(advantages, lower, upper)
