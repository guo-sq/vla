"""advantage_utils.py unit tests - percentile threshold, indicators, binning."""

import numpy as np
import pytest

from openpi.training.advantage_utils import clip_advantages
from openpi.training.advantage_utils import compute_bin_edges
from openpi.training.advantage_utils import compute_indicators
from openpi.training.advantage_utils import compute_percentile_threshold


class TestComputePercentileThreshold:
    def test_basic_percentile(self):
        advantages = np.arange(100, dtype=np.float32)
        threshold = compute_percentile_threshold(advantages, percentile=30.0)
        # Top 30% means threshold at 70th percentile = 70.0
        assert threshold == pytest.approx(70.0, abs=1.0)

    def test_with_clipping(self):
        advantages = np.concatenate(
            [
                np.array([-100.0]),
                np.zeros(98),
                np.array([100.0]),
            ]
        )
        threshold_no_clip = compute_percentile_threshold(advantages, percentile=30.0)
        threshold_clipped = compute_percentile_threshold(advantages, percentile=30.0, clip_percentile=5.0)
        # Clipping should reduce extreme outlier influence
        assert abs(threshold_clipped) < abs(threshold_no_clip) or threshold_clipped == pytest.approx(threshold_no_clip)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_percentile_threshold(np.array([]))

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="NaN or Inf"):
            compute_percentile_threshold(np.array([1.0, float("nan"), 3.0]))

    def test_inf_raises(self):
        with pytest.raises(ValueError, match="NaN or Inf"):
            compute_percentile_threshold(np.array([1.0, float("inf"), 3.0]))


class TestComputeBinEdges:
    def test_single_bin(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        edges = compute_bin_edges(values, num_bins=1)
        np.testing.assert_array_equal(edges, [1.0, 5.0])

    def test_multiple_bins(self):
        values = np.arange(100, dtype=np.float32)
        edges = compute_bin_edges(values, num_bins=4)
        assert len(edges) == 5  # 4 bins = 5 edges
        assert edges[0] == pytest.approx(0.0)
        assert edges[-1] == pytest.approx(99.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_bin_edges(np.array([]))

    def test_invalid_num_bins_raises(self):
        with pytest.raises(ValueError, match="num_bins"):
            compute_bin_edges(np.array([1.0]), num_bins=0)


class TestComputeIndicators:
    def test_global_mode(self):
        """num_bins=1: global percentile threshold."""
        advantages = np.arange(100, dtype=np.float32)
        indicators = compute_indicators(advantages, percentile=30.0)
        # ~30% should be True
        positive_ratio = indicators.sum() / len(indicators)
        assert positive_ratio == pytest.approx(0.30, abs=0.05)

    def test_binned_mode(self):
        """num_bins>1: per-bin thresholds."""
        np.random.seed(42)
        advantages = np.random.randn(1000).astype(np.float32)
        values = np.random.randn(1000).astype(np.float32)
        indicators = compute_indicators(advantages, percentile=30.0, values=values, num_bins=5)
        # Should still have roughly 30% positive overall
        positive_ratio = indicators.sum() / len(indicators)
        assert positive_ratio == pytest.approx(0.30, abs=0.10)

    def test_binned_requires_values(self):
        with pytest.raises(ValueError, match="values must be provided"):
            compute_indicators(np.arange(100, dtype=np.float32), num_bins=5)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length"):
            compute_indicators(
                np.arange(100, dtype=np.float32),
                values=np.arange(50, dtype=np.float32),
                num_bins=5,
            )

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_indicators(np.array([]))

    def test_with_precomputed_bin_edges(self):
        advantages = np.arange(100, dtype=np.float32)
        values = np.arange(100, dtype=np.float32)
        bin_edges = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
        indicators = compute_indicators(advantages, percentile=30.0, values=values, num_bins=4, bin_edges=bin_edges)
        assert indicators.dtype == bool
        assert len(indicators) == 100


class TestClipAdvantages:
    def test_basic(self):
        advantages = np.arange(100, dtype=np.float32)
        result = clip_advantages(advantages, clip_percentile=5.0)
        assert result.min() >= np.percentile(advantages, 5.0) - 1e-5
        assert result.max() <= np.percentile(advantages, 95.0) + 1e-5

    def test_zero_percentile(self):
        advantages = np.array([1.0, 2.0, 3.0])
        result = clip_advantages(advantages, clip_percentile=0.0)
        np.testing.assert_array_equal(result, advantages)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            clip_advantages(np.array([]))

    def test_invalid_percentile_raises(self):
        with pytest.raises(ValueError, match="clip_percentile"):
            clip_advantages(np.array([1.0, 2.0]), clip_percentile=55.0)

    def test_single_element(self):
        result = clip_advantages(np.array([5.0]))
        np.testing.assert_array_equal(result, [5.0])
