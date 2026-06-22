"""Tests for extended benchmark metrics — binned MSE and comprehensive metrics."""

import numpy as np
import pytest

pytestmark = [pytest.mark.unit]

from scripts.benchmark.metrics import compute_binned_mse
from scripts.benchmark.metrics import compute_comprehensive_metrics


@pytest.mark.unit
class TestComputeBinnedMse:
    """compute_binned_mse divides GT range into bins and computes per-bin MSE."""

    def test_basic_binning(self):
        pred = np.array([-0.9, -0.5, -0.1])
        gt = np.array([-1.0, -0.5, 0.0])
        result = compute_binned_mse(pred, gt, value_range=(-1.0, 0.0), num_bins=5)
        assert len(result["bin_centers"]) == 5
        assert len(result["bin_edges"]) == 6
        assert result["count_per_bin"].sum() == 3

    def test_perfect_prediction_zero_mse(self):
        pred = gt = np.linspace(-1, 0, 100)
        result = compute_binned_mse(pred, gt, num_bins=5)
        for i in range(5):
            if result["count_per_bin"][i] > 0:
                assert result["mse_per_bin"][i] == pytest.approx(0.0, abs=1e-10)

    def test_empty_bins_are_nan(self):
        pred = gt = np.array([-0.5])
        result = compute_binned_mse(pred, gt, num_bins=10)
        nan_count = np.isnan(result["mse_per_bin"]).sum()
        assert nan_count >= 8

    def test_bin_labels_format(self):
        result = compute_binned_mse(np.zeros(10), np.linspace(-1, 0, 10), num_bins=5)
        assert len(result["bin_labels"]) == 5
        assert "[" in result["bin_labels"][0]

    def test_std_per_bin_computed(self):
        pred = np.linspace(-1, 0, 100) + np.random.default_rng(42).normal(0, 0.1, 100)
        gt = np.linspace(-1, 0, 100)
        result = compute_binned_mse(pred, gt, num_bins=5)
        for i in range(5):
            if result["count_per_bin"][i] > 1:
                assert not np.isnan(result["std_per_bin"][i])


@pytest.mark.unit
class TestComputeComprehensiveMetrics:
    """compute_comprehensive_metrics provides rich metric suite."""

    def test_perfect_prediction(self):
        pred = gt = np.linspace(-1, 0, 100)
        result = compute_comprehensive_metrics(pred, gt)
        assert result["overall_mse"] == pytest.approx(0.0, abs=1e-10)
        assert result["overall_mae"] == pytest.approx(0.0, abs=1e-10)
        assert result["overall_rmse"] == pytest.approx(0.0, abs=1e-10)
        assert result["pearson_corr"] == pytest.approx(1.0, abs=1e-6)
        assert result["r_squared"] == pytest.approx(1.0, abs=1e-6)

    def test_has_spearman(self):
        pred = np.linspace(-1, 0, 50)
        gt = np.linspace(-1, 0, 50)
        result = compute_comprehensive_metrics(pred, gt)
        assert "spearman_corr" in result
        assert result["spearman_corr"] == pytest.approx(1.0, abs=1e-6)

    def test_within_tolerance(self):
        pred = np.zeros(100)
        gt = np.full(100, 0.02)
        result = compute_comprehensive_metrics(pred, gt, tolerances=[0.01, 0.05])
        assert result["within_tolerance"]["within_0.01"]["percentage"] == 0.0
        assert result["within_tolerance"]["within_0.05"]["percentage"] == 100.0

    def test_has_binned_stats(self):
        result = compute_comprehensive_metrics(np.zeros(50), np.linspace(-1, 0, 50))
        assert "binned_stats" in result
        assert "mse_per_bin" in result["binned_stats"]

    def test_n_samples(self):
        result = compute_comprehensive_metrics(np.zeros(42), np.zeros(42))
        assert result["n_samples"] == 42

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            compute_comprehensive_metrics(np.zeros(10), np.zeros(20))

    def test_bias(self):
        pred = np.full(100, 0.1)
        gt = np.zeros(100)
        result = compute_comprehensive_metrics(pred, gt)
        assert result["bias"] == pytest.approx(0.1, abs=1e-6)

    def test_max_error(self):
        pred = np.zeros(100)
        gt = np.zeros(100)
        gt[50] = 0.5
        result = compute_comprehensive_metrics(pred, gt)
        assert result["max_error"] == pytest.approx(0.5, abs=1e-6)
