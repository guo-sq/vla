"""Tests for benchmark viz curve plotting utilities."""

from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.unit]

from scripts.benchmark.viz.curve_plots import save_value_curve
from scripts.benchmark.viz.curve_plots import save_value_curve_with_advantage


@pytest.mark.unit
class TestSaveValueCurve:
    """save_value_curve generates pred vs GT line plot."""

    def test_creates_file(self, tmp_path):
        pred = np.linspace(-1, 0, 100)
        gt = np.linspace(-1, 0, 100)
        out = str(tmp_path / "curve.png")
        save_value_curve(pred, gt, out, title="test curve")
        assert Path(out).exists()

    def test_creates_parent_dirs(self, tmp_path):
        out = str(tmp_path / "sub" / "dir" / "curve.png")
        save_value_curve(np.zeros(50), np.zeros(50), out)
        assert Path(out).exists()

    def test_noisy_data(self, tmp_path):
        rng = np.random.default_rng(42)
        pred = np.linspace(-1, 0, 200) + rng.normal(0, 0.1, 200)
        gt = np.linspace(-1, 0, 200)
        out = str(tmp_path / "noisy.png")
        save_value_curve(pred, gt, out, title="noisy")
        assert Path(out).exists()

    def test_short_sequence(self, tmp_path):
        out = str(tmp_path / "short.png")
        save_value_curve(np.array([-0.5, -0.3]), np.array([-0.5, 0.0]), out)
        assert Path(out).exists()


@pytest.mark.unit
class TestSaveValueCurveWithAdvantage:
    """save_value_curve_with_advantage adds advantage subplot below curve."""

    def test_creates_file(self, tmp_path):
        pred = np.linspace(-0.8, -0.1, 200)
        gt = np.linspace(-1, 0, 200)
        out = str(tmp_path / "adv_curve.png")
        save_value_curve_with_advantage(pred, gt, out)
        assert Path(out).exists()

    def test_custom_horizon(self, tmp_path):
        pred = np.linspace(-1, 0, 300)
        gt = np.linspace(-1, 0, 300)
        out = str(tmp_path / "custom.png")
        save_value_curve_with_advantage(pred, gt, out, advantage_horizon=30)
        assert Path(out).exists()

    def test_short_sequence_no_advantage(self, tmp_path):
        """Sequence shorter than horizon should still produce output."""
        pred = np.linspace(-1, 0, 10)
        gt = np.linspace(-1, 0, 10)
        out = str(tmp_path / "short_adv.png")
        save_value_curve_with_advantage(pred, gt, out, advantage_horizon=50)
        assert Path(out).exists()
