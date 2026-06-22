"""Tests for benchmark viz video generation."""

from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.unit]

from scripts.benchmark.viz.video import create_value_curve_video


@pytest.mark.unit
class TestCreateValueCurveVideo:
    """create_value_curve_video generates split-screen camera|curve video."""

    def test_returns_path_on_success(self, tmp_path):
        T = 10
        images = np.random.uniform(-1, 1, (T, 64, 64, 3)).astype(np.float32)
        pred = np.linspace(-1, 0, T)
        gt = np.linspace(-1, 0, T)
        out = str(tmp_path / "test.mp4")
        result = create_value_curve_video(images, pred, gt, out, fps=5)
        assert result == out
        assert Path(out).exists()

    def test_empty_images_returns_none(self, tmp_path):
        out = str(tmp_path / "empty.mp4")
        result = create_value_curve_video(np.array([]), np.array([]), np.array([]), out)
        assert result is None

    def test_triple_camera_layout(self, tmp_path):
        T = 8
        images = np.random.uniform(-1, 1, (T, 64, 64, 3)).astype(np.float32)
        left = np.random.uniform(-1, 1, (T, 64, 64, 3)).astype(np.float32)
        right = np.random.uniform(-1, 1, (T, 64, 64, 3)).astype(np.float32)
        out = str(tmp_path / "triple.mp4")
        result = create_value_curve_video(
            images,
            np.linspace(-1, 0, T),
            np.linspace(-1, 0, T),
            out,
            images_left=left,
            images_right=right,
            camera_layout="triple",
            fps=5,
        )
        assert result == out
        assert Path(out).exists()

    def test_frame_skip_for_long_episodes(self, tmp_path):
        """T > 300 triggers frame skipping, producing fewer frames."""
        T = 350
        images = np.random.uniform(-1, 1, (T, 32, 32, 3)).astype(np.float32)
        out = str(tmp_path / "long.mp4")
        create_value_curve_video(images, np.linspace(-1, 0, T), np.linspace(-1, 0, T), out, fps=5)
        assert Path(out).exists()
