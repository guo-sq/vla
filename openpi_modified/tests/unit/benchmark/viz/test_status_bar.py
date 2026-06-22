"""Tests for benchmark viz status bar."""

from PIL import Image
import pytest

pytestmark = [pytest.mark.unit]

from scripts.benchmark.viz.status_bar import draw_status_bar


@pytest.mark.unit
class TestDrawStatusBar:
    """draw_status_bar renders a color-coded prediction status bar."""

    def test_returns_pil_image(self):
        result = draw_status_bar(640, frame_idx=10, pred_val=-0.5, gt_val=-0.3)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_correct_width(self):
        result = draw_status_bar(800, frame_idx=0, pred_val=0.0, gt_val=0.0)
        assert result.width == 800

    def test_default_height(self):
        result = draw_status_bar(640, frame_idx=10, pred_val=-0.5, gt_val=-0.3)
        assert result.height == 36

    def test_custom_height(self):
        result = draw_status_bar(640, frame_idx=10, pred_val=-0.5, gt_val=-0.3, bar_height=48)
        assert result.height == 48

    def test_with_mse(self):
        result = draw_status_bar(640, frame_idx=10, pred_val=-0.5, gt_val=-0.3, mse_val=0.04)
        assert result.width == 640
        assert result.height > 0

    def test_extreme_values(self):
        result = draw_status_bar(640, frame_idx=99999, pred_val=-1.0, gt_val=0.0, mse_val=1.0)
        assert isinstance(result, Image.Image)
