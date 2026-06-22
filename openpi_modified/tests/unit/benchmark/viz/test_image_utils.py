"""Tests for benchmark viz image utilities."""

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

pytestmark = [pytest.mark.unit]

from scripts.benchmark.viz.image_utils import make_image_row
from scripts.benchmark.viz.image_utils import save_composite_frame
from scripts.benchmark.viz.image_utils import to_pil_image
from scripts.benchmark.viz.image_utils import to_uint8_array


@pytest.mark.unit
class TestToPilImage:
    """to_pil_image converts [-1,1] arrays to PIL RGB images."""

    def test_hwc_array_mid_value(self):
        img = np.zeros((64, 64, 3), dtype=np.float32)
        result = to_pil_image(img)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == (64, 64)
        pixel = result.getpixel((0, 0))
        assert all(125 <= v <= 130 for v in pixel)

    def test_hwc_array_min_value(self):
        img = np.full((32, 32, 3), -1.0, dtype=np.float32)
        result = to_pil_image(img)
        pixel = result.getpixel((0, 0))
        assert all(v <= 2 for v in pixel)

    def test_hwc_array_max_value(self):
        img = np.full((32, 32, 3), 1.0, dtype=np.float32)
        result = to_pil_image(img)
        pixel = result.getpixel((0, 0))
        assert all(v >= 253 for v in pixel)

    def test_hw_grayscale(self):
        img = np.full((32, 32), -1.0, dtype=np.float32)
        result = to_pil_image(img)
        assert result.mode == "RGB"
        assert result.size == (32, 32)
        pixel = result.getpixel((0, 0))
        assert all(v <= 2 for v in pixel)

    def test_single_channel(self):
        img = np.ones((32, 32, 1), dtype=np.float32)
        result = to_pil_image(img)
        assert result.mode == "RGB"
        pixel = result.getpixel((0, 0))
        assert all(v >= 253 for v in pixel)

    def test_preserves_spatial_dims(self):
        img = np.zeros((100, 200, 3), dtype=np.float32)
        result = to_pil_image(img)
        assert result.size == (200, 100)  # PIL (W, H)


@pytest.mark.unit
class TestToUint8Array:
    """to_uint8_array converts [-1,1] arrays to uint8 numpy."""

    def test_output_dtype(self):
        img = np.zeros((64, 64, 3), dtype=np.float32)
        result = to_uint8_array(img)
        assert result.dtype == np.uint8

    def test_output_range(self):
        img = np.random.uniform(-1, 1, (64, 64, 3)).astype(np.float32)
        result = to_uint8_array(img)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_shape_preserved(self):
        img = np.zeros((48, 96, 3), dtype=np.float32)
        result = to_uint8_array(img)
        assert result.shape == (48, 96, 3)


@pytest.mark.unit
class TestSaveCompositeFrame:
    """save_composite_frame stitches left|head|right with pred/gt title."""

    def test_creates_file(self, tmp_path):
        imgs = [np.zeros((64, 64, 3), dtype=np.float32) for _ in range(3)]
        out = str(tmp_path / "composite.png")
        save_composite_frame(imgs[0], imgs[1], imgs[2], 0.5, -0.3, out)
        assert Path(out).exists()

    def test_output_width_is_triple(self, tmp_path):
        imgs = [np.zeros((64, 64, 3), dtype=np.float32) for _ in range(3)]
        out = str(tmp_path / "composite.png")
        save_composite_frame(imgs[0], imgs[1], imgs[2], 0.5, -0.3, out)
        result = Image.open(out)
        assert result.width == 64 * 3

    def test_output_has_title_bar(self, tmp_path):
        imgs = [np.zeros((64, 64, 3), dtype=np.float32) for _ in range(3)]
        out = str(tmp_path / "composite.png")
        save_composite_frame(imgs[0], imgs[1], imgs[2], 0.5, -0.3, out)
        result = Image.open(out)
        assert result.height == 64 + 28  # title bar height

    def test_different_sized_images(self, tmp_path):
        img_l = np.zeros((64, 80, 3), dtype=np.float32)
        img_h = np.zeros((64, 100, 3), dtype=np.float32)
        img_r = np.zeros((64, 60, 3), dtype=np.float32)
        out = str(tmp_path / "composite.png")
        save_composite_frame(img_l, img_h, img_r, -0.1, -0.9, out)
        assert Path(out).exists()


@pytest.mark.unit
class TestMakeImageRow:
    """make_image_row stitches images horizontally."""

    def test_same_size_images(self):
        imgs = [np.zeros((64, 64, 3), dtype=np.float32) for _ in range(3)]
        result = make_image_row(imgs)
        assert isinstance(result, Image.Image)
        assert result.width == 64 * 3
        assert result.height == 64

    def test_single_image(self):
        imgs = [np.zeros((64, 128, 3), dtype=np.float32)]
        result = make_image_row(imgs)
        assert result.width == 128
        assert result.height == 64

    def test_different_heights_normalized(self):
        imgs = [np.zeros((64, 64, 3), dtype=np.float32), np.zeros((32, 64, 3), dtype=np.float32)]
        result = make_image_row(imgs, target_height=64)
        assert result.height == 64

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            make_image_row([])
