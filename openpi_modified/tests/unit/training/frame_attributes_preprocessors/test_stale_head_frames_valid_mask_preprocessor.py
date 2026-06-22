"""StaleHeadFramesValidMaskPreprocessor unit tests."""

from unittest.mock import Mock

import numpy as np
import pytest

from openpi.training.frame_attributes_preprocessors import StaleHeadFramesValidMaskPreprocessor
from openpi.training.frame_attributes_preprocessors.base import FrameAttributes


def _make_ctx(total: int, boundaries: list[tuple[int, int]]):
    ctx = Mock()
    ctx.repo_id = "fake/ds"
    ctx.extras = {}
    hf = Mock()
    hf.__len__ = Mock(return_value=total)
    ctx.hf_dataset = hf
    ctx.episode_data_index = {
        "from": np.array([s for s, _ in boundaries], dtype=np.int64),
        "to": np.array([e for _, e in boundaries], dtype=np.int64),
    }
    return ctx


class TestStaleHeadFramesValidMaskPreprocessor:
    def test_basic_two_episodes_only_episode_zero_masked(self):
        ctx = _make_ctx(total=20, boundaries=[(0, 10), (10, 20)])
        attrs = FrameAttributes()

        StaleHeadFramesValidMaskPreprocessor(first_n=3)(ctx, attrs)

        expected = np.array([False] * 3 + [True] * 7 + [True] * 10)
        np.testing.assert_array_equal(attrs.valid_mask, expected)

    def test_first_n_zero_is_noop(self):
        ctx = _make_ctx(total=20, boundaries=[(0, 10), (10, 20)])
        attrs = FrameAttributes()

        StaleHeadFramesValidMaskPreprocessor(first_n=0)(ctx, attrs)

        assert attrs.valid_mask is None

    def test_combines_with_existing_valid_mask_via_and(self):
        ctx = _make_ctx(total=10, boundaries=[(0, 10)])
        attrs = FrameAttributes()
        upstream = np.array([True, True, True, False, True, True, True, True, True, True])
        attrs.valid_mask = upstream.copy()

        StaleHeadFramesValidMaskPreprocessor(first_n=3)(ctx, attrs)

        # first 3 newly False; index 3 stays False (from upstream); rest stays True
        expected = np.array([False, False, False, False, True, True, True, True, True, True])
        np.testing.assert_array_equal(attrs.valid_mask, expected)

    def test_first_n_exceeds_episode_zero_length(self):
        ctx = _make_ctx(total=12, boundaries=[(0, 2), (2, 12)])
        attrs = FrameAttributes()

        StaleHeadFramesValidMaskPreprocessor(first_n=5)(ctx, attrs)

        # only the 2 frames of episode 0 can be invalidated
        expected = np.array([False, False] + [True] * 10)
        np.testing.assert_array_equal(attrs.valid_mask, expected)

    def test_single_episode(self):
        ctx = _make_ctx(total=10, boundaries=[(0, 10)])
        attrs = FrameAttributes()

        StaleHeadFramesValidMaskPreprocessor(first_n=3)(ctx, attrs)

        expected = np.array([False] * 3 + [True] * 7)
        np.testing.assert_array_equal(attrs.valid_mask, expected)

    def test_empty_dataset(self):
        ctx = _make_ctx(total=0, boundaries=[])
        attrs = FrameAttributes()

        StaleHeadFramesValidMaskPreprocessor(first_n=3)(ctx, attrs)

        assert attrs.valid_mask is None

    def test_negative_first_n_raises(self):
        with pytest.raises(ValueError, match="first_n must be non-negative"):
            StaleHeadFramesValidMaskPreprocessor(first_n=-1)
