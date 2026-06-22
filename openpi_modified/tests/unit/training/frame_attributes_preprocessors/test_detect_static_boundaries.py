"""Unit tests for detect_static_boundaries."""

import numpy as np

from openpi.training.frame_attributes_preprocessors.utils import detect_static_boundaries


class TestPruneTrailingTrue:
    """Default behavior: prune both leading and trailing static frames."""

    def test_basic_leading_and_trailing(self):
        # S S S M M M M S S
        is_static = np.array([1, 1, 1, 0, 0, 0, 0, 1, 1], dtype=bool)
        valid = detect_static_boundaries(is_static, head_margin_frames=0, trailing_margin_frames=0)
        # leading 3 pruned, trailing 2 pruned (min trailing=1)
        expected = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0], dtype=bool)
        np.testing.assert_array_equal(valid, expected)

    def test_head_margin_keeps_transition_frames(self):
        # S S S M M M
        is_static = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
        valid = detect_static_boundaries(is_static, head_margin_frames=1, trailing_margin_frames=0)
        # leading=3, margin=1 -> prune first 2; no trailing static -> no trailing prune
        expected = np.array([0, 0, 1, 1, 1, 1], dtype=bool)
        np.testing.assert_array_equal(valid, expected)

    def test_trailing_margin(self):
        # M M M S S S
        is_static = np.array([0, 0, 0, 1, 1, 1], dtype=bool)
        valid = detect_static_boundaries(is_static, head_margin_frames=0, trailing_margin_frames=1)
        # trailing=3, margin=1 -> prune last 2
        expected = np.array([1, 1, 1, 1, 0, 0], dtype=bool)
        np.testing.assert_array_equal(valid, expected)

    def test_no_static_frames(self):
        is_static = np.array([0, 0, 0, 0, 0], dtype=bool)
        valid = detect_static_boundaries(is_static, head_margin_frames=0, trailing_margin_frames=0)
        # No static frames at all -> nothing pruned
        np.testing.assert_array_equal(valid, np.ones(5, dtype=bool))

    def test_all_static(self):
        is_static = np.array([1, 1, 1, 1, 1], dtype=bool)
        valid = detect_static_boundaries(is_static, head_margin_frames=0, trailing_margin_frames=0)
        # All leading pruned + trailing min=1 -> all False
        np.testing.assert_array_equal(valid, np.zeros(5, dtype=bool))


class TestPruneTrailingFalse:
    """When prune_trailing=False, only leading static frames are pruned."""

    def test_trailing_kept_valid(self):
        # S S S M M M M S S
        is_static = np.array([1, 1, 1, 0, 0, 0, 0, 1, 1], dtype=bool)
        valid = detect_static_boundaries(
            is_static, head_margin_frames=0, trailing_margin_frames=0, prune_trailing=False
        )
        # leading 3 pruned, trailing kept
        expected = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=bool)
        np.testing.assert_array_equal(valid, expected)

    def test_no_leading_static(self):
        # M M M M S S
        is_static = np.array([0, 0, 0, 0, 1, 1], dtype=bool)
        valid = detect_static_boundaries(
            is_static, head_margin_frames=0, trailing_margin_frames=0, prune_trailing=False
        )
        # No leading to prune, trailing kept
        np.testing.assert_array_equal(valid, np.ones(6, dtype=bool))

    def test_all_static_prune_trailing_false(self):
        is_static = np.array([1, 1, 1, 1], dtype=bool)
        valid = detect_static_boundaries(
            is_static, head_margin_frames=0, trailing_margin_frames=0, prune_trailing=False
        )
        # All leading static pruned, trailing not pruned -> all False (leading consumes all)
        np.testing.assert_array_equal(valid, np.zeros(4, dtype=bool))

    def test_head_margin_with_no_trailing_prune(self):
        # S S M M S S
        is_static = np.array([1, 1, 0, 0, 1, 1], dtype=bool)
        valid = detect_static_boundaries(
            is_static, head_margin_frames=1, trailing_margin_frames=0, prune_trailing=False
        )
        # leading=2, margin=1 -> prune first 1; trailing kept
        expected = np.array([0, 1, 1, 1, 1, 1], dtype=bool)
        np.testing.assert_array_equal(valid, expected)


class TestEdgeCases:
    """Edge cases: empty input, single frame."""

    def test_empty_input(self):
        is_static = np.array([], dtype=bool)
        valid = detect_static_boundaries(is_static, head_margin_frames=0, trailing_margin_frames=0)
        assert len(valid) == 0

    def test_single_frame_static(self):
        is_static = np.array([True])
        valid = detect_static_boundaries(is_static, head_margin_frames=0, trailing_margin_frames=0)
        # Leading prunes it, trailing also would -> all False
        np.testing.assert_array_equal(valid, np.array([False]))

    def test_single_frame_moving(self):
        is_static = np.array([False])
        valid = detect_static_boundaries(is_static, head_margin_frames=0, trailing_margin_frames=0)
        # No static frames -> frame stays valid
        np.testing.assert_array_equal(valid, np.array([True]))

    def test_single_frame_prune_trailing_false(self):
        is_static = np.array([False])
        valid = detect_static_boundaries(
            is_static, head_margin_frames=0, trailing_margin_frames=0, prune_trailing=False
        )
        # No leading, no trailing prune -> valid
        np.testing.assert_array_equal(valid, np.array([True]))

    def test_large_margin_does_not_go_negative(self):
        # S M M M
        is_static = np.array([1, 0, 0, 0], dtype=bool)
        valid = detect_static_boundaries(is_static, head_margin_frames=10, trailing_margin_frames=10)
        # leading=1, margin=10 -> max(1-10, 0)=0 -> no leading prune
        # no trailing static -> no trailing prune
        np.testing.assert_array_equal(valid, np.ones(4, dtype=bool))
