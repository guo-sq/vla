"""RepoNameMatchSampleWeightPreprocessor unit tests."""

from unittest.mock import Mock

import numpy as np
import pytest

from openpi.training.frame_attributes_preprocessors.base import FrameAttributes
from openpi.training.frame_attributes_preprocessors.sample_weight_preprocessor import (
    RepoNameMatchSampleWeightPreprocessor,
)


def _make_ctx(repo_id: str, total_frames: int):
    ctx = Mock()
    ctx.repo_id = repo_id
    ctx.hf_dataset = Mock()
    ctx.hf_dataset.__len__ = Mock(return_value=total_frames)
    ctx.episode_data_index = {
        "from": np.array([0], dtype=np.int64),
        "to": np.array([total_frames], dtype=np.int64),
    }
    return ctx


class TestRepoNameMatchSampleWeightPreprocessor:
    # --- validation ---

    def test_empty_substring_raises(self):
        with pytest.raises(ValueError, match="substring must be non-empty"):
            RepoNameMatchSampleWeightPreprocessor(substring="")

    def test_empty_substring_list_raises(self):
        with pytest.raises(ValueError, match="substring list must be non-empty"):
            RepoNameMatchSampleWeightPreprocessor(substring=[])

    def test_substring_list_empty_entry_raises(self):
        with pytest.raises(ValueError, match="substring list entries must be non-empty"):
            RepoNameMatchSampleWeightPreprocessor(substring=["a", ""])

    def test_weight_zero_raises(self):
        with pytest.raises(ValueError, match="weight must be in"):
            RepoNameMatchSampleWeightPreprocessor(substring="foo", weight=0)

    def test_weight_negative_raises(self):
        with pytest.raises(ValueError, match="weight must be in"):
            RepoNameMatchSampleWeightPreprocessor(substring="foo", weight=-1)

    def test_weight_exceeds_max_raises(self):
        with pytest.raises(ValueError, match="weight must be in"):
            RepoNameMatchSampleWeightPreprocessor(substring="foo", weight=1001)

    def test_frame_skip_zero_raises(self):
        with pytest.raises(ValueError, match="frame_skip must be >= 1"):
            RepoNameMatchSampleWeightPreprocessor(substring="foo", frame_skip=0)

    def test_weight_at_max_ok(self):
        proc = RepoNameMatchSampleWeightPreprocessor(substring="foo", weight=1000)
        assert proc.weight == 1000

    # --- matching behaviour ---

    def test_match_multiplies_existing_weights(self):
        proc = RepoNameMatchSampleWeightPreprocessor(substring="insert", weight=3)
        ctx = _make_ctx("my_insert_dataset", 4)
        attrs = FrameAttributes(sample_weight=np.array([1, 0, 2, 1], dtype=np.int32))

        proc(ctx, attrs)

        np.testing.assert_array_equal(attrs.sample_weight, [3, 0, 6, 3])

    def test_match_preserves_zeros_from_prior_preprocessor(self):
        """Zeros set by StaticRatioSampleWeightPreprocessor should stay zero."""
        proc = RepoNameMatchSampleWeightPreprocessor(substring="tube", weight=5)
        ctx = _make_ctx("tube_repo", 3)
        attrs = FrameAttributes(sample_weight=np.array([0, 1, 0], dtype=np.int32))

        proc(ctx, attrs)

        np.testing.assert_array_equal(attrs.sample_weight, [0, 5, 0])

    def test_match_sample_weight_none_creates_full_array(self):
        proc = RepoNameMatchSampleWeightPreprocessor(substring="insert", weight=2)
        ctx = _make_ctx("insert_tube", 5)
        attrs = FrameAttributes()

        proc(ctx, attrs)

        assert attrs.sample_weight is not None
        np.testing.assert_array_equal(attrs.sample_weight, [2, 2, 2, 2, 2])
        assert attrs.sample_weight.dtype == np.int32

    def test_no_match_keeps_existing_weights(self):
        proc = RepoNameMatchSampleWeightPreprocessor(substring="insert")
        ctx = _make_ctx("seatbelt_dataset", 3)
        original = np.array([1, 0, 1], dtype=np.int32)
        attrs = FrameAttributes(sample_weight=original.copy())

        proc(ctx, attrs)

        np.testing.assert_array_equal(attrs.sample_weight, original)

    def test_no_match_sample_weight_none_initializes_ones(self):
        proc = RepoNameMatchSampleWeightPreprocessor(substring="insert")
        ctx = _make_ctx("seatbelt_dataset", 4)
        attrs = FrameAttributes()

        proc(ctx, attrs)

        assert attrs.sample_weight is not None
        np.testing.assert_array_equal(attrs.sample_weight, [1, 1, 1, 1])

    # --- case sensitivity ---

    def test_case_insensitive_by_default(self):
        proc = RepoNameMatchSampleWeightPreprocessor(substring="INSERT", weight=2)
        ctx = _make_ctx("my_insert_data", 3)
        attrs = FrameAttributes()

        proc(ctx, attrs)

        np.testing.assert_array_equal(attrs.sample_weight, [2, 2, 2])

    def test_case_sensitive_no_match(self):
        proc = RepoNameMatchSampleWeightPreprocessor(substring="INSERT", weight=2, case_sensitive=True)
        ctx = _make_ctx("my_insert_data", 3)
        attrs = FrameAttributes()

        proc(ctx, attrs)

        np.testing.assert_array_equal(attrs.sample_weight, [1, 1, 1])

    def test_case_sensitive_match(self):
        proc = RepoNameMatchSampleWeightPreprocessor(substring="INSERT", weight=2, case_sensitive=True)
        ctx = _make_ctx("my_INSERT_data", 3)
        attrs = FrameAttributes()

        proc(ctx, attrs)

        np.testing.assert_array_equal(attrs.sample_weight, [2, 2, 2])

    # --- substring matching ---

    def test_partial_match(self):
        proc = RepoNameMatchSampleWeightPreprocessor(substring="tube", weight=4)
        ctx = _make_ctx("insert_tube_v2", 2)
        attrs = FrameAttributes()

        proc(ctx, attrs)

        np.testing.assert_array_equal(attrs.sample_weight, [4, 4])

    def test_list_substring_matches_any_needle(self):
        proc = RepoNameMatchSampleWeightPreprocessor(substring=["pickout", "placeout"], weight=2)
        ctx = _make_ctx("record.pick.place.placeout.bipiper.v1", 3)
        attrs = FrameAttributes()
        proc(ctx, attrs)
        np.testing.assert_array_equal(attrs.sample_weight, [2, 2, 2])

    def test_list_substring_second_needle_matches(self):
        proc = RepoNameMatchSampleWeightPreprocessor(substring=["pickout", "placeout"], weight=3)
        ctx = _make_ctx("data/record.pick.place.pickout.x", 2)
        attrs = FrameAttributes(sample_weight=np.ones(2, dtype=np.int32))
        proc(ctx, attrs)
        np.testing.assert_array_equal(attrs.sample_weight, [3, 3])

    def test_list_substring_no_match(self):
        proc = RepoNameMatchSampleWeightPreprocessor(substring=["pickout", "placeout"])
        ctx = _make_ctx("record.pick.place.normal.x", 2)
        attrs = FrameAttributes()
        proc(ctx, attrs)
        np.testing.assert_array_equal(attrs.sample_weight, [1, 1])

    def test_list_substring_multiplies_once_when_both_needles_in_name(self):
        """Both needles in repo_id still apply weight once."""
        proc = RepoNameMatchSampleWeightPreprocessor(substring=["pickout", "placeout"], weight=2)
        ctx = _make_ctx("pickout_and_placeout_combo", 2)
        attrs = FrameAttributes()
        proc(ctx, attrs)
        np.testing.assert_array_equal(attrs.sample_weight, [2, 2])

    # --- dtype ---

    def test_output_dtype_is_int32(self):
        proc = RepoNameMatchSampleWeightPreprocessor(substring="foo", weight=2)
        ctx = _make_ctx("foo_bar", 3)
        attrs = FrameAttributes()

        proc(ctx, attrs)

        assert attrs.sample_weight.dtype == np.int32

    def test_weight_one_is_identity(self):
        proc = RepoNameMatchSampleWeightPreprocessor(substring="match", weight=1)
        ctx = _make_ctx("match_repo", 3)
        original = np.array([5, 0, 3], dtype=np.int32)
        attrs = FrameAttributes(sample_weight=original.copy())

        proc(ctx, attrs)

        np.testing.assert_array_equal(attrs.sample_weight, original)

    def test_frame_skip_downsample_matched_repo(self):
        proc = RepoNameMatchSampleWeightPreprocessor(substring="heavy", frame_skip=2, weight=1)
        ctx = _make_ctx("my_heavy_repo", 6)
        attrs = FrameAttributes(sample_weight=np.ones(6, dtype=np.int32))
        proc(ctx, attrs)
        np.testing.assert_array_equal(attrs.sample_weight, [1, 0, 1, 0, 1, 0])

    def test_frame_skip_then_weight_multiply(self):
        proc = RepoNameMatchSampleWeightPreprocessor(substring="heavy", frame_skip=2, weight=2)
        ctx = _make_ctx("heavy_x", 4)
        attrs = FrameAttributes(sample_weight=np.ones(4, dtype=np.int32))
        proc(ctx, attrs)
        np.testing.assert_array_equal(attrs.sample_weight, [2, 0, 2, 0])

    def test_frame_skip_respects_valid_mask(self):
        proc = RepoNameMatchSampleWeightPreprocessor(substring="tube", frame_skip=2, weight=1)
        ctx = _make_ctx("tube_repo", 5)
        vm = np.array([1, 1, 0, 1, 1], dtype=bool)
        w = np.ones(5, dtype=np.int32)
        attrs = FrameAttributes(valid_mask=vm, sample_weight=w)
        proc(ctx, attrs)
        # eligible indices 0,1,3,4 -> keep 0 and 3; idx 2 invalid (weight unchanged, unused by loader)
        np.testing.assert_array_equal(attrs.sample_weight, [1, 0, 1, 1, 0])

    def test_frame_skip_no_match_ignored(self):
        proc = RepoNameMatchSampleWeightPreprocessor(substring="pickout", frame_skip=3, weight=1)
        ctx = _make_ctx("other_repo", 4)
        attrs = FrameAttributes(sample_weight=np.arange(4, dtype=np.int32))
        proc(ctx, attrs)
        np.testing.assert_array_equal(attrs.sample_weight, np.arange(4, dtype=np.int32))
