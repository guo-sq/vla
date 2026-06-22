"""IndicatorPreprocessor unit tests."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from openpi.training.frame_attributes_preprocessors import IndicatorPreprocessor
from openpi.training.frame_attributes_preprocessors.base import FrameAttributes


class MockDatasetContext:
    def __init__(self, root: Path, repo_id: str, total_frames: int, episodes: dict | None = None):
        self.root = root
        self.repo_id = repo_id
        self.hf_dataset = Mock()
        self.hf_dataset.__len__ = Mock(return_value=total_frames)
        self.meta = Mock()
        self.meta.episodes = episodes if episodes is not None else {}


class TestIndicatorPreprocessor:
    def test_load_single_episode(self, tmp_path):
        indicator_dir = tmp_path / "indicators" / "chunk-000"
        indicator_dir.mkdir(parents=True)

        indicator_df = pd.DataFrame({"frame_index": [0, 1, 2], "indicator": [True, False, True]})
        indicator_df.to_parquet(indicator_dir / "episode_000000.parquet", index=False)

        preprocessor = IndicatorPreprocessor(validate_episode_count=False)
        ctx = MockDatasetContext(tmp_path, "test_repo", 3, {0: {}})
        attrs = FrameAttributes()

        preprocessor(ctx, attrs)

        assert attrs.indicator is not None
        assert len(attrs.indicator) == 3
        np.testing.assert_array_equal(attrs.indicator, [True, False, True])
        # Should NOT modify valid_mask
        assert attrs.valid_mask is None

    def test_load_multiple_episodes(self, tmp_path):
        indicator_dir = tmp_path / "indicators" / "chunk-000"
        indicator_dir.mkdir(parents=True)

        for i in range(3):
            indicator_df = pd.DataFrame(
                {
                    "frame_index": list(range(5)),
                    "indicator": [i % 2 == 0] * 5,
                }
            )
            indicator_df.to_parquet(indicator_dir / f"episode_{i:06d}.parquet", index=False)

        preprocessor = IndicatorPreprocessor(validate_episode_count=False)
        ctx = MockDatasetContext(tmp_path, "test_repo", 15, {0: {}, 1: {}, 2: {}})
        attrs = FrameAttributes()

        preprocessor(ctx, attrs)

        assert attrs.indicator is not None
        assert len(attrs.indicator) == 15
        # Episode 0: all True, Episode 1: all False, Episode 2: all True
        np.testing.assert_array_equal(attrs.indicator[:5], [True] * 5)
        np.testing.assert_array_equal(attrs.indicator[5:10], [False] * 5)
        np.testing.assert_array_equal(attrs.indicator[10:15], [True] * 5)

    def test_multiple_chunks(self, tmp_path):
        for chunk_id in range(2):
            chunk_dir = tmp_path / "indicators" / f"chunk-{chunk_id:03d}"
            chunk_dir.mkdir(parents=True)
            indicator_df = pd.DataFrame(
                {
                    "frame_index": [0, 1],
                    "indicator": [True, False],
                }
            )
            indicator_df.to_parquet(chunk_dir / f"episode_{chunk_id:06d}.parquet", index=False)

        preprocessor = IndicatorPreprocessor(validate_episode_count=False)
        ctx = MockDatasetContext(tmp_path, "test_repo", 4, {0: {}, 1: {}})
        attrs = FrameAttributes()

        preprocessor(ctx, attrs)

        assert attrs.indicator is not None
        assert len(attrs.indicator) == 4

    def test_missing_directory_skips(self, tmp_path):
        preprocessor = IndicatorPreprocessor()
        ctx = MockDatasetContext(tmp_path, "test_repo", 10)
        attrs = FrameAttributes()

        preprocessor(ctx, attrs)

        assert attrs.indicator is None

    def test_missing_column_raises(self, tmp_path):
        indicator_dir = tmp_path / "indicators" / "chunk-000"
        indicator_dir.mkdir(parents=True)

        indicator_df = pd.DataFrame({"frame_index": [0, 1], "wrong_column": [True, False]})
        indicator_df.to_parquet(indicator_dir / "episode_000000.parquet", index=False)

        preprocessor = IndicatorPreprocessor(validate_episode_count=False)
        ctx = MockDatasetContext(tmp_path, "test_repo", 2, {0: {}})
        attrs = FrameAttributes()

        with pytest.raises(KeyError, match="indicator"):
            preprocessor(ctx, attrs)

    def test_length_mismatch_raises(self, tmp_path):
        indicator_dir = tmp_path / "indicators" / "chunk-000"
        indicator_dir.mkdir(parents=True)

        indicator_df = pd.DataFrame({"frame_index": [0, 1], "indicator": [True, False]})
        indicator_df.to_parquet(indicator_dir / "episode_000000.parquet", index=False)

        preprocessor = IndicatorPreprocessor(validate_episode_count=False)
        ctx = MockDatasetContext(tmp_path, "test_repo", 5, {0: {}})  # 5 != 2
        attrs = FrameAttributes()

        with pytest.raises(ValueError, match="indicator length"):
            preprocessor(ctx, attrs)

    def test_duplicate_episode_raises(self, tmp_path):
        for chunk_id in range(2):
            chunk_dir = tmp_path / "indicators" / f"chunk-{chunk_id:03d}"
            chunk_dir.mkdir(parents=True)
            indicator_df = pd.DataFrame({"frame_index": [0], "indicator": [True]})
            # Same episode number in different chunks
            indicator_df.to_parquet(chunk_dir / "episode_000000.parquet", index=False)

        preprocessor = IndicatorPreprocessor(validate_episode_count=False)
        ctx = MockDatasetContext(tmp_path, "test_repo", 1, {0: {}})
        attrs = FrameAttributes()

        with pytest.raises(ValueError, match="Duplicate episode"):
            preprocessor(ctx, attrs)

    def test_missing_episodes_with_validation(self, tmp_path):
        indicator_dir = tmp_path / "indicators" / "chunk-000"
        indicator_dir.mkdir(parents=True)

        indicator_df = pd.DataFrame({"frame_index": [0], "indicator": [True]})
        indicator_df.to_parquet(indicator_dir / "episode_000000.parquet", index=False)

        preprocessor = IndicatorPreprocessor(validate_episode_count=True)
        ctx = MockDatasetContext(tmp_path, "test_repo", 1, {0: {}, 1: {}})  # Episode 1 missing
        attrs = FrameAttributes()

        with pytest.raises(ValueError, match="Missing indicator"):
            preprocessor(ctx, attrs)

    def test_backward_compat_flat_directory(self, tmp_path):
        """Test loading from flat directory without chunk subdirs."""
        indicator_dir = tmp_path / "indicators"
        indicator_dir.mkdir(parents=True)

        indicator_df = pd.DataFrame({"frame_index": [0, 1, 2], "indicator": [True, True, False]})
        indicator_df.to_parquet(indicator_dir / "episode_000000.parquet", index=False)

        preprocessor = IndicatorPreprocessor(validate_episode_count=False)
        ctx = MockDatasetContext(tmp_path, "test_repo", 3, {0: {}})
        attrs = FrameAttributes()

        preprocessor(ctx, attrs)

        assert attrs.indicator is not None
        assert len(attrs.indicator) == 3
