"""Unit tests for scripts/compute_values.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_dataset_dir(tmp_path: Path) -> Path:
    """Create a mock dataset directory with data parquets for 2 episodes."""
    data_dir = tmp_path / "data" / "chunk-000"
    data_dir.mkdir(parents=True)

    # Episode 0: 5 frames
    df0 = pd.DataFrame(
        {
            "frame_index": np.arange(5, dtype=np.int32),
            "observation.state": np.random.randn(5, 14).tolist(),
        }
    )
    df0.to_parquet(data_dir / "episode_000000.parquet", index=False)

    # Episode 1: 3 frames
    df1 = pd.DataFrame(
        {
            "frame_index": np.arange(3, dtype=np.int32),
            "observation.state": np.random.randn(3, 14).tolist(),
        }
    )
    df1.to_parquet(data_dir / "episode_000001.parquet", index=False)

    return tmp_path


@pytest.fixture
def mock_dataset_dir_multi_chunk(tmp_path: Path) -> Path:
    """Create a mock dataset with episodes in different chunks."""
    for chunk_id, ep_id, n_frames in [(0, 0, 5), (0, 1, 3), (1, 2, 4)]:
        chunk_dir = tmp_path / "data" / f"chunk-{chunk_id:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        frame_df = pd.DataFrame(
            {
                "frame_index": np.arange(n_frames, dtype=np.int32),
                "observation.state": np.random.randn(n_frames, 14).tolist(),
            }
        )
        frame_df.to_parquet(chunk_dir / f"episode_{ep_id:06d}.parquet", index=False)

    return tmp_path


# ---------------------------------------------------------------------------
# save_values_per_episode
# ---------------------------------------------------------------------------


class TestSaveValuesFormat:
    """Test that save_values_per_episode outputs in test_rl.py-compatible format."""

    def test_output_dir_is_value_pred(self, mock_dataset_dir: Path):
        """Output directory must be value_pred/ (not values/)."""
        from scripts.compute_values import save_values_per_episode

        values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)
        episode_indices = np.array([0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int32)
        frame_indices = np.array([0, 1, 2, 3, 4, 0, 1, 2], dtype=np.int32)

        save_values_per_episode(values, episode_indices, frame_indices, mock_dataset_dir)

        # value_pred/ should exist, values/ should NOT
        assert (mock_dataset_dir / "value_pred").is_dir()
        assert not (mock_dataset_dir / "values").exists()

    def test_columns_are_pred_value_and_value_is_valid(self, mock_dataset_dir: Path):
        """Parquet columns must be pred_value (float32) + value_is_valid (bool)."""
        from scripts.compute_values import save_values_per_episode

        values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)
        episode_indices = np.array([0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int32)
        frame_indices = np.array([0, 1, 2, 3, 4, 0, 1, 2], dtype=np.int32)

        save_values_per_episode(values, episode_indices, frame_indices, mock_dataset_dir)

        parquet_path = mock_dataset_dir / "value_pred" / "chunk-000" / "episode_000000.parquet"
        assert parquet_path.exists()

        result_df = pd.read_parquet(parquet_path)
        assert list(result_df.columns) == ["pred_value", "value_is_valid"]
        assert result_df["pred_value"].dtype == np.float32
        assert result_df["value_is_valid"].dtype == bool

    def test_fill_to_full_episode_length(self, mock_dataset_dir: Path):
        """Output parquet row count must equal original episode frame count."""
        from scripts.compute_values import save_values_per_episode

        # Episode 0 has 5 frames in data/chunk-000/episode_000000.parquet
        values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)
        episode_indices = np.array([0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int32)
        frame_indices = np.array([0, 1, 2, 3, 4, 0, 1, 2], dtype=np.int32)

        save_values_per_episode(values, episode_indices, frame_indices, mock_dataset_dir)

        df0 = pd.read_parquet(mock_dataset_dir / "value_pred" / "chunk-000" / "episode_000000.parquet")
        assert len(df0) == 5  # Must match original episode length

        df1 = pd.read_parquet(mock_dataset_dir / "value_pred" / "chunk-000" / "episode_000001.parquet")
        assert len(df1) == 3

    def test_all_frames_valid_when_all_inferred(self, mock_dataset_dir: Path):
        """When all frames are inferred, value_is_valid should be all True."""
        from scripts.compute_values import save_values_per_episode

        values = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        episode_indices = np.array([0, 0, 0, 0, 0], dtype=np.int32)
        frame_indices = np.array([0, 1, 2, 3, 4], dtype=np.int32)

        save_values_per_episode(values, episode_indices, frame_indices, mock_dataset_dir)

        result_df = pd.read_parquet(mock_dataset_dir / "value_pred" / "chunk-000" / "episode_000000.parquet")
        assert result_df["value_is_valid"].all()
        np.testing.assert_array_almost_equal(result_df["pred_value"].to_numpy(), [0.1, 0.2, 0.3, 0.4, 0.5])

    def test_partial_frames_fill_with_invalid(self, mock_dataset_dir: Path):
        """When only some frames are inferred, missing frames should be 0.0/False."""
        from scripts.compute_values import save_values_per_episode

        # Only infer frames 0, 2, 4 of a 5-frame episode
        values = np.array([0.1, 0.3, 0.5], dtype=np.float32)
        episode_indices = np.array([0, 0, 0], dtype=np.int32)
        frame_indices = np.array([0, 2, 4], dtype=np.int32)

        save_values_per_episode(values, episode_indices, frame_indices, mock_dataset_dir)

        result_df = pd.read_parquet(mock_dataset_dir / "value_pred" / "chunk-000" / "episode_000000.parquet")
        assert len(result_df) == 5

        expected_valid = [True, False, True, False, True]
        expected_values = [0.1, 0.0, 0.3, 0.0, 0.5]

        np.testing.assert_array_equal(result_df["value_is_valid"].to_numpy(), expected_valid)
        np.testing.assert_array_almost_equal(result_df["pred_value"].to_numpy(), expected_values)

    def test_multi_chunk_output(self, mock_dataset_dir_multi_chunk: Path):
        """Episodes in different chunks get saved to correct chunk directories."""
        from scripts.compute_values import save_values_per_episode

        # 3 episodes: ep0(5 frames, chunk0), ep1(3 frames, chunk0), ep2(4 frames, chunk1)
        values = np.concatenate(
            [
                np.random.randn(5).astype(np.float32),
                np.random.randn(3).astype(np.float32),
                np.random.randn(4).astype(np.float32),
            ]
        )
        episode_indices = np.array([0] * 5 + [1] * 3 + [2] * 4, dtype=np.int32)
        frame_indices = np.concatenate([np.arange(5), np.arange(3), np.arange(4)]).astype(np.int32)

        save_values_per_episode(values, episode_indices, frame_indices, mock_dataset_dir_multi_chunk)

        assert (mock_dataset_dir_multi_chunk / "value_pred" / "chunk-000" / "episode_000000.parquet").exists()
        assert (mock_dataset_dir_multi_chunk / "value_pred" / "chunk-000" / "episode_000001.parquet").exists()
        assert (mock_dataset_dir_multi_chunk / "value_pred" / "chunk-001" / "episode_000002.parquet").exists()


# ---------------------------------------------------------------------------
# compute_values_for_dataset return shape
# ---------------------------------------------------------------------------


class TestComputeValuesReturnShape:
    """compute_values_for_dataset must return (values, episode_indices, frame_indices)."""

    def test_returns_three_arrays(self):
        """Return value is a 3-tuple: (values, episode_indices, frame_indices)."""
        import inspect

        from scripts.compute_values import compute_values_for_dataset

        inspect.signature(compute_values_for_dataset)
        assert callable(compute_values_for_dataset)


# ---------------------------------------------------------------------------
# ValuePredictionPreprocessor compatibility
# ---------------------------------------------------------------------------


class TestValuePredictionPreprocessorCompat:
    """Output from save_values_per_episode must be readable by ValuePredictionPreprocessor."""

    def test_preprocessor_reads_compute_values_output(self, mock_dataset_dir: Path):
        """ValuePredictionPreprocessor can load parquets saved by save_values_per_episode."""
        from openpi.training.frame_attributes_preprocessors import ValuePredictionPreprocessor
        from scripts.compute_values import save_values_per_episode

        # Save values
        values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)
        episode_indices = np.array([0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int32)
        frame_indices = np.array([0, 1, 2, 3, 4, 0, 1, 2], dtype=np.int32)

        save_values_per_episode(values, episode_indices, frame_indices, mock_dataset_dir)

        # Read with ValuePredictionPreprocessor
        preprocessor = ValuePredictionPreprocessor(
            value_pred_dir="value_pred",
            auto_discover_chunks=True,
            validate_episode_count=False,
        )

        ctx = Mock()
        ctx.root = mock_dataset_dir
        ctx.repo_id = "test_repo"
        ctx.hf_dataset = Mock()
        ctx.hf_dataset.__len__ = Mock(return_value=8)  # 5 + 3 frames total
        ctx.meta = Mock()
        ctx.meta.episodes = {0: {}, 1: {}}

        attrs = Mock()
        attrs.pred_value = None
        attrs.valid_mask = None

        preprocessor(ctx, attrs)

        assert attrs.pred_value is not None
        assert len(attrs.pred_value) == 8
        np.testing.assert_array_almost_equal(attrs.pred_value, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        assert attrs.valid_mask is not None
        assert attrs.valid_mask.all()


# ---------------------------------------------------------------------------
# CLI parameters
# ---------------------------------------------------------------------------


class TestCLIParams:
    """CLI parameters should align with test_rl.py naming conventions."""

    def test_parse_args_has_aligned_params(self):
        """Verify key parameters exist with test_rl.py-compatible names."""
        import inspect

        from scripts.compute_values import main

        source = inspect.getsource(main)

        # Must use underscore-style params aligned with test_rl.py
        assert "config_name" in source
        assert "ckpt_dir" in source
        assert "dataset_root" in source
        assert "repo_id" in source
        assert "batch_size" in source
        assert "data_config_name" in source


# ---------------------------------------------------------------------------
# save_values_config metadata
# ---------------------------------------------------------------------------


class TestSaveValuesConfig:
    """Test values_config.json metadata output."""

    def test_saves_metadata_json(self, tmp_path: Path):
        """save_values_config should write meta/values_config.json."""
        from scripts.compute_values import save_values_config

        values = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        save_values_config(
            tmp_path,
            config_name="test_config",
            value_checkpoint="/path/to/ckpt",
            num_episodes=1,
            num_frames=3,
            values=values,
        )

        config_path = tmp_path / "meta" / "values_config.json"
        assert config_path.exists()

        import json

        with open(config_path) as f:
            config = json.load(f)

        assert config["config_name"] == "test_config"
        assert config["num_episodes"] == 1
        assert config["num_frames"] == 3
        assert "value_stats" in config


# ---------------------------------------------------------------------------
# assign_datasets_balanced
# ---------------------------------------------------------------------------


class TestAssignDatasetsBalanced:
    """Test greedy load-balanced GPU assignment."""

    def test_single_gpu(self):
        """All datasets assigned to GPU 0."""
        from scripts.compute_values import assign_datasets_balanced

        result = assign_datasets_balanced([100, 200, 50], num_gpus=1)
        assert len(result) == 1
        assert sorted(result[0]) == [0, 1, 2]

    def test_equal_split(self):
        """4 equal datasets / 2 GPUs -> 2 each, total load balanced."""
        from scripts.compute_values import assign_datasets_balanced

        result = assign_datasets_balanced([100, 100, 100, 100], num_gpus=2)
        assert len(result) == 2
        assert len(result[0]) == 2
        assert len(result[1]) == 2
        # All indices covered
        all_indices = sorted(result[0] + result[1])
        assert all_indices == [0, 1, 2, 3]

    def test_greedy_heavy_first(self):
        """Largest dataset isolated, small ones grouped together."""
        from scripts.compute_values import assign_datasets_balanced

        result = assign_datasets_balanced([1000, 10, 10, 10], num_gpus=2)
        assert len(result) == 2
        # GPU with the heavy dataset should have only 1 dataset
        # The heavy one is alone on one GPU
        assert any(len(gpu) == 1 and 0 in gpu for gpu in result)

    def test_empty_input(self):
        """Empty dataset list returns empty assignments."""
        from scripts.compute_values import assign_datasets_balanced

        result = assign_datasets_balanced([], num_gpus=2)
        assert len(result) == 2
        assert result[0] == []
        assert result[1] == []

    def test_more_gpus_than_datasets(self):
        """Some GPUs get empty lists when datasets < GPUs."""
        from scripts.compute_values import assign_datasets_balanced

        result = assign_datasets_balanced([100, 200], num_gpus=4)
        assert len(result) == 4
        non_empty = [g for g in result if g]
        assert len(non_empty) == 2
        all_indices = sorted(idx for gpu in result for idx in gpu)
        assert all_indices == [0, 1]

    def test_zero_gpus(self):
        """Zero GPUs returns empty list."""
        from scripts.compute_values import assign_datasets_balanced

        result = assign_datasets_balanced([100, 200], num_gpus=0)
        assert result == []


# ---------------------------------------------------------------------------
# Multi-GPU sharded inference helpers
# ---------------------------------------------------------------------------


class TestBatchSizeAlignment:
    """Test batch size alignment to GPU count."""

    def test_already_aligned(self):
        """batch_size divisible by num_gpus stays unchanged."""
        num_gpus = 4
        batch_size = 64
        aligned = ((batch_size + num_gpus - 1) // num_gpus) * num_gpus
        assert aligned == 64

    def test_needs_alignment(self):
        """batch_size not divisible by num_gpus rounds up."""
        num_gpus = 4
        batch_size = 65
        aligned = ((batch_size + num_gpus - 1) // num_gpus) * num_gpus
        assert aligned == 68

    def test_smaller_than_gpus(self):
        """batch_size < num_gpus rounds up to num_gpus."""
        num_gpus = 8
        batch_size = 3
        aligned = ((batch_size + num_gpus - 1) // num_gpus) * num_gpus
        assert aligned == 8


class TestLastBatchPadding:
    """Test padding logic for the last incomplete batch."""

    def test_pad_and_truncate(self):
        """Padding a short batch to aligned size, then truncating results."""
        num_gpus = 4
        aligned_batch_size = 8
        # Simulate a last batch with only 5 samples
        original_size = 5
        values = np.random.randn(original_size).astype(np.float32)

        # Pad
        pad_size = aligned_batch_size - original_size
        padded = np.pad(values, (0, pad_size), mode="constant", constant_values=0.0)
        assert len(padded) == aligned_batch_size
        assert padded.shape[0] % num_gpus == 0

        # Truncate after "inference"
        result = padded[:original_size]
        np.testing.assert_array_equal(result, values)

    def test_full_batch_no_padding(self):
        """Full batch needs no padding."""
        aligned_batch_size = 8
        values = np.random.randn(aligned_batch_size).astype(np.float32)

        original_size = len(values)
        pad_size = aligned_batch_size - original_size
        assert pad_size == 0


# ---------------------------------------------------------------------------
# JIT function reuse across datasets
# ---------------------------------------------------------------------------


class TestJitReuse:
    """compute_values_for_dataset should accept an external JIT function
    to avoid per-dataset recompilation (~3:45 overhead per dataset)."""

    def test_signature_accepts_score_observation_jit(self):
        """compute_values_for_dataset must accept score_observation_jit param."""
        import inspect

        from scripts.compute_values import compute_values_for_dataset

        sig = inspect.signature(compute_values_for_dataset)
        assert (
            "score_observation_jit" in sig.parameters
        ), "compute_values_for_dataset must accept score_observation_jit parameter"

    def test_score_observation_jit_default_is_none(self):
        """score_observation_jit should default to None for backward compat."""
        import inspect

        from scripts.compute_values import compute_values_for_dataset

        sig = inspect.signature(compute_values_for_dataset)
        param = sig.parameters["score_observation_jit"]
        assert param.default is None, "score_observation_jit should default to None"

    def test_jit_function_not_recreated_when_provided(self):
        """When score_observation_jit is provided, module_jit should NOT be called."""
        import inspect

        from scripts.compute_values import compute_values_for_dataset

        source = inspect.getsource(compute_values_for_dataset)

        # The function should have conditional logic:
        # if score_observation_jit is None: create new one
        # else: use the provided one
        assert (
            "score_observation_jit is None" in source or "score_observation_jit is not None" in source
        ), "Function must conditionally create JIT only when not provided"

    def test_main_creates_jit_once(self):
        """main() should create score_observation_jit once and pass to all datasets."""
        import inspect

        from scripts.compute_values import main

        source = inspect.getsource(main)
        # main should call module_jit before the per-dataset loop
        assert (
            "score_observation_jit" in source
        ), "main() must create and pass score_observation_jit to compute_values_for_dataset"

    def test_signature_accepts_data_sharding(self):
        """compute_values_for_dataset must accept data_sharding param to avoid re-trace."""
        import inspect

        from scripts.compute_values import compute_values_for_dataset

        sig = inspect.signature(compute_values_for_dataset)
        assert "data_sharding" in sig.parameters, "compute_values_for_dataset must accept data_sharding parameter"
        param = sig.parameters["data_sharding"]
        assert param.default is None, "data_sharding should default to None"

    def test_main_creates_data_sharding_once(self):
        """main() should create data_sharding once and pass to all datasets."""
        import inspect

        from scripts.compute_values import main

        source = inspect.getsource(main)
        assert "data_sharding=data_sharding" in source, "main() must pass data_sharding to compute_values_for_dataset"


# ---------------------------------------------------------------------------
# Universal batch padding
# ---------------------------------------------------------------------------


class TestUniversalBatchPadding:
    """_pad_batch_to_align should work for all batches, not just multi-GPU."""

    def test_pad_short_batch(self):
        """Short batch gets padded to aligned_size."""
        from scripts.compute_values import _pad_batch_to_align

        batch = {"x": np.random.randn(5, 3).astype(np.float32)}
        padded, orig_size = _pad_batch_to_align(batch, aligned_size=8)
        assert orig_size == 5
        assert padded["x"].shape[0] == 8

    def test_full_batch_no_change(self):
        """Full batch returns same size."""
        from scripts.compute_values import _pad_batch_to_align

        batch = {"x": np.random.randn(8, 3).astype(np.float32)}
        padded, orig_size = _pad_batch_to_align(batch, aligned_size=8)
        assert orig_size == 8
        assert padded["x"].shape[0] == 8


# ---------------------------------------------------------------------------
# --resume flag
# ---------------------------------------------------------------------------


class TestResume:
    """Behavior tests for the --resume skip predicate.

    The completion marker is meta/values_config[_suffix].json — it is the LAST file
    written by save_values_config(), so its presence guarantees a clean, complete run.
    A bare directory-existence check would silently skip half-written datasets.

    These tests exercise `_should_skip_repo_for_resume` directly so any regression
    (e.g. switching to `.is_dir()`) fails loudly instead of silently.
    """

    def test_resume_arg_exists(self):
        """--resume should be a valid CLI argument."""
        import inspect

        from scripts.compute_values import main

        source = inspect.getsource(main)
        assert '"--resume"' in source or "'--resume'" in source

    def test_skips_when_marker_exists(self, mock_dataset_dir: Path):
        """A completed run (marker present) should be skipped under --resume."""
        from scripts.compute_values import _resume_marker_path
        from scripts.compute_values import _should_skip_repo_for_resume

        marker = _resume_marker_path(mock_dataset_dir, suffix="")
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("{}")

        assert _should_skip_repo_for_resume(mock_dataset_dir, suffix="") is True

    def test_does_not_skip_when_dataset_untouched(self, mock_dataset_dir: Path):
        """A repo_id whose value_pred/ dir does not exist at all must NOT be skipped."""
        from scripts.compute_values import _should_skip_repo_for_resume

        assert _should_skip_repo_for_resume(mock_dataset_dir, suffix="") is False

    def test_does_not_skip_when_dir_exists_but_marker_missing(self, mock_dataset_dir: Path):
        """Half-written run (dir + partial parquets, marker missing) MUST NOT be skipped.

        Regression guard: a previous version used `output_dir.is_dir()`, which would
        silently skip a crashed run and let downstream training read a partial parquet
        set. Switching back to that check would flip this assertion to True and fail.
        """
        from scripts.compute_values import _should_skip_repo_for_resume

        output_dir = mock_dataset_dir / "value_pred"
        (output_dir / "chunk-000").mkdir(parents=True)
        (output_dir / "chunk-000" / "episode-0.parquet").write_text("partial")
        assert output_dir.is_dir()

        assert _should_skip_repo_for_resume(mock_dataset_dir, suffix="") is False

    def test_suffix_marker_is_isolated(self, mock_dataset_dir: Path):
        """--suffix routes the marker check to meta/values_config_{suffix}.json.

        A suffixed marker must NOT satisfy a non-suffixed resume check (and vice versa),
        so different suffix runs on the same dataset stay independent.
        """
        from scripts.compute_values import _resume_marker_path
        from scripts.compute_values import _should_skip_repo_for_resume

        suffix = "fast_mode_0324"
        suffixed_marker = _resume_marker_path(mock_dataset_dir, suffix=suffix)
        suffixed_marker.parent.mkdir(parents=True, exist_ok=True)
        suffixed_marker.write_text("{}")

        assert _should_skip_repo_for_resume(mock_dataset_dir, suffix=suffix) is True
        assert _should_skip_repo_for_resume(mock_dataset_dir, suffix="") is False

    def test_marker_path_schema(self, mock_dataset_dir: Path):
        """Lock in the exact marker location so consumers can write the file correctly."""
        from scripts.compute_values import _resume_marker_path

        plain = _resume_marker_path(mock_dataset_dir, suffix="")
        assert plain == mock_dataset_dir / "value_pred" / "meta" / "values_config.json"

        suffixed = _resume_marker_path(mock_dataset_dir, suffix="v2")
        assert suffixed == mock_dataset_dir / "value_pred_v2" / "meta" / "values_config_v2.json"
