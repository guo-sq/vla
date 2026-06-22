"""Unit tests for MultiRLAnyverseDataset.

Tests that MultiRLAnyverseDataset properly merges per-task norm lengths
from sub-datasets when using the per_task normalization strategy.
"""

from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

from openpi.training.rl_dataset import LeRobotRLDataset
from openpi.training.rl_dataset import MultiRLAnyverseDataset


def _make_mock_data_config(
    repo_ids: list[str],
    value_net_cfg: dict | None = None,
):
    """Create a mock data config for testing."""
    mock = Mock()
    mock.root_dir = "/fake/root"
    mock.repo_id = repo_ids
    mock.action_sequence_keys = ["action"]
    mock.episode = None
    mock.episode_fail = None
    mock.dataset_length = None
    mock.robot_align_info = Mock()
    mock.align_dim = None
    mock.unify_action_space = False
    mock.base_config = None
    mock.value_net_cfg = value_net_cfg or {
        "returns_norm_strategy": "fixed",
        "returns_norm_length": 1000,
    }
    return mock


def _make_mock_subdataset(
    length: int = 100,
    episode_mapping: dict | None = None,
    task_to_norm_length: dict | None = None,
    meta_episodes: list | None = None,
):
    """Create a mock sub-dataset."""
    mock = Mock()
    mock.__len__ = Mock(return_value=length)
    mock.num_frames = length
    mock.num_episodes = len(episode_mapping) if episode_mapping else 5
    mock.episode_mapping = episode_mapping or {}
    mock.task_to_norm_length = task_to_norm_length or {}
    mock.meta = Mock()
    mock.meta.episodes = meta_episodes or []
    return mock


@pytest.mark.rl
class TestMultiRLAnyverseDatasetPerTask:
    """Tests for per_task strategy merging across sub-datasets."""

    def test_per_task_merges_across_datasets(self):
        """Per-task strategy merges episode lengths from all sub-datasets."""
        mock_config = _make_mock_data_config(
            repo_ids=["ds1", "ds2"],
            value_net_cfg={
                "returns_norm_strategy": "per_task",
                "returns_norm_percentile": 1.0,
            },
        )

        # ds1: task_A episodes with lengths 100, 200
        mock_ds1 = _make_mock_subdataset(
            length=300,
            episode_mapping={0: (0, 100), 100: (100, 300)},
            meta_episodes=[{"tasks": ["task_A"]}, {"tasks": ["task_A"]}],
        )

        # ds2: task_A episode with length 500
        mock_ds2 = _make_mock_subdataset(
            length=500,
            episode_mapping={0: (0, 500)},
            meta_episodes=[{"tasks": ["task_A"]}],
        )

        with patch("openpi.training.rl_dataset.LeRobotRLDataset", side_effect=[mock_ds1, mock_ds2]):
            dataset = MultiRLAnyverseDataset(
                data_config=mock_config,
                action_horizon=1,
                download_videos=False,
            )

        # task_A max across both datasets: max(100, 200, 500) = 500
        assert dataset.task_to_norm_length["task_A"] == 500

    def test_per_task_distributes_back_to_subdatasets(self):
        """Merged norm lengths are distributed back to each sub-dataset."""
        mock_config = _make_mock_data_config(
            repo_ids=["ds1", "ds2"],
            value_net_cfg={
                "returns_norm_strategy": "per_task",
                "returns_norm_percentile": 1.0,
            },
        )

        mock_ds1 = _make_mock_subdataset(
            length=100,
            episode_mapping={0: (0, 100)},
            meta_episodes=[{"tasks": ["task_A"]}],
        )
        mock_ds2 = _make_mock_subdataset(
            length=500,
            episode_mapping={0: (0, 500)},
            meta_episodes=[{"tasks": ["task_A"]}],
        )

        with patch("openpi.training.rl_dataset.LeRobotRLDataset", side_effect=[mock_ds1, mock_ds2]):
            MultiRLAnyverseDataset(
                data_config=mock_config,
                action_horizon=1,
                download_videos=False,
            )

        # Both sub-datasets should have the merged norm length
        assert mock_ds1.task_to_norm_length["task_A"] == 500
        assert mock_ds2.task_to_norm_length["task_A"] == 500


@pytest.mark.rl
class TestMultiRLAnyverseDatasetFixedAndPerEpisode:
    """Tests for fixed and per_episode strategies (no cross-dataset merging needed)."""

    def test_fixed_strategy_no_merging(self):
        """Fixed strategy doesn't need cross-dataset merging."""
        mock_config = _make_mock_data_config(
            repo_ids=["ds1"],
            value_net_cfg={
                "returns_norm_strategy": "fixed",
                "returns_norm_length": 3000,
            },
        )

        mock_ds1 = _make_mock_subdataset(length=100)

        with patch("openpi.training.rl_dataset.LeRobotRLDataset", return_value=mock_ds1):
            dataset = MultiRLAnyverseDataset(
                data_config=mock_config,
                action_horizon=1,
                download_videos=False,
            )

        assert not hasattr(dataset, "task_to_norm_length") or not dataset.task_to_norm_length

    def test_per_episode_strategy_no_merging(self):
        """Per-episode strategy doesn't need cross-dataset merging."""
        mock_config = _make_mock_data_config(
            repo_ids=["ds1"],
            value_net_cfg={
                "returns_norm_strategy": "per_episode",
            },
        )

        mock_ds1 = _make_mock_subdataset(length=100)

        with patch("openpi.training.rl_dataset.LeRobotRLDataset", return_value=mock_ds1):
            dataset = MultiRLAnyverseDataset(
                data_config=mock_config,
                action_horizon=1,
                download_videos=False,
            )

        assert not hasattr(dataset, "task_to_norm_length") or not dataset.task_to_norm_length


@pytest.mark.rl
class TestCalcEpisodeMetaKey:
    """Regression for the C1/C2 index-drift bug.

    Before the fix, `__init__`/`_precompute_returns` iterated with `enumerate`
    over the filtered `episode_mapping` and used the positional index against
    `self.meta.episodes`, so any episode dropped by `exclude_failures` shifted
    every later task lookup by one slot. `calc_episode` now records an explicit
    `ep_from → meta_key` map that is pulled from `list(meta.episodes.keys())`,
    so subsequent lookups hit the real meta key no matter how many episodes
    were dropped.
    """

    def _make_rl_dataset_shell(
        self,
        *,
        valid_frame_indices: torch.Tensor,
        episode_data_index: dict,
        meta_episodes: dict,
    ) -> LeRobotRLDataset:
        """Build a bare LeRobotRLDataset without running __init__ (heavy)."""
        ds = LeRobotRLDataset.__new__(LeRobotRLDataset)
        ds._valid_frame_indices = valid_frame_indices  # noqa: SLF001
        ds.episode_data_index = episode_data_index
        ds.meta = Mock()
        ds.meta.episodes = meta_episodes
        return ds

    def test_calc_episode_records_real_meta_key_for_contiguous_keys(self):
        """With no exclusions and 0..N-1 meta keys, ep_from maps straight to i."""
        ds = self._make_rl_dataset_shell(
            valid_frame_indices=torch.arange(110),
            episode_data_index={
                "from": torch.tensor([0, 40, 70]),
                "to": torch.tensor([40, 70, 110]),
            },
            meta_episodes={
                0: {"tasks": ["hang"]},
                1: {"tasks": ["tie"]},
                2: {"tasks": ["take_off"]},
            },
        )

        episode_mapping = ds.calc_episode()

        assert list(episode_mapping.keys()) == [0, 40, 70]
        assert ds._episode_from_to_meta_key == {0: 0, 40: 1, 70: 2}  # noqa: SLF001

    def test_calc_episode_skips_excluded_episodes_and_keeps_real_meta_key(self):
        """Middle episode dropped by exclude_failures: remaining entries still map to real meta keys."""
        # Episode 1 (frames 40..69) dropped because its frames are absent from valid_frame_indices.
        valid = torch.cat([torch.arange(0, 40), torch.arange(70, 110)])
        ds = self._make_rl_dataset_shell(
            valid_frame_indices=valid,
            episode_data_index={
                "from": torch.tensor([0, 40, 70]),
                "to": torch.tensor([40, 70, 110]),
            },
            meta_episodes={
                0: {"tasks": ["hang"]},
                1: {"tasks": ["tie"]},  # dropped
                2: {"tasks": ["take_off"]},
            },
        )

        episode_mapping = ds.calc_episode()

        # Only episodes 0 and 2 survive.
        assert set(episode_mapping.keys()) == {0, 70}
        # And ep_from=70 must map to meta key 2 — NOT to 1 (the old enumerate bug).
        assert ds._episode_from_to_meta_key == {0: 0, 70: 2}  # noqa: SLF001

    def test_calc_episode_resolves_non_contiguous_meta_keys(self):
        """meta.episodes keys aren't required to start at 0 or be contiguous."""
        ds = self._make_rl_dataset_shell(
            valid_frame_indices=torch.arange(110),
            episode_data_index={
                "from": torch.tensor([0, 40, 70]),
                "to": torch.tensor([40, 70, 110]),
            },
            # Real LeRobotDatasetMeta can expose arbitrary episode_index as keys.
            meta_episodes={
                12: {"tasks": ["hang"]},
                17: {"tasks": ["tie"]},
                29: {"tasks": ["take_off"]},
            },
        )

        ds.calc_episode()

        assert ds._episode_from_to_meta_key == {0: 12, 40: 17, 70: 29}  # noqa: SLF001
