"""Unit tests for LeRobotRLDataset episode mapping and returns normalization.

These tests verify that LeRobotRLDataset correctly uses episode_mapping
and the new returns_norm_strategy configuration.
"""

from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest


@pytest.mark.rl
class TestLeRobotRLDatasetEpisodeMapping:
    """Tests for LeRobotRLDataset episode_mapping attribute."""

    def test_init_uses_episode_mapping_not_episode_from_to(self):
        """Test that __init__ uses episode_mapping instead of episode_from_to."""
        from openpi.training.rl_dataset import LeRobotRLDataset

        with patch.object(LeRobotRLDataset, "__init__", lambda self, *args, **kwargs: None):
            dataset = LeRobotRLDataset.__new__(LeRobotRLDataset)
            dataset.value_net_cfg = {
                "returns_norm_strategy": "per_task",
                "returns_norm_percentile": 1.0,
            }
            dataset.returns_norm_strategy = "per_task"
            dataset.returns_norm_percentile = 1.0
            dataset.meta = MagicMock()
            dataset.meta.episodes = [{"tasks": ["test_task"]} for _ in range(3)]

            dataset.episode_mapping = {0: (0, 3), 100: (3, 6), 200: (6, 9)}

            # Simulate per_task norm length computation
            task_to_lengths: dict[str, list[int]] = {}
            for i, (_start_idx, (valid_start, valid_end)) in enumerate(dataset.episode_mapping.items()):
                episode_length = valid_end - valid_start
                task = dataset.meta.episodes[i]["tasks"][0]
                task_to_lengths.setdefault(task, []).append(episode_length)

            task_to_norm_length = {task: max(lengths) for task, lengths in task_to_lengths.items()}

            assert "test_task" in task_to_norm_length
            assert task_to_norm_length["test_task"] == 3

    def test_per_task_multi_task_episode_mapping(self):
        """Test per_task strategy with multiple tasks uses episode_mapping correctly."""
        from openpi.training.rl_dataset import LeRobotRLDataset

        with patch.object(LeRobotRLDataset, "__init__", lambda self, *args, **kwargs: None):
            dataset = LeRobotRLDataset.__new__(LeRobotRLDataset)
            dataset.returns_norm_strategy = "per_task"
            dataset.returns_norm_percentile = 1.0
            dataset.meta = MagicMock()
            dataset.meta.episodes = [
                {"tasks": ["task_A"]},
                {"tasks": ["task_B"]},
                {"tasks": ["task_A"]},
            ]

            dataset.episode_mapping = {
                0: (0, 50),  # Episode 0: task_A, length 50
                100: (50, 120),  # Episode 1: task_B, length 70
                200: (120, 200),  # Episode 2: task_A, length 80
            }

            # Simulate per_task computation
            task_to_lengths: dict[str, list[int]] = {}
            for i, (_start_idx, (valid_start, valid_end)) in enumerate(dataset.episode_mapping.items()):
                episode_length = valid_end - valid_start
                task = dataset.meta.episodes[i]["tasks"][0]
                task_to_lengths.setdefault(task, []).append(episode_length)

            task_to_norm_length = {task: max(lengths) for task, lengths in task_to_lengths.items()}

            assert task_to_norm_length["task_A"] == 80  # max(50, 80)
            assert task_to_norm_length["task_B"] == 70

    def test_per_task_percentile(self):
        """Test per_task strategy with percentile < 1.0."""
        from openpi.training.rl_dataset import LeRobotRLDataset

        with patch.object(LeRobotRLDataset, "__init__", lambda self, *args, **kwargs: None):
            dataset = LeRobotRLDataset.__new__(LeRobotRLDataset)
            dataset.returns_norm_strategy = "per_task"
            dataset.returns_norm_percentile = 0.9
            dataset.meta = MagicMock()
            dataset.meta.episodes = [{"tasks": ["task_A"]} for _ in range(10)]

            # 10 episodes with lengths: 100, 200, ..., 1000
            episode_mapping = {}
            offset = 0
            for i in range(10):
                length = (i + 1) * 100
                episode_mapping[offset] = (offset, offset + length)
                offset += length + 50
            dataset.episode_mapping = episode_mapping

            # Compute with p90
            task_to_lengths: dict[str, list[int]] = {}
            for i, (_start_idx, (valid_start, valid_end)) in enumerate(dataset.episode_mapping.items()):
                episode_length = valid_end - valid_start
                task = dataset.meta.episodes[i]["tasks"][0]
                task_to_lengths.setdefault(task, []).append(episode_length)

            lengths = task_to_lengths["task_A"]
            p90_value = int(np.percentile(lengths, 90))

            assert p90_value < max(lengths)  # p90 < max
            assert p90_value > min(lengths)  # p90 > min


@pytest.mark.rl
class TestLeRobotRLDatasetNoDeadCode:
    """Tests to verify dead code has been removed."""

    def test_no_ori_episode_from_to_attribute_access(self):
        """Test that ori_episode_from_to is not accessed in __init__."""
        import inspect

        from openpi.training.rl_dataset import LeRobotRLDataset

        source = inspect.getsource(LeRobotRLDataset.__init__)

        lines = source.split("\n")
        active_code_lines = [line for line in lines if line.strip() and not line.strip().startswith("#")]

        for line in active_code_lines:
            if "ori_episode_from_to" in line and "self.ori_episode_from_to" in line:
                pytest.fail(f"Found deprecated 'self.ori_episode_from_to' access in active code: {line}")
            if (
                "episode_from_to" in line
                and "self.episode_from_to" in line
                and ("items()" in line or "keys()" in line or "values()" in line)
            ):
                pytest.fail(f"Found deprecated 'self.episode_from_to' access in active code: {line}")

    def test_no_old_param_names_in_init(self):
        """Test that old parameter names are not used in __init__."""
        import inspect

        from openpi.training.rl_dataset import LeRobotRLDataset

        source = inspect.getsource(LeRobotRLDataset.__init__)
        lines = source.split("\n")
        active_code_lines = [line for line in lines if line.strip() and not line.strip().startswith("#")]

        for line in active_code_lines:
            if "use_global_max_returns" in line:
                pytest.fail(f"Found deprecated 'use_global_max_returns' in active code: {line}")
            if "max_episode_length_for_normalization" in line:
                pytest.fail(f"Found deprecated 'max_episode_length_for_normalization' in active code: {line}")
