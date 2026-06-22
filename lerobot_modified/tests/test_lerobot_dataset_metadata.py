# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""测试 LeRobotDataset 的 success 元数据功能"""

import json
import sys
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock 依赖模块"""
    mock_modules = {
        "piper_sdk": MagicMock(),
        "pinocchio": MagicMock(),
        "lerobot.robots.piper": MagicMock(),
        "lerobot.robots.piper.piper": MagicMock(),
        "lerobot.robots.piper.piper_sdk_interface": MagicMock(),
        "lerobot.robots.bi_piper_follower": MagicMock(),
        "lerobot.robots.bi_piper_follower.bi_piper_follower": MagicMock(),
    }

    with patch.dict(sys.modules, mock_modules):
        yield


class TestEpisodeSuccessMetadata:
    """测试 episode success 元数据保存"""

    def test_save_episode_with_success_true(self, tmp_path):
        """保存 episode 时 success=True 应写入 episodes.jsonl"""
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        dataset_path = tmp_path / "dataset1"
        meta = LeRobotDatasetMetadata.create(
            repo_id="test/dataset",
            root=dataset_path,
            fps=30,
            features={},
        )

        meta.save_episode(
            episode_index=0,
            episode_length=100,
            episode_tasks=["test task"],
            episode_stats={},
            episode_metadata={"success": True},
        )

        episodes_file = dataset_path / "meta" / "episodes.jsonl"
        with open(episodes_file) as f:
            episode = json.loads(f.readline())

        assert episode["success"] is True

    def test_save_episode_with_success_false(self, tmp_path):
        """保存 episode 时 success=False 应写入 episodes.jsonl"""
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        dataset_path = tmp_path / "dataset2"
        meta = LeRobotDatasetMetadata.create(
            repo_id="test/dataset",
            root=dataset_path,
            fps=30,
            features={},
        )

        meta.save_episode(
            episode_index=0,
            episode_length=100,
            episode_tasks=["test task"],
            episode_stats={},
            episode_metadata={"success": False},
        )

        episodes_file = dataset_path / "meta" / "episodes.jsonl"
        with open(episodes_file) as f:
            episode = json.loads(f.readline())

        assert episode["success"] is False

    def test_save_episode_without_metadata(self, tmp_path):
        """不传 episode_metadata 时 success 字段应不存在"""
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        dataset_path = tmp_path / "dataset3"
        meta = LeRobotDatasetMetadata.create(
            repo_id="test/dataset",
            root=dataset_path,
            fps=30,
            features={},
        )

        meta.save_episode(
            episode_index=0,
            episode_length=100,
            episode_tasks=["test task"],
            episode_stats={},
        )

        episodes_file = dataset_path / "meta" / "episodes.jsonl"
        with open(episodes_file) as f:
            episode = json.loads(f.readline())

        assert "success" not in episode

    def test_save_episode_with_none_success(self, tmp_path):
        """传 episode_metadata 但 success=None 时 success 字段应不存在"""
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        dataset_path = tmp_path / "dataset4"
        meta = LeRobotDatasetMetadata.create(
            repo_id="test/dataset",
            root=dataset_path,
            fps=30,
            features={},
        )

        meta.save_episode(
            episode_index=0,
            episode_length=100,
            episode_tasks=["test task"],
            episode_stats={},
            episode_metadata={"success": None},
        )

        episodes_file = dataset_path / "meta" / "episodes.jsonl"
        with open(episodes_file) as f:
            episode = json.loads(f.readline())

        assert "success" not in episode


class TestReadOldDataset:
    """测试读取旧数据集的兼容性"""

    def test_read_episode_success_from_old_dataset(self, tmp_path):
        """读取旧数据集（无 success 字段）应返回 None"""
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        meta_dir = tmp_path / "meta"
        meta_dir.mkdir(parents=True)
        episodes_file = meta_dir / "episodes.jsonl"
        with open(episodes_file, "w") as f:
            f.write('{"episode_index": 0, "tasks": ["test"], "length": 100}\n')

        info_file = meta_dir / "info.json"
        with open(info_file, "w") as f:
            json.dump(
                {
                    "fps": 30,
                    "robot_type": "so100",
                    "codebase_version": "v2.1",
                    "total_episodes": 1,
                    "total_frames": 100,
                    "total_tasks": 1,
                    "total_chunks": 1,
                    "chunks_size": 1000,
                    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                    "video_path": None,
                    "features": {},
                    "total_videos": 0,
                    "splits": {"train": "0:1"},
                },
                f,
            )

        tasks_file = meta_dir / "tasks.jsonl"
        with open(tasks_file, "w") as f:
            f.write('{"task_index": 0, "task": "test"}\n')

        ep_stats_file = meta_dir / "episodes_stats.jsonl"
        with open(ep_stats_file, "w") as f:
            f.write('{"episode_index": 0, "stats": {}}\n')

        meta = LeRobotDatasetMetadata(repo_id="test/dataset", root=tmp_path)

        assert meta.episodes[0].get("success") is None
