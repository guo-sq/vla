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

"""测试 record_unified.py 配置和功能"""

import sys
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import fields


@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock 依赖模块"""
    mock_modules = {
        "piper_sdk": MagicMock(),
        "pinocchio": MagicMock(),
        "pygame": MagicMock(),
        "rerun": MagicMock(),
        "rerun_bindings": MagicMock(),
        "websockets": MagicMock(),
        "websockets.sync": MagicMock(),
        "websockets.sync.client": MagicMock(),
        "openpi_client": MagicMock(),
        "openpi_client.websocket_client_policy": MagicMock(),
        "lerobot.robots.piper": MagicMock(),
        "lerobot.robots.piper.piper": MagicMock(),
        "lerobot.robots.piper.piper_sdk_interface": MagicMock(),
        "lerobot.robots.bi_piper_follower": MagicMock(),
        "lerobot.robots.bi_piper_follower.bi_piper_follower": MagicMock(),
    }

    with patch.dict(sys.modules, mock_modules):
        yield


class TestRecordConfigParameters:
    """测试 RecordConfig 新增参数"""

    def test_auto_success_field_exists(self):
        """auto_success 字段应存在于 RecordConfig"""
        from lerobot.record_unified import RecordConfig

        field_names = [f.name for f in fields(RecordConfig)]
        assert "auto_success" in field_names

    def test_auto_success_default_false(self):
        """auto_success 默认值应为 False"""
        from lerobot.record_unified import RecordConfig

        field_dict = {f.name: f for f in fields(RecordConfig)}
        assert field_dict["auto_success"].default is False


class TestResolveEpisodeSuccess:
    """测试 resolve_episode_success 生产代码路径"""

    def test_auto_success_defaults_untagged_to_true(self):
        """auto_success=True 时，未标记的 episode 应为 success=True"""
        from lerobot.record_unified import resolve_episode_success

        task_success, metadata = resolve_episode_success(None, auto_success=True)
        assert task_success is True
        assert metadata == {"success": True}

    def test_auto_success_false_leaves_untagged_as_none(self):
        """auto_success=False 时，未标记的 episode 不应生成 metadata"""
        from lerobot.record_unified import resolve_episode_success

        task_success, metadata = resolve_episode_success(None, auto_success=False)
        assert task_success is None
        assert metadata is None

    def test_explicit_success_not_overridden(self):
        """Ctrl+→ 标记的成功不应被 auto_success=False 覆盖"""
        from lerobot.record_unified import resolve_episode_success

        task_success, metadata = resolve_episode_success(True, auto_success=False)
        assert task_success is True
        assert metadata == {"success": True}

    def test_explicit_failure_not_overridden(self):
        """Ctrl+↓ 标记的失败不应被 auto_success=True 覆盖"""
        from lerobot.record_unified import resolve_episode_success

        task_success, metadata = resolve_episode_success(False, auto_success=True)
        assert task_success is False
        assert metadata == {"success": False}
