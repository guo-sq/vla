"""ValuePredictionPreprocessor 单元测试"""

import logging
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from openpi.training.frame_attributes_preprocessors import ValuePredictionPreprocessor


class MockDatasetContext:
    """模拟 DatasetContext"""

    def __init__(self, root: Path, repo_id: str, total_frames: int, episodes: dict | None = None):
        self.root = root
        self.repo_id = repo_id
        self.hf_dataset = Mock()
        self.hf_dataset.__len__ = Mock(return_value=total_frames)
        self.meta = Mock()
        self.meta.episodes = episodes if episodes is not None else {}


class MockFrameAttributes:
    """模拟 FrameAttributes"""

    def __init__(self):
        self.pred_value = None
        self.valid_mask = None


class TestValuePredictionPreprocessor:
    """ValuePredictionPreprocessor 单元测试"""

    # ========== 测试:正常流程 ==========

    def test_load_single_episode_success(self, tmp_path):
        """测试:成功加载单个 episode 的预测值"""
        # 准备测试数据
        value_pred_dir = tmp_path / "value_pred" / "chunk-000"
        value_pred_dir.mkdir(parents=True)

        # 创建 parquet 文件
        pred_df = pd.DataFrame({"pred_value": [0.5, 0.6, 0.7], "value_is_valid": [True, True, True]})
        pred_df.to_parquet(value_pred_dir / "episode_000000.parquet", index=False)

        # 创建预处理器和上下文
        preprocessor = ValuePredictionPreprocessor(
            value_pred_dir="value_pred",
            auto_discover_chunks=True,
            validate_episode_count=False,  # 禁用验证因为没有 meta
        )
        ctx = MockDatasetContext(tmp_path, "test_repo", 3, {0: {}})
        attrs = MockFrameAttributes()

        # 执行
        preprocessor(ctx, attrs)

        # 验证
        assert attrs.pred_value is not None
        assert len(attrs.pred_value) == 3
        np.testing.assert_array_almost_equal(attrs.pred_value, [0.5, 0.6, 0.7])
        np.testing.assert_array_equal(attrs.valid_mask, [True, True, True])

    def test_load_multiple_episodes_success(self, tmp_path):
        """测试:成功加载多个 episode"""
        value_pred_dir = tmp_path / "value_pred" / "chunk-000"
        value_pred_dir.mkdir(parents=True)

        # 创建 2 个 episode
        for i in range(2):
            pred_df = pd.DataFrame(
                {
                    "pred_value": [0.5 * (i + 1), 0.6 * (i + 1)],
                    "value_is_valid": [True, True],
                }
            )
            pred_df.to_parquet(value_pred_dir / f"episode_{i:06d}.parquet", index=False)

        preprocessor = ValuePredictionPreprocessor(
            value_pred_dir="value_pred",
            auto_discover_chunks=True,
            validate_episode_count=False,
        )
        ctx = MockDatasetContext(tmp_path, "test_repo", 4, {0: {}, 1: {}})
        attrs = MockFrameAttributes()

        preprocessor(ctx, attrs)

        assert len(attrs.pred_value) == 4
        np.testing.assert_array_almost_equal(attrs.pred_value, [0.5, 0.6, 1.0, 1.2])

    def test_load_with_valid_mask_intersection(self, tmp_path):
        """测试:valid_mask 与现有 mask 的交集操作"""
        value_pred_dir = tmp_path / "value_pred" / "chunk-000"
        value_pred_dir.mkdir(parents=True)

        pred_df = pd.DataFrame({"pred_value": [0.5, 0.6, 0.7], "value_is_valid": [True, False, True]})
        pred_df.to_parquet(value_pred_dir / "episode_000000.parquet", index=False)

        preprocessor = ValuePredictionPreprocessor(
            value_pred_dir="value_pred",
            auto_discover_chunks=True,
            validate_episode_count=False,
        )
        ctx = MockDatasetContext(tmp_path, "test_repo", 3, {0: {}})
        attrs = MockFrameAttributes()

        # 设置现有的 valid_mask
        attrs.valid_mask = np.array([True, True, False])

        preprocessor(ctx, attrs)

        # 应该是交集
        np.testing.assert_array_equal(attrs.valid_mask, [True, False, False])

    # ========== 测试:多 chunk 支持 ==========

    def test_auto_discover_multiple_chunks(self, tmp_path):
        """测试:自动发现多个 chunk 的 parquet 文件"""
        # 创建 2 个 chunk
        for chunk_id in ["chunk-000", "chunk-001"]:
            chunk_dir = tmp_path / "value_pred" / chunk_id
            chunk_dir.mkdir(parents=True)

            episode_id = 0 if chunk_id == "chunk-000" else 1
            pred_df = pd.DataFrame({"pred_value": [0.5 * (episode_id + 1)], "value_is_valid": [True]})
            pred_df.to_parquet(chunk_dir / f"episode_{episode_id:06d}.parquet", index=False)

        preprocessor = ValuePredictionPreprocessor(
            value_pred_dir="value_pred",
            auto_discover_chunks=True,
            validate_episode_count=False,
        )
        ctx = MockDatasetContext(tmp_path, "test_repo", 2, {0: {}, 1: {}})
        attrs = MockFrameAttributes()

        preprocessor(ctx, attrs)

        assert len(attrs.pred_value) == 2
        np.testing.assert_array_almost_equal(attrs.pred_value, [0.5, 1.0])

    def test_backward_compat_direct_parquet_files(self, tmp_path):
        """测试:向后兼容直接在 value_pred_dir 下有 parquet 文件"""
        value_pred_dir = tmp_path / "value_pred"
        value_pred_dir.mkdir(parents=True)

        # 直接在 value_pred 下创建 parquet
        pred_df = pd.DataFrame({"pred_value": [0.5, 0.6], "value_is_valid": [True, True]})
        pred_df.to_parquet(value_pred_dir / "episode_000000.parquet", index=False)

        preprocessor = ValuePredictionPreprocessor(
            value_pred_dir="value_pred",
            auto_discover_chunks=True,
            validate_episode_count=False,
        )
        ctx = MockDatasetContext(tmp_path, "test_repo", 2, {0: {}})
        attrs = MockFrameAttributes()

        preprocessor(ctx, attrs)

        assert len(attrs.pred_value) == 2
        np.testing.assert_array_almost_equal(attrs.pred_value, [0.5, 0.6])

    # ========== 测试:边界情况 ==========

    def test_missing_value_pred_dir(self, tmp_path, caplog):
        """测试:value_pred 目录不存在"""
        preprocessor = ValuePredictionPreprocessor(
            value_pred_dir="value_pred",
            auto_discover_chunks=True,
        )
        ctx = MockDatasetContext(tmp_path, "test_repo", 10, {0: {}})
        attrs = MockFrameAttributes()

        with caplog.at_level(logging.WARNING):
            preprocessor(ctx, attrs)

        # 应该记录警告
        assert "value_pred directory not found" in caplog.text
        # 不应该设置 pred_value
        assert attrs.pred_value is None

    def test_empty_parquet_files(self, tmp_path, caplog):
        """测试:目录存在但没有 parquet 文件"""
        value_pred_dir = tmp_path / "value_pred" / "chunk-000"
        value_pred_dir.mkdir(parents=True)

        preprocessor = ValuePredictionPreprocessor(
            value_pred_dir="value_pred",
            auto_discover_chunks=True,
        )
        ctx = MockDatasetContext(tmp_path, "test_repo", 10, {0: {}})
        attrs = MockFrameAttributes()

        with caplog.at_level(logging.WARNING):
            preprocessor(ctx, attrs)

        assert "No parquet files found" in caplog.text
        assert attrs.pred_value is None

    def test_missing_required_columns(self, tmp_path):
        """测试:parquet 文件缺少必需列"""
        value_pred_dir = tmp_path / "value_pred" / "chunk-000"
        value_pred_dir.mkdir(parents=True)

        # 创建缺少 pred_value 列的 parquet
        pred_df = pd.DataFrame({"wrong_column": [1, 2, 3]})
        pred_df.to_parquet(value_pred_dir / "episode_000000.parquet", index=False)

        preprocessor = ValuePredictionPreprocessor(
            value_pred_dir="value_pred",
            auto_discover_chunks=True,
            validate_episode_count=False,
        )
        ctx = MockDatasetContext(tmp_path, "test_repo", 3, {0: {}})
        attrs = MockFrameAttributes()

        # 应该抛出 KeyError
        with pytest.raises(KeyError, match="Missing required columns"):
            preprocessor(ctx, attrs)

    # ========== 测试:严重问题修复验证 ==========

    def test_episode_count_mismatch_detection(self, tmp_path):
        """测试:检测 episode 数量不匹配"""
        value_pred_dir = tmp_path / "value_pred" / "chunk-000"
        value_pred_dir.mkdir(parents=True)

        # 只创建 1 个 episode,但数据集有 2 个
        pred_df = pd.DataFrame({"pred_value": [0.5, 0.6], "value_is_valid": [True, True]})
        pred_df.to_parquet(value_pred_dir / "episode_000000.parquet", index=False)

        preprocessor = ValuePredictionPreprocessor(
            value_pred_dir="value_pred",
            auto_discover_chunks=True,
            validate_episode_count=True,  # 启用验证
        )
        ctx = MockDatasetContext(tmp_path, "test_repo", 4, {0: {}, 1: {}})
        attrs = MockFrameAttributes()

        # 应该抛出 ValueError
        with pytest.raises(ValueError, match="Missing parquet files for episodes"):
            preprocessor(ctx, attrs)

    def test_length_mismatch_detection(self, tmp_path):
        """测试:检测预测值长度与数据集长度不匹配"""
        value_pred_dir = tmp_path / "value_pred" / "chunk-000"
        value_pred_dir.mkdir(parents=True)

        # 创建 2 帧的预测值
        pred_df = pd.DataFrame({"pred_value": [0.5, 0.6], "value_is_valid": [True, True]})
        pred_df.to_parquet(value_pred_dir / "episode_000000.parquet", index=False)

        preprocessor = ValuePredictionPreprocessor(
            value_pred_dir="value_pred",
            auto_discover_chunks=True,
            validate_episode_count=False,
        )
        # 但数据集有 100 帧
        ctx = MockDatasetContext(tmp_path, "test_repo", 100, {0: {}})
        attrs = MockFrameAttributes()

        # 应该抛出 ValueError
        with pytest.raises(ValueError, match="pred_value length 2 != dataset length 100"):
            preprocessor(ctx, attrs)

    def test_duplicate_episode_detection(self, tmp_path):
        """测试:检测重复的 episode"""
        # 在两个 chunk 中创建相同的 episode
        for chunk_id in ["chunk-000", "chunk-001"]:
            chunk_dir = tmp_path / "value_pred" / chunk_id
            chunk_dir.mkdir(parents=True)

            pred_df = pd.DataFrame({"pred_value": [0.5], "value_is_valid": [True]})
            pred_df.to_parquet(chunk_dir / "episode_000000.parquet", index=False)

        preprocessor = ValuePredictionPreprocessor(
            value_pred_dir="value_pred",
            auto_discover_chunks=True,
            validate_episode_count=False,
        )
        ctx = MockDatasetContext(tmp_path, "test_repo", 2, {0: {}, 1: {}})
        attrs = MockFrameAttributes()

        # 应该抛出 ValueError
        with pytest.raises(ValueError, match="Duplicate episode"):
            preprocessor(ctx, attrs)

    def test_invalid_parquet_filename(self, tmp_path):
        """测试:无效的 parquet 文件名会被忽略(不匹配 episode_*.parquet 模式)"""
        value_pred_dir = tmp_path / "value_pred" / "chunk-000"
        value_pred_dir.mkdir(parents=True)

        # 创建无效文件名的 parquet(不会被 glob 匹配)
        pred_df = pd.DataFrame({"pred_value": [0.5], "value_is_valid": [True]})
        pred_df.to_parquet(value_pred_dir / "invalid_name.parquet", index=False)

        preprocessor = ValuePredictionPreprocessor(
            value_pred_dir="value_pred",
            auto_discover_chunks=True,
            validate_episode_count=False,
        )
        ctx = MockDatasetContext(tmp_path, "test_repo", 1, {0: {}})
        attrs = MockFrameAttributes()

        # 无效文件名不会被加载,所以应该没有数据
        preprocessor(ctx, attrs)

        # 因为没有有效文件,pred_value 应该是 None
        assert attrs.pred_value is None

    def test_invalid_episode_number_in_filename(self, tmp_path):
        """测试:文件名格式正确但 episode 编号无效"""
        value_pred_dir = tmp_path / "value_pred" / "chunk-000"
        value_pred_dir.mkdir(parents=True)

        # 创建一个能被 glob 匹配但编号解析可能出问题的文件
        # 例如 episode_abc.parquet
        pred_df = pd.DataFrame({"pred_value": [0.5], "value_is_valid": [True]})
        pred_df.to_parquet(value_pred_dir / "episode_abc.parquet", index=False)

        preprocessor = ValuePredictionPreprocessor(
            value_pred_dir="value_pred",
            auto_discover_chunks=True,
            validate_episode_count=False,
        )
        ctx = MockDatasetContext(tmp_path, "test_repo", 1, {0: {}})
        attrs = MockFrameAttributes()

        # 应该抛出 ValueError(无法解析 episode 编号)
        with pytest.raises(ValueError, match="Invalid parquet filename"):
            preprocessor(ctx, attrs)

    # ========== 测试:值范围验证 ==========

    def test_value_range_warning(self, tmp_path, caplog):
        """测试:值范围超出预期时发出警告"""
        value_pred_dir = tmp_path / "value_pred" / "chunk-000"
        value_pred_dir.mkdir(parents=True)

        # 创建超出范围的值
        pred_df = pd.DataFrame(
            {
                "pred_value": [0.5, 2.0, -1.5],  # 超出 [-1, 1]
                "value_is_valid": [True, True, True],
            }
        )
        pred_df.to_parquet(value_pred_dir / "episode_000000.parquet", index=False)

        preprocessor = ValuePredictionPreprocessor(
            value_pred_dir="value_pred",
            auto_discover_chunks=True,
            validate_episode_count=False,
        )
        ctx = MockDatasetContext(tmp_path, "test_repo", 3, {0: {}})
        attrs = MockFrameAttributes()

        with caplog.at_level(logging.WARNING):
            preprocessor(ctx, attrs)

        assert "pred_value range" in caplog.text
        assert "outside expected" in caplog.text

    # ========== 测试:配置选项 ==========

    def test_explicit_path_mode(self, tmp_path):
        """测试:显式路径模式(auto_discover_chunks=False)"""
        value_pred_dir = tmp_path / "value_pred" / "chunk-000"
        value_pred_dir.mkdir(parents=True)

        pred_df = pd.DataFrame({"pred_value": [0.5, 0.6], "value_is_valid": [True, True]})
        pred_df.to_parquet(value_pred_dir / "episode_000000.parquet", index=False)

        # 使用显式路径
        preprocessor = ValuePredictionPreprocessor(
            value_pred_dir="value_pred/chunk-000",
            auto_discover_chunks=False,
            validate_episode_count=False,
        )
        ctx = MockDatasetContext(tmp_path, "test_repo", 2, {0: {}})
        attrs = MockFrameAttributes()

        preprocessor(ctx, attrs)

        assert len(attrs.pred_value) == 2

    def test_validate_episode_count_disabled(self, tmp_path):
        """测试:禁用 episode 数量验证"""
        value_pred_dir = tmp_path / "value_pred" / "chunk-000"
        value_pred_dir.mkdir(parents=True)

        # 只创建 1 个 episode
        pred_df = pd.DataFrame({"pred_value": [0.5], "value_is_valid": [True]})
        pred_df.to_parquet(value_pred_dir / "episode_000000.parquet", index=False)

        preprocessor = ValuePredictionPreprocessor(
            value_pred_dir="value_pred",
            auto_discover_chunks=True,
            validate_episode_count=False,  # 禁用验证
        )
        # 数据集期望有 2 个 episode
        ctx = MockDatasetContext(tmp_path, "test_repo", 1, {0: {}, 1: {}})
        attrs = MockFrameAttributes()

        # 应该不抛出异常
        preprocessor(ctx, attrs)

        assert attrs.pred_value is not None


class TestValuePredictionPreprocessorIntegration:
    """集成测试(需要更复杂的 mock)"""

    def test_full_pipeline_with_valid_data(self, tmp_path):
        """测试:完整流程使用有效数据"""
        # 创建多 chunk 多 episode 的完整场景
        for chunk_idx in range(2):
            chunk_dir = tmp_path / "value_pred" / f"chunk-{chunk_idx:03d}"
            chunk_dir.mkdir(parents=True)

            for ep_in_chunk in range(3):
                ep_idx = chunk_idx * 3 + ep_in_chunk
                pred_df = pd.DataFrame(
                    {
                        "pred_value": np.random.uniform(-0.5, 0.5, 10).astype(np.float32),
                        "value_is_valid": [True] * 10,
                    }
                )
                pred_df.to_parquet(chunk_dir / f"episode_{ep_idx:06d}.parquet", index=False)

        preprocessor = ValuePredictionPreprocessor(
            value_pred_dir="value_pred",
            auto_discover_chunks=True,
            validate_episode_count=False,
        )

        episodes = {i: {} for i in range(6)}
        ctx = MockDatasetContext(tmp_path, "test_repo", 60, episodes)
        attrs = MockFrameAttributes()

        preprocessor(ctx, attrs)

        assert len(attrs.pred_value) == 60
        assert attrs.valid_mask is not None
        assert attrs.valid_mask.all()  # 所有帧都有效
