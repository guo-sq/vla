"""Value prediction preprocessor: ValuePredictionPreprocessor."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from openpi.training.frame_attributes_preprocessors.base import DatasetContext
from openpi.training.frame_attributes_preprocessors.base import FrameAttributeProcessor
from openpi.training.frame_attributes_preprocessors.base import FrameAttributes

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ValuePredictionPreprocessor(FrameAttributeProcessor):
    """从 parquet 文件加载值网络预测的 value。

    期望的目录结构:
    value_pred/chunk-000/episode_000000.parquet
    value_pred/chunk-000/episode_000001.parquet
    value_pred/chunk-001/episode_000000.parquet  (如果 auto_discover_chunks=True)
    ...

    每个 parquet 文件包含:
    - pred_value: float32,预测的值
    - value_is_valid: bool,是否有效

    配置选项:
    - value_pred_dir: parquet 文件的根目录(默认:"value_pred")
    - auto_discover_chunks: 是否自动发现所有 chunk 子目录(默认:True)
    - validate_episode_count: 是否验证 episode 数量匹配(默认:True)
    """

    value_pred_dir: str = "value_pred"
    auto_discover_chunks: bool = True
    validate_episode_count: bool = True

    def _discover_parquet_files(self, root: Path) -> list[Path]:
        """自动发现所有 parquet 文件。

        Args:
            root: 数据集根目录

        Returns:
            排序后的 parquet 文件列表
        """
        value_pred_root = root / self.value_pred_dir

        if not value_pred_root.exists():
            return []

        if self.auto_discover_chunks:
            # 自动发现所有 chunk-* 子目录
            parquet_files = []
            chunk_dirs = sorted(value_pred_root.glob("chunk-*"))

            if not chunk_dirs:
                # 向后兼容:检查是否直接在 value_pred_dir 下有 parquet 文件
                direct_files = sorted(value_pred_root.glob("episode_*.parquet"))
                if direct_files:
                    logger.info(
                        "[ValuePredictionPreprocessor] Found %d parquet files directly in %s",
                        len(direct_files),
                        value_pred_root,
                    )
                    return direct_files

                logger.warning(
                    "[ValuePredictionPreprocessor] No chunk-* directories found in %s",
                    value_pred_root,
                )
                return []

            for chunk_dir in chunk_dirs:
                chunk_files = sorted(chunk_dir.glob("episode_*.parquet"))
                parquet_files.extend(chunk_files)
                logger.debug(
                    "[ValuePredictionPreprocessor] Found %d parquet files in %s",
                    len(chunk_files),
                    chunk_dir.name,
                )

            return parquet_files
        # 显式路径模式(向后兼容)
        # 如果 value_pred_dir 包含 chunk-*,直接使用
        explicit_path = value_pred_root
        if explicit_path.exists():
            return sorted(explicit_path.glob("episode_*.parquet"))
        return []

    def __call__(self, ctx: DatasetContext, attrs: FrameAttributes) -> None:
        """加载值预测数据。

        Args:
            ctx: 数据集上下文
            attrs: 帧属性对象

        Raises:
            ValueError: 如果数据完整性验证失败
            KeyError: 如果 parquet 文件缺少必需列
        """
        value_pred_root = Path(ctx.root) / self.value_pred_dir

        if not value_pred_root.exists():
            logger.warning(
                "[%s] value_pred directory not found: %s, skipping value prediction loading",
                ctx.repo_id,
                value_pred_root,
            )
            return

        # 发现所有 parquet 文件
        parquet_files = self._discover_parquet_files(Path(ctx.root))

        if not parquet_files:
            logger.warning(
                "[%s] No parquet files found in %s, skipping value prediction loading",
                ctx.repo_id,
                value_pred_root,
            )
            return

        logger.info(
            "[%s] Loading value predictions from %d parquet files",
            ctx.repo_id,
            len(parquet_files),
        )

        # 读取每个 parquet 文件,按 episode 编号存储
        episode_to_data: dict[int, dict] = {}

        for parquet_file in parquet_files:
            # 提取 episode 编号
            try:
                episode_num = int(parquet_file.stem.split("_")[1])
            except (IndexError, ValueError) as e:
                raise ValueError(
                    f"[{ctx.repo_id}] Invalid parquet filename: {parquet_file.name}\n"
                    f"Expected format: episode_XXXXXX.parquet\n"
                    f"Error: {e}"
                ) from e

            # 读取 parquet
            pred_df = pd.read_parquet(parquet_file)

            # 验证必需列
            required_columns = ["pred_value", "value_is_valid"]
            missing_columns = [col for col in required_columns if col not in pred_df.columns]
            if missing_columns:
                raise KeyError(
                    f"\n[Data Integrity Error]\n"
                    f"Repository: {ctx.repo_id}\n"
                    f"File: {parquet_file}\n"
                    f"Missing required columns: {missing_columns}\n"
                    f"Available columns: {pred_df.columns.tolist()}\n"
                    f"Please regenerate parquet files using:\n"
                    f"  bash tools/value_model_tools/auto_test_value_model.sh label"
                )

            # 检查重复的 episode
            if episode_num in episode_to_data:
                raise ValueError(
                    f"[{ctx.repo_id}] Duplicate episode {episode_num} found!\n"
                    f"First occurrence: {episode_to_data[episode_num]['file']}\n"
                    f"Second occurrence: {parquet_file}\n"
                    f"Please check for duplicate parquet files across chunk directories."
                )

            episode_to_data[episode_num] = {
                "pred_value": pred_df["pred_value"].to_numpy(),
                "value_is_valid": pred_df["value_is_valid"].to_numpy(),
                "file": parquet_file,
            }

            logger.debug(
                "[%s] Loaded episode %d: %d frames from %s",
                ctx.repo_id,
                episode_num,
                len(pred_df),
                parquet_file.name,
            )

        # 获取数据集的 episode 信息
        if hasattr(ctx, "meta") and hasattr(ctx.meta, "episodes"):
            dataset_episodes = set(ctx.meta.episodes.keys())
        else:
            # 如果没有 meta 信息,使用 parquet 文件中的 episode 编号
            dataset_episodes = set(episode_to_data.keys())
            logger.warning(
                "[%s] Dataset metadata not available, skipping episode count validation",
                ctx.repo_id,
            )
            self.validate_episode_count = False

        # 验证 episode 数量
        if self.validate_episode_count:
            parquet_episodes = set(episode_to_data.keys())

            # 检查缺失的 episodes
            missing_episodes = dataset_episodes - parquet_episodes
            if missing_episodes:
                raise ValueError(
                    f"\n[Data Integrity Error]\n"
                    f"Repository: {ctx.repo_id}\n"
                    f"Missing parquet files for episodes: {sorted(missing_episodes)}\n"
                    f"Expected {len(dataset_episodes)} episodes, found {len(parquet_episodes)} parquet files.\n"
                    f"Please regenerate parquet files using:\n"
                    f"  bash tools/value_model_tools/auto_test_value_model.sh label"
                )

            # 检查多余的 parquet 文件
            extra_episodes = parquet_episodes - dataset_episodes
            if extra_episodes:
                logger.warning(
                    "[%s] Found %d extra parquet files for episodes not in dataset: %s",
                    ctx.repo_id,
                    len(extra_episodes),
                    sorted(extra_episodes)[:10],  # 只显示前 10 个
                )

        # 按 episode 顺序重组数据
        all_pred_values = []
        all_value_is_valid = []

        # 使用数据集的 episode 顺序
        if hasattr(ctx, "meta") and hasattr(ctx.meta, "episodes"):
            episode_order = sorted(ctx.meta.episodes.keys())
        else:
            episode_order = sorted(episode_to_data.keys())

        for ep_idx in episode_order:
            if ep_idx not in episode_to_data:
                if self.validate_episode_count:
                    # 这不应该发生,因为前面已经检查过了
                    raise ValueError(f"[{ctx.repo_id}] Missing data for episode {ep_idx}")
                # 跳过缺失的 episode
                continue

            all_pred_values.extend(episode_to_data[ep_idx]["pred_value"])
            all_value_is_valid.extend(episode_to_data[ep_idx]["value_is_valid"])

        # 转换为 numpy 数组
        pred_value = np.array(all_pred_values, dtype=np.float32)
        value_is_valid = np.array(all_value_is_valid, dtype=bool)

        # 验证长度
        total_frames = len(ctx.hf_dataset)
        if len(pred_value) != total_frames:
            raise ValueError(
                f"\n[Data Integrity Error]\n"
                f"Repository: {ctx.repo_id}\n"
                f"pred_value length {len(pred_value)} != dataset length {total_frames}\n"
                f"Possible causes:\n"
                f"  1. Parquet files are corrupted or incomplete\n"
                f"  2. Episode count mismatch\n"
                f"  3. Dataset was modified after parquet generation\n"
                f"\nPlease regenerate parquet files using:\n"
                f"  bash tools/value_model_tools/auto_test_value_model.sh label"
            )

        # 验证 pred_value 范围
        valid_pred_values = pred_value[value_is_valid]
        if len(valid_pred_values) > 0:
            min_val = valid_pred_values.min()
            max_val = valid_pred_values.max()
            if min_val < -1.0 or max_val > 1.0:
                logger.warning(
                    "[%s] pred_value range [%.4f, %.4f] is outside expected [-1.0, 1.0]",
                    ctx.repo_id,
                    min_val,
                    max_val,
                )

        # 统计有效帧比例
        valid_ratio = value_is_valid.sum() / len(value_is_valid)
        logger.info(
            "[%s] Loaded %d value predictions (%.2f%% valid frames)",
            ctx.repo_id,
            len(pred_value),
            valid_ratio * 100,
        )

        attrs.pred_value = pred_value
        # 使用交集操作,保留其他 preprocessor 设置的 valid_mask
        if attrs.valid_mask is None:
            attrs.valid_mask = value_is_valid
        else:
            attrs.valid_mask &= value_is_valid
