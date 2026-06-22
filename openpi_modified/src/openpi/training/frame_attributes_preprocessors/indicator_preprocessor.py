"""Indicator preprocessor: IndicatorPreprocessor."""

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
class IndicatorPreprocessor(FrameAttributeProcessor):
    """从 parquet 文件加载预计算的 advantage indicator.

    期望的目录结构:
    indicators/chunk-000/episode_000000.parquet
    indicators/chunk-000/episode_000001.parquet
    indicators/chunk-001/episode_000000.parquet  (如果 auto_discover_chunks=True)
    ...

    每个 parquet 文件包含:
    - frame_index: int, 帧索引
    - indicator: bool, advantage indicator

    注意: 此 preprocessor 不修改 attrs.valid_mask,
    indicator 用于条件训练(advantage-conditioned policy), 不是数据过滤.
    """

    indicator_dir: str = "indicators"
    auto_discover_chunks: bool = True
    validate_episode_count: bool = True

    def _discover_parquet_files(self, root: Path) -> list[Path]:
        """自动发现所有 parquet 文件.

        Args:
            root: 数据集根目录

        Returns:
            排序后的 parquet 文件列表
        """
        indicator_root = root / self.indicator_dir

        if not indicator_root.exists():
            return []

        if self.auto_discover_chunks:
            parquet_files = []
            chunk_dirs = sorted(indicator_root.glob("chunk-*"))

            if not chunk_dirs:
                # 向后兼容: 检查是否直接在 indicator_dir 下有 parquet 文件
                direct_files = sorted(indicator_root.glob("episode_*.parquet"))
                if direct_files:
                    logger.info(
                        "[IndicatorPreprocessor] Found %d parquet files directly in %s",
                        len(direct_files),
                        indicator_root,
                    )
                    return direct_files

                logger.warning(
                    "[IndicatorPreprocessor] No chunk-* directories found in %s",
                    indicator_root,
                )
                return []

            for chunk_dir in chunk_dirs:
                chunk_files = sorted(chunk_dir.glob("episode_*.parquet"))
                parquet_files.extend(chunk_files)
                logger.debug(
                    "[IndicatorPreprocessor] Found %d parquet files in %s",
                    len(chunk_files),
                    chunk_dir.name,
                )

            return parquet_files

        explicit_path = indicator_root
        if explicit_path.exists():
            return sorted(explicit_path.glob("episode_*.parquet"))
        return []

    def __call__(self, ctx: DatasetContext, attrs: FrameAttributes) -> None:
        """加载 indicator 数据.

        Args:
            ctx: 数据集上下文
            attrs: 帧属性对象

        Raises:
            ValueError: 如果数据完整性验证失败
            KeyError: 如果 parquet 文件缺少必需列
        """
        indicator_root = Path(ctx.root) / self.indicator_dir

        if not indicator_root.exists():
            logger.warning(
                "[%s] indicator directory not found: %s, skipping indicator loading",
                ctx.repo_id,
                indicator_root,
            )
            return

        parquet_files = self._discover_parquet_files(Path(ctx.root))

        if not parquet_files:
            logger.warning(
                "[%s] No parquet files found in %s, skipping indicator loading",
                ctx.repo_id,
                indicator_root,
            )
            return

        logger.info(
            "[%s] Loading indicators from %d parquet files",
            ctx.repo_id,
            len(parquet_files),
        )

        episode_to_data: dict[int, dict] = {}

        for parquet_file in parquet_files:
            try:
                episode_num = int(parquet_file.stem.split("_")[1])
            except (IndexError, ValueError) as e:
                raise ValueError(
                    f"[{ctx.repo_id}] Invalid parquet filename: {parquet_file.name}\n"
                    f"Expected format: episode_XXXXXX.parquet\n"
                    f"Error: {e}"
                ) from e

            indicator_df = pd.read_parquet(parquet_file)

            required_columns = ["indicator"]
            missing_columns = [col for col in required_columns if col not in indicator_df.columns]
            if missing_columns:
                raise KeyError(
                    f"\n[Data Integrity Error]\n"
                    f"Repository: {ctx.repo_id}\n"
                    f"File: {parquet_file}\n"
                    f"Missing required columns: {missing_columns}\n"
                    f"Available columns: {indicator_df.columns.tolist()}\n"
                    f"Please regenerate using: python scripts/compute_advantages.py"
                )

            if episode_num in episode_to_data:
                raise ValueError(
                    f"[{ctx.repo_id}] Duplicate episode {episode_num} found!\n"
                    f"First occurrence: {episode_to_data[episode_num]['file']}\n"
                    f"Second occurrence: {parquet_file}\n"
                    f"Please check for duplicate parquet files across chunk directories."
                )

            episode_to_data[episode_num] = {
                "indicator": indicator_df["indicator"].to_numpy().astype(bool),
                "file": parquet_file,
            }

            logger.debug(
                "[%s] Loaded episode %d: %d frames from %s",
                ctx.repo_id,
                episode_num,
                len(indicator_df),
                parquet_file.name,
            )

        # 获取数据集的 episode 信息
        if hasattr(ctx, "meta") and hasattr(ctx.meta, "episodes") and ctx.meta is not None:
            dataset_episodes = set(ctx.meta.episodes.keys())
        else:
            dataset_episodes = set(episode_to_data.keys())
            logger.warning(
                "[%s] Dataset metadata not available, skipping episode count validation",
                ctx.repo_id,
            )
            self.validate_episode_count = False

        # 验证 episode 数量
        if self.validate_episode_count:
            parquet_episodes = set(episode_to_data.keys())

            missing_episodes = dataset_episodes - parquet_episodes
            if missing_episodes:
                raise ValueError(
                    f"\n[Data Integrity Error]\n"
                    f"Repository: {ctx.repo_id}\n"
                    f"Missing indicator parquet files for episodes: {sorted(missing_episodes)}\n"
                    f"Expected {len(dataset_episodes)} episodes, found {len(parquet_episodes)} parquet files.\n"
                    f"Please regenerate using: python scripts/compute_advantages.py"
                )

            extra_episodes = parquet_episodes - dataset_episodes
            if extra_episodes:
                logger.warning(
                    "[%s] Found %d extra indicator parquet files for episodes not in dataset: %s",
                    ctx.repo_id,
                    len(extra_episodes),
                    sorted(extra_episodes)[:10],
                )

        # 按 episode 顺序重组数据
        all_indicators = []

        if hasattr(ctx, "meta") and hasattr(ctx.meta, "episodes") and ctx.meta is not None:
            episode_order = sorted(ctx.meta.episodes.keys())
        else:
            episode_order = sorted(episode_to_data.keys())

        for ep_idx in episode_order:
            if ep_idx not in episode_to_data:
                if self.validate_episode_count:
                    raise ValueError(f"[{ctx.repo_id}] Missing indicator data for episode {ep_idx}")
                continue

            all_indicators.extend(episode_to_data[ep_idx]["indicator"])

        indicator = np.array(all_indicators, dtype=bool)

        # 验证长度
        total_frames = len(ctx.hf_dataset)
        if len(indicator) != total_frames:
            raise ValueError(
                f"\n[Data Integrity Error]\n"
                f"Repository: {ctx.repo_id}\n"
                f"indicator length {len(indicator)} != dataset length {total_frames}\n"
                f"Possible causes:\n"
                f"  1. Parquet files are corrupted or incomplete\n"
                f"  2. Episode count mismatch\n"
                f"  3. Dataset was modified after indicator generation\n"
                f"\nPlease regenerate using: python scripts/compute_advantages.py"
            )

        # 统计 positive 比例
        positive_ratio = indicator.sum() / len(indicator)
        logger.info(
            "[%s] Loaded %d indicators (%.2f%% positive)",
            ctx.repo_id,
            len(indicator),
            positive_ratio * 100,
        )

        attrs.indicator = indicator
