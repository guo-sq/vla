"""Temporal weight preprocessor: TemporalWeightProcessor."""

from __future__ import annotations

import dataclasses
import logging

import numpy as np

from openpi.training.frame_attributes_preprocessors.base import DatasetContext
from openpi.training.frame_attributes_preprocessors.base import FrameAttributeProcessor
from openpi.training.frame_attributes_preprocessors.base import FrameAttributes

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TemporalWeightProcessor(FrameAttributeProcessor):
    """根据时间配置调整 sample_weight。

    配置格式:
    temporal_weight_config = [
        (start_time, end_time, weight, fps),  # 时间段格式
        ...
    ]
    例如:
    temporal_weight_config = [
        (0.0, 5.0, 1, 30.0),   # 0-5秒,权重1
        (5.0, 10.0, 3, 30.0),  # 5-10秒,权重3
    ]

    时间段会按照以下规则应用:
    - 时间范围可能重叠,后面的配置会覆盖前面的
    - 未被任何时间段覆盖的帧使用权重1
    """

    repo_id_with_weights: list[tuple[str, list[tuple[float, float, int, float]]]] | None = None

    def __call__(self, ctx: DatasetContext, attrs: FrameAttributes) -> None:
        if self.repo_id_with_weights is None:
            return

        # 初始化权重数组(使用1表示默认权重)
        sample_weight = np.ones(ctx.hf_dataset.num_rows, dtype=np.int32)

        # 应用每个时间段配置
        for repo_id, temporal_weight_config in self.repo_id_with_weights:
            if repo_id != ctx.repo_id:
                continue
            logger.info(f"Applying temporal weights for repo_id={repo_id} with config={temporal_weight_config}")
            for start_time, end_time, weight, fps in temporal_weight_config:
                # 计算时间段的帧索引范围
                start_weight_frame = int(start_time * fps)
                end_weight_frame = int(end_time * fps)
                logger.info(f"Configuring weight={weight} for time range [{start_time}s, {end_time}s] ")
                for start, end in zip(ctx.episode_data_index["from"], ctx.episode_data_index["to"], strict=True):
                    frame_start = start + start_weight_frame
                    frame_end = start + end_weight_frame
                    valid_start = min(end, frame_start)
                    valid_end = min(end, frame_end)
                    sample_weight[valid_start:valid_end] = weight
                    logger.debug(
                        f"Set weight={weight} for frames [{valid_start}, {valid_end}) in episode [{start}, {end})"
                    )

        # 与已有的 sample_weight 取乘积(如果已有)
        if attrs.sample_weight is not None:
            sample_weight = sample_weight * attrs.sample_weight
            logger.info(
                f"[{ctx.repo_id}] Combined temporal weights with existing sample_weight: "
                f"min={sample_weight.min()}, max={sample_weight.max()}, mean={sample_weight.mean():.2f}"
            )

        logger.info(
            f"[{ctx.repo_id}] Applied temporal weights: {(sample_weight > 0).sum()} frames, "
            f"weight_range=[{sample_weight.min()}, {sample_weight.max()}]"
        )

        attrs.sample_weight = sample_weight
