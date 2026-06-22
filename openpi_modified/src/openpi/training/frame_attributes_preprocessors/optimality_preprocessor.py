"""Optimality preprocessor: OptimalityProcessor."""

from __future__ import annotations

import dataclasses
import logging

import numpy as np

from openpi.training.frame_attributes_preprocessors.base import DatasetContext
from openpi.training.frame_attributes_preprocessors.base import FrameAttributeProcessor
from openpi.training.frame_attributes_preprocessors.base import FrameAttributes

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class OptimalityProcessor(FrameAttributeProcessor):
    """处理器用于计算每一帧的 optimality(优化性/质量标记)。

    默认情况下,optimality 为全 True,表示所有帧都是最优的。
    通过传入 bad_repo_id_with_weight 来标记对应数据集的帧为非最优。

    示例用例:
    - 标记人为干预帧为非最优
    - 根据状态偏离度标记低质量帧
    - 根据时间段标记特定帧为非最优
    """

    bad_repo_id_with_weight: list = dataclasses.field(default_factory=list)

    def __call__(self, ctx: DatasetContext, attrs: FrameAttributes) -> None:
        optimality = np.ones(ctx.hf_dataset.num_rows, dtype=bool)
        bad_repo_ids = [repo_id for repo_id, _ in self.bad_repo_id_with_weight]
        for bad_repo_id in bad_repo_ids:
            if bad_repo_id == ctx.repo_id:
                optimality[:] = False
                logger.info(f"[{ctx.repo_id}] Marked all frames as non-optimal.")
                break
        attrs.optimality = optimality
