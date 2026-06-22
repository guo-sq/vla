"""Invalidate the first N frames of episode 0 to skip stale MP4 head frames.

Background: see lerobot_modified/docs/mp4_stale_first_frame.md. The stale-head-frame
issue (camera buffer backlog between warmup and recording) typically only manifests
in the first episode of each batch — the background capture thread stays running
after that, so subsequent episodes are unaffected.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np

from openpi.training.frame_attributes_preprocessors.base import DatasetContext
from openpi.training.frame_attributes_preprocessors.base import FrameAttributeProcessor
from openpi.training.frame_attributes_preprocessors.base import FrameAttributes

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class StaleHeadFramesValidMaskPreprocessor(FrameAttributeProcessor):
    """Set valid_mask=False for the first ``first_n`` frames of episode 0."""

    first_n: int = 3

    def __post_init__(self) -> None:
        if self.first_n < 0:
            raise ValueError(f"first_n must be non-negative, got {self.first_n}")

    def __call__(self, ctx: DatasetContext, attrs: FrameAttributes) -> None:
        if self.first_n == 0:
            return

        total = len(ctx.hf_dataset)
        num_episodes = len(ctx.episode_data_index["from"])
        if num_episodes == 0:
            return

        s = int(ctx.episode_data_index["from"][0])
        e = int(ctx.episode_data_index["to"][0])
        head_end = min(s + self.first_n, e)
        invalidated = head_end - s

        mask = np.ones(total, dtype=bool)
        mask[s:head_end] = False

        if attrs.valid_mask is None:
            attrs.valid_mask = mask
        else:
            attrs.valid_mask &= mask

        logger.info(
            "[%s] StaleHeadFrames(first_n=%d): invalidated frames [%d:%d) of episode 0 (%d/%d total)",
            ctx.repo_id,
            self.first_n,
            s,
            head_end,
            invalidated,
            total,
        )
