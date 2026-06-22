"""Base classes for frame attributes pipeline.

Processors compute is_static, valid_mask, sample_weight, segment_id for AnyverseDataset at init time.
Pipeline runs a list of atomic FrameAttributeProcessor instances sequentially.
"""

from __future__ import annotations

import abc
import dataclasses
from enum import IntEnum
import logging
from typing import Any

import numpy as np


class EpisodeBoundary(IntEnum):
    """Episode boundary confirmation status.

    START_CONFIRMED/END_CONFIRMED/BOTH_CONFIRMED use bit flags (bit 0 = start,
    bit 1 = end). The two UNCONFIRMED_* values carry an orthogonal signal —
    neither endpoint is trusted, but the visual terminal state biases the
    constant GT either toward failure (-1) or toward success (0):

    - UNCONFIRMED_NEGATIVE_END: FAILURE_FP (builder-failure) — terminal state
      typically NOT at target → constant -1.
    - UNCONFIRMED_POSITIVE_END: FAILURE_FN (destroyer-failure) — terminal state
      typically AT target ('pull failed') → constant 0.

    Both are heuristic approximations (see PR description C1/C5).
    """

    UNCONFIRMED_NEGATIVE_END = 0  # FAILURE_FP heuristic GT=-1
    START_CONFIRMED = 1  # 0b01 - only start GT known (reserved for future use)
    END_CONFIRMED = 2  # 0b10 - only end GT known (self-play success)
    BOTH_CONFIRMED = 3  # 0b11 - both start and end GT known (human demo)
    UNCONFIRMED_POSITIVE_END = 4  # FAILURE_FN heuristic GT=0


logger = logging.getLogger(__name__)

# Constants for ctx.extras keys
EXTRA_SKIP_STATIC_WEIGHT = "skip_static_weight"
EXTRA_EPISODE_PROMPT_MAP = "episode_prompt_map"

# Sentinel for episodes whose task metadata is missing/unreadable.
# Shared between ValueReturnsPreprocessor and LeRobotRLDataset so per_task
# norm_length lookups agree on the fallback key.
UNKNOWN_TASK = "unknown"


def get_episode_task(meta: Any, ep_key: int) -> str:
    """Extract task string from LeRobotDatasetMeta by real episode key.

    Returns UNKNOWN_TASK if meta is missing, the key is absent, or the tasks
    field is empty. ep_key must be a real meta.episodes key, NOT an enumerate
    index over filtered episode_mapping.

    Supports both dict-like meta.episodes (real LeRobotDatasetMeta, keyed by
    episode index) and list-like (used by mock tests and legacy fallbacks).
    """
    if meta is None or not hasattr(meta, "episodes"):
        return UNKNOWN_TASK
    episodes = meta.episodes
    try:
        if isinstance(episodes, dict):
            ep_meta = episodes.get(ep_key)
        elif isinstance(episodes, list):
            ep_meta = episodes[ep_key] if 0 <= ep_key < len(episodes) else None
        else:
            return UNKNOWN_TASK
    except (IndexError, KeyError, AttributeError, TypeError):
        return UNKNOWN_TASK
    if ep_meta is None:
        return UNKNOWN_TASK
    try:
        raw = ep_meta.get("tasks", "") if hasattr(ep_meta, "get") else ep_meta["tasks"]
    except (KeyError, TypeError):
        return UNKNOWN_TASK
    if isinstance(raw, list):
        return raw[0] if raw else UNKNOWN_TASK
    if raw:
        return str(raw)
    return UNKNOWN_TASK


@dataclasses.dataclass
class FrameAttributes:
    """预处理器计算的逐帧属性. None 字段由 finalize() 填充为默认值并校验."""

    is_static: np.ndarray | None = None  # bool (N,) 可选, 无默认值
    valid_mask: np.ndarray | None = None  # bool (N,) 默认: 全 True
    sample_weight: np.ndarray | None = None  # int32 (N,) 默认: 全 1
    segment_id: np.ndarray | None = None  # int64 (N,) 默认: 全 0
    pred_value: np.ndarray | None = None  # float32 (N,) 可选, 值网络预测的 value, 只有部分预处理器会写入
    indicator: np.ndarray | None = None  # bool (N,) 可选, advantage indicator, 由 IndicatorPreprocessor 写入
    advantage: np.ndarray | None = None  # float32 (N,) 可选, 原始 advantage 值, 调试用
    optimality: np.ndarray | None = None  # bool (N,) 默认: 全 True
    is_negative_episode: np.ndarray | None = None  # bool (N,) 可选, 该帧是否属于负样本 episode
    episode_boundary: np.ndarray | None = None  # int8 (N,) 可选, EpisodeBoundary enum values

    def finalize(self, n: int, repo_id: str = "") -> FrameAttributes:
        """填充 None 字段为默认值并校验, 返回新实例."""
        filled = FrameAttributes(
            is_static=self.is_static,
            valid_mask=self.valid_mask if self.valid_mask is not None else np.ones(n, dtype=bool),
            sample_weight=self.sample_weight if self.sample_weight is not None else np.ones(n, dtype=np.int32),
            segment_id=self.segment_id if self.segment_id is not None else np.zeros(n, dtype=np.int64),
            pred_value=self.pred_value,  # Keep None, don't fill default value
            indicator=self.indicator,  # Keep None, don't fill default value
            advantage=self.advantage,  # Keep None, don't fill default value
            optimality=self.optimality if self.optimality is not None else np.ones(n, dtype=bool),
            is_negative_episode=self.is_negative_episode,  # Keep None
            episode_boundary=self.episode_boundary,  # Keep None
        )
        prefix = f"[{repo_id}] " if repo_id else ""
        for name, arr in [
            ("valid_mask", filled.valid_mask),
            ("sample_weight", filled.sample_weight),
            ("segment_id", filled.segment_id),
            ("optimality", filled.optimality),  # 新增
        ]:
            if arr is None:
                raise ValueError(f"{prefix}FrameAttributes.{name} is None after fill")
            if len(arr) != n:
                raise ValueError(f"{prefix}FrameAttributes.{name} length {len(arr)} != {n}")
        assert filled.valid_mask is not None
        assert filled.sample_weight is not None
        assert filled.segment_id is not None
        assert filled.optimality is not None
        if not np.issubdtype(filled.valid_mask.dtype, np.bool_):
            raise ValueError(f"{prefix}FrameAttributes.valid_mask dtype {filled.valid_mask.dtype} != bool")
        if not np.issubdtype(filled.sample_weight.dtype, np.integer):
            raise ValueError(f"{prefix}FrameAttributes.sample_weight dtype {filled.sample_weight.dtype} not integer")
        if not np.issubdtype(filled.segment_id.dtype, np.integer):
            raise ValueError(f"{prefix}FrameAttributes.segment_id dtype {filled.segment_id.dtype} not integer")
        if not np.issubdtype(filled.optimality.dtype, np.bool_):
            raise ValueError(f"{prefix}FrameAttributes.optimality dtype {filled.optimality.dtype} != bool")
        if filled.is_static is not None and len(filled.is_static) != n:
            raise ValueError(f"{prefix}FrameAttributes.is_static length {len(filled.is_static)} != {n}")
        if filled.is_static is not None and not np.issubdtype(filled.is_static.dtype, np.bool_):
            raise ValueError(f"{prefix}FrameAttributes.is_static dtype {filled.is_static.dtype} != bool")
        if filled.pred_value is not None:
            if len(filled.pred_value) != n:
                raise ValueError(f"{prefix}FrameAttributes.pred_value length {len(filled.pred_value)} != {n}")
            if not np.issubdtype(filled.pred_value.dtype, np.floating):
                raise ValueError(f"{prefix}FrameAttributes.pred_value dtype {filled.pred_value.dtype} not float")
        if filled.indicator is not None:
            if len(filled.indicator) != n:
                raise ValueError(f"{prefix}FrameAttributes.indicator length {len(filled.indicator)} != {n}")
            if not np.issubdtype(filled.indicator.dtype, np.bool_):
                raise ValueError(f"{prefix}FrameAttributes.indicator dtype {filled.indicator.dtype} != bool")
        if filled.advantage is not None:
            if len(filled.advantage) != n:
                raise ValueError(f"{prefix}FrameAttributes.advantage length {len(filled.advantage)} != {n}")
            if not np.issubdtype(filled.advantage.dtype, np.floating):
                raise ValueError(f"{prefix}FrameAttributes.advantage dtype {filled.advantage.dtype} not float")
        if filled.is_negative_episode is not None:
            if len(filled.is_negative_episode) != n:
                raise ValueError(
                    f"{prefix}FrameAttributes.is_negative_episode length {len(filled.is_negative_episode)} != {n}"
                )
            if not np.issubdtype(filled.is_negative_episode.dtype, np.bool_):
                raise ValueError(
                    f"{prefix}FrameAttributes.is_negative_episode dtype {filled.is_negative_episode.dtype} != bool"
                )
        if filled.episode_boundary is not None:
            if len(filled.episode_boundary) != n:
                raise ValueError(
                    f"{prefix}FrameAttributes.episode_boundary length {len(filled.episode_boundary)} != {n}"
                )
            if not np.issubdtype(filled.episode_boundary.dtype, np.integer):
                raise ValueError(
                    f"{prefix}FrameAttributes.episode_boundary dtype {filled.episode_boundary.dtype} not integer"
                )
        valid_ratio = filled.valid_mask.sum() / max(n, 1) * 100
        if valid_ratio < 1e-6:
            logger.warning("%svalid_mask: 0%% frames valid (all invalid)", prefix)
        elif valid_ratio < 10:
            logger.warning("%svalid_mask: only %.1f%% frames valid", prefix, valid_ratio)
        return filled


@dataclasses.dataclass
class DatasetContext:
    """Context passed to processors; avoids direct dependency on AnyverseDataset.

    extras: mutable dict for processor-to-processor intermediate data.
    """

    repo_id: str
    hf_dataset: Any  # HF Dataset
    episode_data_index: dict  # {"from": Tensor, "to": Tensor}
    meta: Any  # LeRobotDatasetMeta
    delta_indices: dict | None
    robot_type: str = ""  # Robot type for validation (e.g., "arxx5_bimanual", "bi_piper_follower")
    root: str | None = None  # Dataset root directory for loading external files
    extras: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class FrameAttributeProcessor(abc.ABC):
    """Atomic processor unit for the frame attributes pipeline."""

    @abc.abstractmethod
    def __call__(self, ctx: DatasetContext, attrs: FrameAttributes) -> None:
        """Process ctx and attrs in-place. May read from ctx.extras, attrs; write to attrs, ctx.extras."""
        ...


def run_frame_attr_preprocessor_pipeline(
    processors: list[FrameAttributeProcessor],
    ctx: DatasetContext,
) -> FrameAttributes:
    """Run pipeline of processors and return finalized FrameAttributes."""
    attrs = FrameAttributes()
    for proc in processors:
        proc(ctx, attrs)
    return attrs.finalize(len(ctx.hf_dataset), ctx.repo_id)


def run_pipeline_single_episode(
    processors: list[FrameAttributeProcessor],
    states: np.ndarray,
    repo_id: str,
    action_horizon: int,
    human_intervention: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Run pipeline on a single episode. For tools (e.g. visualize_masks).

    Returns (valid_mask, sample_weight, is_static).
    """
    from datasets import Dataset

    n = len(states)
    data: dict = {"observation.state": states.tolist()}
    if human_intervention is not None:
        data["is_human_intervention"] = human_intervention.tolist()
    hf = Dataset.from_dict(data)
    ctx = DatasetContext(
        repo_id=repo_id,
        hf_dataset=hf,
        episode_data_index={"from": [0], "to": [n]},
        meta=None,
        delta_indices={"action": list(range(action_horizon))},
    )
    attrs = run_frame_attr_preprocessor_pipeline(processors, ctx)
    assert attrs.valid_mask is not None
    assert attrs.sample_weight is not None
    return (
        attrs.valid_mask,
        attrs.sample_weight,
        attrs.is_static,
    )
