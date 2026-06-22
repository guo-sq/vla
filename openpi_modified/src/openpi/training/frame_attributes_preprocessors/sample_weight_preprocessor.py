"""Sample weight preprocessor: StaticRatioSampleWeightPreprocessor."""

from __future__ import annotations

import dataclasses
import logging

import numpy as np

from openpi.training.frame_attributes_preprocessors.base import EXTRA_SKIP_STATIC_WEIGHT
from openpi.training.frame_attributes_preprocessors.base import DatasetContext
from openpi.training.frame_attributes_preprocessors.base import FrameAttributeProcessor
from openpi.training.frame_attributes_preprocessors.base import FrameAttributes
from openpi.training.frame_attributes_preprocessors.utils import _get_states
from openpi.training.frame_attributes_preprocessors.utils import compute_static_ratios
from openpi.training.frame_attributes_preprocessors.utils import detect_gripper_events

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class StaticRatioSampleWeightPreprocessor(FrameAttributeProcessor):
    """Compute sample_weight from is_static and valid_mask (filter inter-static). Requires both from prior processors."""

    ratio_thre: float = 2 / 3

    def __post_init__(self) -> None:
        if not 0 <= self.ratio_thre <= 1:
            raise ValueError(f"ratio_thre must be in [0, 1], got {self.ratio_thre}")

    def __call__(self, ctx: DatasetContext, attrs: FrameAttributes) -> None:
        if attrs.is_static is None:
            raise ValueError(
                "StaticRatioSampleWeightPreprocessor requires attrs.is_static; "
                "add VelocityBasedStaticDetector before StaticRatioSampleWeightPreprocessor."
            )
        if attrs.valid_mask is None:
            raise ValueError(
                "StaticRatioSampleWeightPreprocessor requires attrs.valid_mask; "
                "add PruneHeadTailStaticValidMaskPreprocessor before StaticRatioSampleWeightPreprocessor."
            )
        action_horizon = len(ctx.delta_indices["action"]) if ctx.delta_indices else 1
        total = len(attrs.valid_mask)
        sample_weight = np.ones(total, dtype=np.int32)
        num_episodes = len(ctx.episode_data_index["from"])
        skip_static_weight = ctx.extras.get(EXTRA_SKIP_STATIC_WEIGHT, False)

        for ep in range(num_episodes):
            s = int(ctx.episode_data_index["from"][ep])
            e = int(ctx.episode_data_index["to"][ep])
            valid_mask = attrs.valid_mask[s:e]
            if skip_static_weight:
                sample_weight[s:e] = valid_mask.astype(np.int32)
            else:
                is_static = attrs.is_static[s:e]
                ratios = compute_static_ratios(is_static, valid_mask, action_horizon)
                meet_mask = ratios <= self.ratio_thre
                sample_weight[s:e] = (valid_mask & meet_mask).astype(np.int32)

        attrs.sample_weight = sample_weight


_MAX_WEIGHT = 1000


@dataclasses.dataclass
class RepoNameMatchSampleWeightPreprocessor(FrameAttributeProcessor):
    """Match ``repo_id`` against ``substring`` (str or list, any needle contains).

    ``frame_skip``>1: matched repos only, keep every k-th eligible frame per episode (valid & weight>0).
    ``weight``>1: multiply after that. Zeros from prior preprocessors stay zero. No ``anyverse_dataset`` edits.
    """

    substring: str | list[str]
    weight: int = 1
    case_sensitive: bool = False
    frame_skip: int = 1
    _needles: tuple[str, ...] = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.substring, str):
            if not self.substring:
                raise ValueError("substring must be non-empty")
            needles = (self.substring,)
        else:
            if not self.substring:
                raise ValueError("substring list must be non-empty")
            if any(not isinstance(s, str) or not s for s in self.substring):
                raise ValueError("substring list entries must be non-empty strings")
            needles = tuple(self.substring)
        self._needles = needles
        if not 1 <= self.weight <= _MAX_WEIGHT:
            raise ValueError(f"weight must be in [1, {_MAX_WEIGHT}], got {self.weight}")
        if self.frame_skip < 1:
            raise ValueError(f"frame_skip must be >= 1, got {self.frame_skip}")

    @classmethod
    def _apply_episode_stride_downsample(
        cls,
        sample_weight: np.ndarray,
        valid_mask: np.ndarray | None,
        episode_data_index: dict,
        frame_skip: int,
    ) -> None:
        """Zero weights on all but every ``frame_skip``-th eligible frame per episode (time order)."""
        if frame_skip <= 1:
            return
        n = len(sample_weight)
        vm = valid_mask if valid_mask is not None else np.ones(n, dtype=bool)
        from_arr = episode_data_index["from"]
        to_arr = episode_data_index["to"]
        for ep in range(len(from_arr)):
            s = int(from_arr[ep])
            e = int(to_arr[ep])
            eligible_idx = np.where(vm[s:e] & (sample_weight[s:e] > 0))[0] + s
            m = len(eligible_idx)
            if m <= 1:
                continue
            drop = (np.arange(m, dtype=np.intp) % frame_skip) != 0
            sample_weight[eligible_idx[drop]] = 0

    def __call__(self, ctx: DatasetContext, attrs: FrameAttributes) -> None:
        repo_id = ctx.repo_id if self.case_sensitive else ctx.repo_id.lower()
        raw_needles = self._needles
        needles = raw_needles if self.case_sensitive else tuple(n.lower() for n in raw_needles)
        matched = any(n in repo_id for n in needles)
        total = len(ctx.hf_dataset)
        if matched:
            if attrs.sample_weight is None:
                attrs.sample_weight = np.ones(total, dtype=np.int32)
            else:
                attrs.sample_weight = attrs.sample_weight.astype(np.int32, copy=True)
            if self.frame_skip > 1:
                self._apply_episode_stride_downsample(
                    attrs.sample_weight,
                    attrs.valid_mask,
                    ctx.episode_data_index,
                    self.frame_skip,
                )
            if self.weight > 1:
                attrs.sample_weight = (attrs.sample_weight * self.weight).astype(np.int32)
        elif attrs.sample_weight is None:
            attrs.sample_weight = np.ones(total, dtype=np.int32)


# ---------------------------------------------------------------------------
# Gripper-count-based sample weight
# ---------------------------------------------------------------------------

_GRIPPER_INDEX_BY_DIM = {
    14: {"left": 6, "right": 13},  # 6-DOF arm: 6 joints + 1 gripper per arm
    16: {"left": 7, "right": 15},  # 7-DOF arm: 7 joints + 1 gripper per arm
}


def _get_gripper_index(state_dim: int, gripper: str) -> int:
    """Return the gripper column index for the given state dimensionality."""
    if state_dim not in _GRIPPER_INDEX_BY_DIM:
        raise ValueError(
            f"Unsupported state_dim={state_dim}. "
            f"Only {sorted(_GRIPPER_INDEX_BY_DIM.keys())}-DOF bimanual states are supported."
        )
    return _GRIPPER_INDEX_BY_DIM[state_dim][gripper]


@dataclasses.dataclass
class GripperCountSampleWeightRule:
    """A single rule for adjusting sample weight based on gripper open/close count.

    Attributes:
        batch_contains: substring to match in repo_id (case-sensitive).
        gripper: which gripper to monitor, "left" or "right".
        event: event type to count, "open" or "close".
        count: the Nth occurrence of the event. Positive = count from front
            (1 = first, 2 = second, ...). Negative = count from back
            (-1 = last, -2 = second to last, ...).
        region: which side of the Nth event to apply the weight to, "before" or "after".
        weight: multiplier to apply to sample_weight in the target region.
        duration_s: if set, only apply weight to frames within this many seconds of
            the event instead of all frames in the direction. Requires fps on the
            preprocessor.
    """

    batch_contains: str
    gripper: str  # "left" | "right"
    event: str  # "open" | "close"
    count: int  # positive = from front, negative = from back
    region: str  # "before" | "after"
    weight: int = 2
    duration_s: float | None = None

    def __post_init__(self) -> None:
        if self.gripper not in ("left", "right"):
            raise ValueError(f"gripper must be 'left' or 'right', got {self.gripper!r}")
        if self.event not in ("open", "close"):
            raise ValueError(f"event must be 'open' or 'close', got {self.event!r}")
        if self.count == 0:
            raise ValueError("count must be non-zero (positive = from front, negative = from back)")
        if self.region not in ("before", "after"):
            raise ValueError(f"region must be 'before' or 'after', got {self.region!r}")
        if not 1 <= self.weight <= _MAX_WEIGHT:
            raise ValueError(f"weight must be in [1, {_MAX_WEIGHT}], got {self.weight}")
        if self.duration_s is not None and self.duration_s <= 0:
            raise ValueError(f"duration_s must be positive, got {self.duration_s}")


@dataclasses.dataclass
class GripperCountSampleWeightPreprocessor(FrameAttributeProcessor):
    """Adjust sample_weight based on how many times a gripper has opened/closed.

    Rules are matched by substring in repo_id. Multiple rules can match and
    are applied cumulatively (multiplied). For episodes where no rule matches,
    sample_weight is left unchanged.

    Supports 14-DOF (6-DOF arm) and 16-DOF (7-DOF arm) bimanual states.
    Gripper indices are resolved automatically from state dimensionality.

    Gripper convention: 0 = fully open, high value = fully closed.
    open_threshold: value below which gripper is considered open (default 0.5).
    close_threshold: value above which gripper is considered closed (default 3.0).
    close_threshold must be > open_threshold (hysteresis band).

    head_margin_s / tail_margin_s: seconds of frames at episode start/end where
    weight is always kept at 1 (unmodified) regardless of rules (default 0).

    Example usage in config::

        GripperCountSampleWeightPreprocessor(
            open_threshold=0.5,
            close_threshold=3.0,
            fps=30,
            head_margin_s=0.3,
            rules=[
                # weight=2 for 2 seconds before the last right open
                GripperCountSampleWeightRule(
                    batch_contains="recover_2_n",
                    gripper="right", event="open", count=-1,
                    region="before", duration_s=2.0, weight=2,
                ),
                # weight=3 for all frames after the 3rd right open
                GripperCountSampleWeightRule(
                    batch_contains="recover_2_n",
                    gripper="right", event="open", count=3,
                    region="after", weight=3,
                ),
            ],
        )
    """

    rules: list[GripperCountSampleWeightRule] = dataclasses.field(default_factory=list)
    open_threshold: float = 0.5
    close_threshold: float = 3.0
    fps: int = 30
    head_margin_s: float = 0.0
    tail_margin_s: float = 0.0

    def __post_init__(self) -> None:
        if self.close_threshold <= self.open_threshold:
            raise ValueError(
                f"close_threshold ({self.close_threshold}) must be > " f"open_threshold ({self.open_threshold})"
            )
        if self.head_margin_s < 0:
            raise ValueError(f"head_margin_s must be non-negative, got {self.head_margin_s}")
        if self.tail_margin_s < 0:
            raise ValueError(f"tail_margin_s must be non-negative, got {self.tail_margin_s}")

    def _matching_rules(self, repo_id: str) -> list[GripperCountSampleWeightRule]:
        return [r for r in self.rules if r.batch_contains in repo_id]

    def _apply_rule_to_episode(
        self,
        rule: GripperCountSampleWeightRule,
        ep_states: np.ndarray,
    ) -> np.ndarray:
        """Return an int32 weight multiplier array for one episode given one rule."""
        n = len(ep_states)
        if n == 0:
            return np.ones(0, dtype=np.int32)
        gripper_idx = _get_gripper_index(ep_states.shape[1], rule.gripper)
        gripper_values = ep_states[:, gripper_idx]
        open_frames, close_frames = detect_gripper_events(gripper_values, self.open_threshold, self.close_threshold)

        event_frames = open_frames if rule.event == "open" else close_frames

        # Resolve count: positive = from front (1-indexed), negative = from back (-1 = last)
        idx = rule.count - 1 if rule.count > 0 else len(event_frames) + rule.count

        if idx < 0 or idx >= len(event_frames):
            logger.warning(
                "Rule %s/%s/%s: only %d events found, need count=%d; weight unchanged",
                rule.gripper,
                rule.event,
                rule.batch_contains,
                len(event_frames),
                rule.count,
            )
            return np.ones(n, dtype=np.int32)

        pivot = event_frames[idx]
        multiplier = np.ones(n, dtype=np.int32)
        if rule.region == "before" and rule.duration_s is not None:
            start = max(pivot - int(rule.duration_s * self.fps), 0)
            multiplier[start:pivot] = rule.weight
        elif rule.region == "before":
            multiplier[:pivot] = rule.weight
        elif rule.duration_s is not None:  # "after" with duration
            end = min(pivot + 1 + int(rule.duration_s * self.fps), n)
            multiplier[pivot + 1 : end] = rule.weight
        else:  # "after" without duration
            multiplier[pivot + 1 :] = rule.weight
        return multiplier

    def __call__(self, ctx: DatasetContext, attrs: FrameAttributes) -> None:
        matched_rules = self._matching_rules(ctx.repo_id)
        if not matched_rules:
            return

        all_states = _get_states(ctx)
        total = len(all_states)
        state_dim = all_states.shape[1]
        if state_dim not in _GRIPPER_INDEX_BY_DIM:
            raise ValueError(
                f"[{ctx.repo_id}] GripperCountSampleWeightPreprocessor supports state_dim "
                f"{sorted(_GRIPPER_INDEX_BY_DIM.keys())}, got {state_dim}"
            )
        combined = np.ones(total, dtype=np.int32)
        num_episodes = len(ctx.episode_data_index["from"])
        head_frames = int(self.head_margin_s * self.fps)
        tail_frames = int(self.tail_margin_s * self.fps)

        for ep in range(num_episodes):
            s = int(ctx.episode_data_index["from"][ep])
            e = int(ctx.episode_data_index["to"][ep])
            ep_states = all_states[s:e]
            ep_mult = np.ones(e - s, dtype=np.int32)
            for rule in matched_rules:
                ep_mult *= self._apply_rule_to_episode(rule, ep_states)
            # Restore head/tail margins to weight 1
            if head_frames > 0:
                ep_mult[:head_frames] = 1
            if tail_frames > 0:
                ep_mult[-tail_frames:] = 1
            combined[s:e] = ep_mult

        if attrs.sample_weight is None:
            attrs.sample_weight = combined
        else:
            attrs.sample_weight = (attrs.sample_weight * combined).astype(np.int32)

        logger.info(
            "[%s] GripperCountWeight: %d rules matched, mean_weight=%.2f",
            ctx.repo_id,
            len(matched_rules),
            combined.mean(),
        )


@dataclasses.dataclass
class FrameWeightByDimThresholdProcessor(FrameAttributeProcessor):
    """根据维度阈值区间调整 sample_weight"""

    dim_thresh_config: list[tuple[int, list[tuple[float, float]]]] | None = None
    repeat_weight: int = 2
    match_mode: str = "any"  # "any": OR across joints; "all": AND across joints

    def __post_init__(self) -> None:
        if self.match_mode not in {"any", "all"}:
            raise ValueError(f"match_mode must be 'any' or 'all', got {self.match_mode!r}")
        if self.repeat_weight < 0:
            raise ValueError(f"repeat_weight must be int >= 0, got {self.repeat_weight}")
        if self.dim_thresh_config is None:
            return
        if not isinstance(self.dim_thresh_config, list) or len(self.dim_thresh_config) == 0:
            raise ValueError("dim_thresh_config must be a non-empty list like [(dim_index, [(low, high), ...]), ...]")
        for i, (dim_index, intervals) in enumerate(self.dim_thresh_config):
            if not isinstance(dim_index, int) or dim_index < 0:
                raise ValueError(f"dim_thresh_config[{i}] dim_index must be non-negative int, got {dim_index!r}")
            if not isinstance(intervals, list) or len(intervals) == 0:
                raise ValueError(f"dim_thresh_config[{i}] intervals must be a non-empty list of (low, high)")
            for j, (low, high) in enumerate(intervals):
                if not np.isfinite(low) or not np.isfinite(high):
                    raise ValueError(
                        f"dim_thresh_config[{i}][{j}] low/high must be finite floats, got low={low}, high={high}"
                    )
                if low > high:
                    raise ValueError(f"dim_thresh_config[{i}][{j}] low={low} > high={high}")

    def __call__(self, ctx: DatasetContext, attrs: FrameAttributes) -> None:
        if self.dim_thresh_config is None:
            return

        states = _get_states(ctx)
        if states.ndim != 2:
            raise ValueError(f"[{ctx.repo_id}] observation.state must be 2D (N, dim), got shape={states.shape}")
        n, dim = states.shape

        repeat_mask: np.ndarray | None = None
        for i, (dim_index, intervals) in enumerate(self.dim_thresh_config):
            if dim_index >= dim:
                raise ValueError(
                    f"[{ctx.repo_id}] dim_thresh_config[{i}] dim_index={dim_index} out of range for state_dim={dim}"
                )
            x = states[:, dim_index]
            in_any = np.zeros(n, dtype=bool)
            for low, high in intervals:
                in_any |= (x >= low) & (x <= high)
            if repeat_mask is None:
                repeat_mask = in_any
            elif self.match_mode == "any":
                repeat_mask |= in_any
            else:
                repeat_mask &= in_any
        if repeat_mask is None:
            return

        sample_weight = np.ones(n, dtype=np.int32)
        if self.repeat_weight != 1:
            sample_weight[repeat_mask] = np.int32(self.repeat_weight)

        if attrs.sample_weight is not None:
            if len(attrs.sample_weight) != n:
                raise ValueError(f"[{ctx.repo_id}] attrs.sample_weight length {len(attrs.sample_weight)} != {n}")
            sample_weight = sample_weight * attrs.sample_weight

        logger.info(
            "[%s] Applied dim-threshold repeat weights: mode=%s, repeated=%d/%d, repeat_weight=%d, weight_range=[%d, %d]",
            ctx.repo_id,
            self.match_mode,
            int(repeat_mask.sum()),
            n,
            int(self.repeat_weight),
            int(sample_weight.min()) if n else 0,
            int(sample_weight.max()) if n else 0,
        )

        attrs.sample_weight = sample_weight
