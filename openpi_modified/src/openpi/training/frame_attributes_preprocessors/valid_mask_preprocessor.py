"""Valid mask preprocessor: PruneHeadTailStaticValidMaskPreprocessor/ HF dataset column"""

from __future__ import annotations

import dataclasses
import fnmatch
import logging

import numpy as np

from openpi.training.frame_attributes_preprocessors.base import EXTRA_SKIP_STATIC_WEIGHT
from openpi.training.frame_attributes_preprocessors.base import DatasetContext
from openpi.training.frame_attributes_preprocessors.base import FrameAttributeProcessor
from openpi.training.frame_attributes_preprocessors.base import FrameAttributes
from openpi.training.frame_attributes_preprocessors.utils import _get_states
from openpi.training.frame_attributes_preprocessors.utils import detect_gripper_events
from openpi.training.frame_attributes_preprocessors.utils import detect_static_boundaries
from openpi.training.frame_attributes_preprocessors.utils import get_intervention_intervals
from openpi.training.frame_attributes_preprocessors.utils import remove_reset_frames

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ValidMaskGroupParams:
    """Parameters for valid_mask computation for one group of repo_ids."""

    name: str = "default"
    match: list[str] = dataclasses.field(default_factory=list)
    head_margin_s: float = 0.3
    trailing_margin_s: float = 0.0
    prune_trailing: bool = True
    remove_reset_frames: bool = False
    use_human_intervention_mask: bool = False
    skip_static_processing: bool = False

    def __post_init__(self) -> None:
        if self.head_margin_s < 0:
            raise ValueError(f"head_margin_s must be non-negative, got {self.head_margin_s}")
        if self.trailing_margin_s < 0:
            raise ValueError(f"trailing_margin_s must be non-negative, got {self.trailing_margin_s}")


def _get_group_for_repo(repo_id: str, groups: list[ValidMaskGroupParams]) -> ValidMaskGroupParams:
    """Return the first matching group for *repo_id*."""
    default: ValidMaskGroupParams | None = None
    for group in groups:
        if not group.match:
            if default is None:
                default = group
            continue
        for pattern in group.match:
            if fnmatch.fnmatch(repo_id, pattern):
                return group
    if default is None:
        raise ValueError(
            f"No matching ValidMaskGroupParams found for repo_id='{repo_id}'. "
            f"Please add a group with matching pattern or an empty 'match' list as default."
        )
    return default


def _compute_episode_valid_mask(
    states: np.ndarray,
    is_static: np.ndarray,
    group: ValidMaskGroupParams,
    fps: int,
    human_intervention: np.ndarray | None,
    robot_type: str = "",
) -> np.ndarray:
    """Compute valid_mask for a single episode."""
    head_margin_frames = int(group.head_margin_s * fps)
    trailing_margin_frames = int(group.trailing_margin_s * fps)
    valid_mask = detect_static_boundaries(
        is_static, head_margin_frames, trailing_margin_frames, prune_trailing=group.prune_trailing
    )

    if group.remove_reset_frames:
        valid_mask &= remove_reset_frames(states, robot_type=robot_type)

    if group.use_human_intervention_mask and human_intervention is not None:
        valid_mask &= human_intervention
        for start, end in get_intervention_intervals(human_intervention):
            seg_is_static = is_static[start:end]
            seg_valid = detect_static_boundaries(seg_is_static, 0, 0)
            valid_mask[start:end] &= seg_valid

    return valid_mask


@dataclasses.dataclass
class PruneHeadTailStaticValidMaskPreprocessor(FrameAttributeProcessor):
    """Compute valid_mask from is_static (prune head/tail static). Requires attrs.is_static from VelocityBasedStaticDetector."""

    fps: int = 30
    groups: list[ValidMaskGroupParams] = dataclasses.field(default_factory=lambda: [ValidMaskGroupParams()])

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")

    def __call__(self, ctx: DatasetContext, attrs: FrameAttributes) -> None:
        if attrs.is_static is None:
            raise ValueError(
                "PruneHeadTailStaticValidMaskPreprocessor requires attrs.is_static; "
                "add VelocityBasedStaticDetector before PruneHeadTailStaticValidMaskPreprocessor in pipeline."
            )
        group = _get_group_for_repo(ctx.repo_id, self.groups)
        all_states = _get_states(ctx)
        has_intervention = group.use_human_intervention_mask and "is_human_intervention" in ctx.hf_dataset.features
        all_intervention = np.array(ctx.hf_dataset["is_human_intervention"], dtype=bool) if has_intervention else None

        total = len(all_states)
        if all_intervention is not None and len(all_intervention) != total:
            raise ValueError(
                f"[{ctx.repo_id}] human_intervention length {len(all_intervention)} " f"!= dataset length {total}"
            )
        valid_mask = np.zeros(total, dtype=bool)
        num_episodes = len(ctx.episode_data_index["from"])

        if group.skip_static_processing:
            ctx.extras[EXTRA_SKIP_STATIC_WEIGHT] = True
            valid_mask[:] = True
        else:
            for ep in range(num_episodes):
                s = int(ctx.episode_data_index["from"][ep])
                e = int(ctx.episode_data_index["to"][ep])
                ep_states = all_states[s:e]
                is_static = attrs.is_static[s:e]
                ep_intervention = all_intervention[s:e] if all_intervention is not None else None
                valid_mask[s:e] = _compute_episode_valid_mask(
                    ep_states,
                    is_static,
                    group,
                    self.fps,
                    ep_intervention,
                    robot_type=ctx.robot_type,
                )

        logger.info(
            "[%s] group=%s, valid=%d/%d (%.1f%%)",
            ctx.repo_id,
            group.name,
            valid_mask.sum(),
            total,
            valid_mask.sum() / max(total, 1) * 100,
        )
        attrs.valid_mask = valid_mask


@dataclasses.dataclass
class HfColumnIsValidPreprocessor(FrameAttributeProcessor):
    """Set or tighten attrs.valid_mask from ctx.hf_dataset[column_name].

    If the column is absent, skips with a debug log (other processors may still set mask).
    """

    column_name: str = "is_valid"

    @staticmethod
    def _hf_column_to_bool_array(hf_dataset, column_name: str) -> np.ndarray:
        """Read a full column as length-N bool (handles torch / numpy scalars in cells)."""
        col = hf_dataset[column_name]
        lst = col.to_list() if hasattr(col, "to_list") else list(col)
        out = np.empty(len(lst), dtype=bool)
        for i, v in enumerate(lst):
            if hasattr(v, "item"):
                out[i] = bool(np.asarray(v).item())
            else:
                out[i] = bool(v)
        return out

    def __call__(self, ctx: DatasetContext, attrs: FrameAttributes) -> None:
        if self.column_name not in ctx.hf_dataset.features:
            logger.debug(
                "[%s] HF dataset has no column %r; HfColumnIsValidPreprocessor skipped",
                ctx.repo_id,
                self.column_name,
            )
            return

        n = len(ctx.hf_dataset)
        col_mask = self._hf_column_to_bool_array(ctx.hf_dataset, self.column_name)
        if len(col_mask) != n:
            raise ValueError(
                f"[{ctx.repo_id}] column {self.column_name!r} length {len(col_mask)} " f"!= dataset length {n}"
            )

        if attrs.valid_mask is None:
            attrs.valid_mask = col_mask
        else:
            attrs.valid_mask &= col_mask

        logger.info(
            "[%s] Applied HF column %r: %d/%d frames valid",
            ctx.repo_id,
            self.column_name,
            int(col_mask.sum()),
            n,
        )


# ---------------------------------------------------------------------------
# Gripper-count-based valid mask
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
class GripperCountRule:
    """A single rule for invalidating frames based on gripper open/close count.

    Attributes:
        batch_contains: substring to match in repo_id (case-sensitive).
        gripper: which gripper to monitor, "left" or "right".
        event: event type to count, "open" or "close".
        count: the Nth occurrence of the event. Positive = count from front
            (1 = first, 2 = second, ...). Negative = count from back
            (-1 = last, -2 = second to last, ...).
        invalidate: which side of the Nth event to invalidate, "before" or "after".
        duration_s: if set, only invalidate frames within this many seconds of the
            event instead of all frames in the direction. Requires fps on the
            preprocessor.
    """

    batch_contains: str
    gripper: str  # "left" | "right"
    event: str  # "open" | "close"
    count: int  # positive = from front, negative = from back
    invalidate: str  # "before" | "after"
    duration_s: float | None = None

    def __post_init__(self) -> None:
        if self.gripper not in ("left", "right"):
            raise ValueError(f"gripper must be 'left' or 'right', got {self.gripper!r}")
        if self.event not in ("open", "close"):
            raise ValueError(f"event must be 'open' or 'close', got {self.event!r}")
        if self.count == 0:
            raise ValueError("count must be non-zero (positive = from front, negative = from back)")
        if self.invalidate not in ("before", "after"):
            raise ValueError(f"invalidate must be 'before' or 'after', got {self.invalidate!r}")
        if self.duration_s is not None and self.duration_s <= 0:
            raise ValueError(f"duration_s must be positive, got {self.duration_s}")


@dataclasses.dataclass
class GripperCountValidMaskPreprocessor(FrameAttributeProcessor):
    """Invalidate frames based on how many times a gripper has opened/closed.

    Rules are matched by substring in repo_id. Multiple rules can match and
    are applied cumulatively (AND). For episodes where no rule matches, all
    frames keep their current valid_mask.

    Supports 14-DOF (6-DOF arm) and 16-DOF (7-DOF arm) bimanual states.
    Gripper indices are resolved automatically from state dimensionality.

    Gripper convention: 0 = fully open, high value = fully closed.
    open_threshold: value below which gripper is considered open (default 0.5).
    close_threshold: value above which gripper is considered closed (default 3.0).
    close_threshold must be > open_threshold (hysteresis band).

    head_margin_s / tail_margin_s: seconds of frames at episode start/end that
    are always kept valid regardless of rules (default 0).

    Example usage in config::

        GripperCountValidMaskPreprocessor(
            open_threshold=0.5,
            close_threshold=3.0,
            fps=30,
            head_margin_s=0.3,
            rules=[
                # invalidate all frames before the 3rd right open
                GripperCountRule(
                    batch_contains="recover_2_n",
                    gripper="right", event="open", count=3, invalidate="before",
                ),
                # invalidate 2 seconds before the last right open
                GripperCountRule(
                    batch_contains="recover_2_n",
                    gripper="right", event="open", count=-1,
                    invalidate="before", duration_s=2.0,
                ),
            ],
        )
    """

    rules: list[GripperCountRule] = dataclasses.field(default_factory=list)
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

    def _matching_rules(self, repo_id: str) -> list[GripperCountRule]:
        return [r for r in self.rules if r.batch_contains in repo_id]

    def _apply_rule_to_episode(
        self,
        rule: GripperCountRule,
        ep_states: np.ndarray,
    ) -> np.ndarray:
        """Return a bool mask (True=valid) for one episode given one rule."""
        n = len(ep_states)
        if n == 0:
            return np.ones(0, dtype=bool)
        gripper_idx = _get_gripper_index(ep_states.shape[1], rule.gripper)
        gripper_values = ep_states[:, gripper_idx]
        open_frames, close_frames = detect_gripper_events(gripper_values, self.open_threshold, self.close_threshold)

        event_frames = open_frames if rule.event == "open" else close_frames

        # Resolve count: positive = from front (1-indexed), negative = from back (-1 = last)
        idx = rule.count - 1 if rule.count > 0 else len(event_frames) + rule.count

        if idx < 0 or idx >= len(event_frames):
            logger.warning(
                "Rule %s/%s/%s: only %d events found, need count=%d; keeping all frames valid",
                rule.gripper,
                rule.event,
                rule.batch_contains,
                len(event_frames),
                rule.count,
            )
            return np.ones(n, dtype=bool)

        pivot = event_frames[idx]
        mask = np.ones(n, dtype=bool)
        if rule.invalidate == "before" and rule.duration_s is not None:
            start = max(pivot - int(rule.duration_s * self.fps), 0)
            mask[start:pivot] = False
        elif rule.invalidate == "before":
            mask[:pivot] = False
        elif rule.duration_s is not None:  # "after" with duration
            end = min(pivot + 1 + int(rule.duration_s * self.fps), n)
            mask[pivot + 1 : end] = False
        else:  # "after" without duration
            mask[pivot + 1 :] = False
        return mask

    def __call__(self, ctx: DatasetContext, attrs: FrameAttributes) -> None:
        matched_rules = self._matching_rules(ctx.repo_id)
        if not matched_rules:
            return

        all_states = _get_states(ctx)
        total = len(all_states)
        state_dim = all_states.shape[1]
        if state_dim not in _GRIPPER_INDEX_BY_DIM:
            raise ValueError(
                f"[{ctx.repo_id}] GripperCountValidMaskPreprocessor supports state_dim "
                f"{sorted(_GRIPPER_INDEX_BY_DIM.keys())}, got {state_dim}"
            )
        mask = np.ones(total, dtype=bool)
        num_episodes = len(ctx.episode_data_index["from"])
        head_frames = int(self.head_margin_s * self.fps)
        tail_frames = int(self.tail_margin_s * self.fps)

        for ep in range(num_episodes):
            s = int(ctx.episode_data_index["from"][ep])
            e = int(ctx.episode_data_index["to"][ep])
            ep_states = all_states[s:e]
            ep_mask = np.ones(e - s, dtype=bool)
            for rule in matched_rules:
                ep_mask &= self._apply_rule_to_episode(rule, ep_states)
            # Restore head/tail margins to valid
            if head_frames > 0:
                ep_mask[:head_frames] = True
            if tail_frames > 0:
                ep_mask[-tail_frames:] = True
            mask[s:e] &= ep_mask

        if attrs.valid_mask is None:
            attrs.valid_mask = mask
        else:
            attrs.valid_mask &= mask

        logger.info(
            "[%s] GripperCount: %d rules matched, valid=%d/%d (%.1f%%)",
            ctx.repo_id,
            len(matched_rules),
            mask.sum(),
            total,
            mask.sum() / max(total, 1) * 100,
        )
