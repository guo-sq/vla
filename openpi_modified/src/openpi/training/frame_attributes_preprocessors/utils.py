"""Shared utilities for frame attributes preprocessors.

Functions for velocity computation, static boundary detection, etc.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _get_states(ctx: Any) -> np.ndarray:
    """Get observation.state as (N, dim) array. Handles both HF Dataset and list."""
    col = ctx.hf_dataset["observation.state"]
    lst = col.to_list() if hasattr(col, "to_list") else list(col)
    return np.stack([np.asarray(s) for s in lst], axis=0)


def moving_average(arr: np.ndarray, h: int, axis: int = -1) -> np.ndarray:
    """Sliding-window average with half-window *h* (window width ``2h+1``)."""
    arr = np.asarray(arr)
    n = arr.shape[axis]
    a = np.moveaxis(arr, axis, -1)

    cum = np.cumsum(a, axis=-1)
    right = np.clip(np.arange(n) + h, 0, n - 1)
    left = np.clip(np.arange(n) - h, 0, n - 1)
    window_sum = cum[..., right] - np.where(left > 0, cum[..., left - 1], 0)
    window_size = right - left + 1
    avg = window_sum / window_size
    return np.moveaxis(avg, -1, axis)


def compute_smoothed_velocities(states: np.ndarray, fps: int, smoothing_h: int) -> np.ndarray:
    """Compute velocity with smoothing."""
    raw = np.gradient(states, 1.0 / fps, axis=0)
    return moving_average(raw, h=smoothing_h, axis=0)


def build_velocity_threshold(
    state_dim: int,
    joint_velocity_threshold: float = 0.1,
    gripper_velocity_threshold: float = 0.2,
) -> np.ndarray:
    """Build per-DOF velocity threshold for 14-dim bimanual state."""
    if state_dim == 14:
        return np.array(
            [
                *[joint_velocity_threshold] * 6,
                gripper_velocity_threshold,
                *[joint_velocity_threshold] * 6,
                gripper_velocity_threshold,
            ]
        )
    raise ValueError(
        f"Unknown state_dim={state_dim}. Only 14-dim bimanual state is supported. "
        f"Please configure velocity_threshold manually or extend this function."
    )


def detect_static_boundaries(
    is_static: np.ndarray,
    head_margin_frames: int,
    trailing_margin_frames: int,
    *,
    prune_trailing: bool = True,
) -> np.ndarray:
    """Mark leading / trailing static frames as invalid.

    Args:
        prune_trailing: If False, only leading static frames are pruned and
            trailing static frames are kept valid.
    """
    n = len(is_static)
    if n == 0:
        return np.zeros(0, dtype=bool)

    leading = 0
    while leading < n and is_static[leading]:
        leading += 1
    leading = max(leading - head_margin_frames, 0)

    valid = np.ones(n, dtype=bool)
    if leading > 0:
        valid[:leading] = False

    trailing = 0
    if prune_trailing:
        while trailing < n and is_static[n - 1 - trailing]:
            trailing += 1
        if trailing > 0:
            # Only prune when trailing static frames are actually detected;
            # margin reduces the count but always prune at least 1 static frame.
            trailing = max(trailing - trailing_margin_frames, 1)
            valid[-trailing:] = False

    if valid.sum() < n // 3:
        logger.warning(
            "Abnormal mask: only %d / %d frames valid (leading=%d, trailing=%d)",
            valid.sum(),
            n,
            leading,
            trailing,
        )
    return valid


def remove_reset_frames(
    states: np.ndarray,
    gripper_close_thre: float = 3.0,
    robot_type: str = "",
) -> np.ndarray:
    """Remove trailing reset frames where grippers are opened.

    Only supports bimanual robots with 14-DOF state (left arm 7-DOF + right arm 7-DOF).
    Gripper indices are hardcoded at [6, 13] for left and right grippers respectively.

    Args:
        states: (N, 14) state array
        gripper_close_thre: threshold for gripper open/close detection
        robot_type: robot type for validation, must be one of BIMANUAL_ROBOT_TYPES

    Returns:
        (N,) bool mask where True indicates valid frames
    """
    bimanual_robot_types = {"arxx5_bimanual", "bi_piper_follower"}

    if robot_type not in bimanual_robot_types:
        raise ValueError(
            f"remove_reset_frames only supports bimanual robots {bimanual_robot_types}, "
            f"got robot_type='{robot_type}'. Disable remove_reset_frames for this robot type."
        )
    if states.shape[1] != 14:
        raise ValueError(
            f"remove_reset_frames expects 14-dim state for bimanual robots, " f"got state_dim={states.shape[1]}"
        )
    left_gripper = states[:, 6]
    right_gripper = states[:, 13]
    open_indices = np.where((left_gripper > gripper_close_thre) | (right_gripper > gripper_close_thre))[0]
    if len(open_indices) == 0:
        return np.ones(len(states), dtype=bool)
    mask = np.ones(len(states), dtype=bool)
    mask[open_indices[-1] :] = False
    return mask


def get_intervention_intervals(
    human_intervention: np.ndarray,
) -> list[tuple[int, int]]:
    """Return (start, end) intervals of consecutive True runs."""
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for i in range(len(human_intervention)):
        if human_intervention[i] and (i == 0 or not human_intervention[i - 1]):
            start = i
        if start is not None and (not human_intervention[i] or i == len(human_intervention) - 1):
            end = i + 1 if human_intervention[i] else i
            intervals.append((start, end))
            start = None
    return intervals


def detect_gripper_events(
    gripper_values: np.ndarray,
    open_threshold: float,
    close_threshold: float,
) -> tuple[list[int], list[int]]:
    """Detect gripper open/close events using hysteresis.

    Gripper convention: 0 = fully open, high value = fully closed.
    An "open" event is recorded when the value drops below *open_threshold*.
    A "close" event is recorded when the value rises above *close_threshold*.
    Uses two thresholds (close_threshold > open_threshold) to avoid bouncing.

    Args:
        gripper_values: (N,) array of gripper state values (0=open, high=closed).
        open_threshold: value below which gripper is considered open.
        close_threshold: value above which gripper is considered closed.

    Returns:
        (open_frames, close_frames): lists of frame indices where each event occurs.
    """
    if close_threshold <= open_threshold:
        raise ValueError(f"close_threshold ({close_threshold}) must be > open_threshold ({open_threshold})")
    if len(gripper_values) == 0:
        return [], []
    open_frames: list[int] = []
    close_frames: list[int] = []
    # Determine initial state from first frame
    is_open = bool(gripper_values[0] < open_threshold)
    for i in range(len(gripper_values)):
        v = gripper_values[i]
        if not is_open and v < open_threshold:
            is_open = True
            open_frames.append(i)
        elif is_open and v > close_threshold:
            is_open = False
            close_frames.append(i)
    return open_frames, close_frames


def compute_static_ratios(
    is_static: np.ndarray,
    valid_mask: np.ndarray,
    action_horizon: int,
) -> np.ndarray:
    """For each frame, compute the fraction of static frames *among valid frames*
    in the future ``action_horizon`` window. Returns float array in [0, 1]."""
    n = len(is_static)
    static_and_valid = (is_static & valid_mask).astype(np.float64)
    valid_float = valid_mask.astype(np.float64)

    pad = np.zeros(action_horizon)
    cum_sv = np.cumsum(np.concatenate([static_and_valid, pad]))
    cum_v = np.cumsum(np.concatenate([valid_float, pad]))

    starts = np.arange(n)
    ends = np.minimum(starts + action_horizon, n)
    static_counts = cum_sv[ends] - cum_sv[starts]
    valid_counts = cum_v[ends] - cum_v[starts]

    zero_valid_count = np.sum(valid_counts == 0)
    if zero_valid_count > 0:
        logger.warning(
            "Found %d frames with zero valid frames in action_horizon window",
            zero_valid_count,
        )

    return np.where(valid_counts == 0, np.nan, static_counts / valid_counts)
