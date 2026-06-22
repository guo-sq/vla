"""Velocity-based static detector: VelocityBasedStaticDetector."""

from __future__ import annotations

import dataclasses

import numpy as np

from openpi.training.frame_attributes_preprocessors.base import (
    DatasetContext,
    FrameAttributeProcessor,
    FrameAttributes,
)
from openpi.training.frame_attributes_preprocessors.utils import (
    _get_states,
    build_velocity_threshold,
    compute_smoothed_velocities,
)


def _compute_is_static_episode(
    states: np.ndarray,
    fps: int,
    smoothing_half_window: int,
    joint_velocity_threshold: float,
    gripper_velocity_threshold: float,
) -> np.ndarray:
    """Compute per-frame is_static from velocity for a single episode."""
    state_dim = states.shape[1] if states.ndim == 2 else 14
    threshold = build_velocity_threshold(
        state_dim,
        joint_velocity_threshold,
        gripper_velocity_threshold,
    )
    velocities = compute_smoothed_velocities(
        states, fps, smoothing_half_window
    )
    return np.all(np.abs(velocities) < threshold, axis=1)


@dataclasses.dataclass
class VelocityBasedStaticDetector(FrameAttributeProcessor):
    """Compute is_static from observation.state velocity. Per-episode to avoid cross-episode gradient."""

    fps: int = 30
    joint_velocity_threshold: float = 0.1
    gripper_velocity_threshold: float = 0.2
    smoothing_half_window: int = 2

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.joint_velocity_threshold < 0:
            raise ValueError(
                f"joint_velocity_threshold must be non-negative, got {self.joint_velocity_threshold}"
            )
        if self.gripper_velocity_threshold < 0:
            raise ValueError(
                f"gripper_velocity_threshold must be non-negative, got {self.gripper_velocity_threshold}"
            )
        if self.smoothing_half_window < 0:
            raise ValueError(
                f"smoothing_half_window must be non-negative, got {self.smoothing_half_window}"
            )

    def __call__(self, ctx: DatasetContext, attrs: FrameAttributes) -> None:
        all_states = _get_states(ctx)
        total = len(all_states)
        is_static = np.zeros(total, dtype=bool)
        num_episodes = len(ctx.episode_data_index["from"])

        for ep in range(num_episodes):
            s = int(ctx.episode_data_index["from"][ep])
            e = int(ctx.episode_data_index["to"][ep])
            ep_states = all_states[s:e]
            is_static[s:e] = _compute_is_static_episode(
                ep_states,
                self.fps,
                self.smoothing_half_window,
                self.joint_velocity_threshold,
                self.gripper_velocity_threshold,
            )

        attrs.is_static = is_static
