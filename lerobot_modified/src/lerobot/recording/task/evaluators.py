"""Pure evaluators for home position, value score, timeout, and sub-tasks.

All evaluators are side-effect-free: they take state and return results.
No logging, no announcements, no state transitions.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from lerobot.recording.task.task_spec import ResetConfig, SuccessCondition
from lerobot.recording.utils.logging import debug_print


class HomeEvaluator:
    """Evaluates whether the robot is at its home (reset) position.

    Combines pose check (all joints within threshold of reference) and
    speed check (all joint velocities below speed_threshold).
    """

    def __init__(
        self,
        reset_config: ResetConfig,
        robot_state_keys: list[str],
        robot_speed_keys: list[str] | None = None,
    ):
        self._ref = np.array(reset_config.reference_pose, dtype=np.float64)
        self._threshold = reset_config.threshold
        self._speed_threshold = reset_config.speed_threshold
        self._state_keys = robot_state_keys
        self._speed_keys = robot_speed_keys or []

    def _extract_state(self, observation: dict) -> np.ndarray | None:
        """Extract joint state vector from observation dict.

        Tries aggregated key first (observation.state), then falls back
        to individual joint keys (e.g., left_joint_1.pos).
        """
        # Try aggregated state vector first
        state = observation.get("observation.state")
        if state is not None:
            return np.asarray(state, dtype=np.float64)
        # Fall back to individual joint keys
        if self._state_keys:
            vals = [observation.get(k) for k in self._state_keys]
            if any(v is not None for v in vals):
                return np.array([float(v) if v is not None else 0.0 for v in vals], dtype=np.float64)
        return None

    def _extract_velocity(self, observation: dict) -> np.ndarray | None:
        """Extract velocity vector from observation dict."""
        velocity = observation.get("observation.velocity")
        if velocity is not None:
            return np.asarray(velocity, dtype=np.float64)
        # Fall back to individual velocity keys
        if self._speed_keys:
            vals = [observation.get(k) for k in self._speed_keys]
            if any(v is not None for v in vals):
                return np.array([float(v) if v is not None else 0.0 for v in vals], dtype=np.float64)
        return None

    def is_home_pose(self, observation: dict) -> bool:
        """Position within threshold (pose only, ignores velocity)."""
        if len(self._ref) == 0:
            return True
        state = self._extract_state(observation)
        if state is None:
            return False  # Cannot determine → assume NOT home (safe default)
        n = min(len(state), len(self._ref))
        if n == 0:
            return False
        diff = np.abs(state[:n] - self._ref[:n])
        result = bool(np.all(diff <= self._threshold))
        # Debug: log first comparison
        if not hasattr(self, '_debug_logged'):
            self._debug_logged = True
            max_diff_idx = int(np.argmax(diff))
            debug_print(f"HomeEvaluator first check:")
            debug_print(f"  state_keys: {self._state_keys}")
            debug_print(f"  state:     {state.tolist()}")
            debug_print(f"  reference: {self._ref[:n].tolist()}")
            debug_print(f"  diff:      {diff.tolist()}")
            debug_print(f"  threshold: {self._threshold}")
            debug_print(f"  max_diff:  idx={max_diff_idx}, val={diff[max_diff_idx]:.6f}, key={self._state_keys[max_diff_idx] if max_diff_idx < len(self._state_keys) else '?'}")
            debug_print(f"  is_home:   {result}")
        return result

    def is_home(self, observation: dict) -> bool:
        """Pose within threshold AND speed is low."""
        if not self.is_home_pose(observation):
            return False
        velocity = self._extract_velocity(observation)
        if velocity is None:
            return True  # No velocity info → trust pose check alone
        return bool(np.all(np.abs(velocity) <= self._speed_threshold))


class ValueEvaluator:
    """Evaluates task success based on a value model score.

    Builder succeeds when score >= value_gte.
    Destroyer succeeds when score <= value_lte.
    If both are set, either condition triggers success.
    """

    def __init__(self, success_condition: SuccessCondition):
        self._gte = success_condition.value_gte
        self._lte = success_condition.value_lte

    def is_success(self, score: float) -> bool:
        if self._gte is not None and score >= self._gte:
            return True
        if self._lte is not None and score <= self._lte:
            return True
        return False


class TimeoutResult(Enum):
    NONE = "none"         # Not timed out
    SOFT = "soft"         # Timed out but not at home — recovery needed
    HARD = "hard"         # Timed out and at home — can end immediately


class TimeoutEvaluator:
    """Checks whether an episode has exceeded its time limit."""

    def check(self, elapsed_s: float, max_time_s: float, is_home: bool) -> TimeoutResult:
        if elapsed_s < max_time_s:
            return TimeoutResult.NONE
        if is_home:
            return TimeoutResult.HARD
        return TimeoutResult.SOFT


class SubTaskEvaluator:
    """Manages sub-task timing within an episode.

    Given a list of sub-task durations [10, 5, 20], tracks which sub-task
    is active based on elapsed time and announces transitions.
    """

    def __init__(self, durations: list[float] | None = None):
        self.durations = durations or []
        self.enabled = bool(self.durations) and all(d > 0 for d in self.durations)

        if self.enabled:
            self.timestamps: list[float] = []
            cumsum = 0.0
            for d in self.durations:
                cumsum += d
                self.timestamps.append(cumsum)
            self.total_duration = self.timestamps[-1]
        else:
            self.timestamps = []
            self.total_duration = 0.0

        self._current_index = -1
        self._announced = [False] * len(self.durations) if self.enabled else []
        self._finished = False

    def update(self, timestamp: float) -> int:
        """Update with current timestamp. Returns current sub-task index or -1."""
        if not self.enabled:
            return -1

        if timestamp >= self.total_duration:
            if not self._finished:
                self._finished = True
                self._current_index = -1
            return -1

        new_index = 0
        for i, ts in enumerate(self.timestamps):
            if timestamp < ts:
                new_index = i
                break

        if not self._announced[new_index]:
            self._announced[new_index] = True
            self._current_index = new_index

        return self._current_index

    def get_current_index(self) -> int:
        return self._current_index if self.enabled else -1

    def is_finished(self) -> bool:
        return self._finished
