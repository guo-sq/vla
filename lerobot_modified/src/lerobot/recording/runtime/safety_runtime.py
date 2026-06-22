"""Collision detection, hold-in-place, and N1/N2/N3 escalation.

Self-play only — for modes without safety, simply don't instantiate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from lerobot.recording.task.task_spec import SafetyConfig
from lerobot.recording.utils.logging import debug_print

# Keys used to extract per-joint current readings from the observation dict.
# Set by the caller (record.py) after constructing SafetyRuntime.
# Falls back to "observation.current" aggregated key if not set.


@dataclass
class CollisionEvent:
    dim_idx: str
    value: float
    bounds: tuple[float | None, float | None]


class EscalationAction(Enum):
    PAUSE = "pause"                             # below N1: just pause inference
    N1_SAFE_PROMPT = "n1_safe_prompt"           # switch to safe recovery prompt
    N2_HOME_PROMPT = "n2_home_prompt"           # switch to home recovery prompt
    N3_FORCE_TAKEOVER = "n3_force_takeover"     # require human takeover
    DESTROYER_FORCE_TAKEOVER = "destroyer_force_takeover"


class SafetyRuntime:
    """Collision detection and escalation state machine.

    Checks observation currents against configured bounds.  Tracks
    collision count and returns the appropriate escalation action.
    """

    def __init__(self, safety_config: SafetyConfig, robot: Any = None, current_keys: list[str] | None = None):
        self._config = safety_config
        self._robot = robot
        self._current_keys = current_keys or []
        self.collision_count: int = 0
        self.inference_paused: bool = False
        self._collision_active: bool = False  # True while current is still out of bounds

    def _extract_current(self, observation: dict) -> np.ndarray | None:
        """Extract current readings.

        Prefers robot.get_current_vector() (direct SDK call, per PR #28),
        falls back to observation dict keys.
        """
        # Prefer direct SDK call (most reliable)
        if self._robot is not None and hasattr(self._robot, "get_current_vector"):
            try:
                vec = self._robot.get_current_vector()
                return np.array(vec, dtype=np.float64)
            except Exception as e:
                if not hasattr(self, '_sdk_warn_logged'):
                    self._sdk_warn_logged = True
                    logging.warning(f"SafetyRuntime: get_current_vector() failed: {e}. "
                                   "Falling back to observation dict.")
        # Fall back to observation dict
        current = observation.get("observation.current")
        if current is not None:
            return np.asarray(current, dtype=np.float64)
        if self._current_keys:
            vals = [observation.get(k) for k in self._current_keys]
            if any(v is not None for v in vals):
                return np.array([float(v) if v is not None else 0.0 for v in vals], dtype=np.float64)
        return None

    def check_collision(self, observation: dict) -> CollisionEvent | None:
        """Check current bounds. Returns CollisionEvent if triggered, None otherwise."""
        current = self._extract_current(observation)
        if current is None:
            if not hasattr(self, '_current_warn_logged'):
                self._current_warn_logged = True
                logging.warning("SafetyRuntime: no current readings found. "
                               "Collision detection is DISABLED.")
            return None
        if not hasattr(self, '_current_debug_logged'):
            self._current_debug_logged = True
            monitored = {k: float(current[int(k)]) for k in self._config.collision_current_bounds if int(k) < len(current)}
            debug_print(f"SafetyRuntime first check (source={'robot.get_current_vector' if self._robot and hasattr(self._robot, 'get_current_vector') else 'observation'}): current_len={len(current)}, monitored_dims={monitored}")

        any_out_of_bounds = False
        triggered_event = None
        for dim_key, (lower, upper) in self._config.collision_current_bounds.items():
            idx = int(dim_key)
            if idx >= len(current):
                continue
            val = float(current[idx])

            if (lower is not None and val < lower) or (upper is not None and val > upper):
                any_out_of_bounds = True
                if not self._collision_active:
                    # First tick of this collision event
                    triggered_event = CollisionEvent(dim_idx=dim_key, value=val, bounds=(lower, upper))
                break

        if any_out_of_bounds:
            self._collision_active = True
        else:
            # Current back within bounds — allow next collision to trigger
            self._collision_active = False

        return triggered_event

    def escalate(self, role: str) -> EscalationAction:
        """Increment collision count and return the appropriate escalation action.

        For destroyer role, always returns DESTROYER_FORCE_TAKEOVER.
        For builder role, escalates through N1 → N2 → N3 based on thresholds.
        """
        self.collision_count += 1
        self.inference_paused = True

        if role == "destroyer":
            return EscalationAction.DESTROYER_FORCE_TAKEOVER

        recovery = self._config.recovery
        n3 = recovery.get("n3")
        n2 = recovery.get("n2")
        n1 = recovery.get("n1")

        if n3 and self.collision_count >= n3.threshold:
            return EscalationAction.N3_FORCE_TAKEOVER
        if n2 and self.collision_count >= n2.threshold:
            return EscalationAction.N2_HOME_PROMPT
        if n1 and self.collision_count >= n1.threshold:
            return EscalationAction.N1_SAFE_PROMPT
        return EscalationAction.PAUSE

    def can_resume(self) -> bool:
        return self.inference_paused

    def resume(self) -> None:
        self.inference_paused = False

    def hold_position(self, robot: Any) -> dict:
        """Read current joint positions and return as action dict."""
        positions = robot.get_joint_positions()
        return {k: positions[k] for k in robot.action_features if k in positions}

    def reset(self) -> None:
        """Reset for new episode."""
        self.collision_count = 0
        self.inference_paused = False
        self._collision_active = False

    @property
    def n1_prompt(self) -> str | None:
        n1 = self._config.recovery.get("n1")
        return n1.prompt if n1 else None

    @property
    def n2_prompt(self) -> str | None:
        n2 = self._config.recovery.get("n2")
        return n2.prompt if n2 else None

    @property
    def n1_timeout_s(self) -> float | None:
        n1 = self._config.recovery.get("n1")
        return n1.timeout_s if n1 else None

    @property
    def n2_timeout_s(self) -> float | None:
        n2 = self._config.recovery.get("n2")
        return n2.timeout_s if n2 else None
