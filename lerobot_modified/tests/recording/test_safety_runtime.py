"""Tests for recording.runtime.safety_runtime — collision detection + escalation."""

import numpy as np
import pytest

from lerobot.recording.runtime.safety_runtime import (
    CollisionEvent,
    EscalationAction,
    SafetyRuntime,
)
from lerobot.recording.task.task_spec import RecoveryLevel, SafetyConfig


def _make_safety_config(
    bounds=None,
    n1_threshold=1,
    n2_threshold=2,
    n3_threshold=3,
):
    if bounds is None:
        bounds = {"7": (-3.5, None), "9": (-0.5, None)}
    recovery = {
        "n1": RecoveryLevel(threshold=n1_threshold, prompt="go safe", timeout_s=8.0),
        "n2": RecoveryLevel(threshold=n2_threshold, prompt="go home", timeout_s=12.0),
        "n3": RecoveryLevel(threshold=n3_threshold, action="force_human_takeover"),
    }
    return SafetyConfig(collision_current_bounds=bounds, recovery=recovery)


class TestCollisionDetection:
    def test_no_collision(self):
        cfg = _make_safety_config()
        sr = SafetyRuntime(cfg)
        obs = {"observation.current": np.array([0.0] * 14)}
        result = sr.check_collision(obs)
        assert result is None

    def test_collision_lower_bound(self):
        cfg = _make_safety_config(bounds={"7": (-3.5, None)})
        sr = SafetyRuntime(cfg)
        current = np.zeros(14)
        current[7] = -4.0  # below -3.5
        obs = {"observation.current": current}
        result = sr.check_collision(obs)
        assert result is not None
        assert result.dim_idx == "7"
        assert result.value == -4.0

    def test_collision_upper_bound(self):
        cfg = _make_safety_config(bounds={"3": (None, 2.0)})
        sr = SafetyRuntime(cfg)
        current = np.zeros(14)
        current[3] = 3.0  # above 2.0
        obs = {"observation.current": current}
        result = sr.check_collision(obs)
        assert result is not None
        assert result.dim_idx == "3"

    def test_no_current_in_observation(self):
        cfg = _make_safety_config()
        sr = SafetyRuntime(cfg)
        obs = {"observation.state": np.zeros(14)}
        result = sr.check_collision(obs)
        assert result is None

    def test_null_bounds_ignored(self):
        cfg = _make_safety_config(bounds={"5": (None, None)})
        sr = SafetyRuntime(cfg)
        current = np.zeros(14)
        current[5] = 999.0
        obs = {"observation.current": current}
        result = sr.check_collision(obs)
        assert result is None

    def test_dim_out_of_range(self):
        cfg = _make_safety_config(bounds={"99": (-1.0, 1.0)})
        sr = SafetyRuntime(cfg)
        obs = {"observation.current": np.zeros(14)}
        result = sr.check_collision(obs)
        assert result is None


class TestEscalation:
    def test_first_collision_below_n1(self):
        cfg = _make_safety_config(n1_threshold=2)
        sr = SafetyRuntime(cfg)
        action = sr.escalate(role="builder")
        assert action == EscalationAction.PAUSE
        assert sr.collision_count == 1

    def test_n1_escalation(self):
        cfg = _make_safety_config(n1_threshold=2, n2_threshold=3, n3_threshold=4)
        sr = SafetyRuntime(cfg)
        sr.escalate(role="builder")  # count=1
        action = sr.escalate(role="builder")  # count=2 → N1
        assert action == EscalationAction.N1_SAFE_PROMPT
        assert sr.collision_count == 2

    def test_n2_escalation(self):
        cfg = _make_safety_config(n1_threshold=1, n2_threshold=2, n3_threshold=3)
        sr = SafetyRuntime(cfg)
        sr.escalate(role="builder")  # count=1 → N1
        action = sr.escalate(role="builder")  # count=2 → N2
        assert action == EscalationAction.N2_HOME_PROMPT

    def test_n3_escalation(self):
        cfg = _make_safety_config(n1_threshold=1, n2_threshold=2, n3_threshold=3)
        sr = SafetyRuntime(cfg)
        sr.escalate(role="builder")  # N1
        sr.escalate(role="builder")  # N2
        action = sr.escalate(role="builder")  # count=3 → N3
        assert action == EscalationAction.N3_FORCE_TAKEOVER

    def test_destroyer_always_force_takeover(self):
        cfg = _make_safety_config()
        sr = SafetyRuntime(cfg)
        action = sr.escalate(role="destroyer")
        assert action == EscalationAction.DESTROYER_FORCE_TAKEOVER

    def test_reset_clears_count(self):
        cfg = _make_safety_config()
        sr = SafetyRuntime(cfg)
        sr.escalate(role="builder")
        sr.escalate(role="builder")
        assert sr.collision_count == 2
        sr.reset()
        assert sr.collision_count == 0
        assert sr.inference_paused is False

    def test_pause_and_resume(self):
        cfg = _make_safety_config()
        sr = SafetyRuntime(cfg)
        sr.inference_paused = True
        assert sr.can_resume() is True
        sr.resume()
        assert sr.inference_paused is False


class TestHoldPosition:
    def test_hold_returns_current_joints(self):
        cfg = _make_safety_config()
        sr = SafetyRuntime(cfg)

        class FakeRobot:
            action_features = {"j1.pos": True, "j2.pos": True}
            def get_joint_positions(self):
                return {"j1.pos": 1.0, "j2.pos": 2.0, "extra": 99.0}

        action = sr.hold_position(FakeRobot())
        assert action == {"j1.pos": 1.0, "j2.pos": 2.0}
