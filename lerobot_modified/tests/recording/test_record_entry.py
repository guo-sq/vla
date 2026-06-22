"""Tests for the record entry point logic — task spec resolution and mode composition.

Note: record.py imports hardware-dependent lerobot modules (cv2, serial, etc.)
so we test the composition logic through the underlying modules directly.
"""

import json
import os
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lerobot.recording.task.task_spec import TaskSpec
from lerobot.recording.runtime.control_loop import TeleopSource
from lerobot.recording.runtime.safety_runtime import SafetyRuntime
from lerobot.recording.task.state_machine import StateMachine


SELF_PLAY_JSON = {
    "task_id": "seatbelt",
    "roles": {
        "builder": {
            "prompt": "build it",
            "max_time_s": 60,
            "success_when": {"at_home": True, "value_gte": 0.8},
        },
        "destroyer": {
            "prompt": "destroy it",
            "max_time_s": 45,
            "success_when": {"at_home": True, "value_lte": -0.9},
        },
    },
    "reset": {
        "reference_pose": [0.0, 0.0],
        "threshold": 0.1,
        "speed_threshold": 0.01,
    },
    "safety": {
        "collision_current_bounds": {"0": [-3.0, None]},
        "recovery": {
            "n1": {"threshold": 1, "prompt": "safe", "timeout_s": 5.0},
            "n2": {"threshold": 2, "prompt": "home", "timeout_s": 8.0},
            "n3": {"threshold": 3, "action": "force_human_takeover"},
        },
    },
}


class TestResolveTaskSpec:
    def test_from_json_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "task.json")
            with open(path, "w") as f:
                json.dump(SELF_PLAY_JSON, f)
            spec = TaskSpec.from_json(path)
            assert spec.task_id == "seatbelt"

    def test_default_single_role(self):
        spec = TaskSpec.default_single_role(
            prompt="do it",
            episode_time_s=30,
        )
        assert spec.task_id == "default"
        assert len(spec.roles) == 1
        assert spec.roles["operator"].max_time_s == 30


class TestModeComposition:
    def test_record_mode_uses_teleop(self):
        class FakeRobot:
            action_features = {"j0.pos": True}
            def get_joint_positions(self):
                return {"j0.pos": 0.0}
            def send_action(self, a):
                return a

        class FakeTeleop:
            def get_action(self):
                return {"j0.pos": 1.0}

        src = TeleopSource(FakeTeleop(), FakeRobot())
        action = src.get_action({}, 0)
        assert action is not None

    def test_self_play_has_safety_and_state_machine(self):
        spec = TaskSpec.from_dict(SELF_PLAY_JSON)
        assert spec.has_safety
        assert spec.is_self_play
        safety = SafetyRuntime(spec.safety)
        sm = StateMachine(spec)
        assert safety is not None
        assert sm is not None

    def test_record_mode_no_safety_needed(self):
        spec = TaskSpec.default_single_role("do it", 30)
        assert not spec.has_safety
        assert not spec.is_self_play

    def test_infer_record_no_safety(self):
        spec = TaskSpec.default_single_role("do it", 30)
        # No safety config → no safety runtime needed
        assert spec.safety is None


@pytest.fixture(scope="module")
def select_action_source_fn():
    """Import select_action_source once; restoring sys.modules between tests
    causes a PyO3 re-init failure in rerun_bindings."""
    mock_modules = {
        "piper_sdk": MagicMock(),
        "pinocchio": MagicMock(),
        "scipy": MagicMock(),
        "scipy.interpolate": MagicMock(),
        "lerobot.robots.piper": MagicMock(),
        "lerobot.robots.piper.piper": MagicMock(),
        "lerobot.robots.piper.piper_sdk_interface": MagicMock(),
        "lerobot.robots.bi_piper_follower": MagicMock(),
        "lerobot.robots.bi_piper_follower.bi_piper_follower": MagicMock(),
        "lerobot.sensors.paxini_tactile_sensor": MagicMock(),
        "lerobot.sensors.paxini_tactile_sensor.paxini_tactile": MagicMock(),
    }
    with patch.dict(sys.modules, mock_modules):
        from lerobot.recording.record import select_action_source as _sas
    return _sas


class TestSelectActionSource:
    """select_action_source picks the per-episode action source for the
    non-self-play recording flow."""

    def test_record_mode_returns_teleop_source(self, select_action_source_fn):
        cfg = SimpleNamespace(mode="record")
        teleop, robot = MagicMock(), MagicMock()
        src = select_action_source_fn(cfg, teleop, robot, policy_runtime=None, prompt="x")
        assert isinstance(src, TeleopSource)

    def test_infer_record_returns_policy_runtime_and_reinits_with_prompt(self, select_action_source_fn):
        cfg = SimpleNamespace(mode="infer_record")
        runtime = MagicMock()
        src = select_action_source_fn(cfg, MagicMock(), MagicMock(), policy_runtime=runtime, prompt="hello")
        assert src is runtime
        runtime.reinit.assert_called_once_with("hello")

    def test_falls_back_to_teleop_when_runtime_missing(self, select_action_source_fn):
        cfg = SimpleNamespace(mode="infer_record")
        src = select_action_source_fn(cfg, MagicMock(), MagicMock(), policy_runtime=None, prompt="x")
        assert isinstance(src, TeleopSource)


@pytest.fixture(scope="module")
def record_module():
    """Import record.py once with hardware-only modules mocked."""
    mock_modules = {
        "piper_sdk": MagicMock(),
        "pinocchio": MagicMock(),
        "scipy": MagicMock(),
        "scipy.interpolate": MagicMock(),
        "lerobot.robots.piper": MagicMock(),
        "lerobot.robots.piper.piper": MagicMock(),
        "lerobot.robots.piper.piper_sdk_interface": MagicMock(),
        "lerobot.robots.bi_piper_follower": MagicMock(),
        "lerobot.robots.bi_piper_follower.bi_piper_follower": MagicMock(),
        "lerobot.sensors.paxini_tactile_sensor": MagicMock(),
        "lerobot.sensors.paxini_tactile_sensor.paxini_tactile": MagicMock(),
    }
    with patch.dict(sys.modules, mock_modules):
        from lerobot.recording import record as _record
    return _record


class TestEpisodeAudioPrompts:
    def test_announces_reset_before_next_episode(self, record_module):
        say = MagicMock()

        announced = record_module._announce_reset_before_next_episode(
            say,
            current_episode_index=0,
            num_episodes=2,
            stop_recording=False,
            rerecord_episode=False,
            reset_time_s=5,
        )

        assert announced is True
        # Blocking so the TTS voice doesn't overlap the next episode's
        # start prompt during the reset sleep.
        say.assert_called_once_with("请重置环境", blocking=True)

    def test_does_not_announce_reset_after_last_episode(self, record_module):
        say = MagicMock()

        announced = record_module._announce_reset_before_next_episode(
            say,
            current_episode_index=1,
            num_episodes=2,
            stop_recording=False,
            rerecord_episode=False,
            reset_time_s=5,
        )

        assert announced is False
        say.assert_not_called()

    def test_announces_reset_before_rerecord_even_on_last_episode(self, record_module):
        say = MagicMock()

        announced = record_module._announce_reset_before_next_episode(
            say,
            current_episode_index=1,
            num_episodes=2,
            stop_recording=False,
            rerecord_episode=True,
            reset_time_s=5,
        )

        assert announced is True
        say.assert_called_once_with("请重置环境", blocking=True)

    def test_suppresses_reset_announcement_when_reset_time_is_zero(self, record_module):
        say = MagicMock()

        announced = record_module._announce_reset_before_next_episode(
            say,
            current_episode_index=0,
            num_episodes=2,
            stop_recording=False,
            rerecord_episode=False,
            reset_time_s=0,
        )

        assert announced is False
        say.assert_not_called()

    def test_suppresses_reset_announcement_when_reset_time_is_zero_float(self, record_module):
        say = MagicMock()

        announced = record_module._announce_reset_before_next_episode(
            say,
            current_episode_index=0,
            num_episodes=2,
            stop_recording=False,
            rerecord_episode=True,
            reset_time_s=0.0,
        )

        assert announced is False
        say.assert_not_called()
