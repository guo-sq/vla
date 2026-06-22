"""Tests for recording.task.task_spec — structured task configuration."""

import json
import os
import tempfile

import pytest

from lerobot.recording.task.task_spec import (
    RecoveryLevel,
    ResetConfig,
    RoleSpec,
    SafetyConfig,
    SuccessCondition,
    TaskSpec,
)


SEATBELT_SELF_PLAY_JSON = {
    "task_id": "seatbelt",
    "roles": {
        "builder": {
            "prompt": "Hang the seatbelt with right hand under 20 seconds.",
            "max_time_s": 60,
            "action_mask": "right_hand_only",
            "success_when": {"at_home": True, "value_gte": -0.7},
        },
        "destroyer": {
            "prompt": "Take the seatbelt off under 20 seconds.",
            "max_time_s": 60,
            "action_mask": "left_hand_only",
            "success_when": {"at_home": True, "value_lte": -0.97},
        },
    },
    "reset": {
        "reference_pose": [0.0, -0.12, 0.07, -0.04, 0.0, 0.0, 0.0, 0.0, -0.12, 0.07, -0.04, 0.0, 0.0, 0.0],
        "threshold": 0.12,
        "speed_threshold": 0.01,
        "home_wait_s": 2.0,
    },
    "safety": {
        "collision_current_bounds": {"7": [-3.5, None], "9": [-0.5, None], "10": [-1, None]},
        "recovery": {
            "n1": {"threshold": 2, "prompt": "Collision happened. Return to a safe position.", "timeout_s": 8.0},
            "n2": {"threshold": 3, "prompt": "Back to home position.", "timeout_s": 12.0},
            "n3": {"threshold": 4, "action": "force_human_takeover"},
        },
    },
}

SIMPLE_RECORD_JSON = {
    "task_id": "seatbelt_teleop",
    "roles": {
        "operator": {
            "prompt": "Hang the seatbelt with right hand.",
            "max_time_s": 60,
        }
    },
    "reset": {
        "reference_pose": [0.0, -0.12, 0.07, -0.04, 0.0, 0.0, 0.0, 0.0, -0.12, 0.07, -0.04, 0.0, 0.0, 0.0],
        "threshold": 0.12,
        "speed_threshold": 0.01,
    },
}

PIPER_CLOTH_JSON = {
    "task_id": "fold_cloth",
    "roles": {
        "folder": {
            "prompt": "Fold the T-shirt.",
            "max_time_s": 120,
        },
        "disturber": {
            "prompt": "Disarrange the folded T-shirts.",
            "max_time_s": 60,
        },
    },
    "reset": {
        "reference_pose": [-2.809, -100.0, 67.248, 0.0, -78.75, 0.0, 0.0, 2.809, -100.0, 67.248, 0.0, 78.75, 0.0, 0.0],
        "threshold": 2.5,
        "speed_threshold": 0.01,
    },
}


def _write_json(data, tmpdir, filename="task.json"):
    path = os.path.join(tmpdir, filename)
    with open(path, "w") as f:
        json.dump(data, f)
    return path


class TestSuccessCondition:
    def test_defaults(self):
        sc = SuccessCondition()
        assert sc.at_home is True
        assert sc.value_gte is None
        assert sc.value_lte is None

    def test_from_dict_builder(self):
        sc = SuccessCondition.from_dict({"at_home": True, "value_gte": -0.7})
        assert sc.at_home is True
        assert sc.value_gte == -0.7
        assert sc.value_lte is None

    def test_from_dict_destroyer(self):
        sc = SuccessCondition.from_dict({"at_home": True, "value_lte": -0.97})
        assert sc.value_lte == -0.97

    def test_from_dict_empty(self):
        sc = SuccessCondition.from_dict({})
        assert sc.at_home is True  # default

    def test_from_dict_none(self):
        sc = SuccessCondition.from_dict(None)
        assert sc.at_home is True


class TestRoleSpec:
    def test_basic_role(self):
        role = RoleSpec(prompt="do something", max_time_s=60)
        assert role.prompt == "do something"
        assert role.max_time_s == 60
        assert role.action_mask is None
        assert role.success_when.value_gte is None

    def test_from_dict_with_success_when(self):
        role = RoleSpec.from_dict({
            "prompt": "build it",
            "max_time_s": 45,
            "action_mask": "right_hand_only",
            "success_when": {"at_home": True, "value_gte": 0.9},
        })
        assert role.action_mask == "right_hand_only"
        assert role.success_when.value_gte == 0.9

    def test_from_dict_minimal(self):
        role = RoleSpec.from_dict({"prompt": "do it", "max_time_s": 30})
        assert role.action_mask is None
        assert role.success_when.at_home is True


class TestResetConfig:
    def test_basic(self):
        rc = ResetConfig(
            reference_pose=[0.0, 1.0, 2.0],
            threshold=0.1,
            speed_threshold=0.01,
        )
        assert len(rc.reference_pose) == 3
        assert rc.home_wait_s == 2.0  # default

    def test_from_dict(self):
        rc = ResetConfig.from_dict({
            "reference_pose": [1.0, 2.0],
            "threshold": 0.5,
            "speed_threshold": 0.02,
            "home_wait_s": 3.0,
        })
        assert rc.home_wait_s == 3.0

    def test_from_dict_default_home_wait(self):
        rc = ResetConfig.from_dict({
            "reference_pose": [0.0],
            "threshold": 0.1,
            "speed_threshold": 0.01,
        })
        assert rc.home_wait_s == 2.0


class TestRecoveryLevel:
    def test_n1(self):
        rl = RecoveryLevel.from_dict({
            "threshold": 2,
            "prompt": "go safe",
            "timeout_s": 8.0,
        })
        assert rl.threshold == 2
        assert rl.prompt == "go safe"
        assert rl.timeout_s == 8.0
        assert rl.action is None

    def test_n3_force_takeover(self):
        rl = RecoveryLevel.from_dict({
            "threshold": 4,
            "action": "force_human_takeover",
        })
        assert rl.action == "force_human_takeover"
        assert rl.prompt is None


class TestSafetyConfig:
    def test_from_dict(self):
        sc = SafetyConfig.from_dict(SEATBELT_SELF_PLAY_JSON["safety"])
        assert "7" in sc.collision_current_bounds
        assert sc.collision_current_bounds["7"] == (-3.5, None)
        assert "n1" in sc.recovery
        assert sc.recovery["n1"].threshold == 2
        assert sc.recovery["n3"].action == "force_human_takeover"

    def test_null_bounds_converted(self):
        sc = SafetyConfig.from_dict({
            "collision_current_bounds": {"0": [None, 5.0], "1": [-2.0, None]},
            "recovery": {},
        })
        assert sc.collision_current_bounds["0"] == (None, 5.0)
        assert sc.collision_current_bounds["1"] == (-2.0, None)


class TestTaskSpec:
    def test_from_json_self_play(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_json(SEATBELT_SELF_PLAY_JSON, tmpdir)
            spec = TaskSpec.from_json(path)

        assert spec.task_id == "seatbelt"
        assert "builder" in spec.roles
        assert "destroyer" in spec.roles
        assert spec.roles["builder"].prompt == "Hang the seatbelt with right hand under 20 seconds."
        assert spec.roles["builder"].action_mask == "right_hand_only"
        assert spec.roles["builder"].success_when.value_gte == -0.7
        assert spec.roles["destroyer"].success_when.value_lte == -0.97
        assert spec.safety is not None
        assert spec.safety.recovery["n1"].timeout_s == 8.0
        assert len(spec.reset.reference_pose) == 14

    def test_from_json_simple_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_json(SIMPLE_RECORD_JSON, tmpdir)
            spec = TaskSpec.from_json(path)

        assert spec.task_id == "seatbelt_teleop"
        assert len(spec.roles) == 1
        assert "operator" in spec.roles
        assert spec.safety is None

    def test_from_json_piper_cloth(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_json(PIPER_CLOTH_JSON, tmpdir)
            spec = TaskSpec.from_json(path)

        assert spec.task_id == "fold_cloth"
        assert "folder" in spec.roles
        assert "disturber" in spec.roles
        assert spec.reset.threshold == 2.5
        assert spec.safety is None

    def test_from_dict(self):
        spec = TaskSpec.from_dict(SEATBELT_SELF_PLAY_JSON)
        assert spec.task_id == "seatbelt"

    def test_role_names(self):
        spec = TaskSpec.from_dict(SEATBELT_SELF_PLAY_JSON)
        assert spec.role_names == ["builder", "destroyer"]

    def test_initial_role(self):
        spec = TaskSpec.from_dict(SEATBELT_SELF_PLAY_JSON)
        assert spec.initial_role == "builder"

    def test_initial_prompt(self):
        spec = TaskSpec.from_dict(SEATBELT_SELF_PLAY_JSON)
        assert spec.initial_prompt == "Hang the seatbelt with right hand under 20 seconds."

    def test_is_self_play(self):
        sp = TaskSpec.from_dict(SEATBELT_SELF_PLAY_JSON)
        assert sp.is_self_play is True
        simple = TaskSpec.from_dict(SIMPLE_RECORD_JSON)
        assert simple.is_self_play is False

    def test_has_safety(self):
        sp = TaskSpec.from_dict(SEATBELT_SELF_PLAY_JSON)
        assert sp.has_safety is True
        simple = TaskSpec.from_dict(SIMPLE_RECORD_JSON)
        assert simple.has_safety is False

    def test_default_single_role(self):
        spec = TaskSpec.default_single_role(
            prompt="do the task",
            episode_time_s=30,
            reference_pose=[0.0, 0.0, 0.0],
        )
        assert spec.task_id == "default"
        assert len(spec.roles) == 1
        assert "operator" in spec.roles
        assert spec.roles["operator"].max_time_s == 30
        assert spec.safety is None

    def test_missing_task_id_raises(self):
        with pytest.raises((KeyError, ValueError)):
            TaskSpec.from_dict({"roles": {}, "reset": {"reference_pose": [], "threshold": 0.1, "speed_threshold": 0.01}})

    def test_missing_roles_raises(self):
        with pytest.raises((KeyError, ValueError)):
            TaskSpec.from_dict({"task_id": "x", "reset": {"reference_pose": [], "threshold": 0.1, "speed_threshold": 0.01}})

    def test_no_reset_is_valid(self):
        spec = TaskSpec.from_dict({"task_id": "x", "roles": {"op": {"prompt": "p", "max_time_s": 10}}})
        assert spec.reset is None
        assert spec.has_reset is False

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            TaskSpec.from_json("/nonexistent/path.json")

    def test_to_dict_roundtrip(self):
        spec = TaskSpec.from_dict(SEATBELT_SELF_PLAY_JSON)
        d = spec.to_dict()
        spec2 = TaskSpec.from_dict(d)
        assert spec2.task_id == spec.task_id
        assert spec2.roles["builder"].prompt == spec.roles["builder"].prompt
        assert spec2.safety.recovery["n1"].threshold == spec.safety.recovery["n1"].threshold
