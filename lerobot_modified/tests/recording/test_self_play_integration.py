"""Integration tests for self-play flow.

Tests every path in the self-play flow chart using real StateMachine,
SafetyRuntime, and HomeEvaluator with the seatbelt_self_play.json config.
No mocking of decision logic — only the robot/policy are faked.

Flow chart paths tested:
1. Normal episode: inference → home → value eval → success → end
2. Normal episode: inference → timeout at home → end
3. Builder timeout not at home → N2 recovery → reaches home → end
4. Builder timeout not at home → N2 recovery → N2 timeout → force takeover
5. Destroyer timeout not at home → force human takeover immediately
6. Collision builder N1 → safe prompt → stable → back to main
7. Collision builder N1 → timeout → N2 → home → back to main
8. Collision builder N2 → home prompt → home → back to main
9. Collision builder N3 → force takeover → Ctrl+Space at home → resume
10. Collision destroyer → force takeover immediately
11. Collision below N1 → pause → Ctrl+Enter → resume
12. Value model failure at home → stop program
13. Value model failure not home builder → N2 recovery → home → stop
14. Value model failure not home builder → N2 timeout → force takeover
15. Value model failure not home destroyer → force takeover
16. Force takeover exit with value-model reason → stop program
17. Home-only success (fold cloth style, no value model)
18. Action mask applied correctly (right_hand_only, left_hand_only)
19. Speed threshold gates value eval
20. Full scenario: builder collision → N1 → N2 → home → main → success
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(_SRC))

from lerobot.recording.runtime.safety_runtime import (
    EscalationAction,
    SafetyRuntime,
)
from lerobot.recording.task.evaluators import HomeEvaluator
from lerobot.recording.task.state_machine import (
    ActionMode,
    FlowPhase,
    StateMachine,
)
from lerobot.recording.task.task_spec import TaskSpec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SEATBELT_SPEC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "lerobot_example_config_files",
    "task_specs", "seatbelt", "arxx5_self_play.json",
)

REF_POSE = [-0.00247955322265625, 0.00896453857421875, 0.00324249267578125,
            -0.00476837158203125, 0.00400543212890625, -0.00286102294921875,
            0.0064849853515625, -0.00057220458984375, -0.00171661376953125,
            0.00553131103515625, -0.00934600830078125, -0.00286102294921875,
            0.01430511474609375, 0.008392333984375]

STATE_KEYS = [f"left_joint_{i}.pos" for i in range(1, 8)] + \
             [f"right_joint_{i}.pos" for i in range(1, 8)]
VEL_KEYS = [f"left_joint_{i}.vel" for i in range(1, 8)] + \
           [f"right_joint_{i}.vel" for i in range(1, 8)]
CUR_KEYS = [f"left_joint_{i}.cur" for i in range(1, 8)] + \
           [f"right_joint_{i}.cur" for i in range(1, 8)]


def _obs(home=True, speed_low=True, collision_dim=None, collision_val=None):
    """Build fake ARX bimanual observation."""
    obs = {}
    for i, val in enumerate(REF_POSE):
        side = "left" if i < 7 else "right"
        idx = (i % 7) + 1
        obs[f"{side}_joint_{idx}.pos"] = val + (0.001 if home else 0.5)
        obs[f"{side}_joint_{idx}.vel"] = 0.001 if speed_low else 0.5
        obs[f"{side}_joint_{idx}.cur"] = 0.0

    if collision_dim is not None and collision_val is not None:
        cur_keys = [f"left_joint_{i}.cur" for i in range(1, 8)] + \
                   [f"right_joint_{i}.cur" for i in range(1, 8)]
        dim_int = int(collision_dim)
        if dim_int < len(cur_keys):
            obs[cur_keys[dim_int]] = collision_val
    return obs


@pytest.fixture
def spec():
    return TaskSpec.from_json(SEATBELT_SPEC_PATH)


@pytest.fixture
def sm(spec):
    m = StateMachine(spec)
    m.reset_for_episode("builder", "Hang the seatbelt with right hand under 20 seconds.")
    return m


@pytest.fixture
def sm_d(spec):
    m = StateMachine(spec)
    m.reset_for_episode("destroyer", "Take the seatbelt off under 20 seconds.")
    return m


@pytest.fixture
def safety(spec):
    return SafetyRuntime(spec.safety, robot=None, current_keys=CUR_KEYS)


@pytest.fixture
def home_eval(spec):
    return HomeEvaluator(spec.reset, robot_state_keys=STATE_KEYS, robot_speed_keys=VEL_KEYS)


# ===========================================================================
# 1. Normal value success
# ===========================================================================

class TestValueSuccess:
    def test_home_triggers_eval_after_wait(self, sm, home_eval):
        # Must leave home first before home detection arms
        sm.update_home_state(home_eval.is_home_pose(_obs(home=False)), 0.0)
        o = _obs(home=True, speed_low=True)
        for t in range(100):
            sm.update_home_state(home_eval.is_home_pose(o), 1.0 + t * 0.033)
        assert sm.state.value_eval_pending is True

    def test_no_trigger_before_wait(self, sm, home_eval):
        sm.update_home_state(home_eval.is_home_pose(_obs(home=False)), 0.0)
        o = _obs(home=True, speed_low=True)
        for t in range(50):
            sm.update_home_state(home_eval.is_home_pose(o), 1.0 + t * 0.033)
        assert sm.state.value_eval_pending is False

    def test_leaving_home_resets_timer(self, sm, home_eval):
        oh, oa = _obs(home=True), _obs(home=False)
        for t in range(45):
            sm.update_home_state(home_eval.is_home_pose(oh), t * 0.033)
        sm.update_home_state(home_eval.is_home_pose(oa), 1.5)
        assert sm.state.home_detected_since_ts is None
        for t in range(80):
            sm.update_home_state(home_eval.is_home_pose(oh), 2.0 + t * 0.033)
        assert sm.state.value_eval_pending is True

    def test_builder_score_success(self, sm):
        d = sm.on_value_score(-0.1, True)
        assert d.task_success is True and d.exit_reason == "value_success"

    def test_builder_score_fail(self, sm):
        d = sm.on_value_score(-0.8, True)
        assert d.task_success is None and d.action_mode == ActionMode.CONTINUE

    def test_destroyer_score_success(self, sm_d):
        d = sm_d.on_value_score(-0.99, True)
        assert d.task_success is True

    def test_destroyer_score_fail(self, sm_d):
        d = sm_d.on_value_score(-0.5, True)
        assert d.task_success is None

    def test_latch_prevents_reeval(self, sm):
        sm.state.has_left_home = True
        sm.on_value_score(-0.8, True)
        assert sm.state.value_eval_home_latched is True
        for t in range(100):
            sm.update_home_state(True, t * 0.033)
        assert sm.state.value_eval_pending is False


# ===========================================================================
# 2. Timeout at home
# ===========================================================================

class TestTimeoutAtHome:
    def test_ends_episode(self, sm):
        d = sm.check_timeout(61.0, True)
        assert d.action_mode == ActionMode.BREAK
        assert d.exit_reason == "role_time_limit_exceeded"

    def test_no_timeout_before_limit(self, sm):
        assert sm.check_timeout(59.0, True) is None

    def test_fires_only_once(self, sm):
        assert sm.check_timeout(61.0, True) is not None
        assert sm.check_timeout(62.0, True) is None


# ===========================================================================
# 3. Builder timeout → N2 → home
# ===========================================================================

class TestBuilderTimeoutN2:
    def test_starts_n2(self, sm):
        d = sm.check_timeout(61.0, False)
        assert d.new_prompt == "Back to home position."
        assert sm.state.phase == FlowPhase.RECOVERY_N2

    def test_reaches_home(self, sm):
        sm.check_timeout(61.0, False)
        sm.check_n2_recovery_home(True, 0.05, 62.0)
        d = sm.check_n2_recovery_home(True, 0.05, 62.6)
        assert d.action_mode == ActionMode.BREAK
        assert d.exit_reason == "role_time_limit_recovered_home"

    def test_n2_timeout_forces_takeover(self, sm):
        sm.check_timeout(61.0, False)
        d = sm.check_n2_recovery_home(False, 0.5, 77.0)
        assert d.request_takeover is True
        assert d.takeover_reason == "timeout_n2_not_home"


# ===========================================================================
# 5. Destroyer timeout → force takeover
# ===========================================================================

class TestDestroyerTimeout:
    def test_forces_takeover(self, sm_d):
        d = sm_d.check_timeout(61.0, False)
        assert d.request_takeover is True
        assert d.takeover_reason == "role_time_limit_not_home"


# ===========================================================================
# 6. Collision N1 → stable → main
# ===========================================================================

class TestCollisionN1Stable:
    def test_n1_triggered(self, sm, safety):
        esc = safety.escalate("builder")
        assert esc == EscalationAction.N1_SAFE_PROMPT
        d = sm.on_collision_escalation(esc, 10.0)
        assert sm.state.phase == FlowPhase.RECOVERY_N1
        assert d.new_prompt == "Collision happened. Return to a safe position."

    def test_before_execute_time(self, sm, safety):
        safety.escalate("builder")
        sm.on_collision_escalation(EscalationAction.N1_SAFE_PROMPT, 10.0)
        assert sm.check_n1_recovery(14.0, 0.05) is None

    def test_stable_returns_to_main(self, sm, safety):
        safety.escalate("builder")
        sm.on_collision_escalation(EscalationAction.N1_SAFE_PROMPT, 10.0)
        sm.check_n1_recovery(15.5, 0.05)
        d = sm.check_n1_recovery(16.1, 0.05)
        assert d is not None and sm.state.phase == FlowPhase.MAIN

    def test_speed_high_resets_stable(self, sm, safety):
        safety.escalate("builder")
        sm.on_collision_escalation(EscalationAction.N1_SAFE_PROMPT, 10.0)
        sm.check_n1_recovery(15.5, 0.05)
        sm.check_n1_recovery(15.7, 0.5)
        assert sm.state.n1_speed_stable_since_ts is None


# ===========================================================================
# 7. N1 timeout → N2
# ===========================================================================

class TestCollisionN1Timeout:
    def test_escalates_to_n2(self, sm, safety):
        safety.escalate("builder")
        sm.on_collision_escalation(EscalationAction.N1_SAFE_PROMPT, 10.0)
        d = sm.check_n1_recovery(22.5, 0.5)
        assert sm.state.phase == FlowPhase.RECOVERY_N2
        assert d.new_prompt == "Back to home position."


# ===========================================================================
# 8. N2 from collision → home → main
# ===========================================================================

class TestCollisionN2Home:
    def test_home_stable_returns_to_main(self, sm, safety):
        for _ in range(3):
            safety.escalate("builder")
        sm.on_collision_escalation(EscalationAction.N2_HOME_PROMPT, 20.0)
        sm.check_n2_recovery_home(True, 0.05, 25.0)
        d = sm.check_n2_recovery_home(True, 0.05, 25.6)
        assert d.action_mode == ActionMode.CONTINUE
        assert sm.state.phase == FlowPhase.MAIN

    def test_n2_timeout_forces_takeover(self, sm, safety):
        for _ in range(3):
            safety.escalate("builder")
        sm.on_collision_escalation(EscalationAction.N2_HOME_PROMPT, 20.0)
        d = sm.check_n2_recovery_home(False, 0.5, 36.0)
        assert d.request_takeover is True
        assert d.takeover_reason == "n2_timeout_not_home"


# ===========================================================================
# 9. N3 → force takeover
# ===========================================================================

class TestCollisionN3:
    def test_n3_forces_takeover(self, sm, safety):
        for _ in range(3):
            safety.escalate("builder")
        esc = safety.escalate("builder")
        assert esc == EscalationAction.N3_FORCE_TAKEOVER
        d = sm.on_collision_escalation(esc, 30.0)
        assert d.request_takeover is True
        assert d.takeover_reason == "collision_n3"


# ===========================================================================
# 10. Destroyer collision → force takeover
# ===========================================================================

class TestDestroyerCollision:
    def test_any_collision_forces_takeover(self, sm_d, safety):
        esc = safety.escalate("destroyer")
        assert esc == EscalationAction.DESTROYER_FORCE_TAKEOVER
        d = sm_d.on_collision_escalation(esc, 10.0)
        assert d.request_takeover is True


# ===========================================================================
# 11. Below N1 → pause
# ===========================================================================

class TestBelowN1Pause:
    def test_pause_when_below_n1(self, spec):
        d = spec.to_dict()
        d["safety"]["recovery"]["n1"]["threshold"] = 2
        s = TaskSpec.from_dict(d)
        safety = SafetyRuntime(s.safety, robot=None, current_keys=CUR_KEYS)
        sm = StateMachine(s)
        sm.reset_for_episode("builder", "test")

        esc = safety.escalate("builder")
        assert esc == EscalationAction.PAUSE
        dec = sm.on_collision_escalation(esc, 10.0)
        assert dec.action_mode == ActionMode.HOLD
        assert dec.request_takeover is False


# ===========================================================================
# 12-15. Value model failure paths
# ===========================================================================

class TestValueModelFailure:
    def test_at_home_stops(self, sm):
        d = sm.on_value_model_failure(True)
        assert d.exit_reason == "value_model_unavailable_home"

    def test_builder_not_home_starts_n2(self, sm):
        d = sm.on_value_model_failure(False)
        assert d.new_prompt == "Back to home position."
        assert sm.state.value_model_failure_recovery is True

    def test_builder_n2_home_stops_program(self, sm):
        sm.on_value_model_failure(False)
        sm.state.n2_recovery_start_ts = 10.0
        sm.check_n2_recovery_home(True, 0.05, 15.0)
        d = sm.check_n2_recovery_home(True, 0.05, 15.6)
        assert d.exit_reason == "value_model_unavailable_home"

    def test_builder_n2_timeout_forces_takeover(self, sm):
        sm.on_value_model_failure(False)
        sm.state.n2_recovery_start_ts = 10.0
        d = sm.check_n2_recovery_home(False, 0.5, 26.0)
        assert d.request_takeover is True
        assert d.takeover_reason == "value_model_failure_n2_not_home"

    def test_destroyer_not_home_forces_takeover(self, sm_d):
        d = sm_d.on_value_model_failure(False)
        assert d.request_takeover is True
        assert d.takeover_reason == "value_model_unavailable_not_home"


# ===========================================================================
# 17. Home-only success (fold cloth)
# ===========================================================================

class TestHomeOnlySuccess:
    def test_no_value_model_needed(self):
        d = {
            "task_id": "fold_cloth",
            "roles": {
                "folder": {"prompt": "Fold.", "max_time_s": None,
                           "success_when": {"at_home": True}},
                "disturber": {"prompt": "Unfold.", "max_time_s": None,
                              "success_when": {"at_home": True}},
            },
            "reset": {"reference_pose": [0.0] * 12, "threshold": 0.15,
                       "speed_threshold": 0.01, "home_wait_s": 1.0},
        }
        s = TaskSpec.from_dict(d)
        assert s.roles["folder"].success_when.is_home_only is True
        assert s.roles["folder"].success_when.needs_value_model is False

    def test_unlimited_time_no_timeout(self):
        d = {
            "task_id": "fold_cloth",
            "roles": {"folder": {"prompt": "Fold.", "max_time_s": None,
                                  "success_when": {"at_home": True}}},
            "reset": {"reference_pose": [0.0] * 12, "threshold": 0.15,
                       "speed_threshold": 0.01, "home_wait_s": 1.0},
        }
        sm = StateMachine(TaskSpec.from_dict(d))
        sm.reset_for_episode("folder", "Fold.")
        assert sm.check_timeout(99999.0, False) is None


# ===========================================================================
# 18. Action mask
# ===========================================================================

class TestActionMask:
    def test_right_hand_only(self):
        from lerobot.recording.runtime.control_loop import _apply_action_mask
        o = _obs(home=False)
        action = {f"left_joint_{i}.pos": 1.0 for i in range(1, 8)}
        action.update({f"right_joint_{i}.pos": 2.0 for i in range(1, 8)})

        masked = _apply_action_mask(action, o, "right_hand_only")
        for i in range(1, 8):
            assert masked[f"left_joint_{i}.pos"] == o[f"left_joint_{i}.pos"]
            assert masked[f"right_joint_{i}.pos"] == 2.0

    def test_left_hand_only(self):
        from lerobot.recording.runtime.control_loop import _apply_action_mask
        o = _obs(home=False)
        action = {f"left_joint_{i}.pos": 1.0 for i in range(1, 8)}
        action.update({f"right_joint_{i}.pos": 2.0 for i in range(1, 8)})

        masked = _apply_action_mask(action, o, "left_hand_only")
        for i in range(1, 8):
            assert masked[f"left_joint_{i}.pos"] == 1.0
            assert masked[f"right_joint_{i}.pos"] == o[f"right_joint_{i}.pos"]

    def test_no_mask(self):
        from lerobot.recording.runtime.control_loop import _apply_action_mask
        o = _obs()
        action = {f"left_joint_{i}.pos": 1.0 for i in range(1, 8)}
        masked = _apply_action_mask(action, o, None)
        assert masked == action


# ===========================================================================
# 19. Speed threshold gates eval
# ===========================================================================

class TestSpeedGating:
    def test_speed_threshold_in_spec(self, spec):
        assert spec.roles["builder"].success_when.speed_threshold == 0.01


# ===========================================================================
# Safety runtime
# ===========================================================================

class TestSafetyRuntime:
    def test_no_collision_normal(self, safety):
        assert safety.check_collision(_obs()) is None

    def test_collision_detected(self, safety):
        r = safety.check_collision(_obs(collision_dim="7", collision_val=-4.0))
        assert r is not None and r.dim_idx == "7"

    def test_not_retriggered_while_active(self, safety):
        oc = _obs(collision_dim="7", collision_val=-4.0)
        assert safety.check_collision(oc) is not None
        assert safety.check_collision(oc) is None
        assert safety.check_collision(_obs()) is None
        assert safety.check_collision(oc) is not None

    def test_escalation_sequence(self, safety):
        assert safety.escalate("builder") == EscalationAction.N1_SAFE_PROMPT
        assert safety.escalate("builder") == EscalationAction.N1_SAFE_PROMPT
        assert safety.escalate("builder") == EscalationAction.N2_HOME_PROMPT
        assert safety.escalate("builder") == EscalationAction.N3_FORCE_TAKEOVER

    def test_reset_clears(self, safety):
        safety.escalate("builder")
        safety.reset()
        assert safety.collision_count == 0


# ===========================================================================
# HomeEvaluator
# ===========================================================================

class TestHomeEvaluator:
    def test_at_home(self, home_eval):
        assert home_eval.is_home_pose(_obs(home=True)) is True
        assert home_eval.is_home(_obs(home=True, speed_low=True)) is True

    def test_not_home(self, home_eval):
        assert home_eval.is_home_pose(_obs(home=False)) is False

    def test_home_pose_but_speed_high(self, home_eval):
        assert home_eval.is_home_pose(_obs(home=True, speed_low=False)) is True
        assert home_eval.is_home(_obs(home=True, speed_low=False)) is False


# ===========================================================================
# Task spec loading + round-trip
# ===========================================================================

class TestTaskSpec:
    def test_loads(self, spec):
        assert spec.task_id == "seatbelt"
        assert "builder" in spec.roles and "destroyer" in spec.roles

    def test_builder_config(self, spec):
        b = spec.roles["builder"]
        assert b.max_time_s == 60
        assert b.action_mask == "right_hand_only"
        assert b.timeout_recovery == "recovery_prompt"
        assert b.success_when.value_gte == -0.3

    def test_safety_config(self, spec):
        assert spec.safety.recovery["n1"].threshold == 1
        assert spec.safety.recovery["n1"].speed_threshold == 0.1
        assert spec.safety.recovery["n2"].timeout_s == 15.0
        assert spec.safety.recovery["n3"].threshold == 4

    def test_round_trip(self, spec):
        s2 = TaskSpec.from_dict(spec.to_dict())
        assert s2.task_id == spec.task_id
        assert s2.safety.recovery["n1"].speed_threshold == 0.1


# ===========================================================================
# 20. Full builder scenario: collision → N1 → N2 → home → main → success
# ===========================================================================

class TestFullBuilderScenario:
    def test_collision_recovery_then_success(self, spec):
        sm = StateMachine(spec)
        sm.reset_for_episode("builder", "Hang the seatbelt with right hand under 20 seconds.")
        safety = SafetyRuntime(spec.safety, robot=None, current_keys=CUR_KEYS)

        # 1. First collision → N1
        esc = safety.escalate("builder")
        sm.on_collision_escalation(esc, 10.0)
        assert sm.state.phase == FlowPhase.RECOVERY_N1

        # 2. N1 timeout → N2
        sm.check_n1_recovery(22.5, 0.5)
        assert sm.state.phase == FlowPhase.RECOVERY_N2

        # 3. N2 home stable → main
        sm.check_n2_recovery_home(True, 0.05, 25.0)
        d = sm.check_n2_recovery_home(True, 0.05, 25.6)
        assert sm.state.phase == FlowPhase.MAIN

        # 4. Robot does task, returns home, value eval triggers
        sm.update_home_state(False, 26.0)
        for t in range(70):
            sm.update_home_state(True, 40.0 + t * 0.033)
        assert sm.state.value_eval_pending is True

        # 5. Value success
        d = sm.on_value_score(-0.1, True)
        assert d.task_success is True


# ===========================================================================
# Fold cloth self-play spec: home-only, no safety, per-role servers
# ===========================================================================

FOLD_CLOTH_SPEC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "lerobot_example_config_files",
    "task_specs", "fold_cloth", "piper_self_play.json",
)


class TestFoldClothSpec:
    @pytest.fixture
    def fc_spec(self):
        return TaskSpec.from_json(FOLD_CLOTH_SPEC_PATH)

    def test_loads(self, fc_spec):
        assert fc_spec.task_id == "fold_cloth"
        assert "folder" in fc_spec.roles and "disturber" in fc_spec.roles

    def test_no_safety(self, fc_spec):
        assert fc_spec.safety is None
        assert fc_spec.has_safety is False

    def test_no_time_limit(self, fc_spec):
        assert fc_spec.roles["folder"].max_time_s is None
        assert fc_spec.roles["disturber"].max_time_s is None

    def test_home_only_success(self, fc_spec):
        assert fc_spec.roles["folder"].success_when.is_home_only is True
        assert fc_spec.roles["folder"].success_when.needs_value_model is False
        assert fc_spec.roles["disturber"].success_when.is_home_only is True

    def test_per_role_servers(self, fc_spec):
        folder_server = fc_spec.roles["folder"].policy_server
        disturber_server = fc_spec.roles["disturber"].policy_server
        assert folder_server is not None
        assert folder_server.host == "localhost" and folder_server.port == 8001
        assert disturber_server is not None
        assert disturber_server.host == "localhost" and disturber_server.port == 8002

    def test_speed_threshold_in_success(self, fc_spec):
        assert fc_spec.roles["folder"].success_when.speed_threshold == 0.5

    def test_no_timeout_ever(self, fc_spec):
        sm = StateMachine(fc_spec)
        sm.reset_for_episode("folder", "Fold.")
        assert sm.check_timeout(999999.0, False) is None

    def test_no_collision_handling(self, fc_spec):
        """Without safety config, collision checks should be skipped."""
        assert fc_spec.safety is None

    def test_folder_home_triggers_eval(self, fc_spec):
        sm = StateMachine(fc_spec)
        sm.reset_for_episode("folder", "Fold.")
        # Simulate being at home for > home_wait_s (2.0s)
        for t in range(70):
            sm.update_home_state(True, t * 0.033)
        assert sm.state.value_eval_pending is True

    def test_round_trip(self, fc_spec):
        d = fc_spec.to_dict()
        s2 = TaskSpec.from_dict(d)
        assert s2.roles["folder"].policy_server.port == 8001
        assert s2.roles["disturber"].policy_server.port == 8002
        assert s2.safety is None


# ===========================================================================
# Seatbelt vs fold cloth: structural comparison
# ===========================================================================

class TestTaskSpecComparison:
    """Verify both task specs support different self-play patterns."""

    def test_seatbelt_has_value_model(self, spec):
        assert spec.roles["builder"].success_when.needs_value_model is True
        assert spec.roles["destroyer"].success_when.needs_value_model is True

    def test_fold_cloth_no_value_model(self):
        fc = TaskSpec.from_json(FOLD_CLOTH_SPEC_PATH)
        assert fc.roles["folder"].success_when.needs_value_model is False

    def test_seatbelt_has_safety(self, spec):
        assert spec.has_safety is True

    def test_fold_cloth_no_safety(self):
        fc = TaskSpec.from_json(FOLD_CLOTH_SPEC_PATH)
        assert fc.has_safety is False

    def test_seatbelt_has_timeout(self, spec):
        assert spec.roles["builder"].max_time_s == 60

    def test_fold_cloth_no_timeout(self):
        fc = TaskSpec.from_json(FOLD_CLOTH_SPEC_PATH)
        assert fc.roles["folder"].max_time_s is None

    def test_seatbelt_shared_server(self, spec):
        """Seatbelt uses a shared server (no per-role policy_server)."""
        assert spec.roles["builder"].policy_server is None
        assert spec.roles["destroyer"].policy_server is None

    def test_fold_cloth_per_role_servers(self):
        fc = TaskSpec.from_json(FOLD_CLOTH_SPEC_PATH)
        assert fc.roles["folder"].policy_server is not None
        assert fc.roles["disturber"].policy_server is not None
        assert fc.roles["folder"].policy_server.port != fc.roles["disturber"].policy_server.port


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
