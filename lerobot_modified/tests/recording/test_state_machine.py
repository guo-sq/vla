"""Tests for recording.task.state_machine — flow phase transitions and decisions."""

import pytest

from lerobot.recording.task.state_machine import (
    ActionMode,
    Decision,
    EpisodeState,
    FlowPhase,
    StateMachine,
)
from lerobot.recording.task.task_spec import (
    RecoveryLevel,
    ResetConfig,
    RoleSpec,
    SafetyConfig,
    SuccessCondition,
    TaskSpec,
)
from lerobot.recording.task.evaluators import HomeEvaluator, ValueEvaluator
from lerobot.recording.runtime.safety_runtime import EscalationAction


def _make_self_play_spec():
    return TaskSpec(
        task_id="test",
        roles={
            "builder": RoleSpec(
                prompt="build it",
                max_time_s=60,
                action_mask="right_hand_only",
                success_when=SuccessCondition(at_home=True, value_gte=0.8),
            ),
            "destroyer": RoleSpec(
                prompt="destroy it",
                max_time_s=45,
                success_when=SuccessCondition(at_home=True, value_lte=-0.9),
            ),
        },
        reset=ResetConfig(reference_pose=[0.0, 0.0], threshold=0.1, speed_threshold=0.01, home_wait_s=1.0),
        safety=SafetyConfig(
            collision_current_bounds={"0": (-3.0, None)},
            recovery={
                "n1": RecoveryLevel(threshold=1, prompt="go safe", timeout_s=5.0),
                "n2": RecoveryLevel(threshold=2, prompt="go home", timeout_s=8.0),
                "n3": RecoveryLevel(threshold=3, action="force_human_takeover"),
            },
        ),
    )


def _make_simple_spec():
    return TaskSpec(
        task_id="simple",
        roles={"operator": RoleSpec(prompt="do it", max_time_s=30)},
        reset=ResetConfig(reference_pose=[0.0], threshold=0.1, speed_threshold=0.01),
    )


class TestStateMachineInit:
    def test_initial_state(self):
        spec = _make_self_play_spec()
        sm = StateMachine(spec)
        sm.reset_for_episode("builder", "build it")
        assert sm.state.phase == FlowPhase.MAIN
        assert sm.state.role == "builder"
        assert sm.state.active_prompt == "build it"

    def test_reset_clears_state(self):
        spec = _make_self_play_spec()
        sm = StateMachine(spec)
        sm.reset_for_episode("builder", "build it")
        sm.state.phase = FlowPhase.RECOVERY_N1
        sm.state.collision_count = 5
        sm.reset_for_episode("destroyer", "destroy it")
        assert sm.state.phase == FlowPhase.MAIN
        assert sm.state.role == "destroyer"
        assert sm.state.collision_count == 0


class TestTimeoutHandling:
    def test_no_timeout(self):
        spec = _make_self_play_spec()
        sm = StateMachine(spec)
        sm.reset_for_episode("builder", "build it")
        decision = sm.check_timeout(elapsed_s=30.0, is_home=False)
        assert decision is None

    def test_timeout_at_home_ends_episode(self):
        spec = _make_self_play_spec()
        sm = StateMachine(spec)
        sm.reset_for_episode("builder", "build it")
        decision = sm.check_timeout(elapsed_s=61.0, is_home=True)
        assert decision is not None
        assert decision.action_mode == ActionMode.BREAK
        assert decision.task_success is False
        assert "time" in decision.exit_reason.lower()

    def test_timeout_not_at_home_builder_goes_n2(self):
        spec = _make_self_play_spec()
        sm = StateMachine(spec)
        sm.reset_for_episode("builder", "build it")
        decision = sm.check_timeout(elapsed_s=61.0, is_home=False)
        assert decision is not None
        assert sm.state.phase == FlowPhase.RECOVERY_N2
        assert decision.new_prompt == "go home"

    def test_timeout_not_at_home_destroyer_requests_takeover(self):
        spec = _make_self_play_spec()
        sm = StateMachine(spec)
        sm.reset_for_episode("destroyer", "destroy it")
        decision = sm.check_timeout(elapsed_s=46.0, is_home=False)
        assert decision is not None
        assert decision.request_takeover is True

    def test_already_exceeded_does_not_repeat(self):
        spec = _make_self_play_spec()
        sm = StateMachine(spec)
        sm.reset_for_episode("builder", "build it")
        d1 = sm.check_timeout(elapsed_s=61.0, is_home=False)
        assert d1 is not None
        d2 = sm.check_timeout(elapsed_s=62.0, is_home=False)
        assert d2 is None  # already handled


class TestValueEvaluation:
    def test_value_success_builder(self):
        spec = _make_self_play_spec()
        sm = StateMachine(spec)
        sm.reset_for_episode("builder", "build it")
        decision = sm.on_value_score(0.9, is_home=True)
        assert decision.action_mode == ActionMode.BREAK
        assert decision.task_success is True

    def test_value_failure_builder(self):
        spec = _make_self_play_spec()
        sm = StateMachine(spec)
        sm.reset_for_episode("builder", "build it")
        decision = sm.on_value_score(0.3, is_home=True)
        assert decision.action_mode == ActionMode.CONTINUE
        assert decision.task_success is None

    def test_value_success_destroyer(self):
        spec = _make_self_play_spec()
        sm = StateMachine(spec)
        sm.reset_for_episode("destroyer", "destroy it")
        decision = sm.on_value_score(-0.95, is_home=True)
        assert decision.action_mode == ActionMode.BREAK
        assert decision.task_success is True

    def test_value_failure_returns_to_main(self):
        spec = _make_self_play_spec()
        sm = StateMachine(spec)
        sm.reset_for_episode("builder", "build it")
        sm.state.phase = FlowPhase.RECOVERY_N2
        decision = sm.on_value_score(0.3, is_home=True)
        assert sm.state.phase == FlowPhase.MAIN
        assert decision.new_prompt == "build it"

    def test_value_model_failure_builder_goes_n2(self):
        spec = _make_self_play_spec()
        sm = StateMachine(spec)
        sm.reset_for_episode("builder", "build it")
        decision = sm.on_value_model_failure(is_home=False)
        assert sm.state.phase == FlowPhase.RECOVERY_N2
        assert decision.new_prompt == "go home"

    def test_value_model_failure_at_home_stops(self):
        spec = _make_self_play_spec()
        sm = StateMachine(spec)
        sm.reset_for_episode("builder", "build it")
        decision = sm.on_value_model_failure(is_home=True)
        assert decision.action_mode == ActionMode.BREAK
        assert decision.task_success is False


class TestHomeDetection:
    def test_home_arms_value_eval(self):
        spec = _make_self_play_spec()
        sm = StateMachine(spec)
        sm.reset_for_episode("builder", "build it")
        # Enter home
        sm.update_home_state(is_home_pose=True, timestamp=10.0)
        # Wait for home_wait_s (1.0)
        sm.update_home_state(is_home_pose=True, timestamp=11.1)
        assert sm.state.value_eval_pending is True

    def test_leave_home_resets_latch(self):
        spec = _make_self_play_spec()
        sm = StateMachine(spec)
        sm.reset_for_episode("builder", "build it")
        sm.update_home_state(is_home_pose=True, timestamp=10.0)
        sm.update_home_state(is_home_pose=True, timestamp=11.1)
        assert sm.state.value_eval_pending is True
        sm.state.value_eval_pending = False
        sm.state.value_eval_home_latched = True
        # Leave home
        sm.update_home_state(is_home_pose=False, timestamp=15.0)
        assert sm.state.value_eval_home_latched is False
        assert sm.state.home_detected_since_ts is None


class TestCollisionEscalation:
    def test_n1_changes_phase(self):
        spec = _make_self_play_spec()
        sm = StateMachine(spec)
        sm.reset_for_episode("builder", "build it")
        decision = sm.on_collision_escalation(EscalationAction.N1_SAFE_PROMPT, timestamp=5.0)
        assert sm.state.phase == FlowPhase.RECOVERY_N1
        assert decision.new_prompt == "go safe"

    def test_n2_changes_phase(self):
        spec = _make_self_play_spec()
        sm = StateMachine(spec)
        sm.reset_for_episode("builder", "build it")
        decision = sm.on_collision_escalation(EscalationAction.N2_HOME_PROMPT, timestamp=5.0)
        assert sm.state.phase == FlowPhase.RECOVERY_N2
        assert decision.new_prompt == "go home"

    def test_n3_requests_takeover(self):
        spec = _make_self_play_spec()
        sm = StateMachine(spec)
        sm.reset_for_episode("builder", "build it")
        decision = sm.on_collision_escalation(EscalationAction.N3_FORCE_TAKEOVER, timestamp=5.0)
        assert decision.request_takeover is True
        assert decision.takeover_reason == "collision_n3"

    def test_destroyer_collision_requests_takeover(self):
        spec = _make_self_play_spec()
        sm = StateMachine(spec)
        sm.reset_for_episode("destroyer", "destroy it")
        decision = sm.on_collision_escalation(EscalationAction.DESTROYER_FORCE_TAKEOVER, timestamp=5.0)
        assert decision.request_takeover is True

    def test_pause_returns_hold(self):
        spec = _make_self_play_spec()
        sm = StateMachine(spec)
        sm.reset_for_episode("builder", "build it")
        decision = sm.on_collision_escalation(EscalationAction.PAUSE, timestamp=5.0)
        assert decision.action_mode == ActionMode.HOLD


class TestN1Recovery:
    def test_n1_stable_returns_to_main(self):
        spec = _make_self_play_spec()
        sm = StateMachine(spec)
        sm.reset_for_episode("builder", "build it")
        sm.on_collision_escalation(EscalationAction.N1_SAFE_PROMPT, timestamp=1.0)
        assert sm.state.phase == FlowPhase.RECOVERY_N1
        # Simulate stable time passing (n1 stable_time = no new collisions for timeout period)
        # n1_timeout_s = 5.0, check at 6.1s => stable
        decision = sm.check_n1_recovery(timestamp=7.0, max_abs_velocity=0.5)
        assert decision is not None
        # Should timeout to N2 after n1_timeout_s
        assert sm.state.phase == FlowPhase.RECOVERY_N2

    def test_n1_stable_within_time(self):
        spec = _make_self_play_spec()
        sm = StateMachine(spec)
        sm.reset_for_episode("builder", "build it")
        sm.on_collision_escalation(EscalationAction.N1_SAFE_PROMPT, timestamp=1.0)
        decision = sm.check_n1_recovery(timestamp=2.0, max_abs_velocity=0.05)
        assert decision is None  # not yet timed out


class TestSimpleMode:
    def test_simple_spec_no_safety(self):
        spec = _make_simple_spec()
        sm = StateMachine(spec)
        sm.reset_for_episode("operator", "do it")
        # Should not crash, no safety config
        decision = sm.check_timeout(elapsed_s=31.0, is_home=True)
        assert decision is not None
        assert decision.action_mode == ActionMode.BREAK
