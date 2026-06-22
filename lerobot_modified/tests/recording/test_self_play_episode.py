"""Tests for run_self_play_episode — the full interleaved self-play control loop.

Tests cover:
- Basic episode runs to completion
- Collision detection triggers escalation
- Home detection triggers value evaluation
- Value success ends episode
- Value failure triggers recovery
- Timeout handling
- Intervention toggle (Ctrl+Space)
- Forced takeover blocks exit until home
- Per-frame is_human_intervention written to dataset
- EpisodeResult carries correct metadata
- Home-only success (no value model)
- Unlimited episode time (max_time_s=null)
- Per-role policy server
"""

import time
import threading

import numpy as np
import pytest

from lerobot.recording.runtime.control_loop import ControlLoop, EpisodeResult
from lerobot.recording.runtime.safety_runtime import SafetyRuntime, CollisionEvent, EscalationAction
from lerobot.recording.runtime.intervention import InterventionRuntime
from lerobot.recording.task.evaluators import HomeEvaluator
from lerobot.recording.task.state_machine import StateMachine, FlowPhase, ActionMode, Decision
from lerobot.recording.task.task_spec import (
    TaskSpec, RoleSpec, SuccessCondition, ResetConfig, SafetyConfig, RecoveryLevel,
)


# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------

class FakeRobot:
    def __init__(self, num_joints=3, observation_override=None):
        self.action_features = {f"j{i}.pos": True for i in range(num_joints)}
        self._num_joints = num_joints
        self._obs_override = observation_override
        self.sent_actions = []

    def get_observation(self):
        if self._obs_override:
            return dict(self._obs_override)
        return {
            "observation.state": np.zeros(self._num_joints),
            "observation.velocity": np.zeros(self._num_joints),
        }

    def get_joint_positions(self):
        return {f"j{i}.pos": 0.0 for i in range(self._num_joints)}

    def send_action(self, action):
        self.sent_actions.append(dict(action))
        return action


class FakeTeleop:
    name = "piper_leader"  # opt into adaptive sync duration in tests

    def __init__(self, num_joints=3):
        self._num_joints = num_joints
        self.action_features = {f"j{i}.pos": float for i in range(num_joints)}
        self.feedback_features = dict(self.action_features)
        self._sent_feedbacks = []

    def get_action(self):
        return {f"j{i}.pos": float(i + 1) for i in range(self._num_joints)}

    def send_feedback(self, pose):
        self._sent_feedbacks.append(dict(pose))


class FakePolicyRuntime:
    """Minimal PolicyRuntime stub for testing."""

    def __init__(self, action=None, value_score=None, num_joints=3):
        self._action = action or {f"j{i}.pos": 0.5 for i in range(num_joints)}
        self._value_score = value_score
        self.lang_prompt = ""
        self.transition_weight = 0.0
        self.transition_steps = 10
        self.reinit_count = 0
        self.reset_for_resume_count = 0
        self._num_joints = num_joints

    def get_action(self, observation, step_id):
        return dict(self._action)

    def get_value_score(self, observation, lang_prompt):
        return self._value_score

    def reinit(self, lang_prompt):
        self.lang_prompt = lang_prompt
        self.reinit_count += 1

    def reset_for_resume(self):
        self.transition_weight = 0.9
        self.reset_for_resume_count += 1

    def cleanup(self):
        pass

    def clear_queues(self):
        pass


class FakeDataset:
    """Captures frames added to the dataset."""

    def __init__(self):
        self.frames = []
        self.features = {}

    def add_frame(self, frame, task=""):
        self.frames.append({"frame": frame, "task": task})


class FakeLogger:
    def __init__(self):
        self.messages = []

    def log(self, msg, **kwargs):
        self.messages.append(msg)


class FakeSelfPlayLogger:
    def __init__(self):
        self.events = []

    def log(self, event, **kwargs):
        self.events.append({"event": event, **kwargs})


def make_seatbelt_task_spec(
    builder_max_time=1.0,
    destroyer_max_time=1.0,
    value_gte=-0.7,
    value_lte=-0.97,
    reference_pose=None,
    collision_bounds=None,
):
    """Create a seatbelt-like self-play task spec for testing."""
    if reference_pose is None:
        reference_pose = [0.0, 0.0, 0.0]
    roles = {
        "builder": RoleSpec(
            prompt="Hang the seatbelt.",
            max_time_s=builder_max_time,
            success_when=SuccessCondition(at_home=True, value_gte=value_gte),
        ),
        "destroyer": RoleSpec(
            prompt="Take the seatbelt off.",
            max_time_s=destroyer_max_time,
            success_when=SuccessCondition(at_home=True, value_lte=value_lte),
        ),
    }
    reset = ResetConfig(
        reference_pose=reference_pose,
        threshold=0.12,
        speed_threshold=0.01,
        home_wait_s=0.0,  # instant for tests
    )
    safety = None
    if collision_bounds:
        safety = SafetyConfig(
            collision_current_bounds=collision_bounds,
            recovery={
                "n1": RecoveryLevel(threshold=2, prompt="Safe recovery.", timeout_s=0.5),
                "n2": RecoveryLevel(threshold=3, prompt="Go home."),
                "n3": RecoveryLevel(threshold=4, action="force_human_takeover"),
            },
        )
    return TaskSpec(task_id="seatbelt", roles=roles, reset=reset, safety=safety)


def run_episode_with_defaults(
    robot=None,
    policy_runtime=None,
    intervention=None,
    safety=None,
    home_evaluator=None,
    state_machine=None,
    events=None,
    control_time_s=0.5,
    lang_prompt="test prompt",
    role="builder",
    fps=100,
    dataset=None,
    log_say_fn=None,
    self_play_logger=None,
):
    """Helper to call run_self_play_episode with sensible defaults."""
    if robot is None:
        robot = FakeRobot()
    if policy_runtime is None:
        policy_runtime = FakePolicyRuntime()
    if events is None:
        events = {}

    task_spec = make_seatbelt_task_spec()
    if intervention is None:
        intervention = InterventionRuntime(robot, None)
    if state_machine is None:
        state_machine = StateMachine(task_spec)
    state_machine.reset_for_episode(role, lang_prompt)

    loop = ControlLoop(robot, fps=fps, dataset=dataset)
    return loop.run_self_play_episode(
        policy_runtime=policy_runtime,
        intervention=intervention,
        safety=safety,
        home_evaluator=home_evaluator,
        state_machine=state_machine,
        events=events,
        control_time_s=control_time_s,
        lang_prompt=lang_prompt,
        role=role,
        log_say_fn=log_say_fn,
        self_play_logger=self_play_logger,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSelfPlayEpisodeBasic:
    """Basic episode lifecycle tests."""

    def test_runs_to_completion(self):
        result = run_episode_with_defaults(control_time_s=0.1)
        assert isinstance(result, EpisodeResult)
        assert result.step_count > 0
        assert result.duration_s >= 0.09
        assert result.end_reason == "completed"
        assert result.task_success is None

    def test_stop_recording_breaks_immediately(self):
        result = run_episode_with_defaults(events={"stop_recording": True})
        assert result.step_count == 0
        assert result.end_reason == "stop_recording"

    def test_exit_early_breaks(self):
        result = run_episode_with_defaults(events={"exit_early": True})
        assert result.step_count == 0
        assert result.end_reason == "exit_early"

    def test_policy_actions_sent_to_robot(self):
        robot = FakeRobot()
        policy = FakePolicyRuntime(action={"j0.pos": 1.5, "j1.pos": 2.5, "j2.pos": 3.5})
        result = run_episode_with_defaults(
            robot=robot, policy_runtime=policy, control_time_s=0.05,
        )
        assert result.step_count > 0
        assert len(robot.sent_actions) > 0
        assert robot.sent_actions[0]["j0.pos"] == 1.5


class TestSelfPlayHomeDetection:
    """Home detection and value evaluation trigger."""

    def test_home_at_zero_triggers_value_eval(self):
        """Robot at reference pose should trigger value evaluation."""
        robot = FakeRobot()  # obs state = zeros = at home
        policy = FakePolicyRuntime(value_score=-0.5)
        task_spec = make_seatbelt_task_spec()
        home_eval = HomeEvaluator(
            task_spec.reset,
            robot_state_keys=["j0.pos", "j1.pos", "j2.pos"],
        )
        sp_logger = FakeSelfPlayLogger()

        result = run_episode_with_defaults(
            robot=robot,
            policy_runtime=policy,
            home_evaluator=home_eval,
            control_time_s=0.2,
            self_play_logger=sp_logger,
        )

        # Value eval should have been triggered and logged
        value_events = [e for e in sp_logger.events if e["event"] == "value_eval"]
        assert len(value_events) > 0
        assert value_events[0]["score"] == -0.5

    def test_value_success_ends_episode(self):
        """Builder + value >= threshold → success."""
        robot = FakeRobot()
        policy = FakePolicyRuntime(value_score=-0.5)  # >= -0.7 threshold
        task_spec = make_seatbelt_task_spec()
        home_eval = HomeEvaluator(
            task_spec.reset,
            robot_state_keys=["j0.pos", "j1.pos", "j2.pos"],
        )

        result = run_episode_with_defaults(
            robot=robot,
            policy_runtime=policy,
            home_evaluator=home_eval,
            control_time_s=5.0,  # long time, should end early via value
        )

        assert result.task_success is True
        assert result.end_reason == "value_success"
        assert result.last_value_score == -0.5
        assert result.duration_s < 5.0  # ended early

    def test_value_failure_continues(self):
        """Builder + value < threshold → not success, episode continues."""
        robot = FakeRobot()
        policy = FakePolicyRuntime(value_score=-0.9)  # < -0.7 threshold → fail
        task_spec = make_seatbelt_task_spec(builder_max_time=0.1)
        home_eval = HomeEvaluator(
            task_spec.reset,
            robot_state_keys=["j0.pos", "j1.pos", "j2.pos"],
        )

        result = run_episode_with_defaults(
            robot=robot,
            policy_runtime=policy,
            home_evaluator=home_eval,
            control_time_s=0.1,
        )

        # Should not be marked as success
        assert result.task_success is not True
        assert result.last_value_score == -0.9

    def test_value_model_failure_at_home_stops(self):
        """Value model returns None at home → stop recording."""
        robot = FakeRobot()
        policy = FakePolicyRuntime(value_score=None)  # failure
        task_spec = make_seatbelt_task_spec()
        home_eval = HomeEvaluator(
            task_spec.reset,
            robot_state_keys=["j0.pos", "j1.pos", "j2.pos"],
        )
        events = {}

        result = run_episode_with_defaults(
            robot=robot,
            policy_runtime=policy,
            home_evaluator=home_eval,
            events=events,
            control_time_s=5.0,
        )

        assert result.task_success is False
        assert "value_model_unavailable" in result.end_reason
        assert events.get("stop_recording") is True

    def test_home_tracking_metadata(self):
        """first_home_time_s and home_duration_s should be tracked."""
        robot = FakeRobot()
        policy = FakePolicyRuntime(value_score=-0.5)  # success → early exit
        task_spec = make_seatbelt_task_spec()
        home_eval = HomeEvaluator(
            task_spec.reset,
            robot_state_keys=["j0.pos", "j1.pos", "j2.pos"],
        )

        result = run_episode_with_defaults(
            robot=robot,
            policy_runtime=policy,
            home_evaluator=home_eval,
            control_time_s=5.0,
        )

        assert result.first_home_time_s is not None
        assert result.home_duration_s >= 0


class TestSelfPlayTimeout:
    """Role time limit handling."""

    def test_timeout_at_home_ends_episode(self):
        """Robot at home + timeout → break with failure."""
        robot = FakeRobot()
        # Don't trigger value eval to test pure timeout
        # (value_score None means no value to trigger early exit)
        task_spec = make_seatbelt_task_spec(builder_max_time=0.05)
        home_eval = HomeEvaluator(
            task_spec.reset,
            robot_state_keys=["j0.pos", "j1.pos", "j2.pos"],
        )
        state_machine = StateMachine(task_spec)

        result = run_episode_with_defaults(
            robot=robot,
            policy_runtime=FakePolicyRuntime(value_score=None),
            home_evaluator=home_eval,
            state_machine=state_machine,
            control_time_s=0.05,
        )

        # Timeout fires on the first tick past max_time_s
        # Since robot is at home and value model fails, either timeout or value failure ends it
        assert result.duration_s <= 0.2  # should end quickly


class TestSelfPlayCollision:
    """Collision detection and escalation."""

    def test_collision_detected_and_logged(self):
        """When observation.current exceeds bounds, collision fires."""
        obs = {
            "observation.state": np.zeros(3),
            "observation.velocity": np.zeros(3),
            "observation.current": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -4.0, 0.0, 0.0]),
        }
        robot = FakeRobot(observation_override=obs)

        task_spec = make_seatbelt_task_spec(
            builder_max_time=0.1,
            collision_bounds={"7": (-3.5, None)},
        )
        safety = SafetyRuntime(task_spec.safety)
        state_machine = StateMachine(task_spec)
        sp_logger = FakeSelfPlayLogger()

        result = run_episode_with_defaults(
            robot=robot,
            safety=safety,
            state_machine=state_machine,
            control_time_s=0.1,
            self_play_logger=sp_logger,
        )

        collision_events = [e for e in sp_logger.events if e["event"] == "collision"]
        assert len(collision_events) > 0
        assert collision_events[0]["dim"] == "7"

    def test_destroyer_collision_triggers_force_takeover(self):
        """Destroyer role + any collision → force human takeover."""
        obs = {
            "observation.state": np.zeros(3),
            "observation.velocity": np.zeros(3),
            "observation.current": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -4.0, 0.0, 0.0]),
        }
        robot = FakeRobot(observation_override=obs)

        task_spec = make_seatbelt_task_spec(
            destroyer_max_time=0.1,
            collision_bounds={"7": (-3.5, None)},
        )
        safety = SafetyRuntime(task_spec.safety)
        state_machine = StateMachine(task_spec)
        log_msgs = []

        result = run_episode_with_defaults(
            robot=robot,
            safety=safety,
            state_machine=state_machine,
            role="destroyer",
            lang_prompt="Take the seatbelt off.",
            control_time_s=0.1,
            log_say_fn=lambda msg, **kw: log_msgs.append(msg),
        )

        # Should have announced forced takeover
        assert any("接管" in m for m in log_msgs)


class TestSelfPlayIntervention:
    """Human intervention (Ctrl+Space toggle) tests."""

    def test_intervention_toggle_enters_and_exits(self):
        """Ctrl+Space toggles intervention on/off."""
        robot = FakeRobot()
        teleop = FakeTeleop()
        intervention = InterventionRuntime(
            robot, teleop, pose_sync_duration_s=0.0,
            waiting_evacuation_time_s=0.0,
        )

        events = {}
        # Toggle on after a few ticks, then off
        step_count = [0]
        orig_get_obs = robot.get_observation

        def patched_get_obs():
            step_count[0] += 1
            if step_count[0] == 3:
                events["switch_infer_mode"] = True  # enter
            elif step_count[0] == 6:
                events["switch_infer_mode"] = True  # exit
            elif step_count[0] == 9:
                events["exit_early"] = True  # end
            return orig_get_obs()

        robot.get_observation = patched_get_obs

        result = run_episode_with_defaults(
            robot=robot,
            intervention=intervention,
            events=events,
            control_time_s=5.0,
        )

        assert result.intervention_count >= 1

    def test_forced_takeover_blocks_exit_until_home(self):
        """When forced takeover is active, cannot exit intervention unless at home."""
        # This is tested implicitly — the forced takeover sets force_takeover_required=True,
        # and the exit check requires is_home. Since we can't easily simulate the full
        # flow with timing in a unit test, we test the InterventionRuntime directly.
        intervention = InterventionRuntime(FakeRobot(), None)
        intervention.enter()
        intervention.request_forced_takeover("collision_n3")

        # Cannot exit forced takeover when not at home
        assert not intervention.can_exit_forced(is_home=False)
        # Can exit when at home
        assert intervention.can_exit_forced(is_home=True)


class TestSelfPlayDatasetWriting:
    """Verify per-frame is_human_intervention labels."""

    def test_writes_is_human_intervention_false(self):
        """Normal policy frames should have is_human_intervention=False."""
        # We can't easily test _write_frame without a real dataset,
        # but we verify the control loop passes the right flag
        robot = FakeRobot()
        policy = FakePolicyRuntime()

        # Patch _write_frame to capture calls
        frames_written = []
        loop = ControlLoop(robot, fps=100)

        orig_write = loop._write_frame
        def capture_write(obs, action, prompt, is_intervention=False):
            frames_written.append({"is_intervention": is_intervention})

        loop._write_frame = capture_write
        loop.dataset = "placeholder"  # non-None so _write_frame is called

        task_spec = make_seatbelt_task_spec(builder_max_time=0.05)
        state_machine = StateMachine(task_spec)
        state_machine.reset_for_episode("builder", "test")
        intervention = InterventionRuntime(robot, None)

        loop.run_self_play_episode(
            policy_runtime=policy,
            intervention=intervention,
            safety=None,
            home_evaluator=None,
            state_machine=state_machine,
            events={},
            control_time_s=0.05,
            lang_prompt="test",
            role="builder",
        )

        assert len(frames_written) > 0
        assert all(f["is_intervention"] is False for f in frames_written)


class TestSelfPlayEpisodeResult:
    """Verify EpisodeResult carries correct metadata."""

    def test_result_has_all_fields(self):
        result = run_episode_with_defaults(control_time_s=0.05)
        assert hasattr(result, "step_count")
        assert hasattr(result, "duration_s")
        assert hasattr(result, "end_reason")
        assert hasattr(result, "task_success")
        assert hasattr(result, "last_value_score")
        assert hasattr(result, "last_is_home")
        assert hasattr(result, "intervention_count")
        assert hasattr(result, "intervention_duration_s")
        assert hasattr(result, "first_home_time_s")
        assert hasattr(result, "home_duration_s")

    def test_value_success_populates_result(self):
        robot = FakeRobot()
        policy = FakePolicyRuntime(value_score=-0.5)
        task_spec = make_seatbelt_task_spec()
        home_eval = HomeEvaluator(
            task_spec.reset,
            robot_state_keys=["j0.pos", "j1.pos", "j2.pos"],
        )

        result = run_episode_with_defaults(
            robot=robot,
            policy_runtime=policy,
            home_evaluator=home_eval,
            control_time_s=5.0,
        )

        assert result.task_success is True
        assert result.end_reason == "value_success"
        assert result.last_value_score == -0.5
        assert result.last_is_home is True
        assert result.intervention_count == 0
        assert result.intervention_duration_s == 0.0


class TestSelfPlayPromptSwitching:
    """Verify prompt changes during collision recovery."""

    def test_collision_escalation_switches_prompt(self):
        """First collision at N1 threshold=1 → prompt switches to safe recovery prompt."""
        obs = {
            "observation.state": np.zeros(3),
            "observation.velocity": np.zeros(3),
            "observation.current": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -4.0, 0.0, 0.0]),
        }
        robot = FakeRobot(observation_override=obs)

        # Use N1 threshold=1 so first collision immediately triggers prompt switch
        task_spec = make_seatbelt_task_spec(
            builder_max_time=0.15,
            collision_bounds={"7": (-3.5, None)},
        )
        # Override N1 threshold to 1 so first collision triggers it
        task_spec.safety.recovery["n1"] = RecoveryLevel(
            threshold=1, prompt="Safe recovery.", timeout_s=0.5,
        )
        safety = SafetyRuntime(task_spec.safety)
        state_machine = StateMachine(task_spec)
        policy = FakePolicyRuntime()

        result = run_episode_with_defaults(
            robot=robot,
            policy_runtime=policy,
            safety=safety,
            state_machine=state_machine,
            control_time_s=0.15,
        )

        # Policy should have been reinited with safe recovery prompt
        assert policy.reinit_count > 0


# ---------------------------------------------------------------------------
# Home-only success (no value model)
# ---------------------------------------------------------------------------

def make_fold_cloth_task_spec(
    max_time_s=None,
    reference_pose=None,
    home_wait_s=0.0,
):
    """Create a fold-cloth-like task spec: home-only success, no value model, no safety."""
    if reference_pose is None:
        reference_pose = [0.0, 0.0, 0.0]
    roles = {
        "folder": RoleSpec(
            prompt="Fold the T-shirt.",
            max_time_s=max_time_s,
            success_when=SuccessCondition(at_home=True),  # no value thresholds
        ),
        "disturber": RoleSpec(
            prompt="Disarrange the T-shirts.",
            max_time_s=max_time_s,
            success_when=SuccessCondition(at_home=True),
        ),
    }
    reset = ResetConfig(
        reference_pose=reference_pose,
        threshold=0.12,
        speed_threshold=0.01,
        home_wait_s=home_wait_s,
    )
    return TaskSpec(task_id="fold_cloth", roles=roles, reset=reset)


class TestHomeOnlySuccess:
    """Home-only success: no value model, reaching home = episode success."""

    def test_home_only_success_at_home(self):
        """Robot at home → immediate success without value model call."""
        robot = FakeRobot()  # obs = zeros = at home
        policy = FakePolicyRuntime(value_score=None)  # no value model
        task_spec = make_fold_cloth_task_spec()
        home_eval = HomeEvaluator(
            task_spec.reset,
            robot_state_keys=["j0.pos", "j1.pos", "j2.pos"],
        )
        state_machine = StateMachine(task_spec)
        sp_logger = FakeSelfPlayLogger()

        state_machine.reset_for_episode("folder", "Fold the T-shirt.")
        loop = ControlLoop(robot, fps=100)
        intervention = InterventionRuntime(robot, None)

        result = loop.run_self_play_episode(
            policy_runtime=policy,
            intervention=intervention,
            safety=None,
            home_evaluator=home_eval,
            state_machine=state_machine,
            events={},
            control_time_s=5.0,
            lang_prompt="Fold the T-shirt.",
            role="folder",
            self_play_logger=sp_logger,
        )

        assert result.task_success is True
        assert result.end_reason == "home_success"
        assert result.duration_s < 5.0  # ended early
        # Should log home_success, NOT value_eval
        home_events = [e for e in sp_logger.events if e["event"] == "home_success"]
        value_events = [e for e in sp_logger.events if e["event"] == "value_eval"]
        assert len(home_events) > 0
        assert len(value_events) == 0

    def test_home_only_no_value_model_call(self):
        """With home-only success, value model should never be called."""
        call_count = [0]

        class TrackingPolicyRuntime(FakePolicyRuntime):
            def get_value_score(self, observation, lang_prompt):
                call_count[0] += 1
                return None  # should never reach here

        robot = FakeRobot()
        policy = TrackingPolicyRuntime()
        task_spec = make_fold_cloth_task_spec()
        home_eval = HomeEvaluator(
            task_spec.reset,
            robot_state_keys=["j0.pos", "j1.pos", "j2.pos"],
        )
        state_machine = StateMachine(task_spec)
        state_machine.reset_for_episode("folder", "test")
        loop = ControlLoop(robot, fps=100)
        intervention = InterventionRuntime(robot, None)

        loop.run_self_play_episode(
            policy_runtime=policy,
            intervention=intervention,
            safety=None,
            home_evaluator=home_eval,
            state_machine=state_machine,
            events={},
            control_time_s=5.0,
            lang_prompt="test",
            role="folder",
        )

        assert call_count[0] == 0, f"Value model called {call_count[0]} times, expected 0"

    def test_seatbelt_still_uses_value_model(self):
        """Seatbelt (with value thresholds) should still call value model."""
        robot = FakeRobot()
        policy = FakePolicyRuntime(value_score=-0.5)
        task_spec = make_seatbelt_task_spec()  # has value_gte/-0.7
        home_eval = HomeEvaluator(
            task_spec.reset,
            robot_state_keys=["j0.pos", "j1.pos", "j2.pos"],
        )
        sp_logger = FakeSelfPlayLogger()

        result = run_episode_with_defaults(
            robot=robot,
            policy_runtime=policy,
            home_evaluator=home_eval,
            control_time_s=5.0,
            self_play_logger=sp_logger,
        )

        # Should use value_success, not home_success
        assert result.task_success is True
        assert result.end_reason == "value_success"
        value_events = [e for e in sp_logger.events if e["event"] == "value_eval"]
        assert len(value_events) > 0


# ---------------------------------------------------------------------------
# Unlimited episode time (max_time_s=null)
# ---------------------------------------------------------------------------

class TestUnlimitedEpisodeTime:
    """max_time_s=None means no timeout."""

    def test_null_max_time_no_timeout(self):
        """With max_time_s=None, check_timeout should never fire."""
        task_spec = make_fold_cloth_task_spec(max_time_s=None)
        sm = StateMachine(task_spec)
        sm.reset_for_episode("folder", "test")

        # Even at very long elapsed time, should not timeout
        decision = sm.check_timeout(9999.0, is_home=False)
        assert decision is None

    def test_effective_max_time_s_with_none(self):
        """RoleSpec with max_time_s=None should return NO_TIME_LIMIT."""
        from lerobot.recording.task.task_spec import NO_TIME_LIMIT
        spec = RoleSpec(prompt="test", max_time_s=None)
        assert spec.effective_max_time_s == NO_TIME_LIMIT

    def test_effective_max_time_s_with_value(self):
        """RoleSpec with max_time_s=60 should return 60."""
        spec = RoleSpec(prompt="test", max_time_s=60)
        assert spec.effective_max_time_s == 60.0

    def test_unlimited_episode_ends_on_home(self):
        """With max_time_s=None, episode should still end when home is detected."""
        robot = FakeRobot()
        policy = FakePolicyRuntime(value_score=None)
        task_spec = make_fold_cloth_task_spec(max_time_s=None)
        home_eval = HomeEvaluator(
            task_spec.reset,
            robot_state_keys=["j0.pos", "j1.pos", "j2.pos"],
        )
        state_machine = StateMachine(task_spec)
        state_machine.reset_for_episode("folder", "test")
        loop = ControlLoop(robot, fps=100)
        intervention = InterventionRuntime(robot, None)

        # Use effective_max_time_s (very large) — should end via home_success
        result = loop.run_self_play_episode(
            policy_runtime=policy,
            intervention=intervention,
            safety=None,
            home_evaluator=home_eval,
            state_machine=state_machine,
            events={},
            control_time_s=task_spec.roles["folder"].effective_max_time_s,
            lang_prompt="test",
            role="folder",
        )

        assert result.task_success is True
        assert result.end_reason == "home_success"


# ---------------------------------------------------------------------------
# Per-role policy server
# ---------------------------------------------------------------------------

class TestPerRolePolicyServer:
    """Per-role policy_server in task spec."""

    def test_policy_server_from_dict(self):
        """PolicyServerConfig parses from dict."""
        from lerobot.recording.task.task_spec import PolicyServerConfig
        config = PolicyServerConfig.from_dict({"host": "10.0.0.1", "port": 9001})
        assert config.host == "10.0.0.1"
        assert config.port == 9001

    def test_role_spec_with_policy_server(self):
        """RoleSpec can carry an optional policy_server."""
        d = {
            "prompt": "Fold.",
            "policy_server": {"host": "localhost", "port": 8001},
            "success_when": {"at_home": True},
        }
        spec = RoleSpec.from_dict(d)
        assert spec.policy_server is not None
        assert spec.policy_server.host == "localhost"
        assert spec.policy_server.port == 8001
        assert spec.max_time_s is None

    def test_role_spec_without_policy_server(self):
        """RoleSpec without policy_server defaults to None."""
        d = {
            "prompt": "Hang seatbelt.",
            "max_time_s": 60,
            "success_when": {"at_home": True, "value_gte": -0.7},
        }
        spec = RoleSpec.from_dict(d)
        assert spec.policy_server is None

    def test_fold_cloth_self_play_json_loads(self):
        """The fold_cloth piper self-play task spec loads correctly."""
        task_spec = TaskSpec.from_json(
            "lerobot_example_config_files/task_specs/fold_cloth/piper_self_play.json"
        )
        assert task_spec.task_id == "fold_cloth"
        assert len(task_spec.roles) == 2
        assert "folder" in task_spec.roles
        assert "disturber" in task_spec.roles
        assert task_spec.roles["folder"].policy_server is not None
        assert task_spec.roles["folder"].policy_server.port == 8001
        assert task_spec.roles["disturber"].policy_server.port == 8002
        assert task_spec.roles["folder"].max_time_s is None
        assert task_spec.roles["folder"].success_when.is_home_only
        assert not task_spec.roles["folder"].success_when.needs_value_model
        assert task_spec.has_reset
        assert not task_spec.has_safety

    def test_seatbelt_self_play_json_still_loads(self):
        """The seatbelt arxx5 self-play task spec still loads correctly."""
        task_spec = TaskSpec.from_json(
            "lerobot_example_config_files/task_specs/seatbelt/arxx5_self_play.json"
        )
        assert task_spec.task_id == "seatbelt"
        assert task_spec.roles["builder"].max_time_s == 60
        assert task_spec.roles["builder"].success_when.value_gte == -0.7
        assert task_spec.roles["builder"].success_when.needs_value_model
        assert not task_spec.roles["builder"].success_when.is_home_only
        assert task_spec.roles["builder"].policy_server is None
        assert task_spec.has_safety


# ---------------------------------------------------------------------------
# SuccessCondition properties
# ---------------------------------------------------------------------------

class TestSuccessConditionProperties:
    """Test needs_value_model and is_home_only properties."""

    def test_home_only_no_thresholds(self):
        sc = SuccessCondition(at_home=True)
        assert sc.is_home_only is True
        assert sc.needs_value_model is False

    def test_needs_value_with_gte(self):
        sc = SuccessCondition(at_home=True, value_gte=-0.7)
        assert sc.is_home_only is False
        assert sc.needs_value_model is True

    def test_needs_value_with_lte(self):
        sc = SuccessCondition(at_home=True, value_lte=-0.97)
        assert sc.is_home_only is False
        assert sc.needs_value_model is True

    def test_needs_value_with_both(self):
        sc = SuccessCondition(at_home=True, value_gte=-0.7, value_lte=-0.97)
        assert sc.needs_value_model is True
        assert sc.is_home_only is False

    def test_not_at_home_not_home_only(self):
        sc = SuccessCondition(at_home=False)
        assert sc.is_home_only is False
