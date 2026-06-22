"""Tests for recording.runtime.control_loop — fixed-rate loop skeleton."""

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from lerobot.recording.runtime.control_loop import (
    ControlLoop,
    EpisodeResult,
    LoopDirective,
    TeleopSource,
)


class FakeRobot:
    def __init__(self, num_joints=3):
        self.action_features = {f"j{i}.pos": True for i in range(num_joints)}
        self._obs = {
            "observation.state": np.zeros(num_joints),
            "observation.velocity": np.zeros(num_joints),
        }
        self.sent_actions = []

    def get_observation(self):
        return dict(self._obs)

    def get_joint_positions(self):
        return {f"j{i}.pos": 0.0 for i in range(len(self.action_features))}

    def send_action(self, action):
        self.sent_actions.append(dict(action))
        return action


class FakeTeleop:
    def get_action(self):
        return {"j0.pos": 1.0, "j1.pos": 2.0, "j2.pos": 3.0}


class ConstantActionSource:
    """Always returns a fixed action (sends to robot like a real source would)."""
    def __init__(self, action, robot=None):
        self._action = action
        self._robot = robot

    def get_action(self, observation, step_id):
        action = dict(self._action)
        if self._robot is not None:
            self._robot.send_action(action)
        return action


class CountingHook:
    """Counts ticks and optionally returns a directive at a given step."""
    def __init__(self, break_at=None, hold_at=None):
        self.tick_count = 0
        self._break_at = break_at
        self._hold_at = hold_at

    def on_tick(self, ctx, events):
        self.tick_count += 1
        if self._break_at is not None and ctx["step_id"] >= self._break_at:
            return LoopDirective.BREAK
        if self._hold_at is not None and ctx["step_id"] == self._hold_at:
            return LoopDirective.SKIP_ACTION
        return LoopDirective.CONTINUE


class TestControlLoop:
    def test_runs_for_duration(self):
        robot = FakeRobot()
        action_src = ConstantActionSource({"j0.pos": 1.0, "j1.pos": 2.0, "j2.pos": 3.0}, robot=robot)
        loop = ControlLoop(robot, fps=30)
        events = {}

        result = loop.run_episode(
            action_source=action_src,
            hooks=[],
            control_time_s=0.1,  # 100ms → ~3 steps at 30fps
            events=events,
        )

        assert isinstance(result, EpisodeResult)
        assert result.step_count > 0
        assert result.duration_s >= 0.09
        assert len(robot.sent_actions) > 0

    def test_stop_recording_breaks(self):
        robot = FakeRobot()
        action_src = ConstantActionSource({"j0.pos": 1.0})
        loop = ControlLoop(robot, fps=30)
        events = {"stop_recording": True}

        result = loop.run_episode(action_src, [], control_time_s=10.0, events=events)
        assert result.step_count == 0

    def test_exit_early_breaks(self):
        robot = FakeRobot()
        action_src = ConstantActionSource({"j0.pos": 1.0})
        loop = ControlLoop(robot, fps=30)
        events = {"exit_early": True}

        result = loop.run_episode(action_src, [], control_time_s=10.0, events=events)
        assert result.step_count == 0

    def test_hook_break_stops_loop(self):
        robot = FakeRobot()
        action_src = ConstantActionSource({"j0.pos": 1.0})
        hook = CountingHook(break_at=2)
        loop = ControlLoop(robot, fps=100)
        events = {}

        result = loop.run_episode(action_src, [hook], control_time_s=10.0, events=events)
        assert result.step_count <= 3

    def test_hook_skip_action(self):
        robot = FakeRobot()
        action_src = ConstantActionSource({"j0.pos": 1.0})
        hook = CountingHook(hold_at=0, break_at=2)
        loop = ControlLoop(robot, fps=100)
        events = {}

        result = loop.run_episode(action_src, [hook], control_time_s=10.0, events=events)
        # Step 0 was skipped, so first sent action should be from step 1
        assert result.step_count > 0

    def test_multiple_hooks(self):
        robot = FakeRobot()
        action_src = ConstantActionSource({"j0.pos": 1.0})
        hook1 = CountingHook()
        hook2 = CountingHook(break_at=3)
        loop = ControlLoop(robot, fps=100)
        events = {}

        result = loop.run_episode(action_src, [hook1, hook2], control_time_s=10.0, events=events)
        assert hook1.tick_count > 0
        assert hook2.tick_count > 0

    def test_none_action_skips_send(self):
        robot = FakeRobot()

        class NoneSource:
            def get_action(self, obs, step_id):
                return None

        loop = ControlLoop(robot, fps=100)
        events = {}
        hook = CountingHook(break_at=3)

        result = loop.run_episode(NoneSource(), [hook], control_time_s=10.0, events=events)
        assert len(robot.sent_actions) == 0

    def test_run_indefinite(self):
        robot = FakeRobot()
        action_src = ConstantActionSource({"j0.pos": 1.0})
        loop = ControlLoop(robot, fps=100)
        events = {}

        # Set stop after a few iterations
        import threading
        def stop_later():
            time.sleep(0.05)
            events["stop_recording"] = True
        t = threading.Thread(target=stop_later, daemon=True)
        t.start()

        result = loop.run_indefinite(action_src, events)
        assert result.step_count > 0
        t.join()


class TestTeleopSource:
    def test_returns_teleop_action(self):
        robot = FakeRobot()
        teleop = FakeTeleop()
        src = TeleopSource(teleop, robot)
        action = src.get_action({}, 0)
        assert action is not None
        assert "j0.pos" in action
        assert len(robot.sent_actions) == 1

    def test_no_teleop_returns_joint_positions(self):
        robot = FakeRobot()
        src = TeleopSource(None, robot)
        action = src.get_action({}, 0)
        assert action is not None


class TestWriteFrameFailureEscalation:
    def test_fatal_flag_set_after_10_failures(self):
        """_write_frame should set _write_frame_fatal after 10 consecutive failures."""
        from unittest.mock import patch
        robot = FakeRobot()
        mock_dataset = MagicMock()
        loop = ControlLoop(robot, fps=30, dataset=mock_dataset)

        # Force build_dataset_frame to raise so every _write_frame call fails
        with patch("lerobot.datasets.utils.build_dataset_frame", side_effect=RuntimeError("boom")):
            for i in range(10):
                loop._write_frame({"obs": 0}, {"act": 0}, "test", False)

        assert loop._write_fail_count >= 10
        assert loop._write_frame_fatal is True

    def test_fatal_flag_resets_on_success(self):
        """A successful write resets the failure counter and fatal flag."""
        from unittest.mock import patch, MagicMock as MM
        robot = FakeRobot()
        mock_dataset = MM()
        loop = ControlLoop(robot, fps=30, dataset=mock_dataset)

        # Set failure state
        loop._write_fail_count = 9
        loop._write_frame_fatal = False

        # One more failure to trigger fatal
        with patch("lerobot.datasets.utils.build_dataset_frame", side_effect=RuntimeError("boom")):
            loop._write_frame({"obs": 0}, {"act": 0}, "test", False)
        assert loop._write_frame_fatal is True

        # Now mock a successful write
        with patch("lerobot.datasets.utils.build_dataset_frame", return_value={}):
            mock_dataset.add_frame = MM()
            loop._write_frame({"obs": 0}, {"act": 0}, "test", False)

        assert loop._write_fail_count == 0
        assert loop._write_frame_fatal is False

    def test_episode_stops_on_write_fatal(self):
        """run_episode should set stop_recording when _write_frame_fatal is True."""
        from unittest.mock import patch
        robot = FakeRobot()
        mock_dataset = MagicMock()
        loop = ControlLoop(robot, fps=100, dataset=mock_dataset)
        events = {}

        # Pre-set failure count to 9 so the first failing _write_frame triggers fatal
        loop._write_fail_count = 9

        action_src = ConstantActionSource({"j0.pos": 1.0, "j1.pos": 2.0, "j2.pos": 3.0})
        with patch("lerobot.datasets.utils.build_dataset_frame", side_effect=RuntimeError("boom")):
            result = loop.run_episode(action_src, [], control_time_s=10.0, events=events)

        # Should have stopped quickly — fatal triggered on first _write_frame,
        # stop_recording checked at top of next iteration
        assert events.get("stop_recording") is True
        assert result.step_count <= 2

    def test_typeerror_is_swallowed(self):
        """Round 4 whitelist: TypeError from camera/sensor backends returning
        None must be swallowed (not propagated), so a single bad frame does
        not crash the recording loop. Increments the failure counter only.

        Locks in the MR !10 round 4 trade-off: tolerating None-induced
        TypeError is the deliberate choice, rationale in commit f51ab04f.
        """
        from unittest.mock import patch
        robot = FakeRobot()
        mock_dataset = MagicMock()
        loop = ControlLoop(robot, fps=30, dataset=mock_dataset)

        # Simulate camera backend returning None triggering TypeError inside
        # the build pipeline. Must NOT raise out of _write_frame.
        with patch(
            "lerobot.datasets.utils.build_dataset_frame",
            side_effect=TypeError("cannot convert None to ndarray"),
        ):
            loop._write_frame({"obs": 0}, {"act": 0}, "test", False)

        # Failure was counted, but recording continues (not fatal yet).
        assert loop._write_fail_count == 1
        assert loop._write_frame_fatal is False

        # Same for AttributeError (e.g. `.shape` on None).
        with patch(
            "lerobot.datasets.utils.build_dataset_frame",
            side_effect=AttributeError("'NoneType' object has no attribute 'shape'"),
        ):
            loop._write_frame({"obs": 0}, {"act": 0}, "test", False)
        assert loop._write_fail_count == 2
        assert loop._write_frame_fatal is False

    def test_nameerror_is_propagated(self):
        """Unambiguous code bugs (NameError, ImportError) must propagate out
        of _write_frame — they cannot be confused with runtime data issues
        and swallowing them would hide developer errors for up to 10 frames.

        Locks the other side of the whitelist trade-off: expanded types are
        TypeError/AttributeError only, never NameError.
        """
        from unittest.mock import patch
        robot = FakeRobot()
        mock_dataset = MagicMock()
        loop = ControlLoop(robot, fps=30, dataset=mock_dataset)

        with patch(
            "lerobot.datasets.utils.build_dataset_frame",
            side_effect=NameError("name 'undefined_var' is not defined"),
        ):
            with pytest.raises(NameError):
                loop._write_frame({"obs": 0}, {"act": 0}, "test", False)

        # ImportError also propagates (missing module = dev bug).
        with patch(
            "lerobot.datasets.utils.build_dataset_frame",
            side_effect=ImportError("No module named 'nonexistent'"),
        ):
            with pytest.raises(ImportError):
                loop._write_frame({"obs": 0}, {"act": 0}, "test", False)


class _StubInterventionRuntime:
    """Minimal InterventionRuntime stub for _enter_intervention tests."""

    def __init__(self, teleop, pose_sync_duration_s=0.0, leader_movement_timeout_s=0.05):
        self.teleop = teleop
        self.pose_sync_duration_s = pose_sync_duration_s
        self.waiting_evacuation_time_s = 0.0
        self.leader_movement_timeout_s = leader_movement_timeout_s
        self.is_active = False
        self.intervention_count = 0
        self.entered_via_helper = False

    def enter(self):
        self.is_active = True
        self.intervention_count += 1
        self.entered_via_helper = True

    def exit(self):
        self.is_active = False


class _MovingTeleop:
    """Leader stub that 'moves' on the second get_action() call."""
    name = "piper_leader"
    action_features = {f"j{i}.pos": float for i in range(3)}
    feedback_features = {f"j{i}.pos": float for i in range(3)}

    def __init__(self):
        self._calls = 0
        self.feedbacks = []

    def get_action(self):
        self._calls += 1
        if self._calls == 1:
            return {"j0.pos": 0.0, "j1.pos": 0.0, "j2.pos": 0.0}
        return {"j0.pos": 100.0, "j1.pos": 0.0, "j2.pos": 0.0}  # > move_threshold

    def send_feedback(self, pose):
        self.feedbacks.append(dict(pose))


class _FeedbackLessTeleop:
    """Leader without force feedback (mirrors so100/koch/keyboard).

    `send_feedback` raises NotImplementedError; if `_enter_intervention` is
    correctly checking `feedback_features` it should never be called.
    """

    name = "so100_leader"
    action_features = {f"j{i}.pos": float for i in range(3)}
    feedback_features = {}  # ← critical: empty means "no feedback support"

    def __init__(self):
        self._calls = 0

    def get_action(self):
        self._calls += 1
        if self._calls == 1:
            return {"j0.pos": 0.0, "j1.pos": 0.0, "j2.pos": 0.0}
        return {"j0.pos": 100.0, "j1.pos": 0.0, "j2.pos": 0.0}

    def send_feedback(self, pose):
        raise NotImplementedError("so100 leaders cannot accept feedback")


class TestEnterIntervention:
    """Tests for ControlLoop._enter_intervention atomic 3-phase sequence (H1+M3)."""

    def test_atomic_enter_on_success(self):
        """On successful return, intervention.enter() must have been called."""
        robot = FakeRobot()
        loop = ControlLoop(robot, fps=100)
        teleop = _MovingTeleop()
        intervention = _StubInterventionRuntime(teleop, pose_sync_duration_s=0.02)

        result = loop._enter_intervention(intervention, events={})

        assert result is True
        assert intervention.is_active is True, (
            "M3: _enter_intervention should call intervention.enter() itself"
        )
        assert intervention.intervention_count == 1

    def test_no_enter_on_interruption(self):
        """When interrupted, intervention.enter() must NOT be called."""
        robot = FakeRobot()
        loop = ControlLoop(robot, fps=100)
        teleop = _MovingTeleop()
        intervention = _StubInterventionRuntime(teleop, pose_sync_duration_s=1.0)

        # exit_early triggers immediately during leader_sync_loop
        result = loop._enter_intervention(intervention, events={"exit_early": True})

        assert result is False
        assert intervention.is_active is False
        assert intervention.intervention_count == 0

    def test_skips_leader_sync_for_feedbackless_teleop(self):
        """H1: leaders without feedback_features should skip Phase 2 entirely.

        Without this guard, so100/so101/koch users would crash on Ctrl+Space
        because their send_feedback raises NotImplementedError.
        """
        robot = FakeRobot()
        loop = ControlLoop(robot, fps=100)
        teleop = _FeedbackLessTeleop()
        intervention = _StubInterventionRuntime(teleop, pose_sync_duration_s=0.02)

        # Should NOT raise NotImplementedError, should still enter successfully
        # (Phase 3 wait_for_leader_movement still runs and detects movement).
        result = loop._enter_intervention(intervention, events={})

        assert result is True
        assert intervention.is_active is True

    def test_leaderless_workflow_skips_both_phases(self):
        """ARX-style leader-less takeover: teleop=None should still succeed."""
        robot = FakeRobot()
        loop = ControlLoop(robot, fps=100)
        intervention = _StubInterventionRuntime(teleop=None, pose_sync_duration_s=0.02)

        result = loop._enter_intervention(intervention, events={})

        assert result is True
        assert intervention.is_active is True


class TestEvacuationHold:
    """Tests for ControlLoop._evacuation_hold per-tick follower refresh (M3).

    Locks the round 2/3 I3+I5 fix: blocking `time.sleep(duration_s)` was
    replaced by a per-tick send_action loop so the follower does not drift
    during the evacuation wait. Guards against a regression back to sleep.
    """

    def test_sends_follower_position_each_tick(self):
        """During evacuation, robot.send_action must be called ≥N times with
        the current get_joint_positions() value each tick.
        """
        robot = FakeRobot()
        loop = ControlLoop(robot, fps=100)
        events = {}

        loop._evacuation_hold(duration_s=0.05, events=events)

        # At 100 fps × 0.05s we expect ~5 ticks (allow wide margin for timing
        # jitter on loaded CI). Must be > 1 to distinguish from a sleep-only
        # implementation.
        assert len(robot.sent_actions) >= 2, (
            f"_evacuation_hold should send follower position each tick, "
            f"got only {len(robot.sent_actions)} calls in 0.05s"
        )
        # Every send was the current joint position (all zeros in FakeRobot).
        expected = {f"j{i}.pos": 0.0 for i in range(3)}
        for call in robot.sent_actions:
            assert call == expected, (
                f"expected {expected}, got {call}"
            )

    def test_noop_for_zero_duration(self):
        """duration_s <= 0 is an explicit no-op — no send_action calls."""
        robot = FakeRobot()
        loop = ControlLoop(robot, fps=100)

        loop._evacuation_hold(duration_s=0.0, events={})
        loop._evacuation_hold(duration_s=-1.0, events={})

        assert robot.sent_actions == []

    def test_interrupted_by_exit_early(self):
        """exit_early mid-evacuation breaks the loop immediately."""
        robot = FakeRobot()
        loop = ControlLoop(robot, fps=100)
        events = {"exit_early": True}

        loop._evacuation_hold(duration_s=1.0, events=events)

        # With exit_early set from the start, the loop should break on the
        # first iteration check and send at most 0 actions.
        assert len(robot.sent_actions) == 0

    def test_interrupted_by_stop_recording(self):
        """stop_recording mid-evacuation also breaks immediately."""
        robot = FakeRobot()
        loop = ControlLoop(robot, fps=100)
        events = {"stop_recording": True}

        loop._evacuation_hold(duration_s=1.0, events=events)

        assert len(robot.sent_actions) == 0

    def test_follower_tracks_moving_joint_positions(self):
        """If get_joint_positions() returns different values across ticks,
        each send_action reflects the *current* reading, not a stale snapshot.

        Guards against a regression that caches joint_positions once at the
        top and reuses it every tick.
        """
        robot = FakeRobot()
        loop = ControlLoop(robot, fps=100)

        # Make get_joint_positions return a monotonically increasing counter.
        counter = {"n": 0}

        def counting_get():
            counter["n"] += 1
            return {f"j{i}.pos": float(counter["n"]) for i in range(3)}

        robot.get_joint_positions = counting_get
        loop._evacuation_hold(duration_s=0.05, events={})

        # Each sent_action must have strictly increasing values if the
        # implementation re-reads per tick.
        assert len(robot.sent_actions) >= 2
        first_val = robot.sent_actions[0]["j0.pos"]
        last_val = robot.sent_actions[-1]["j0.pos"]
        assert last_val > first_val, (
            f"expected joint position to advance across ticks; "
            f"first={first_val}, last={last_val}"
        )


class TestHandleSwitchInferMode:
    """Tests for ControlLoop._handle_switch_infer_mode Ctrl+Space toggle (M4).

    Safety-critical path: if the rollback logic breaks, a failed takeover
    attempt would permanently freeze inference.
    """

    def test_enter_success_sets_both_flags(self):
        """Successful entry → (current_intervention=True, inference_paused=True)."""
        robot = FakeRobot()
        loop = ControlLoop(robot, fps=100)
        teleop = _MovingTeleop()
        intervention = _StubInterventionRuntime(teleop, pose_sync_duration_s=0.02)

        ci, ip = loop._handle_switch_infer_mode(
            current_intervention=False,
            inference_paused=False,
            intervention=intervention,
            events={},
            action_source=object(),
            lang_prompt="",
            log_say_fn=None,
        )

        assert ci is True
        assert ip is True
        assert intervention.is_active is True

    def test_rollback_on_exception(self, monkeypatch):
        """If _enter_intervention raises (e.g. hardware SDK failure), the
        helper must roll back inference_paused to False so inference resumes
        immediately, AND notify the operator via log_say.

        Regression guard: breaking this is the worst failure mode of the
        MR — a silent permanent freeze of policy inference.
        """
        robot = FakeRobot()
        loop = ControlLoop(robot, fps=100)
        intervention = _StubInterventionRuntime(teleop=None)

        def boom(*args, **kwargs):
            raise RuntimeError("simulated Piper SDK comm failure")

        monkeypatch.setattr(loop, "_enter_intervention", boom)

        announcements = []

        def fake_log_say(msg, **kwargs):
            announcements.append(msg)

        ci, ip = loop._handle_switch_infer_mode(
            current_intervention=False,
            inference_paused=False,
            intervention=intervention,
            events={},
            action_source=object(),
            lang_prompt="",
            log_say_fn=fake_log_say,
        )

        # Critical invariants: both flags rolled back to False.
        assert ci is False, "intervention must not activate on exception"
        assert ip is False, (
            "inference_paused MUST roll back — leaving it True permanently "
            "freezes policy inference"
        )
        assert intervention.is_active is False
        # Operator is informed that the takeover attempt failed.
        assert any("接管进入失败" in msg for msg in announcements), (
            f"expected '接管进入失败' in log_say calls, got: {announcements}"
        )

    def test_rollback_on_interruption(self):
        """If _enter_intervention returns False (exit_early / stop_recording
        during Phase 2 or 3), the helper must also roll back — not just on
        exceptions but also on normal early-return interruption.
        """
        robot = FakeRobot()
        loop = ControlLoop(robot, fps=100)
        teleop = _MovingTeleop()
        intervention = _StubInterventionRuntime(teleop, pose_sync_duration_s=1.0)

        ci, ip = loop._handle_switch_infer_mode(
            current_intervention=False,
            inference_paused=False,
            intervention=intervention,
            events={"exit_early": True},
            action_source=object(),
            lang_prompt="",
            log_say_fn=None,
        )

        assert ci is False
        assert ip is False
        assert intervention.is_active is False

    def test_exit_calls_intervention_exit_and_reinit(self):
        """Second Ctrl+Space (exit path) → call intervention.exit(), reinit
        policy, and clear both flags.
        """
        robot = FakeRobot()
        loop = ControlLoop(robot, fps=100)
        teleop = _MovingTeleop()
        intervention = _StubInterventionRuntime(teleop)
        intervention.is_active = True  # simulate already in intervention

        reinit_calls = []
        reset_calls = []

        class FakeActionSource:
            def reinit(self, prompt):
                reinit_calls.append(prompt)

            def reset_for_resume(self):
                reset_calls.append(True)

        announcements = []

        def fake_log_say(msg, **kwargs):
            announcements.append(msg)

        ci, ip = loop._handle_switch_infer_mode(
            current_intervention=True,
            inference_paused=True,
            intervention=intervention,
            events={},
            action_source=FakeActionSource(),
            lang_prompt="pick cube",
            log_say_fn=fake_log_say,
        )

        assert ci is False
        assert ip is False
        assert intervention.is_active is False  # intervention.exit() called
        assert reinit_calls == ["pick cube"]
        assert reset_calls == [True]
        assert any("请立即撤离" in msg for msg in announcements)


class TestHandleResumeInferMode:
    """Tests for Ctrl+Enter resume behavior shared by record/infer_record loops."""

    def test_resume_exits_active_intervention_and_reinitializes_policy(self):
        robot = FakeRobot()
        loop = ControlLoop(robot, fps=100)
        intervention = _StubInterventionRuntime(teleop=None)
        intervention.is_active = True

        reinit_calls = []
        reset_calls = []

        class FakeActionSource:
            def reinit(self, prompt):
                reinit_calls.append(prompt)

            def reset_for_resume(self):
                reset_calls.append(True)

        ci, ip = loop._handle_resume_infer_mode(
            current_intervention=True,
            inference_paused=True,
            intervention=intervention,
            events={},
            action_source=FakeActionSource(),
            lang_prompt="resume prompt",
            log_say_fn=None,
        )

        assert ci is False
        assert ip is False
        assert intervention.is_active is False
        assert reinit_calls == ["resume prompt"]
        assert reset_calls == [True]

    def test_resume_clears_plain_pause_and_reinitializes_policy(self):
        robot = FakeRobot()
        loop = ControlLoop(robot, fps=100)

        reinit_calls = []
        reset_calls = []

        class FakeActionSource:
            def reinit(self, prompt):
                reinit_calls.append(prompt)

            def reset_for_resume(self):
                reset_calls.append(True)

        ci, ip = loop._handle_resume_infer_mode(
            current_intervention=False,
            inference_paused=True,
            intervention=None,
            events={},
            action_source=FakeActionSource(),
            lang_prompt="resume prompt",
            log_say_fn=None,
        )

        assert ci is False
        assert ip is False
        assert reinit_calls == ["resume prompt"]
        assert reset_calls == [True]


class TestEpisodeHotkeyFlow:
    def test_ctrl_enter_resumes_policy_after_ctrl_space_takeover(self):
        robot = FakeRobot()
        loop = ControlLoop(robot, fps=100)
        intervention = _StubInterventionRuntime(teleop=None)

        class HotkeyHook:
            def on_tick(self, ctx, events):
                if ctx["step_id"] == 0:
                    events["switch_infer_mode"] = True
                elif ctx["step_id"] == 1:
                    events["resume_inference"] = True
                elif ctx["step_id"] >= 3:
                    return LoopDirective.BREAK
                return LoopDirective.CONTINUE

        class CountingActionSource:
            def __init__(self):
                self.get_action_count = 0
                self.reinit_calls = []
                self.reset_count = 0

            def get_action(self, observation, step_id):
                self.get_action_count += 1
                return {"j0.pos": float(step_id)}

            def reinit(self, prompt):
                self.reinit_calls.append(prompt)

            def reset_for_resume(self):
                self.reset_count += 1

        action_src = CountingActionSource()
        loop.run_episode(
            action_source=action_src,
            hooks=[HotkeyHook()],
            control_time_s=10.0,
            events={},
            lang_prompt="pick cube",
            intervention=intervention,
        )

        assert intervention.is_active is False
        assert action_src.reinit_calls == ["pick cube"]
        assert action_src.reset_count == 1
        assert action_src.get_action_count >= 2


class TestProgressPrintThrottle:
    """The per-tick ``Episode N | Xs / Xs`` line was originally printed every
    iteration (~30 Hz). Over a 50-min pack-socks episode that's ~90k lines
    on stdout — fine on a fast tty, painful on slow ssh, and a stdout-pipe
    backpressure risk when piped through grep. Throttled to ~1 Hz."""

    def test_prints_once_per_second_when_play_sounds(self, capsys):
        robot = FakeRobot()
        action_src = ConstantActionSource({"j0.pos": 0.0}, robot=robot)
        loop = ControlLoop(robot, fps=30)
        events = {}

        # 0.5 s at 30 Hz → ~15 ticks → expect ~1 line (once-per-30 throttle).
        loop.run_episode(
            action_source=action_src, hooks=[],
            control_time_s=0.5, events=events,
            episode_idx=7, play_sounds=True,
        )
        out = capsys.readouterr().out
        progress_lines = [ln for ln in out.splitlines() if ln.startswith("Episode 7 |")]
        # Allow 1-2 prints to account for the boundary tick at step_id == 0.
        assert 1 <= len(progress_lines) <= 2, (
            f"expected ~1 progress line in 0.5s, got {len(progress_lines)}"
        )

    def test_silent_when_play_sounds_false(self, capsys):
        robot = FakeRobot()
        action_src = ConstantActionSource({"j0.pos": 0.0}, robot=robot)
        loop = ControlLoop(robot, fps=30)
        events = {}

        loop.run_episode(
            action_source=action_src, hooks=[],
            control_time_s=0.3, events=events,
            episode_idx=0, play_sounds=False,
        )
        out = capsys.readouterr().out
        assert "Episode 0 |" not in out

    def test_interval_zero_prints_every_tick(self, capsys):
        """``progress_print_interval_s=0`` reverts to the original ~30 Hz print
        cadence — useful for short episodes or fast-tty debugging where the
        once-per-second throttle is too sparse."""
        robot = FakeRobot()
        action_src = ConstantActionSource({"j0.pos": 0.0}, robot=robot)
        loop = ControlLoop(robot, fps=30)
        events = {}

        # 0.3 s at 30 Hz ≈ 9 ticks. With interval=0 we expect roughly that many
        # lines (give a generous range to account for timing jitter).
        loop.run_episode(
            action_source=action_src, hooks=[],
            control_time_s=0.3, events=events,
            episode_idx=2, play_sounds=True,
            progress_print_interval_s=0,
        )
        out = capsys.readouterr().out
        progress_lines = [ln for ln in out.splitlines() if ln.startswith("Episode 2 |")]
        assert len(progress_lines) >= 5, (
            f"expected ~9 progress lines in 0.3s with interval=0, got {len(progress_lines)}"
        )

    def test_custom_interval_throttles(self, capsys):
        """An explicit interval throttles to that cadence (here 0.2 s ≈ 5 Hz)."""
        robot = FakeRobot()
        action_src = ConstantActionSource({"j0.pos": 0.0}, robot=robot)
        loop = ControlLoop(robot, fps=30)
        events = {}

        # 0.6 s with 0.2 s interval → ~3-4 prints.
        loop.run_episode(
            action_source=action_src, hooks=[],
            control_time_s=0.6, events=events,
            episode_idx=3, play_sounds=True,
            progress_print_interval_s=0.2,
        )
        out = capsys.readouterr().out
        progress_lines = [ln for ln in out.splitlines() if ln.startswith("Episode 3 |")]
        assert 2 <= len(progress_lines) <= 5, (
            f"expected 2-5 progress lines in 0.6s at 0.2s interval, got {len(progress_lines)}"
        )
