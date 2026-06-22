"""Tests for recording.runtime.intervention — human takeover + pose sync."""

import time

import numpy as np
import pytest

from lerobot.recording.runtime.intervention import (
    InterventionRuntime,
    leader_sync_loop,
    wait_for_leader_movement,
    _adaptive_sync_duration,
    pose_sync_loop,
)


class FakeRobot:
    def __init__(self, num_joints=3):
        self.action_features = {f"j{i}.pos": True for i in range(num_joints)}
        self._pos = {f"j{i}.pos": float(i) for i in range(num_joints)}
        self._sent_actions = []

    def get_joint_positions(self):
        return dict(self._pos)

    def get_observation(self):
        return {"observation.state": np.array(list(self._pos.values()))}

    def send_action(self, action):
        self._sent_actions.append(dict(action))
        return action

    def set_gravity_compensation_mode(self):
        pass


class FakeTeleop:
    """Default fake leader: piper-like, supports feedback for the same field set."""

    name = "piper_leader"  # opt into adaptive sync duration in tests

    def __init__(self, action=None, num_joints=3):
        self._action = action or {f"j{i}.pos": float((i + 1) * 10) for i in range(num_joints)}
        self.action_features = {k: float for k in self._action}
        self.feedback_features = dict(self.action_features)
        self._sent_feedbacks = []

    def get_action(self):
        return dict(self._action)

    def send_feedback(self, pose):
        self._sent_feedbacks.append(dict(pose))


class FakeFeedbackLessTeleop:
    """Leader that exposes positions but cannot accept feedback frames.

    Mirrors so100_leader / so101_leader / koch_leader / keyboard / gamepad —
    `feedback_features = {}` and `send_feedback` either raises or no-ops.
    """

    name = "so100_leader"

    def __init__(self, num_joints=3, raise_on_feedback=True):
        self.action_features = {f"j{i}.pos": float for i in range(num_joints)}
        self.feedback_features = {}
        self._raise = raise_on_feedback
        self._sent_feedbacks = []

    def get_action(self):
        return {f"j{i}.pos": 0.0 for i in self.action_features}

    def send_feedback(self, pose):
        if self._raise:
            raise NotImplementedError
        self._sent_feedbacks.append(dict(pose))


class TestInterventionRuntime:
    def test_initial_state(self):
        ir = InterventionRuntime(FakeRobot(), FakeTeleop())
        assert ir.is_active is False
        assert ir.forced is False
        assert ir.intervention_count == 0
        assert ir.intervention_duration_s == 0.0

    def test_enter_intervention(self):
        ir = InterventionRuntime(FakeRobot(), FakeTeleop())
        ir.enter()
        assert ir.is_active is True
        assert ir.intervention_count == 1

    def test_exit_intervention(self):
        ir = InterventionRuntime(FakeRobot(), FakeTeleop())
        ir.enter()
        ir.exit()
        assert ir.is_active is False
        assert ir.intervention_duration_s >= 0.0

    def test_double_enter_does_not_increment(self):
        ir = InterventionRuntime(FakeRobot(), FakeTeleop())
        ir.enter()
        ir.enter()
        assert ir.intervention_count == 1

    def test_exit_when_not_active_is_noop(self):
        ir = InterventionRuntime(FakeRobot(), FakeTeleop())
        ir.exit()  # should not raise
        assert ir.intervention_count == 0

    def test_forced_takeover(self):
        ir = InterventionRuntime(FakeRobot(), FakeTeleop())
        ir.request_forced_takeover("collision_n3")
        assert ir.forced is True
        assert ir.forced_reason == "collision_n3"

    def test_forced_exit_blocked_when_not_home(self):
        ir = InterventionRuntime(FakeRobot(), FakeTeleop())
        ir.request_forced_takeover("test")
        ir.enter()
        allowed = ir.can_exit_forced(is_home=False)
        assert allowed is False

    def test_forced_exit_allowed_when_home(self):
        ir = InterventionRuntime(FakeRobot(), FakeTeleop())
        ir.request_forced_takeover("test")
        ir.enter()
        allowed = ir.can_exit_forced(is_home=True)
        assert allowed is True

    def test_clear_forced(self):
        ir = InterventionRuntime(FakeRobot(), FakeTeleop())
        ir.request_forced_takeover("test")
        ir.clear_forced()
        assert ir.forced is False
        assert ir.forced_reason is None

    def test_get_teleop_action(self):
        teleop = FakeTeleop({"j0.pos": 5.0})
        ir = InterventionRuntime(FakeRobot(), teleop)
        action = ir.get_teleop_action()
        assert action["j0.pos"] == 5.0

    def test_get_teleop_action_no_teleop(self):
        ir = InterventionRuntime(FakeRobot(), teleop=None)
        action = ir.get_teleop_action()
        assert action is None

    def test_get_stats(self):
        ir = InterventionRuntime(FakeRobot(), FakeTeleop())
        ir.enter()
        time.sleep(0.01)
        ir.exit()
        stats = ir.get_stats()
        assert stats["count"] == 1
        assert stats["duration_s"] >= 0.0

    def test_reset(self):
        ir = InterventionRuntime(FakeRobot(), FakeTeleop())
        ir.enter()
        ir.exit()
        ir.request_forced_takeover("test")
        ir.reset()
        assert ir.is_active is False
        assert ir.forced is False
        assert ir.intervention_count == 0
        assert ir.intervention_duration_s == 0.0


class TestPoseSyncLoop:
    def test_basic_sync(self):
        robot = FakeRobot(num_joints=2)
        robot._pos = {"j0.pos": 0.0, "j1.pos": 0.0}
        teleop = FakeTeleop({"j0.pos": 10.0, "j1.pos": 20.0}, num_joints=2)
        events = {}

        result = pose_sync_loop(robot, teleop, events, sync_duration_s=0.05, fps=30)
        assert result is True
        assert len(robot._sent_actions) > 0

    def test_sync_interrupted_by_stop(self):
        robot = FakeRobot(num_joints=2)
        teleop = FakeTeleop({"j0.pos": 10.0, "j1.pos": 20.0}, num_joints=2)
        events = {"stop_recording": True}

        result = pose_sync_loop(robot, teleop, events, sync_duration_s=1.0, fps=30)
        assert result is False

    def test_sync_interrupted_by_exit_early(self):
        robot = FakeRobot(num_joints=2)
        teleop = FakeTeleop({"j0.pos": 10.0, "j1.pos": 20.0}, num_joints=2)
        events = {"exit_early": True}

        result = pose_sync_loop(robot, teleop, events, sync_duration_s=1.0, fps=30)
        assert result is False


class TestLeaderSyncLoop:
    def test_basic_leader_sync(self):
        """Normal completion: returns True, feedback and actions are sent."""
        robot = FakeRobot(num_joints=2)
        robot._pos = {"j0.pos": 0.0, "j1.pos": 0.0}
        teleop = FakeTeleop({"j0.pos": 50.0, "j1.pos": 50.0})
        events = {}

        result = leader_sync_loop(robot, teleop, events, sync_duration_s=0.05, fps=100)
        assert result is True
        assert len(teleop._sent_feedbacks) > 0
        assert len(robot._sent_actions) > 0

    def test_leader_sync_interrupted_by_exit_early(self):
        robot = FakeRobot(num_joints=2)
        teleop = FakeTeleop({"j0.pos": 50.0, "j1.pos": 50.0})
        events = {"exit_early": True}

        result = leader_sync_loop(robot, teleop, events, sync_duration_s=1.0, fps=30)
        assert result is False

    def test_leader_sync_interrupted_by_stop(self):
        robot = FakeRobot(num_joints=2)
        teleop = FakeTeleop({"j0.pos": 50.0, "j1.pos": 50.0})
        events = {"stop_recording": True}

        result = leader_sync_loop(robot, teleop, events, sync_duration_s=1.0, fps=30)
        assert result is False

    def test_leader_sync_interpolation_endpoints(self):
        """First feedback ~= leader start, last feedback ~= follower target."""
        robot = FakeRobot(num_joints=2)
        robot._pos = {"j0.pos": 0.0, "j1.pos": 0.0}
        teleop = FakeTeleop({"j0.pos": 100.0, "j1.pos": 100.0})
        events = {}

        leader_sync_loop(robot, teleop, events, sync_duration_s=0.1, fps=100)

        first_fb = teleop._sent_feedbacks[0]
        last_fb = teleop._sent_feedbacks[-1]
        # Linear: first step t=0 → value = start (100.0), last step t=1 → target (0.0).
        assert first_fb["j0.pos"] == pytest.approx(100.0, abs=5.0)
        assert last_fb["j0.pos"] == pytest.approx(0.0, abs=5.0)

    def test_leader_sync_holds_position_for_unmatched_keys(self):
        """When start has key X but target lacks it, pose[X] should hold start[X]
        — never default to 0.0 (physical safety hazard: would jump joint to zero).
        """
        robot = FakeRobot(num_joints=2)
        # Follower only exposes j0; j1 is missing — no target for j1.
        robot.action_features = {"j0.pos": True}
        robot._pos = {"j0.pos": 50.0}
        # Leader exposes both j0 and j1 (and feedback_features mirrors that).
        teleop = FakeTeleop({"j0.pos": 0.0, "j1.pos": 80.0}, num_joints=2)
        events = {}

        leader_sync_loop(robot, teleop, events, sync_duration_s=0.05, fps=100)

        # Every feedback frame must hold j1 at its leader start (80.0),
        # NOT default it to 0.0.
        assert len(teleop._sent_feedbacks) > 0
        for fb in teleop._sent_feedbacks:
            assert fb["j1.pos"] == pytest.approx(80.0, abs=0.1), (
                f"j1 should hold start position 80.0, got {fb['j1.pos']}"
            )

    def test_leader_sync_uses_feedback_features_not_action_features(self):
        """leader_sync_loop should iterate `feedback_features`, not `action_features`.

        For a leader that has narrower feedback_features than action_features,
        only the feedback_features keys should be sent to send_feedback.
        """
        robot = FakeRobot(num_joints=3)
        robot._pos = {"j0.pos": 10.0, "j1.pos": 20.0, "j2.pos": 30.0}
        teleop = FakeTeleop({"j0.pos": 0.0, "j1.pos": 0.0, "j2.pos": 0.0})
        # Only j0/j1 accept feedback; j2 is read-only.
        teleop.feedback_features = {"j0.pos": float, "j1.pos": float}
        events = {}

        leader_sync_loop(robot, teleop, events, sync_duration_s=0.05, fps=100)

        for fb in teleop._sent_feedbacks:
            assert set(fb.keys()) == {"j0.pos", "j1.pos"}, (
                f"send_feedback should only receive feedback_features keys, got {fb.keys()}"
            )

    def test_leader_sync_skips_adaptive_for_non_piper_teleop(self):
        """Non-Piper teleops use the fixed sync_duration_s — no adaptive scaling.

        Adaptive thresholds (40/80) only translate to Piper's [-100,100] range;
        a degree-based 90° distance must NOT trigger 2× duration on so100.
        """
        from lerobot.recording.runtime.intervention import _adaptive_sync_duration

        robot = FakeRobot(num_joints=2)
        robot._pos = {"j0.pos": 0.0, "j1.pos": 0.0}
        teleop = FakeTeleop({"j0.pos": 90.0, "j1.pos": 0.0})
        teleop.name = "so100_leader"  # not in _PIPER_PCT_TELEOPS
        events = {}

        # The adaptive function would return 2.0 for max_diff=90, but
        # leader_sync_loop should bypass it for non-Piper.
        # Use a tiny base to keep test fast.
        leader_sync_loop(robot, teleop, events, sync_duration_s=0.02, fps=100)

        # Number of steps should be ≈ 0.02 * 100 = 2 (not 4 from 2× scaling).
        assert 1 <= len(teleop._sent_feedbacks) <= 3, (
            f"non-piper teleop should not trigger adaptive 2x scaling; "
            f"got {len(teleop._sent_feedbacks)} feedback frames"
        )

    def test_leader_sync_holds_follower(self):
        """During sync, follower receives its position every tick to stay stable."""
        robot = FakeRobot(num_joints=2)
        robot._pos = {"j0.pos": 5.0, "j1.pos": 10.0}
        teleop = FakeTeleop({"j0.pos": 50.0, "j1.pos": 50.0})
        events = {}

        leader_sync_loop(robot, teleop, events, sync_duration_s=0.05, fps=100)

        # Every sent action should be the follower's original position
        for action in robot._sent_actions:
            assert action["j0.pos"] == pytest.approx(5.0, abs=0.1)
            assert action["j1.pos"] == pytest.approx(10.0, abs=0.1)

    def test_leader_sync_warns_on_empty_target(self):
        """Dual-arm mismatch: prefixed follower keys vs unprefixed teleop.

        Follower has left_/right_ prefixed keys, teleop has plain j0.pos.
        leader_sync_loop should log a warning AND skip the feedback loop
        entirely — proceeding would call send_feedback({}) every tick,
        which trips piper_leader's strict missing-key validation.
        """
        robot = FakeRobot(num_joints=2)
        # Simulate dual-arm follower with left_ prefix — no matching keys
        robot.action_features = {"left_j0.pos": True, "left_j1.pos": True}
        robot._pos = {"left_j0.pos": 0.0, "left_j1.pos": 0.0}
        teleop = FakeTeleop({"j0.pos": 50.0, "j1.pos": 50.0})
        events = {}

        class CapturingLogger:
            def __init__(self):
                self.messages = []

            def log(self, msg):
                self.messages.append(msg)

        logger = CapturingLogger()
        result = leader_sync_loop(
            robot, teleop, events, sync_duration_s=0.05, fps=100, logger=logger
        )
        # Graceful skip: returns True so _enter_intervention does not
        # rollback and continues to Phase 3.
        assert result is True
        # Should have warned about the key mismatch
        assert any("no matching keys" in m for m in logger.messages), (
            f"expected mismatch warning, got: {logger.messages}"
        )
        # Must also log the skip message so operators know the sync was
        # short-circuited, not silently a no-op.
        assert any("skipping feedback loop" in m for m in logger.messages), (
            f"expected skip log, got: {logger.messages}"
        )
        # I1 critical invariant: no send_feedback call was ever made.
        # Prior to the early-return fix this list would contain N empty
        # dicts (one per num_steps), each triggering piper_leader KeyError.
        assert teleop._sent_feedbacks == [], (
            f"expected no feedback frames when keys mismatch, got "
            f"{len(teleop._sent_feedbacks)}: {teleop._sent_feedbacks}"
        )
        # And no follower hold ticks either — the loop body never ran.
        assert robot._sent_actions == [], (
            f"expected no follower actions, got {len(robot._sent_actions)}"
        )


class TestWaitForLeaderMovement:
    def test_detects_movement(self):
        """Returns True when leader moves beyond threshold."""
        robot = FakeRobot(num_joints=2)
        robot._pos = {"j0.pos": 0.0, "j1.pos": 0.0}

        call_count = [0]
        baseline = {"j0.pos": 10.0, "j1.pos": 10.0}

        class MovingTeleop:
            action_features = {"j0.pos": float, "j1.pos": float}
            def get_action(self):
                call_count[0] += 1
                if call_count[0] <= 1:
                    return dict(baseline)  # baseline capture
                # After baseline, simulate movement
                return {"j0.pos": 15.0, "j1.pos": 10.0}  # delta=5 > threshold=2

        result = wait_for_leader_movement(
            robot, MovingTeleop(), {}, fps=100, move_threshold=2.0, timeout_s=5.0,
        )
        assert result is True

    def test_no_movement_timeout(self):
        """Returns True on timeout if no movement detected."""
        robot = FakeRobot(num_joints=2)
        robot._pos = {"j0.pos": 0.0, "j1.pos": 0.0}
        teleop = FakeTeleop({"j0.pos": 10.0, "j1.pos": 10.0})

        result = wait_for_leader_movement(
            robot, teleop, {}, fps=100, move_threshold=2.0, timeout_s=0.05,
        )
        assert result is True

    def test_interrupted_by_exit_early(self):
        robot = FakeRobot(num_joints=2)
        teleop = FakeTeleop({"j0.pos": 10.0, "j1.pos": 10.0})
        events = {"exit_early": True}

        result = wait_for_leader_movement(
            robot, teleop, events, fps=100, move_threshold=2.0, timeout_s=5.0,
        )
        assert result is False

    def test_follower_held_stable(self):
        """During wait, follower keeps receiving its position."""
        robot = FakeRobot(num_joints=2)
        robot._pos = {"j0.pos": 3.0, "j1.pos": 7.0}
        teleop = FakeTeleop({"j0.pos": 10.0, "j1.pos": 10.0})

        wait_for_leader_movement(
            robot, teleop, {}, fps=100, move_threshold=2.0, timeout_s=0.05,
        )

        assert len(robot._sent_actions) > 0
        for action in robot._sent_actions:
            assert action["j0.pos"] == pytest.approx(3.0, abs=0.1)
            assert action["j1.pos"] == pytest.approx(7.0, abs=0.1)


class TestAdaptiveSyncDuration:
    # Thresholds: small < 40, medium 40-80, large > 80 (Piper percentage range)

    def test_small_distance(self):
        start = {"j0.pos": 0.0}
        target = {"j0.pos": 30.0}  # < 40
        assert _adaptive_sync_duration(start, target, 1.0) == 1.0

    def test_medium_distance(self):
        start = {"j0.pos": 0.0}
        target = {"j0.pos": 60.0}  # 40 < 60 <= 80
        assert _adaptive_sync_duration(start, target, 1.0) == 1.5

    def test_large_distance(self):
        start = {"j0.pos": 0.0}
        target = {"j0.pos": 90.0}  # > 80
        assert _adaptive_sync_duration(start, target, 1.0) == 2.0
