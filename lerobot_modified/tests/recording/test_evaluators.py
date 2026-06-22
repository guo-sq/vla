"""Tests for recording.task.evaluators — home, value, timeout, sub-task evaluators."""

import numpy as np
import pytest

from lerobot.recording.task.evaluators import (
    HomeEvaluator,
    SubTaskEvaluator,
    TimeoutEvaluator,
    TimeoutResult,
    ValueEvaluator,
)
from lerobot.recording.task.task_spec import ResetConfig, SuccessCondition


class TestHomeEvaluator:
    def _make_evaluator(self, ref_pose=None, threshold=0.1, speed_threshold=0.05):
        ref = ref_pose or [0.0, 0.0, 0.0]
        keys = ["joint_1.pos", "joint_2.pos", "joint_3.pos"]
        speed_keys = ["joint_1.vel", "joint_2.vel", "joint_3.vel"]
        reset = ResetConfig(
            reference_pose=ref,
            threshold=threshold,
            speed_threshold=speed_threshold,
        )
        return HomeEvaluator(reset, robot_state_keys=keys, robot_speed_keys=speed_keys)

    def test_at_home_exact(self):
        ev = self._make_evaluator()
        obs = {
            "observation.state": np.array([0.0, 0.0, 0.0]),
            "observation.velocity": np.array([0.0, 0.0, 0.0]),
        }
        assert ev.is_home_pose(obs) is True
        assert ev.is_home(obs) is True

    def test_at_home_within_threshold(self):
        ev = self._make_evaluator(threshold=0.1)
        obs = {
            "observation.state": np.array([0.05, -0.05, 0.02]),
            "observation.velocity": np.array([0.0, 0.0, 0.0]),
        }
        assert ev.is_home_pose(obs) is True

    def test_not_at_home_pose(self):
        ev = self._make_evaluator(threshold=0.1)
        obs = {
            "observation.state": np.array([1.0, 0.0, 0.0]),
            "observation.velocity": np.array([0.0, 0.0, 0.0]),
        }
        assert ev.is_home_pose(obs) is False
        assert ev.is_home(obs) is False

    def test_at_home_pose_but_moving(self):
        ev = self._make_evaluator(threshold=0.1, speed_threshold=0.05)
        obs = {
            "observation.state": np.array([0.0, 0.0, 0.0]),
            "observation.velocity": np.array([0.5, 0.0, 0.0]),
        }
        assert ev.is_home_pose(obs) is True
        assert ev.is_home(obs) is False

    def test_no_velocity_in_observation(self):
        ev = self._make_evaluator()
        obs = {
            "observation.state": np.array([0.0, 0.0, 0.0]),
        }
        # Without velocity data, is_home should still work (assume stopped)
        assert ev.is_home_pose(obs) is True
        assert ev.is_home(obs) is True

    def test_different_reference_pose(self):
        ev = self._make_evaluator(ref_pose=[1.0, 2.0, 3.0], threshold=0.1)
        obs = {
            "observation.state": np.array([1.05, 2.03, 2.98]),
            "observation.velocity": np.array([0.0, 0.0, 0.0]),
        }
        assert ev.is_home_pose(obs) is True

    def test_empty_reference_pose(self):
        reset = ResetConfig(reference_pose=[], threshold=0.1, speed_threshold=0.01)
        ev = HomeEvaluator(reset, robot_state_keys=[], robot_speed_keys=[])
        obs = {}
        # Empty reference pose means always at home
        assert ev.is_home_pose(obs) is True
        assert ev.is_home(obs) is True

    def test_partial_dimension_mismatch_uses_available(self):
        """If observation has fewer dims than reference, check available ones."""
        ref = [0.0, 0.0, 0.0, 0.0]
        keys = ["j1.pos", "j2.pos", "j3.pos", "j4.pos"]
        speed_keys = ["j1.vel", "j2.vel", "j3.vel", "j4.vel"]
        reset = ResetConfig(reference_pose=ref, threshold=0.1, speed_threshold=0.05)
        ev = HomeEvaluator(reset, robot_state_keys=keys, robot_speed_keys=speed_keys)
        obs = {
            "observation.state": np.array([0.0, 0.0, 0.0, 0.0]),
            "observation.velocity": np.array([0.0, 0.0, 0.0, 0.0]),
        }
        assert ev.is_home(obs) is True


class TestValueEvaluator:
    def test_builder_success(self):
        cond = SuccessCondition(at_home=True, value_gte=-0.7)
        ev = ValueEvaluator(cond)
        assert ev.is_success(-0.5) is True
        assert ev.is_success(-0.7) is True

    def test_builder_failure(self):
        cond = SuccessCondition(at_home=True, value_gte=-0.7)
        ev = ValueEvaluator(cond)
        assert ev.is_success(-0.8) is False

    def test_destroyer_success(self):
        cond = SuccessCondition(at_home=True, value_lte=-0.97)
        ev = ValueEvaluator(cond)
        assert ev.is_success(-0.98) is True
        assert ev.is_success(-0.97) is True

    def test_destroyer_failure(self):
        cond = SuccessCondition(at_home=True, value_lte=-0.97)
        ev = ValueEvaluator(cond)
        assert ev.is_success(-0.5) is False

    def test_no_thresholds_always_false(self):
        cond = SuccessCondition(at_home=True)
        ev = ValueEvaluator(cond)
        assert ev.is_success(0.5) is False
        assert ev.is_success(-1.0) is False

    def test_both_thresholds(self):
        """Unusual but valid: both gte and lte set. Either triggers success."""
        cond = SuccessCondition(value_gte=0.8, value_lte=0.2)
        ev = ValueEvaluator(cond)
        assert ev.is_success(0.9) is True   # meets gte (0.9 >= 0.8)
        assert ev.is_success(0.1) is True   # meets lte (0.1 <= 0.2)
        assert ev.is_success(0.5) is False  # fails both (0.5 < 0.8 and 0.5 > 0.2)


class TestTimeoutEvaluator:
    def test_no_timeout(self):
        ev = TimeoutEvaluator()
        result = ev.check(elapsed_s=30.0, max_time_s=60.0, is_home=False)
        assert result == TimeoutResult.NONE

    def test_timeout_at_home(self):
        ev = TimeoutEvaluator()
        result = ev.check(elapsed_s=61.0, max_time_s=60.0, is_home=True)
        assert result == TimeoutResult.HARD

    def test_timeout_not_at_home(self):
        ev = TimeoutEvaluator()
        result = ev.check(elapsed_s=61.0, max_time_s=60.0, is_home=False)
        assert result == TimeoutResult.SOFT

    def test_exact_boundary(self):
        ev = TimeoutEvaluator()
        result = ev.check(elapsed_s=60.0, max_time_s=60.0, is_home=False)
        assert result == TimeoutResult.SOFT


class TestSubTaskEvaluator:
    def test_disabled_when_no_durations(self):
        ev = SubTaskEvaluator(durations=None)
        assert ev.enabled is False
        assert ev.update(5.0) == -1

    def test_disabled_when_empty(self):
        ev = SubTaskEvaluator(durations=[])
        assert ev.enabled is False

    def test_basic_transitions(self):
        ev = SubTaskEvaluator(durations=[10.0, 5.0, 20.0])
        assert ev.enabled is True
        assert ev.total_duration == 35.0

        idx = ev.update(0.0)
        assert idx == 0

        idx = ev.update(5.0)
        assert idx == 0

        idx = ev.update(10.0)
        assert idx == 1

        idx = ev.update(14.9)
        assert idx == 1

        idx = ev.update(15.0)
        assert idx == 2

        idx = ev.update(34.9)
        assert idx == 2

    def test_finished(self):
        ev = SubTaskEvaluator(durations=[5.0, 5.0])
        ev.update(0.0)
        assert ev.is_finished() is False
        ev.update(10.0)
        assert ev.is_finished() is True

    def test_after_finish_returns_minus_one(self):
        ev = SubTaskEvaluator(durations=[5.0])
        ev.update(0.0)
        idx = ev.update(6.0)
        assert idx == -1

    def test_single_subtask(self):
        ev = SubTaskEvaluator(durations=[30.0])
        idx = ev.update(0.0)
        assert idx == 0
        idx = ev.update(29.0)
        assert idx == 0
        idx = ev.update(30.0)
        assert idx == -1
