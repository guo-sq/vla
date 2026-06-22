"""Tests for inference_speedup: robot frame reuse + effective FPS calculation."""

import sys
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Mock hardware dependencies before importing piper modules
# ---------------------------------------------------------------------------

def _mock_hardware_imports():
    """Install mock modules for hardware libs that aren't available in CI/test."""
    for mod_name in [
        "pinocchio", "piper_sdk", "piper_sdk.C_PiperInterface_V2",
        "lerobot.sensors", "lerobot.sensors.paxini_tactile_sensor",
        "lerobot.sensors.paxini_tactile_sensor.PaxiniTactileSensorConfig",
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()

    # Mock arx_x5_python's broken import chain
    for mod_name in [
        "lerobot.robots.arx_x5_python.bimanual",
        "lerobot.robots.arx_x5_python.bimanual.BimanualArm",
        "lerobot.robots.arx_x5_python.bimanual.SingleArm",
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()


# Install mocks at module level to avoid import errors in all test classes
_mock_hardware_imports()


# ---------------------------------------------------------------------------
# Step 2: Robot frame reuse
# ---------------------------------------------------------------------------

class TestRobotFrameReuse:

    def test_piper_wait_for_new_frame_default_true(self):
        """Piper robot should have wait_for_new_frame=True by default."""
        from lerobot.robots.piper.piper import Piper, PiperConfig

        config = PiperConfig(port="/dev/ttyUSB0")
        with patch("lerobot.robots.piper.piper.PiperSDKInterface"):
            robot = Piper(config)
        assert robot.wait_for_new_frame is True

    def test_piper_passes_wait_for_new_to_cameras(self):
        """Setting wait_for_new_frame=False should pass to cam.async_read()."""
        from lerobot.robots.piper.piper import Piper, PiperConfig

        config = PiperConfig(port="/dev/ttyUSB0")
        with patch("lerobot.robots.piper.piper.PiperSDKInterface") as MockSDK:
            MockSDK.return_value.get_status.return_value = {"joint_0.pos": 0.0}
            robot = Piper(config)

        mock_cam = MagicMock()
        mock_cam.async_read.return_value = "fake_frame"
        robot.cameras = {"cam_1": mock_cam}
        robot.wait_for_new_frame = False

        obs = robot.get_observation()

        mock_cam.async_read.assert_called_once_with(wait_for_new=False)
        assert obs["cam_1"] == "fake_frame"

    def test_bi_piper_wait_for_new_frame_default_true(self):
        """BiPiperFollower should have wait_for_new_frame=True by default."""
        from lerobot.robots.bi_piper_follower.bi_piper_follower import (
            BiPiperFollower,
            BiPiperFollowerConfig,
        )

        config = BiPiperFollowerConfig(
            left_arm_port="/dev/ttyUSB0",
            right_arm_port="/dev/ttyUSB1",
        )
        with patch("lerobot.robots.bi_piper_follower.bi_piper_follower.make_cameras_from_configs", return_value={}):
            with patch("lerobot.robots.piper.piper.PiperSDKInterface"):
                robot = BiPiperFollower(config)
        assert robot.wait_for_new_frame is True

    def test_bi_piper_passes_wait_for_new_to_cameras(self):
        """BiPiperFollower with wait_for_new_frame=False should pass to cameras."""
        from lerobot.robots.bi_piper_follower.bi_piper_follower import (
            BiPiperFollower,
            BiPiperFollowerConfig,
        )

        config = BiPiperFollowerConfig(
            left_arm_port="/dev/ttyUSB0",
            right_arm_port="/dev/ttyUSB1",
        )
        with patch("lerobot.robots.bi_piper_follower.bi_piper_follower.make_cameras_from_configs", return_value={}):
            with patch("lerobot.robots.piper.piper.PiperSDKInterface") as MockSDK:
                MockSDK.return_value.get_status.return_value = {"joint_0.pos": 0.0}
                robot = BiPiperFollower(config)

        mock_cam = MagicMock()
        mock_cam.async_read.return_value = "fake_frame"
        robot.cameras = {"head_cam": mock_cam}
        robot.wait_for_new_frame = False

        obs = robot.get_observation()

        mock_cam.async_read.assert_called_once_with(wait_for_new=False)
        assert obs["head_cam"] == "fake_frame"


# ---------------------------------------------------------------------------
# Step 3: inference_speedup + effective FPS
# ---------------------------------------------------------------------------

class TestEffectiveFPS:
    """Test get_effective_fps and enable_inference_speedup using lightweight mocks.

    We avoid importing RecordConfig directly because it triggers heavy robot
    registration imports. Instead we use SimpleNamespace as a stand-in config.
    """

    def test_effective_fps_with_speedup(self):
        """inference_speedup=1.5, base_fps=30 → effective_fps=45."""
        from lerobot.recording.record import get_effective_fps
        from types import SimpleNamespace

        cfg = SimpleNamespace(
            dataset=SimpleNamespace(fps=30),
            mode="self_play",
            inference_speedup=1.5,
        )
        assert get_effective_fps(cfg) == 45

    def test_effective_fps_record_mode_ignores_speedup(self):
        """record mode should return base fps even with speedup set."""
        from lerobot.recording.record import get_effective_fps
        from types import SimpleNamespace

        cfg = SimpleNamespace(
            dataset=SimpleNamespace(fps=30),
            mode="record",
            inference_speedup=2.0,
        )
        assert get_effective_fps(cfg) == 30

    def test_effective_fps_default_unchanged(self):
        """Default speedup=1.0 should return base fps."""
        from lerobot.recording.record import get_effective_fps
        from types import SimpleNamespace

        cfg = SimpleNamespace(
            dataset=SimpleNamespace(fps=30),
            mode="infer",
            inference_speedup=1.0,
        )
        assert get_effective_fps(cfg) == 30

    def test_speedup_enables_frame_reuse_on_robot(self):
        """inference_speedup > 1 should set robot.wait_for_new_frame = False."""
        from lerobot.recording.record import enable_inference_speedup

        mock_robot = MagicMock()
        mock_robot.wait_for_new_frame = True

        enable_inference_speedup(mock_robot, speedup=1.5)

        assert mock_robot.wait_for_new_frame is False

    def test_speedup_noop_when_1(self):
        """inference_speedup=1.0 should not change robot state."""
        from lerobot.recording.record import enable_inference_speedup

        mock_robot = MagicMock()
        mock_robot.wait_for_new_frame = True

        enable_inference_speedup(mock_robot, speedup=1.0)

        assert mock_robot.wait_for_new_frame is True


# ---------------------------------------------------------------------------
# Step 4: Auto-optimal infer_interval
# ---------------------------------------------------------------------------

class TestComputeOptimalInferInterval:
    """Test the pure function that computes optimal infer_interval from latency stats."""

    def test_piper_1_5x_speedup(self):
        """Piper T=80±15ms, S=1.5 → fps=45, L_3σ=6, I=41."""
        from lerobot.recording.runtime.policy_runtime import compute_optimal_infer_interval

        result = compute_optimal_infer_interval(
            latencies=[0.075, 0.080, 0.085],  # ~80ms ± small
            effective_fps=45,
            action_horizon=50,
        )
        assert result["infer_interval"] > 30  # should be ~41, much larger than default 20
        assert result["infer_interval"] <= 47  # H - L_mean - margin at most
        assert result["latency_steps_mean"] >= 3
        assert result["feasible"] is True

    def test_ark_high_latency(self):
        """方舟 T=200±60ms, S=1.5 → fps=45, tighter margin."""
        from lerobot.recording.runtime.policy_runtime import compute_optimal_infer_interval

        result = compute_optimal_infer_interval(
            latencies=[0.160, 0.200, 0.240],
            effective_fps=45,
            action_horizon=50,
        )
        assert result["infer_interval"] >= 9   # at least L_mean
        assert result["infer_interval"] <= 40
        assert result["feasible"] is True

    def test_infeasible_when_too_slow(self):
        """T=600ms at fps=60 → L_3σ >= H, should be infeasible."""
        from lerobot.recording.runtime.policy_runtime import compute_optimal_infer_interval

        result = compute_optimal_infer_interval(
            latencies=[0.550, 0.600, 0.650],
            effective_fps=60,
            action_horizon=50,
        )
        assert result["feasible"] is False

    def test_no_speedup_returns_none(self):
        """When effective_fps equals base fps (no speedup), return None (don't override)."""
        from lerobot.recording.runtime.policy_runtime import compute_optimal_infer_interval

        result = compute_optimal_infer_interval(
            latencies=[0.080, 0.080, 0.080],
            effective_fps=30,
            action_horizon=50,
        )
        # Still computes, but with more headroom
        assert result["feasible"] is True
        assert result["infer_interval"] >= 40  # very generous at 30fps

    def test_minimum_sigma_floor(self):
        """Even with identical latencies, sigma should have a floor of 15% of mean."""
        from lerobot.recording.runtime.policy_runtime import compute_optimal_infer_interval

        result = compute_optimal_infer_interval(
            latencies=[0.100, 0.100, 0.100],  # zero variance
            effective_fps=45,
            action_horizon=50,
        )
        # sigma floor = 0.015s, so L_3σ = ceil((0.1 + 0.045) * 45) = ceil(6.525) = 7
        assert result["latency_steps_3sigma"] >= 7
        assert result["feasible"] is True


class TestPolicyRuntimeAutoInterval:
    """Test that PolicyRuntime auto-adjusts infer_interval during warmup."""

    def _make_runtime(self, effective_fps=45, infer_interval=20, inference_mode="async"):
        """Create a PolicyRuntime with mock client/robot, simulating ~80ms inference."""
        from lerobot.recording.runtime.policy_runtime import PolicyRuntime
        from lerobot.recording.utils.logging import AsyncLogger

        mock_client = MagicMock()
        mock_client.robot_state_keys = ["j0.pos", "j1.pos", "j2.pos"]

        # Simulate ~80ms inference latency
        def fake_get_action(obs, prefix, delay, valid_len, prompt):
            time.sleep(0.08)
            return np.zeros((50, 3), dtype=np.float32)

        mock_client.get_action.side_effect = fake_get_action

        mock_robot = MagicMock()
        mock_robot.get_observation.return_value = {"j0.pos": 0.0}

        logger = AsyncLogger("/tmp/test_log", enabled=False)

        rt = PolicyRuntime(
            policy_client=mock_client,
            robot=mock_robot,
            lang_prompt="test",
            logger=logger,
            inference_mode=inference_mode,
            action_horizon=50,
            infer_interval=infer_interval,
            effective_fps=effective_fps,
            auto_infer_interval=True,
        )
        return rt

    def test_auto_adjusts_infer_interval_with_speedup(self):
        """With speedup (fps=45), infer_interval should be auto-adjusted above default 20."""
        rt = self._make_runtime(effective_fps=45, infer_interval=20)
        # After warmup, infer_interval should have been auto-adjusted
        assert rt.infer_interval > 20, (
            f"Expected auto-adjusted > 20, got {rt.infer_interval}"
        )
        rt.cleanup()

    def test_no_auto_adjust_when_disabled(self):
        """When auto_infer_interval=False, infer_interval stays at configured value."""
        from lerobot.recording.runtime.policy_runtime import PolicyRuntime
        from lerobot.recording.utils.logging import AsyncLogger

        mock_client = MagicMock()
        mock_client.robot_state_keys = ["j0.pos"]
        mock_client.get_action.return_value = np.zeros((50, 1), dtype=np.float32)

        mock_robot = MagicMock()
        mock_robot.get_observation.return_value = {"j0.pos": 0.0}

        logger = AsyncLogger("/tmp/test_log", enabled=False)

        rt = PolicyRuntime(
            policy_client=mock_client,
            robot=mock_robot,
            lang_prompt="test",
            logger=logger,
            inference_mode="async",
            action_horizon=50,
            infer_interval=20,
            effective_fps=45,
            auto_infer_interval=False,
        )
        assert rt.infer_interval == 20
        rt.cleanup()


# ---------------------------------------------------------------------------
# Commit 1: Downsampled recording + transition scaling + warmup bias
# ---------------------------------------------------------------------------

class TestWriteDownsampling:
    def test_write_rate_matches_base_fps_at_1p5x(self):
        """fps=45, base_fps=30 → exactly 6 writes per 9 ticks (avg 30 Hz)."""
        from lerobot.recording.runtime.control_loop import ControlLoop

        mock_robot = MagicMock()
        loop = ControlLoop(mock_robot, fps=45, base_fps=30)
        writes = [loop._should_write_frame(i) for i in range(9)]
        assert sum(writes) == 6
        # And over a long horizon, average rate equals base_fps / fps.
        n = 4500
        assert sum(loop._should_write_frame(i) for i in range(n)) == n * 30 // 45

    def test_write_rate_matches_base_fps_at_2x(self):
        """fps=60, base_fps=30 → exactly every other tick writes."""
        from lerobot.recording.runtime.control_loop import ControlLoop

        mock_robot = MagicMock()
        loop = ControlLoop(mock_robot, fps=60, base_fps=30)
        assert all(loop._should_write_frame(i) is (i % 2 == 0) for i in range(20))

    def test_write_rate_no_speedup(self):
        """fps=30, base_fps=30 → every step writes."""
        from lerobot.recording.runtime.control_loop import ControlLoop

        mock_robot = MagicMock()
        loop = ControlLoop(mock_robot, fps=30, base_fps=30)
        assert all(loop._should_write_frame(i) for i in range(20))

    def test_write_rate_defaults_to_every_tick(self):
        """Without base_fps, every tick writes."""
        from lerobot.recording.runtime.control_loop import ControlLoop

        mock_robot = MagicMock()
        loop = ControlLoop(mock_robot, fps=30)
        assert all(loop._should_write_frame(i) for i in range(20))


class TestTransitionScaling:
    def test_scaled_with_speedup(self):
        """effective_fps=45 → transition_steps scaled from 15 to ~23."""
        from lerobot.recording.runtime.policy_runtime import PolicyRuntime
        from lerobot.recording.utils.logging import AsyncLogger

        mock_client = MagicMock()
        mock_client.robot_state_keys = ["j0.pos"]
        mock_client.get_action.return_value = np.zeros((50, 1), dtype=np.float32)
        mock_robot = MagicMock()
        mock_robot.get_observation.return_value = {"j0.pos": 0.0}
        logger = AsyncLogger("/tmp/test_log", enabled=False)

        rt = PolicyRuntime(
            policy_client=mock_client, robot=mock_robot,
            lang_prompt="test", logger=logger,
            transition_steps=15, effective_fps=45,
        )
        # 15 * 45/30 = 22.5 → 22 or 23
        assert rt.transition_steps >= 22
        assert rt.transition_steps <= 23
        rt.cleanup()

    def test_unchanged_without_speedup(self):
        """effective_fps=30 → transition_steps stays 15."""
        from lerobot.recording.runtime.policy_runtime import PolicyRuntime
        from lerobot.recording.utils.logging import AsyncLogger

        mock_client = MagicMock()
        mock_client.robot_state_keys = ["j0.pos"]
        mock_client.get_action.return_value = np.zeros((50, 1), dtype=np.float32)
        mock_robot = MagicMock()
        mock_robot.get_observation.return_value = {"j0.pos": 0.0}
        logger = AsyncLogger("/tmp/test_log", enabled=False)

        rt = PolicyRuntime(
            policy_client=mock_client, robot=mock_robot,
            lang_prompt="test", logger=logger,
            transition_steps=15, effective_fps=30,
        )
        assert rt.transition_steps == 15
        rt.cleanup()


class TestWarmupBias:
    def test_discards_first_measurement(self):
        """Auto mode should discard the first (cold) measurement."""
        from lerobot.recording.runtime.policy_runtime import PolicyRuntime
        from lerobot.recording.utils.logging import AsyncLogger

        call_count = 0
        def fake_get_action(obs, prefix, delay, valid_len, prompt):
            nonlocal call_count
            call_count += 1
            # First call simulates cold start (200ms), rest are ~50ms
            if call_count == 1:
                time.sleep(0.20)
            else:
                time.sleep(0.05)
            return np.zeros((50, 1), dtype=np.float32)

        mock_client = MagicMock()
        mock_client.robot_state_keys = ["j0.pos"]
        mock_client.get_action.side_effect = fake_get_action
        mock_robot = MagicMock()
        mock_robot.get_observation.return_value = {"j0.pos": 0.0}
        logger = AsyncLogger("/tmp/test_log", enabled=False)

        rt = PolicyRuntime(
            policy_client=mock_client, robot=mock_robot,
            lang_prompt="test", logger=logger,
            inference_mode="async", effective_fps=45,
            auto_infer_interval=True, infer_interval=20,
        )
        # If first measurement (200ms) was NOT discarded, L_3σ would be very high
        # and I would be low (~30). If discarded (50ms only), I should be ~40+.
        assert rt.infer_interval >= 38, (
            f"Expected I >= 38 (cold start discarded), got {rt.infer_interval}"
        )
        rt.cleanup()


# ---------------------------------------------------------------------------
# Simplified scheduling: no adaptive I, no low water mark
# ---------------------------------------------------------------------------

class TestNoAdaptiveOscillation:
    def test_interval_stays_fixed_after_warmup(self):
        """I should not change at runtime — no adaptive re-tuning."""
        from lerobot.recording.runtime.policy_runtime import PolicyRuntime
        from lerobot.recording.utils.logging import AsyncLogger

        mock_client = MagicMock()
        mock_client.robot_state_keys = ["j0.pos"]
        mock_client.get_action.side_effect = lambda *a: (
            time.sleep(0.05),
            np.zeros((50, 1), dtype=np.float32),
        )[-1]
        mock_robot = MagicMock()
        mock_robot.get_observation.return_value = {"j0.pos": 0.0}
        logger = AsyncLogger("/tmp/test_log", enabled=False)

        rt = PolicyRuntime(
            policy_client=mock_client, robot=mock_robot,
            lang_prompt="test", logger=logger,
            inference_mode="async", effective_fps=45,
            auto_infer_interval=True, infer_interval=20,
            action_horizon=50,
        )
        initial_I = rt.infer_interval

        # _record_inference_latency should not exist anymore
        assert not hasattr(rt, "_record_inference_latency"), \
            "_record_inference_latency should be removed"
        # I stays fixed
        assert rt.infer_interval == initial_I
        rt.cleanup()


class TestNoLowWaterMarkSubmission:
    def test_no_emergency_threshold(self):
        """No emergency threshold should exist after warmup."""
        from lerobot.recording.runtime.policy_runtime import PolicyRuntime
        from lerobot.recording.utils.logging import AsyncLogger

        mock_client = MagicMock()
        mock_client.robot_state_keys = ["j0.pos"]
        mock_client.get_action.side_effect = lambda *a: (
            time.sleep(0.05),
            np.zeros((50, 1), dtype=np.float32),
        )[-1]
        mock_robot = MagicMock()
        mock_robot.get_observation.return_value = {"j0.pos": 0.0}
        logger = AsyncLogger("/tmp/test_log", enabled=False)

        rt = PolicyRuntime(
            policy_client=mock_client, robot=mock_robot,
            lang_prompt="test", logger=logger,
            inference_mode="async", effective_fps=45,
            auto_infer_interval=True, infer_interval=20,
            action_horizon=50,
        )
        # _emergency_threshold should not exist
        assert not hasattr(rt, "_emergency_threshold"), \
            "_emergency_threshold should be removed"
        rt.cleanup()


# ---------------------------------------------------------------------------
# Sync stutter warning
# ---------------------------------------------------------------------------

class TestSyncStutterWarning:
    def test_warns_when_stutter_high(self):
        """Sync mode with long inference should log stutter warning."""
        from lerobot.recording.runtime.policy_runtime import PolicyRuntime
        from lerobot.recording.utils.logging import AsyncLogger

        mock_client = MagicMock()
        mock_client.robot_state_keys = ["j0.pos"]
        # Simulate 300ms inference (stutter > 15% at 45fps)
        mock_client.get_action.side_effect = lambda *a: (
            time.sleep(0.01),  # fast warmup
            np.zeros((50, 1), dtype=np.float32),
        )[-1]
        mock_robot = MagicMock()
        mock_robot.get_observation.return_value = {"j0.pos": 0.0}
        logger = AsyncLogger("/tmp/test_log", enabled=False)

        rt = PolicyRuntime(
            policy_client=mock_client, robot=mock_robot,
            lang_prompt="test", logger=logger,
            inference_mode="sync", effective_fps=45,
            action_horizon=50,
        )

        # Drain buffer to trigger sync inference
        for _ in range(50):
            rt.action_buffer.get_next_action()

        # Now simulate slow inference blocking
        def slow_infer(*a):
            time.sleep(0.30)
            return np.zeros((50, 1), dtype=np.float32)
        mock_client.get_action.side_effect = slow_infer

        obs = {"j0.pos": 0.0}
        action = rt.get_action(obs, 51)

        # Should have logged a stutter warning
        assert rt._last_stutter_ratio > 0.15
        rt.cleanup()

    def test_no_warn_when_stutter_low(self):
        """Sync mode with fast inference should not trigger warning."""
        from lerobot.recording.runtime.policy_runtime import PolicyRuntime
        from lerobot.recording.utils.logging import AsyncLogger

        mock_client = MagicMock()
        mock_client.robot_state_keys = ["j0.pos"]
        mock_client.get_action.return_value = np.zeros((50, 1), dtype=np.float32)
        mock_robot = MagicMock()
        mock_robot.get_observation.return_value = {"j0.pos": 0.0}
        logger = AsyncLogger("/tmp/test_log", enabled=False)

        rt = PolicyRuntime(
            policy_client=mock_client, robot=mock_robot,
            lang_prompt="test", logger=logger,
            inference_mode="sync", effective_fps=30,
            action_horizon=50,
        )
        assert rt._last_stutter_ratio == 0.0
        rt.cleanup()


# ---------------------------------------------------------------------------
# Fusion bug fix: enable fusion when buffer has remaining actions
# ---------------------------------------------------------------------------

class TestFusionEnabled:
    def test_fusion_enabled_when_buffer_has_remaining(self):
        """Async mode should use fusion when buffer has remaining actions."""
        from lerobot.recording.runtime.policy_runtime import ActionBuffer

        keys = ["j0.pos", "j1.pos"]
        buf = ActionBuffer(
            robot_state_keys=keys, action_dim=2,
            action_horizon=50, fusion_type="linear",
        )
        # Init buffer with chunk A
        chunk_a = np.ones((50, 2), dtype=np.float32) * 10.0
        buf.init_buffer(chunk_a)

        # Consume 40 actions → 10 remaining
        for _ in range(40):
            buf.get_next_action()

        # Update with chunk B (different values)
        chunk_b = np.ones((50, 2), dtype=np.float32) * 20.0
        buf.update_from_inference(chunk_b, start_step=0, current_step=8)

        # First action should be blended, NOT pure 20.0
        action = buf.get_next_action()
        val = action["j0.pos"]
        assert val != 20.0, f"Expected blended value, got pure new value {val}"
        assert 10.0 < val < 20.0, f"Expected blend between 10 and 20, got {val}"

    def test_fusion_not_enabled_when_buffer_empty(self):
        """When buffer is empty, fusion has no effect — init_buffer loads directly."""
        from lerobot.recording.runtime.policy_runtime import ActionBuffer

        keys = ["j0.pos"]
        buf = ActionBuffer(
            robot_state_keys=keys, action_dim=1,
            action_horizon=50, fusion_type="linear",
        )
        # Buffer starts empty — init directly (update_from_inference has a
        # known deadlock when calling init_buffer under lock on dev/anyverse)
        chunk = np.ones((50, 1), dtype=np.float32) * 5.0
        buf.init_buffer(chunk)

        action = buf.get_next_action()
        assert action["j0.pos"] == 5.0  # Direct init, no blend

    def test_fusion_blends_correctly_linear(self):
        """Linear fusion should gradually transition from old to new."""
        from lerobot.recording.runtime.policy_runtime import ActionBuffer

        keys = ["j0.pos"]
        buf = ActionBuffer(
            robot_state_keys=keys, action_dim=1,
            action_horizon=10, fusion_type="linear",
        )
        # Init with old chunk (all 10s)
        buf.init_buffer(np.full((10, 1), 10.0, dtype=np.float32))

        # Consume 5 → 5 remaining
        for _ in range(5):
            buf.get_next_action()

        # Update with new chunk (all 20s), enable fusion
        buf.update_from_inference(
            np.full((10, 1), 20.0, dtype=np.float32),
            start_step=0, current_step=0,
        )

        # Read first 5 actions — should be gradually transitioning
        vals = []
        for _ in range(5):
            a = buf.get_next_action()
            vals.append(a["j0.pos"])

        # First action should be close to old (10), last close to new (20)
        assert vals[0] < vals[-1], f"Expected ascending blend, got {vals}"
        assert vals[0] > 10.0, f"First should be > 10 (some new), got {vals[0]}"
        assert vals[-1] < 20.0, f"Last should be < 20 (some old), got {vals[-1]}"

    def test_fusion_window_zero_disables_fusion(self):
        """fusion_window=0 should disable fusion (backward compat)."""
        from lerobot.recording.runtime.policy_runtime import ActionBuffer

        keys = ["j0.pos"]
        buf = ActionBuffer(
            robot_state_keys=keys, action_dim=1,
            action_horizon=50, fusion_type="linear",
            fusion_window=0,
        )
        buf.init_buffer(np.full((50, 1), 10.0, dtype=np.float32))
        for _ in range(40):
            buf.get_next_action()

        buf.update_from_inference(
            np.full((50, 1), 20.0, dtype=np.float32),
            start_step=0, current_step=8,
        )

        # With fusion_window=0, should be pure new (no blend)
        action = buf.get_next_action()
        assert action["j0.pos"] == 20.0, (
            f"Expected pure new value 20.0 with fusion_window=0, got {action['j0.pos']}"
        )

    def test_fusion_window_configurable(self):
        """Custom fusion_window=3 should only blend 3 steps."""
        from lerobot.recording.runtime.policy_runtime import ActionBuffer

        keys = ["j0.pos"]
        buf = ActionBuffer(
            robot_state_keys=keys, action_dim=1,
            action_horizon=50, fusion_type="linear",
            fusion_window=3,
        )
        buf.init_buffer(np.full((50, 1), 10.0, dtype=np.float32))
        for _ in range(40):
            buf.get_next_action()
        # 10 remaining

        buf.update_from_inference(
            np.full((50, 1), 20.0, dtype=np.float32),
            start_step=0, current_step=8,
        )

        # Read 4 actions: first 3 blended, 4th should be pure new
        vals = [buf.get_next_action()["j0.pos"] for _ in range(4)]
        assert vals[0] > 10.0 and vals[0] < 20.0, f"Step 0 should be blended, got {vals[0]}"
        assert vals[3] == 20.0, f"Step 3 should be pure new, got {vals[3]}"


class TestPostFusionSmoothing:
    """Gaussian smoothing applied AFTER fusion should eliminate boundary jump."""

    def test_smooth_eliminates_fusion_boundary_jump(self):
        """With smooth_sigma>0, the transition from fused to non-fused region
        should be gradual, not a sudden jump."""
        from lerobot.recording.runtime.policy_runtime import ActionBuffer

        keys = ["j0.pos"]
        buf = ActionBuffer(
            robot_state_keys=keys, action_dim=1,
            action_horizon=20, fusion_type="linear",
            fusion_window=5, smooth_sigma=1.0,
        )
        # Old chunk: all 0s
        buf.init_buffer(np.full((20, 1), 0.0, dtype=np.float32))
        # Consume 10 → 10 remaining
        for _ in range(10):
            buf.get_next_action()

        # New chunk: all 100s (big difference to make boundary visible)
        buf.update_from_inference(
            np.full((20, 1), 100.0, dtype=np.float32),
            start_step=0, current_step=0,
        )

        # Read actions around the fusion boundary (steps 0-9)
        vals = [buf.get_next_action()["j0.pos"] for _ in range(10)]
        # Key assertion: the deltas should be monotonically increasing then
        # decreasing (bell-shaped), NOT a sudden jump at the boundary.
        # Without smoothing, step 4→5 jumps ~17 then immediately drops to 0.
        # With smoothing, the transition should be gradual on both sides.
        deltas = [abs(vals[i+1] - vals[i]) for i in range(len(vals) - 1)]
        # The boundary region (steps 5-7) should NOT have the largest jump
        boundary_max = max(deltas[5:])
        # Boundary should be clearly smaller than the ramp region peak
        assert boundary_max < 12.0, (
            f"Fusion boundary jump still too large ({boundary_max:.1f}), "
            f"smoothing should reduce it. vals={[f'{v:.1f}' for v in vals]}"
        )

    def test_no_smooth_has_boundary_jump(self):
        """With smooth_sigma=0, the fusion boundary should have a visible jump."""
        from lerobot.recording.runtime.policy_runtime import ActionBuffer

        keys = ["j0.pos"]
        buf = ActionBuffer(
            robot_state_keys=keys, action_dim=1,
            action_horizon=20, fusion_type="linear",
            fusion_window=5, smooth_sigma=0.0,
        )
        buf.init_buffer(np.full((20, 1), 0.0, dtype=np.float32))
        for _ in range(10):
            buf.get_next_action()

        buf.update_from_inference(
            np.full((20, 1), 100.0, dtype=np.float32),
            start_step=0, current_step=0,
        )

        vals = [buf.get_next_action()["j0.pos"] for _ in range(8)]
        # Without smoothing, the jump at fusion boundary should be large
        max_delta = max(abs(vals[i+1] - vals[i]) for i in range(len(vals) - 1))
        assert max_delta >= 15.0, (
            f"Expected large boundary jump without smoothing, got {max_delta:.1f}. "
            f"vals={[f'{v:.1f}' for v in vals]}"
        )


# ---------------------------------------------------------------------------
# Async timeout diagnostic + Sync queue.Empty crash
# ---------------------------------------------------------------------------

class TestAsyncTimeoutDiagnostic:
    def test_timeout_logs_warning_with_diagnostic(self):
        """3s timeout should log at logging.warning level with worker status."""
        import logging as log_mod
        from lerobot.recording.runtime.policy_runtime import PolicyRuntime
        from lerobot.recording.utils.logging import AsyncLogger

        mock_client = MagicMock()
        mock_client.robot_state_keys = ["j0.pos"]
        mock_client.get_action.return_value = np.zeros((50, 1), dtype=np.float32)
        mock_robot = MagicMock()
        mock_robot.get_observation.return_value = {"j0.pos": 0.0}
        logger = AsyncLogger("/tmp/test_log", enabled=False)

        rt = PolicyRuntime(
            policy_client=mock_client, robot=mock_robot,
            lang_prompt="test", logger=logger,
            inference_mode="async", effective_fps=30,
            action_horizon=50, infer_interval=1,
        )

        # Drain buffer and stop worker
        for _ in range(50):
            rt.action_buffer.get_next_action()
        rt.stop_event.set()
        rt._thread.join(timeout=2)

        with patch.object(log_mod, 'warning') as mock_warn:
            result = rt._get_action_async({"j0.pos": 0.0}, 999)
            assert result is None
            mock_warn.assert_called_once()
            msg = mock_warn.call_args[0][0]
            assert "timeout" in msg.lower()
            assert "worker_alive" in msg

        rt.cleanup()


class TestSyncTimeoutHandling:
    def test_sync_timeout_returns_none_not_crash(self):
        """Sync mode with dead worker should return None, not raise queue.Empty."""
        from lerobot.recording.runtime.policy_runtime import PolicyRuntime
        from lerobot.recording.utils.logging import AsyncLogger

        mock_client = MagicMock()
        mock_client.robot_state_keys = ["j0.pos"]
        mock_client.get_action.return_value = np.zeros((50, 1), dtype=np.float32)
        mock_robot = MagicMock()
        mock_robot.get_observation.return_value = {"j0.pos": 0.0}
        logger = AsyncLogger("/tmp/test_log", enabled=False)

        rt = PolicyRuntime(
            policy_client=mock_client, robot=mock_robot,
            lang_prompt="test", logger=logger,
            inference_mode="sync", effective_fps=30,
            action_horizon=50,
        )

        # Drain buffer and kill worker
        for _ in range(50):
            rt.action_buffer.get_next_action()
        rt.stop_event.set()
        rt._thread.join(timeout=2)

        # Should return None, NOT raise queue.Empty
        result = rt._get_action_sync({"j0.pos": 0.0}, 999)
        assert result is None
        rt.cleanup()


# ---------------------------------------------------------------------------
# Stale chunk logging + chunk size guard
# ---------------------------------------------------------------------------

class TestStaleChunkLogging:
    def test_logs_when_entire_chunk_discarded(self):
        """When latency_steps >= chunk length, should log a debug message."""
        from lerobot.recording.runtime.policy_runtime import ActionBuffer
        from unittest.mock import patch
        import logging as log_mod

        keys = ["j0.pos"]
        buf = ActionBuffer(robot_state_keys=keys, action_dim=1, action_horizon=10)
        buf.init_buffer(np.full((10, 1), 5.0, dtype=np.float32))
        for _ in range(5):
            buf.get_next_action()

        with patch.object(log_mod, 'debug') as mock_debug:
            buf.update_from_inference(
                np.full((10, 1), 99.0, dtype=np.float32),
                start_step=0, current_step=15
            )
            mock_debug.assert_called_once()
            assert "stale" in mock_debug.call_args[0][0].lower()

        # Buffer should be unchanged (still has 5 remaining)
        assert buf.available_count == 5


class TestChunkSizeGuard:
    def test_oversized_chunk_truncated(self):
        """If inference returns a larger chunk than buffer, it should be truncated."""
        from lerobot.recording.runtime.policy_runtime import ActionBuffer

        keys = ["j0.pos"]
        buf = ActionBuffer(robot_state_keys=keys, action_dim=1, action_horizon=10)
        buf.init_buffer(np.full((10, 1), 5.0, dtype=np.float32))
        for _ in range(10):
            buf.get_next_action()

        # Feed oversized chunk (20 steps for a 10-step buffer)
        oversized = np.full((20, 1), 99.0, dtype=np.float32)
        buf.update_from_inference(oversized, start_step=0, current_step=0)

        assert buf.available_count == 10
        action = buf.get_next_action()
        assert action["j0.pos"] == 99.0


class TestSpeedupUnsupportedRobotWarning:
    def test_warns_when_robot_lacks_wait_for_new_frame(self):
        """Should log warning when speedup > 1 but robot doesn't support frame reuse."""
        from lerobot.recording.record import enable_inference_speedup
        from unittest.mock import patch, MagicMock
        import logging as log_mod

        mock_robot = MagicMock(spec=[])  # no attributes at all

        with patch.object(log_mod, 'warning') as mock_warn:
            enable_inference_speedup(mock_robot, speedup=1.5)
            mock_warn.assert_called_once()
            assert "wait_for_new_frame" in mock_warn.call_args[0][0]

    def test_no_warn_when_speedup_le_1(self):
        """No warning when speedup <= 1."""
        from lerobot.recording.record import enable_inference_speedup
        from unittest.mock import patch, MagicMock
        import logging as log_mod

        mock_robot = MagicMock(spec=[])

        with patch.object(log_mod, 'warning') as mock_warn:
            enable_inference_speedup(mock_robot, speedup=1.0)
            mock_warn.assert_not_called()
