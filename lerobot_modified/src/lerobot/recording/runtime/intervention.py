"""Human takeover lifecycle: enter, exit, forced takeover, pose sync."""

from __future__ import annotations

import time
from typing import Any

# Teleops whose joint positions live in the Piper [-100, 100] percentage range.
# `_adaptive_sync_duration` thresholds (40/80) only translate to this scale; for
# any other teleop (degree-based, end-effector etc.) we fall back to the fixed
# duration to avoid mis-classifying distances.
_PIPER_PCT_TELEOPS = ("piper_leader", "bi_piper_leader")


def pose_sync_loop(
    robot: Any,
    teleop: Any,
    events: dict,
    sync_duration_s: float,
    fps: int,
    logger: Any = None,
) -> bool:
    """Smooth pose synchronization from robot to teleoperator.

    Linearly interpolates from the robot's current position to the
    teleoperator's position over sync_duration_s seconds.

    Returns True if completed, False if interrupted.
    """
    if events.get("exit_early") or events.get("stop_recording"):
        return False

    teleop_action = teleop.get_action()
    target = {k: teleop_action[k] for k in robot.action_features if k in teleop_action}

    current_obs = robot.get_joint_positions()
    start = {k: current_obs[k] for k in robot.action_features if k in current_obs}

    num_steps = max(int(sync_duration_s * fps), 1)
    step_duration = 1.0 / fps

    for step in range(num_steps):
        if events.get("exit_early") or events.get("stop_recording"):
            return False

        t0 = time.perf_counter()
        t = step / max(num_steps - 1, 1)

        pose = {}
        for key in robot.action_features:
            if key in start and key in target:
                pose[key] = start[key] + t * (target[key] - start[key])
            else:
                pose[key] = start.get(key, 0.0)

        robot.send_action(pose)

        elapsed = time.perf_counter() - t0
        if elapsed < step_duration:
            time.sleep(step_duration - elapsed)

    if logger:
        logger.log("Pose synchronization completed.")
    return True


# Thresholds tuned for Piper's -100~100 percentage range (NOT degrees).
# Piper leader/follower positions come from PiperSDKInterface.get_status()
# mapped to [-100, 100], so normal arm motion easily exceeds 60 — the old
# 20/60 thresholds almost always triggered the max multiplier.
_SYNC_DISTANCE_MEDIUM = 40.0
_SYNC_DISTANCE_LARGE = 80.0


def _adaptive_sync_duration(start: dict, target: dict, base_duration_s: float) -> float:
    """Calculate sync duration based on max joint distance.

    Thresholds assume Piper percentage units (-100 to 100), not degrees.
    Small (< 40) -> base, Medium (40-80) -> 1.5x, Large (> 80) -> 2x.

    Callers must guard with `_PIPER_PCT_TELEOPS` — for non-Piper teleops the
    thresholds are meaningless and the function should not be invoked.
    """
    max_diff = 0.0
    for key in start:
        if key in target:
            max_diff = max(max_diff, abs(start[key] - target[key]))

    if max_diff > _SYNC_DISTANCE_LARGE:
        return base_duration_s * 2.0
    elif max_diff > _SYNC_DISTANCE_MEDIUM:
        return base_duration_s * 1.5
    return base_duration_s


def _warn_key_mismatch(
    logger: Any,
    where: str,
    expected_keys: list,
    actual_keys: list,
) -> None:
    """Log a warning when teleop/follower key sets don't intersect.

    Typical cause: dual-arm follower exposes `left_joint_0.pos` /
    `right_joint_0.pos` while a single-arm teleop exposes plain `joint_0.pos`.
    Without this warning the sync silently becomes a no-op.
    """
    if logger is None:
        return
    logger.log(
        f"{where}: no matching keys between teleop ({expected_keys}) "
        f"and {actual_keys}. Sync will be a no-op."
    )


def leader_sync_loop(
    robot: Any,
    teleop: Any,
    events: dict,
    sync_duration_s: float,
    fps: int,
    logger: Any = None,
) -> bool:
    """Sync leader arm(s) to follower's current position.

    Linearly interpolates the leader from its current pose to the follower's
    current pose; the follower simultaneously receives its own position so it
    stays put. For Piper teleops the duration is stretched proportionally to
    the joint distance (see `_adaptive_sync_duration`); for other teleops the
    fixed `sync_duration_s` is used.

    Callers must verify `len(teleop.feedback_features) > 0` before calling —
    leaders without force feedback (so100/so101/koch) cannot accept the
    feedback frames produced here.

    Returns True if completed, False if interrupted.
    """
    if events.get("exit_early") or events.get("stop_recording"):
        return False

    # Field set we send back to the leader. `feedback_features` is the
    # authoritative key list for `teleop.send_feedback()`; `action_features`
    # would happen to match for Piper but is the wrong concept here.
    feedback_keys = list(teleop.feedback_features)

    # Target = follower's current joint positions (filtered to leader's
    # feedback fields).
    follower_pos = robot.get_joint_positions()
    target = {k: follower_pos[k] for k in feedback_keys if k in follower_pos}

    # Start = leader's current joint positions.
    leader_pos = teleop.get_action()
    start = {k: leader_pos[k] for k in feedback_keys if k in leader_pos}

    # Warn on dual-arm naming mismatch (silent no-op without this).
    if not target and follower_pos:
        _warn_key_mismatch(
            logger, "leader_sync_loop/follower",
            feedback_keys, list(follower_pos.keys()),
        )
    if not start and leader_pos:
        _warn_key_mismatch(
            logger, "leader_sync_loop/leader",
            feedback_keys, list(leader_pos.keys()),
        )

    # Early-return on key mismatch: proceeding would call send_feedback({})
    # every tick, tripping piper_leader's strict missing-key validation and
    # producing a noisy KeyError traceback + misleading "接管进入失败" TTS.
    # The warnings above are the user-facing signal; return True so
    # _enter_intervention treats this as a graceful skip (no rollback).
    if not start or not target:
        if logger:
            logger.log("Leader sync: empty start/target, skipping feedback loop")
        return True

    # Adaptive duration only meaningful for Piper percentage range.
    teleop_name = getattr(teleop, "name", None)
    if teleop_name in _PIPER_PCT_TELEOPS:
        duration = _adaptive_sync_duration(start, target, sync_duration_s)
    else:
        duration = sync_duration_s
    num_steps = max(int(duration * fps), 1)
    step_duration = 1.0 / fps

    if logger:
        max_diff = max(
            (abs(start.get(k, 0) - target.get(k, 0)) for k in start if k in target),
            default=0.0,
        )
        logger.log(f"Leader sync: max_diff={max_diff:.1f}, duration={duration:.1f}s")

    for step in range(num_steps):
        if events.get("exit_early") or events.get("stop_recording"):
            return False

        t0 = time.perf_counter()
        t = step / max(num_steps - 1, 1)

        pose = {}
        for key in feedback_keys:
            if key in start and key in target:
                pose[key] = start[key] + t * (target[key] - start[key])
            elif key in start:
                # Hold leader's current position for keys we cannot interpolate
                # — never default to 0.0, which would command the joint to its
                # zero pose (physical safety hazard).
                pose[key] = start[key]
            elif key in target:
                pose[key] = target[key]
            # else: key absent from both — skip; downstream send_feedback
            # validation (e.g. piper_leader) will surface incomplete dicts.

        # Move leader toward follower
        teleop.send_feedback(pose)
        # Keep follower in place (refresh position each tick to avoid drift)
        follower_pos = robot.get_joint_positions()
        robot.send_action(follower_pos)

        elapsed = time.perf_counter() - t0
        if elapsed < step_duration:
            time.sleep(step_duration - elapsed)

    if logger:
        logger.log("Leader synced to follower position.")
    return True


def wait_for_leader_movement(
    robot: Any,
    teleop: Any,
    events: dict,
    fps: int,
    move_threshold: float = 2.0,
    timeout_s: float = 30.0,
    logger: Any = None,
) -> bool:
    """Wait until the human starts moving the leader arm.

    After leader sync, the leader sits at the follower position.
    Detects when the operator begins moving the leader (any joint
    changes by more than move_threshold), signaling readiness.

    During waiting, keeps sending follower its current position
    to maintain stability. No frames are recorded.

    Returns True when movement detected or timeout, False if interrupted.
    """
    step_duration = 1.0 / fps
    start_time = time.perf_counter()
    follower_pos = robot.get_joint_positions()

    # Snapshot leader position right after sync (baseline)
    baseline = teleop.get_action()
    movement_keys = list(teleop.action_features)

    # Warn on dual-arm naming mismatch (movement detection would never trigger).
    if baseline and not any(k in baseline for k in movement_keys):
        _warn_key_mismatch(
            logger, "wait_for_leader_movement",
            movement_keys, list(baseline.keys()),
        )

    while True:
        if events.get("exit_early") or events.get("stop_recording"):
            return False

        t0 = time.perf_counter()

        leader_pos = teleop.get_action()
        # Keep follower stable (refresh position each tick to avoid drift)
        follower_pos = robot.get_joint_positions()
        robot.send_action(follower_pos)

        # Check if any joint moved beyond threshold compared to baseline
        max_delta = 0.0
        for key in movement_keys:
            if key in leader_pos and key in baseline:
                max_delta = max(max_delta, abs(leader_pos[key] - baseline[key]))

        if max_delta > move_threshold:
            if logger:
                logger.log(f"Leader movement detected: delta = {max_delta:.1f}")
            return True

        if time.perf_counter() - start_time > timeout_s:
            if logger:
                logger.log(f"Wait timeout ({timeout_s}s), starting takeover anyway")
            return True

        elapsed = time.perf_counter() - t0
        if elapsed < step_duration:
            time.sleep(step_duration - elapsed)


class InterventionRuntime:
    """Manages human takeover lifecycle.

    Tracks enter/exit, forced takeover state, intervention counts
    and durations for episode metadata.
    """

    def __init__(
        self,
        robot: Any,
        teleop: Any | None = None,
        *,
        pose_sync_duration_s: float = 0.0,
        waiting_intervention_time_s: float = 0.0,
        waiting_evacuation_time_s: float = 0.0,
        leader_movement_timeout_s: float = 30.0,
    ):
        self.robot = robot
        self.teleop = teleop
        self.pose_sync_duration_s = pose_sync_duration_s
        self.waiting_intervention_time_s = waiting_intervention_time_s
        self.waiting_evacuation_time_s = waiting_evacuation_time_s
        self.leader_movement_timeout_s = leader_movement_timeout_s

        self.is_active: bool = False
        self.forced: bool = False
        self.forced_reason: str | None = None
        self.intervention_count: int = 0
        self.intervention_duration_s: float = 0.0
        self._entry_ts: float | None = None

    def enter(self) -> None:
        """Enter human intervention mode."""
        if self.is_active:
            return
        self.is_active = True
        self.intervention_count += 1
        self._entry_ts = time.perf_counter()

    def exit(self) -> None:
        """Exit human intervention mode."""
        if not self.is_active:
            return
        self.is_active = False
        if self._entry_ts is not None:
            self.intervention_duration_s += max(0.0, time.perf_counter() - self._entry_ts)
            self._entry_ts = None

    def request_forced_takeover(self, reason: str) -> None:
        """Request mandatory human takeover (from safety or state machine)."""
        self.forced = True
        self.forced_reason = reason

    def can_exit_forced(self, is_home: bool) -> bool:
        """Check if forced takeover can be exited (requires home position)."""
        if not self.forced:
            return True
        return is_home

    def clear_forced(self) -> None:
        """Clear forced takeover state after successful exit."""
        self.forced = False
        self.forced_reason = None

    def get_teleop_action(self) -> dict | None:
        """Get action from teleoperator. Returns None if no teleop."""
        if self.teleop is None:
            return None
        return self.teleop.get_action()

    def get_stats(self) -> dict:
        """Return intervention statistics for episode metadata."""
        duration = self.intervention_duration_s
        if self.is_active and self._entry_ts is not None:
            duration += max(0.0, time.perf_counter() - self._entry_ts)
        return {
            "count": self.intervention_count,
            "duration_s": round(duration, 3),
        }

    def reset(self) -> None:
        """Reset for new episode."""
        self.is_active = False
        self.forced = False
        self.forced_reason = None
        self.intervention_count = 0
        self.intervention_duration_s = 0.0
        self._entry_ts = None
