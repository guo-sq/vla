"""Fixed-rate control loop skeleton.

The metronome of the recording system: reads observations, dispatches
action sources, writes dataset frames, and rate-limits to target FPS.
"""

from __future__ import annotations

from lerobot.recording.task.state_machine import FlowPhase  # noqa: E402

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol

import numpy as np

from lerobot.recording.runtime.intervention import (
    leader_sync_loop,
    wait_for_leader_movement,
)
from lerobot.recording.utils.logging import debug_print

# Collision recovery severity ordering for max_recovery tracking
_RECOVERY_ORDER = {"none": 0, "pause": 1, "n1": 2, "n2": 3, "force_takeover": 4}
_ESC_TO_RECOVERY = {
    "pause": "pause",
    "n1_safe_prompt": "n1",
    "n2_home_prompt": "n2",
    "n3_force_takeover": "force_takeover",
    "destroyer_force_takeover": "force_takeover",
}


class LoopDirective(Enum):
    CONTINUE = "continue"
    SKIP_ACTION = "skip_action"
    BREAK = "break"


class ActionSource(Protocol):
    def get_action(self, observation: dict, step_id: int) -> dict | None: ...


class LoopHook(Protocol):
    def on_tick(self, ctx: dict, events: dict) -> LoopDirective: ...


@dataclass
class EpisodeResult:
    step_count: int
    duration_s: float
    end_reason: str = "completed"
    task_success: Optional[bool] = None
    last_value_score: Optional[float] = None
    last_is_home: bool = False
    intervention_count: int = 0
    intervention_duration_s: float = 0.0
    first_home_time_s: Optional[float] = None
    home_duration_s: float = 0.0
    collision_count: int = 0
    collision_max_recovery: str = "none"  # "none", "n1", "n2", "force_takeover"
    collision_events: list = field(default_factory=list)
    # Ordered timeline of which prompt was active when. Each entry:
    #   {"step": int, "t_s": float, "prompt": str, "source": "initial" | "user" | "<recovery_reason>"}
    # The first entry is always the initial prompt at step 0.
    prompt_switches: list = field(default_factory=list)


def _handle_prompt_switch(
    *,
    events: dict,
    prompt_switcher: Any | None,
    action_source: Any,
    current_prompt: str,
    step_id: int,
    t_s: float,
    prompt_switches: list,
    logger: Any,
    self_play_logger: Any = None,
    role: str | None = None,
) -> tuple[str, bool]:
    """Consume a pending Ctrl+<key> event, route it to the policy, append to
    timeline. Returns ``(new_current_prompt, switched)``.

    Safe to call with ``prompt_switcher=None`` (no-op).
    """
    if prompt_switcher is None:
        return current_prompt, False
    new = prompt_switcher.consume(events)
    if new is None or new == current_prompt:
        return current_prompt, False
    if hasattr(action_source, "update_prompt"):
        action_source.update_prompt(new)
    entry = {"step": step_id, "t_s": round(t_s, 3), "prompt": new, "source": "user"}
    prompt_switches.append(entry)
    if logger is not None:
        logger.log(f"Prompt switch @ step={step_id} t={t_s:.2f}s → {new!r}")
    if self_play_logger is not None:
        self_play_logger.log("prompt_switch", step=step_id, t_s=t_s, prompt=new, role=role)
    return new, True


def _record_recovery_prompt(
    *,
    new_prompt: str,
    step_id: int,
    t_s: float,
    prompt_switches: list,
    self_play_logger: Any = None,
    role: str | None = None,
    reason: str = "recovery",
) -> None:
    """Append a recovery-driven prompt change to the timeline. The safety/state
    machine has already mutated the runtime's prompt; this is timeline
    bookkeeping so the per-episode audit reflects what the policy actually saw.
    """
    entry = {"step": step_id, "t_s": round(t_s, 3), "prompt": new_prompt, "source": reason}
    prompt_switches.append(entry)
    if self_play_logger is not None:
        self_play_logger.log(
            "prompt_switch", step=step_id, t_s=t_s, prompt=new_prompt, role=role, source=reason,
        )


class TeleopSource:
    """ActionSource backed by a teleoperator (or joint readback if no teleop).

    When no teleop is provided, returns joint positions WITHOUT calling
    send_action() — this keeps robots like ARX in gravity compensation
    mode instead of switching to stiff position control.
    """

    def __init__(self, teleop: Any | None, robot: Any):
        self.teleop = teleop
        self.robot = robot

    def get_action(self, observation: dict, step_id: int) -> dict | None:
        if self.teleop is not None:
            action = self.teleop.get_action()
            return self.robot.send_action(action)
        # No teleop: read-only. Do NOT send_action() — preserves
        # gravity compensation mode on robots that support it (e.g. ARX).
        return self.robot.get_joint_positions()


def _apply_action_mask(
    action: dict,
    observation: dict,
    mask_type: str,
) -> dict:
    """Apply role action mask: hold the masked side at current observation values.

    mask_type:
      "right_hand_only" → left arm holds position (builder)
      "left_hand_only"  → right arm holds position (destroyer)
    """
    if mask_type == "right_hand_only":
        masked_side = "left"
    elif mask_type == "left_hand_only":
        masked_side = "right"
    else:
        return action

    masked = dict(action)
    for key in list(masked.keys()):
        if masked_side in key and key in observation:
            masked[key] = float(observation[key])
    return masked


def _busy_wait(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


class ControlLoop:
    """Fixed-rate control loop.

    Each tick:
    1. Check stop/exit_early events
    2. Read observation from robot
    3. Call hooks — any can return SKIP_ACTION or BREAK
    4. Get action from action_source
    5. Send action to robot
    6. Write frame to dataset (if provided)
    7. Rate-limit to target FPS
    """

    def __init__(
        self,
        robot: Any,
        fps: int,
        dataset: Any | None = None,
        logger: Any | None = None,
        display_data: bool = False,
        base_fps: int | None = None,
    ):
        self.robot = robot
        self.fps = fps
        self.base_fps = base_fps or fps
        self.dataset = dataset
        self.logger = logger
        self._write_fail_count: int = 0
        self._write_frame_fatal: bool = False
        self.display_data = display_data
        self._log_rerun = None
        # Throttle camera frames to ~10 Hz regardless of effective fps so the
        # Rerun viewer can keep up; otherwise queued frames accumulate and
        # produce growing display lag (and eventual freeze) at high fps.
        self._image_decimate = max(1, fps // 10)
        if display_data:
            try:
                from lerobot.utils.visualization_utils import log_rerun_data
                self._log_rerun = log_rerun_data
            except ImportError:
                logging.info("Visualization requested but rerun-sdk not available. Install rerun-sdk.")

    def _should_write_frame(self, step_id: int) -> bool:
        """Bresenham-style downsampling so on average exactly base_fps frames
        are written per second, even for non-integer speedup ratios.

        Writes when (step_id * base_fps) % fps < base_fps. For base==fps this
        is always true. For 30/45 (1.5× speedup) the pattern is
        Y N Y Y N Y Y N Y ... (6 writes per 9 ticks ≈ 0.667 = 30/45).
        """
        return (step_id * self.base_fps) % self.fps < self.base_fps

    def _evacuation_hold(self, duration_s: float, events: dict) -> None:
        """Hold follower at current position during evacuation wait.

        Unlike a blocking sleep, this loop keeps sending the current
        joint positions to the follower each tick, preventing drift.
        """
        if duration_s <= 0:
            return
        step_duration = 1.0 / self.fps
        start = time.perf_counter()
        while time.perf_counter() - start < duration_s:
            if events.get("exit_early") or events.get("stop_recording"):
                break
            t0 = time.perf_counter()
            pos = self.robot.get_joint_positions()
            self.robot.send_action(pos)
            elapsed = time.perf_counter() - t0
            if elapsed < step_duration:
                time.sleep(step_duration - elapsed)

    def _enter_intervention(
        self,
        intervention: Any,
        events: dict | None = None,
        log_say_fn: Any = None,
    ) -> bool:
        """Execute the 3-phase intervention entry sequence atomically.

        Phase 1: Enable gravity compensation (if supported)
        Phase 2: Sync leader arm to follower position (only if leader supports
                 force feedback)
        Phase 3: Wait for human to start moving leader arm

        On success, calls `intervention.enter()` so callers don't have to
        remember a separate bookkeeping step. Returns False (without entering)
        if any phase is interrupted by exit_early / stop_recording, so the
        caller can roll back `inference_paused`.
        """
        if events is None:
            events = {}

        # Phase 1: gravity comp (graceful skip for robots without it, e.g. Piper)
        if hasattr(self.robot, "set_gravity_compensation_mode"):
            if self.logger:
                self.logger.log("Intervention phase 1: enabling gravity compensation")
            self.robot.set_gravity_compensation_mode()

        # Decide once whether the leader can accept feedback frames. SO100 /
        # SO101 / Koch leaders raise NotImplementedError from send_feedback;
        # keyboard / gamepad / bi_so100_leader expose `feedback_features = {}`.
        # In all those cases the leader_sync_loop would either raise or be a
        # silent no-op — skip Phase 2 entirely.
        teleop = intervention.teleop
        leader_supports_feedback = (
            teleop is not None
            and len(getattr(teleop, "feedback_features", {})) > 0
        )

        if log_say_fn:
            if leader_supports_feedback and intervention.pose_sync_duration_s > 0:
                log_say_fn("开始主臂同步")
            else:
                log_say_fn("接管准备中")

        # Phase 2: leader sync — only when the leader can be commanded back.
        if leader_supports_feedback and intervention.pose_sync_duration_s > 0:
            if self.logger:
                self.logger.log(
                    f"Intervention phase 2: syncing leader to follower "
                    f"({intervention.pose_sync_duration_s}s)"
                )
            if not leader_sync_loop(
                self.robot, teleop, events,
                intervention.pose_sync_duration_s, self.fps, self.logger,
            ):
                return False
        elif teleop is not None and not leader_supports_feedback and self.logger:
            self.logger.log(
                f"Leader {type(teleop).__name__} has no feedback support; "
                "skipping leader sync phase"
            )

        # Phase 3: wait for leader movement — only meaningful when there is a
        # leader to monitor. For leader-less workflows (e.g. ARX bimanual with
        # gravity comp) we let the operator move the follower directly.
        if teleop is not None:
            if self.logger:
                self.logger.log(
                    f"Intervention phase 3: waiting for leader movement "
                    f"(timeout={intervention.leader_movement_timeout_s}s)"
                )
            if log_say_fn and leader_supports_feedback:
                log_say_fn("同步完成，开始移动主臂即可接管")
            if not wait_for_leader_movement(
                self.robot, teleop, events,
                self.fps, move_threshold=2.0,
                timeout_s=intervention.leader_movement_timeout_s,
                logger=self.logger,
            ):
                return False

        if log_say_fn:
            log_say_fn("接管已启动", blocking=True)
        # Atomically mark intervention as active — caller no longer needs to
        # call intervention.enter() separately.
        intervention.enter()
        return True

    def _handle_switch_infer_mode(
        self,
        current_intervention: bool,
        inference_paused: bool,
        intervention: Any,
        events: dict,
        action_source: Any,
        lang_prompt: str,
        log_say_fn: Any = None,
    ) -> tuple[bool, bool]:
        """Handle one Ctrl+Space toggle. Returns (current_intervention, inference_paused).

        Shared between `run_episode` and `run_self_play_episode`. Force-takeover
        and home-zone gating live in `run_self_play_episode` and call this only
        once they've decided the toggle is allowed.
        """
        if not current_intervention:
            # Set inference_paused BEFORE the potentially-failing call but roll
            # it back on failure to avoid a permanent freeze. Catch Exception
            # (not specific types) because hardware SDKs (e.g. Piper) do not
            # wrap their errors — anything can surface here.
            inference_paused = True
            try:
                entered = self._enter_intervention(
                    intervention, events=events, log_say_fn=log_say_fn,
                )
            except Exception as exc:
                logging.error(
                    "_enter_intervention failed: %s", exc, exc_info=True
                )
                entered = False
                if log_say_fn:
                    log_say_fn("接管进入失败，恢复推理")
            if entered:
                current_intervention = True
            else:
                inference_paused = False
        else:
            current_intervention = False
            intervention.exit()
            if log_say_fn:
                log_say_fn("请立即撤离", blocking=True)
            # Non-blocking evacuation: hold follower stable while waiting.
            # A blocking sleep would let the follower drift because no
            # send_action calls reach the SDK during the sleep window.
            self._evacuation_hold(intervention.waiting_evacuation_time_s, events)
            # Reinit policy for smooth resume. Catch Exception broadly because
            # hardware/policy SDKs (websocket, CUDA, etc.) surface arbitrary
            # errors here, and a raised exception would kill the whole episode.
            # On failure we keep inference_paused=False so the main loop's next
            # tick retries via action_source.get_action().
            try:
                if hasattr(action_source, "reinit"):
                    action_source.reinit(lang_prompt)
                if hasattr(action_source, "reset_for_resume"):
                    action_source.reset_for_resume()
            except Exception as exc:
                logging.error(
                    "action_source reinit failed after intervention exit: %s",
                    exc, exc_info=True,
                )
                if log_say_fn:
                    log_say_fn("推理重启失败，将继续尝试", blocking=False)
            inference_paused = False
        return current_intervention, inference_paused

    def _reinit_action_source_for_resume(
        self,
        action_source: Any,
        lang_prompt: str,
        log_say_fn: Any = None,
    ) -> None:
        try:
            if hasattr(action_source, "reinit"):
                action_source.reinit(lang_prompt)
            if hasattr(action_source, "reset_for_resume"):
                action_source.reset_for_resume()
        except Exception as exc:
            logging.error(
                "action_source reinit failed while resuming inference: %s",
                exc,
                exc_info=True,
            )
            if log_say_fn:
                log_say_fn("推理重启失败，将继续尝试", blocking=False)

    def _handle_resume_infer_mode(
        self,
        current_intervention: bool,
        inference_paused: bool,
        intervention: Any | None,
        events: dict,
        action_source: Any,
        lang_prompt: str,
        log_say_fn: Any = None,
        safety: Any | None = None,
        allow_intervention_exit: bool = True,
    ) -> tuple[bool, bool]:
        """Handle Ctrl+Enter resume. Returns (current_intervention, inference_paused)."""
        if current_intervention:
            if not allow_intervention_exit:
                if log_say_fn:
                    log_say_fn("请先将机械臂带回初始位，再恢复推理", blocking=True)
                return current_intervention, inference_paused

            if intervention is not None:
                intervention.exit()
                if log_say_fn:
                    log_say_fn("请立即撤离", blocking=True)
                self._evacuation_hold(intervention.waiting_evacuation_time_s, events)

            self._reinit_action_source_for_resume(action_source, lang_prompt, log_say_fn)
            if safety:
                safety.resume()
            if log_say_fn:
                log_say_fn("恢复模型推理")
            return False, False

        if inference_paused:
            self._reinit_action_source_for_resume(action_source, lang_prompt, log_say_fn)
            if safety:
                safety.resume()
            if log_say_fn:
                log_say_fn("恢复模型推理")
            return current_intervention, False

        return current_intervention, inference_paused

    def run_episode(
        self,
        action_source: Any,
        hooks: list[Any],
        control_time_s: float,
        events: dict,
        lang_prompt: str = "",
        intervention: Any | None = None,
        log_say_fn: Any = None,
        subtask_manager: Any | None = None,
        episode_idx: int = 0,
        play_sounds: bool = False,
        progress_print_interval_s: float = 1.0,
        prompt_switcher: Any | None = None,
    ) -> EpisodeResult:
        """Run a timed episode for record/infer_record modes.

        For self_play mode, use run_self_play_episode instead.

        When ``subtask_manager`` is provided and enabled, its ``update(timestamp)``
        is called each tick and the returned index is written to the dataset
        frame as ``subtask_index``.

        When ``prompt_switcher`` is provided, Ctrl+<key> presses during the
        episode swap the policy's prompt on the next chunk and the change is
        recorded in the returned ``EpisodeResult.prompt_switches`` timeline.
        The active prompt is also written into the dataset frame's ``task``
        column, so the dataset reflects exactly what the policy saw.
        """
        start_time = time.perf_counter()
        step_id = 0
        timestamp = 0.0
        current_intervention = False
        inference_paused = False
        # Negative sentinel ensures the first tick's progress line fires
        # regardless of episode length (otherwise short episodes print nothing).
        last_progress_print = -1.0
        current_prompt = lang_prompt
        prompt_switches: list = [
            {"step": 0, "t_s": 0.0, "prompt": current_prompt, "source": "initial"}
        ]

        while timestamp < control_time_s:
            loop_start = time.perf_counter()

            if events.get("stop_recording", False):
                break
            if events.get("exit_early", False):
                events["exit_early"] = False
                break

            current_prompt, _ = _handle_prompt_switch(
                events=events, prompt_switcher=prompt_switcher,
                action_source=action_source, current_prompt=current_prompt,
                step_id=step_id, t_s=timestamp, prompt_switches=prompt_switches,
                logger=self.logger,
            )

            # ---- Ctrl+Space intervention toggle ----
            if intervention is not None and events.get("switch_infer_mode", False):
                events["switch_infer_mode"] = False
                current_intervention, inference_paused = self._handle_switch_infer_mode(
                    current_intervention, inference_paused,
                    intervention, events, action_source, lang_prompt,
                    log_say_fn=log_say_fn,
                )

            # ---- Ctrl+Enter resume inference ----
            if events.get("resume_inference", False):
                events["resume_inference"] = False
                current_intervention, inference_paused = self._handle_resume_infer_mode(
                    current_intervention=current_intervention,
                    inference_paused=inference_paused,
                    intervention=intervention,
                    events=events,
                    action_source=action_source,
                    lang_prompt=lang_prompt,
                    log_say_fn=log_say_fn,
                )

            observation = self.robot.get_observation()

            ctx = {
                "timestamp": timestamp,
                "step_id": step_id,
                "observation": observation,
                "dt_s": 0.0,
            }

            # Run hooks
            directive = LoopDirective.CONTINUE
            for hook in hooks:
                d = hook.on_tick(ctx, events)
                if d == LoopDirective.BREAK:
                    directive = LoopDirective.BREAK
                    break
                if d == LoopDirective.SKIP_ACTION:
                    directive = LoopDirective.SKIP_ACTION

            if directive == LoopDirective.BREAK:
                break

            action = None
            if directive != LoopDirective.SKIP_ACTION:
                if current_intervention:
                    if intervention.teleop is not None:
                        teleop_action = intervention.teleop.get_action()
                        action = self.robot.send_action(teleop_action)
                    else:
                        action = self.robot.get_joint_positions()
                elif inference_paused:
                    action = self.robot.get_joint_positions()
                else:
                    action = action_source.get_action(observation, step_id)
                    if action is not None and not isinstance(action_source, TeleopSource):
                        self.robot.send_action(action)

            # Update subtask manager and snapshot index for this frame
            subtask_index = subtask_manager.update(timestamp) if subtask_manager is not None else -1

            # Write frame to dataset (downsample to base_fps when speedup active)
            if self.dataset is not None and action is not None and self._should_write_frame(step_id):
                self._write_frame(observation, action, current_prompt, current_intervention, subtask_index)
                if self._write_frame_fatal:
                    events["stop_recording"] = True

            # Visualize in rerun
            if self._log_rerun and action is not None:
                self._log_rerun(observation, action, step_id, self._image_decimate)

            step_id += 1
            dt_s = time.perf_counter() - loop_start
            wait_time = 1.0 / self.fps - dt_s
            if wait_time > 0:
                _busy_wait(wait_time)

            timestamp = time.perf_counter() - start_time

            # Throttle the per-tick progress line.
            #   progress_print_interval_s == 0 → print every tick (original ~30 Hz
            #     behaviour); useful for short episodes or when piping through a
            #     fast log collector.
            #   progress_print_interval_s > 0  → throttle to that interval so a
            #     slow operator terminal can't pin the recording loop via stdout
            #     backpressure (default 1.0 s — long episodes were previously
            #     hitting 30 Hz × 50 min ≈ 90k lines/episode).
            # ``play_sounds=False`` disables the print entirely regardless of the
            # interval.
            if (
                play_sounds
                and timestamp - last_progress_print >= progress_print_interval_s
            ):
                # ``flush=True`` is critical: when ``run_session.sh`` pipes the
                # recorder through ``grep --line-buffered -v ARX方舟无限``,
                # Python's ``sys.stdout`` switches to block buffering and the
                # operator sees nothing for the whole episode until Ctrl+Right
                # triggers a logging-module call that happens to flush. Force
                # a flush so the line lands on the terminal each tick.
                print(
                    f"Episode {episode_idx} | {timestamp:.1f}s / {control_time_s}s | "
                    f"Intervention: {current_intervention} | SubTask: {subtask_index}",
                    flush=True,
                )
                last_progress_print = timestamp

        return EpisodeResult(
            step_count=step_id, duration_s=timestamp, prompt_switches=prompt_switches,
        )

    def run_self_play_episode(
        self,
        policy_runtime: Any,
        intervention: Any,
        safety: Any | None,
        home_evaluator: Any | None,
        state_machine: Any,
        events: dict,
        control_time_s: float,
        lang_prompt: str,
        role: str,
        log_say_fn: Any = None,
        self_play_logger: Any = None,
        prompt_switcher: Any | None = None,
    ) -> EpisodeResult:
        """Run one self-play episode with full collision/home/value/intervention logic.

        This implements the interleaved per-tick logic that the old
        record_self_play.py had inline, but using the modular components.
        """
        start_time = time.perf_counter()
        step_id = 0
        timestamp = 0.0

        # Per-episode tracking
        current_intervention = False
        inference_paused = False
        force_takeover_required = False
        force_takeover_reason = None
        first_home_time_s = None
        home_duration_s = 0.0
        home_start_ts = None
        last_value_score = None
        last_is_home = False
        active_prompt = lang_prompt
        end_reason = "completed"
        task_success = None
        # Recording time: only counts when inference is active (not paused/intervention)
        recording_time_s = 0.0
        last_recording_tick_ts = None
        # Collision tracking
        collision_events = []
        collision_max_recovery = "none"  # "none" < "n1" < "n2" < "force_takeover"
        post_timeout_recovery = False

        # Prompt timeline. ``active_prompt`` tracks what the policy is currently
        # receiving (may temporarily be a safety/recovery prompt). User Ctrl+<key>
        # switches and safety overrides both append to this timeline; the source
        # field distinguishes them.
        prompt_switches: list = [
            {"step": 0, "t_s": 0.0, "prompt": active_prompt, "source": "initial"}
        ]

        while recording_time_s < control_time_s or post_timeout_recovery:
            loop_start = time.perf_counter()

            if events.get("stop_recording", False):
                end_reason = "stop_recording"
                break
            if events.get("exit_early", False):
                events["exit_early"] = False
                end_reason = "exit_early"
                break

            # ---- Ctrl+<key> live prompt switch ----
            new_active, switched = _handle_prompt_switch(
                events=events, prompt_switcher=prompt_switcher,
                action_source=policy_runtime, current_prompt=active_prompt,
                step_id=step_id, t_s=timestamp, prompt_switches=prompt_switches,
                logger=self.logger, self_play_logger=self_play_logger, role=role,
            )
            if switched:
                active_prompt = new_active

            observation = self.robot.get_observation()

            # ---- Home detection + velocity extraction ----
            is_home_pose = False
            is_home = False
            max_abs_velocity = 0.0
            if home_evaluator is not None:
                is_home_pose = home_evaluator.is_home_pose(observation)
                is_home = home_evaluator.is_home(observation)
                velocity = home_evaluator._extract_velocity(observation)
                if velocity is not None:
                    max_abs_velocity = float(np.max(np.abs(velocity)))
            last_is_home = is_home_pose

            # Track home timing
            if is_home_pose and not current_intervention:
                if first_home_time_s is None:
                    first_home_time_s = timestamp
                if home_start_ts is None:
                    home_start_ts = time.perf_counter()
            else:
                if home_start_ts is not None:
                    home_duration_s += time.perf_counter() - home_start_ts
                    home_start_ts = None

            # ---- Ctrl+Space intervention toggle ----
            if events.get("switch_infer_mode", False):
                events["switch_infer_mode"] = False

                # Force-takeover gating: when collision recovery requires the
                # operator to land in home before resuming, refuse exit toggles
                # outside the home zone. Skip the helper entirely in that case.
                if (
                    current_intervention
                    and force_takeover_required
                    and home_evaluator is not None
                    and not is_home
                ):
                    if log_say_fn:
                        log_say_fn("请先将机械臂带回初始位，再退出接管", blocking=True)
                else:
                    was_in_intervention = current_intervention
                    current_intervention, inference_paused = self._handle_switch_infer_mode(
                        current_intervention, inference_paused,
                        intervention, events, policy_runtime, active_prompt,
                        log_say_fn=log_say_fn,
                    )

                    # Self-play specific exit bookkeeping (only when we actually
                    # transitioned out of intervention).
                    if was_in_intervention and not current_intervention:
                        if force_takeover_required:
                            saved_reason = force_takeover_reason
                            force_takeover_required = False
                            force_takeover_reason = None

                            # Value-model failure reason → stop program
                            if saved_reason and "value_model" in saved_reason:
                                end_reason = "value_model_unavailable_home"
                                task_success = False
                                if log_say_fn:
                                    log_say_fn("Value model不可用，停止程序", blocking=True)
                                break
                            else:
                                # After forced takeover exit, trigger value eval
                                state_machine.state.value_eval_home_latched = False
                                state_machine.state.value_eval_pending = True
                                state_machine.state.phase = FlowPhase.MAIN
                                state_machine.state.value_model_failure_recovery = False

                        if self_play_logger:
                            self_play_logger.log("intervention_exit", step=step_id, role=role)

            # ---- Collision detection (only when not in intervention and not paused) ----
            if (
                safety is not None
                and not current_intervention
                and not inference_paused
            ):
                collision = safety.check_collision(observation)
                if collision is not None:
                    escalation = safety.escalate(role)
                    decision = state_machine.on_collision_escalation(escalation, timestamp)

                    debug_print(f"Collision: step={step_id}, dim={collision.dim_idx}, val={collision.value:.3f}, "
                               f"escalation={escalation.value}, count={safety.collision_count}")

                    # Track collision event for episode metadata
                    collision_events.append({
                        "t": round(timestamp, 2),
                        "level": escalation.value,
                        "dim": collision.dim_idx,
                    })
                    # Update max recovery level
                    esc_recovery = _ESC_TO_RECOVERY.get(escalation.value, "none")
                    if _RECOVERY_ORDER.get(esc_recovery, 0) > _RECOVERY_ORDER.get(collision_max_recovery, 0):
                        collision_max_recovery = esc_recovery

                    if log_say_fn:
                        log_say_fn("检测到碰撞")

                    if self_play_logger:
                        self_play_logger.log(
                            "collision", step=step_id, role=role,
                            dim=collision.dim_idx, value=collision.value,
                            escalation=escalation.value,
                            collision_count=safety.collision_count,
                            decision_action=decision.action_mode.value,
                            decision_prompt=decision.new_prompt,
                            decision_takeover=decision.request_takeover,
                        )

                    if decision.request_takeover:
                        # N3 or destroyer: stop inference, hold position
                        # Don't enter gravity comp yet — wait for Ctrl+Space
                        force_takeover_required = True
                        force_takeover_reason = decision.takeover_reason
                        inference_paused = True
                        if log_say_fn:
                            log_say_fn("碰撞过多，请人工接管", blocking=True)
                        debug_print(f"Force takeover: reason={decision.takeover_reason}")
                    elif decision.new_prompt:
                        # N1 or N2: switch to recovery prompt and auto-resume
                        active_prompt = decision.new_prompt
                        policy_runtime.reinit(active_prompt)
                        _record_recovery_prompt(
                            new_prompt=active_prompt, step_id=step_id, t_s=timestamp,
                            prompt_switches=prompt_switches,
                            self_play_logger=self_play_logger, role=role,
                            reason="collision_recovery",
                        )
                        if log_say_fn:
                            log_say_fn("切换到恢复提示词")
                        debug_print(f"Recovery prompt: {active_prompt[:60]}")
                    elif decision.action_mode.value == "hold":
                        # Below N1: pause, hold position. Ctrl+Enter to resume, Ctrl+Space for takeover
                        inference_paused = True
                        debug_print(f"Collision pause: awaiting Ctrl+Enter or Ctrl+Space")

                    if decision.announce and log_say_fn:
                        log_say_fn(decision.announce)

            # ---- Resume inference (Ctrl+Enter) ----
            if events.get("resume_inference", False):
                events["resume_inference"] = False
                if force_takeover_required and not current_intervention:
                    if log_say_fn:
                        log_say_fn("当前必须人工接管，请按控制加空格进入接管", blocking=True)
                elif (
                    current_intervention
                    and force_takeover_required
                    and home_evaluator is not None
                    and not is_home
                ):
                    if log_say_fn:
                        log_say_fn("请先将机械臂带回初始位，再恢复推理", blocking=True)
                elif current_intervention or inference_paused:
                    was_in_intervention = current_intervention
                    current_intervention, inference_paused = self._handle_resume_infer_mode(
                        current_intervention=current_intervention,
                        inference_paused=inference_paused,
                        intervention=intervention,
                        events=events,
                        action_source=policy_runtime,
                        lang_prompt=active_prompt,
                        log_say_fn=log_say_fn,
                        safety=safety,
                    )

                    if was_in_intervention and not current_intervention and force_takeover_required:
                        saved_reason = force_takeover_reason
                        force_takeover_required = False
                        force_takeover_reason = None

                        if saved_reason and "value_model" in saved_reason:
                            end_reason = "value_model_unavailable_home"
                            task_success = False
                            if log_say_fn:
                                log_say_fn("Value model不可用，停止程序", blocking=True)
                            break

                        state_machine.state.value_eval_home_latched = False
                        state_machine.state.value_eval_pending = True
                        state_machine.state.phase = FlowPhase.MAIN
                        state_machine.state.value_model_failure_recovery = False

                    if self_play_logger:
                        self_play_logger.log("resume_inference", step=step_id, role=role)

            # ---- N1 recovery check (wait timeout → check speed stable → main or N2) ----
            if safety is not None and not current_intervention:
                n1_decision = state_machine.check_n1_recovery(timestamp, max_abs_velocity)
                if n1_decision is not None:
                    debug_print(f"N1 recovery: phase→{state_machine.state.phase.value}, prompt={n1_decision.new_prompt and n1_decision.new_prompt[:50]}")
                    # Track timeout escalation to N2
                    if state_machine.state.phase == FlowPhase.RECOVERY_N2:
                        if _RECOVERY_ORDER.get("n2", 0) > _RECOVERY_ORDER.get(collision_max_recovery, 0):
                            collision_max_recovery = "n2"
                    if n1_decision.new_prompt:
                        active_prompt = n1_decision.new_prompt
                        policy_runtime.reinit(active_prompt)
                        _record_recovery_prompt(
                            new_prompt=active_prompt, step_id=step_id, t_s=timestamp,
                            prompt_switches=prompt_switches,
                            self_play_logger=self_play_logger, role=role,
                            reason="n1_recovery",
                        )
                    if n1_decision.announce and log_say_fn:
                        log_say_fn(n1_decision.announce)
                    if self_play_logger:
                        self_play_logger.log(
                            "n1_recovery_transition", step=step_id, role=role,
                            new_phase=state_machine.state.phase.value,
                            new_prompt=n1_decision.new_prompt,
                        )

            # ---- N2 recovery: check home + speed stable → back to main ----
            if safety is not None and not current_intervention and not inference_paused:
                n2_home = state_machine.check_n2_recovery_home(is_home, max_abs_velocity, timestamp)
                if n2_home is not None:
                    debug_print(f"N2 recovery home+stable: back to main prompt")
                    # Track force takeover from N2 timeout
                    if n2_home.request_takeover:
                        if _RECOVERY_ORDER.get("force_takeover", 0) > _RECOVERY_ORDER.get(collision_max_recovery, 0):
                            collision_max_recovery = "force_takeover"
                    if n2_home.new_prompt:
                        active_prompt = n2_home.new_prompt
                        policy_runtime.reinit(active_prompt)
                        _record_recovery_prompt(
                            new_prompt=active_prompt, step_id=step_id, t_s=timestamp,
                            prompt_switches=prompt_switches,
                            self_play_logger=self_play_logger, role=role,
                            reason="n2_recovery_home",
                        )
                    if n2_home.announce and log_say_fn:
                        log_say_fn(n2_home.announce)
                    if self_play_logger:
                        self_play_logger.log("n2_recovery_home", step=step_id, role=role)

                    # Handle BREAK decisions (timeout recovery home, value model failure recovery)
                    if n2_home.action_mode.value == "break":
                        end_reason = n2_home.exit_reason or "n2_recovery_home"
                        task_success = n2_home.task_success
                        break

                    # Handle force takeover from N2 timeout
                    if n2_home.request_takeover:
                        force_takeover_required = True
                        force_takeover_reason = n2_home.takeover_reason
                        inference_paused = True

            # ---- Home state update → value eval trigger ----
            if not current_intervention:
                state_machine.update_home_state(is_home_pose, timestamp)
                if step_id % 60 == 0:
                    debug_print(f"Home: step={step_id}, t={timestamp:.1f}s, is_home_pose={is_home_pose}, is_home={is_home}, "
                               f"pending={state_machine.state.value_eval_pending}, latched={state_machine.state.value_eval_home_latched}, "
                               f"phase={state_machine.state.phase.value}, paused={inference_paused}, intervention={current_intervention}")

            # ---- Value evaluation or home-only success ----
            if (
                state_machine.state.value_eval_pending
                and not current_intervention
                and not inference_paused
            ):
                role_spec = state_machine._spec.roles.get(role)

                # Check speed threshold — if too high, wait for robot to settle
                success_speed_thresh = role_spec.success_when.speed_threshold if role_spec else None
                if success_speed_thresh is not None and max_abs_velocity > success_speed_thresh:
                    pass  # keep pending=True, retry next tick when speed drops
                else:
                    # Speed OK — consume pending and evaluate
                    debug_print(f"VALUE EVAL TRIGGERED: step={step_id}, velocity={max_abs_velocity:.4f}")
                    state_machine.state.value_eval_pending = False
                    state_machine.state.value_eval_home_latched = True

                    home_only = (
                        role_spec is not None
                        and role_spec.success_when.is_home_only
                    )
                    if home_only:
                        debug_print(f"HOME-ONLY SUCCESS: step={step_id}, role={role}")
                        if self_play_logger:
                            self_play_logger.log(
                                "home_success", step=step_id, role=role,
                            )
                        end_reason = "home_success"
                        task_success = True
                        break
                    else:
                        # Value-based success: call value model with ROLE prompt (not recovery prompt)
                        role_prompt = state_machine.state.role_prompt
                        debug_print(f"Calling value model: step={step_id}, role={role}, prompt={role_prompt[:50]}...")
                        score = policy_runtime.get_value_score(observation, role_prompt)
                        debug_print(f"Value model returned: score={score}")

                        if score is not None:
                            last_value_score = score
                            decision = state_machine.on_value_score(score, is_home)

                            if self_play_logger:
                                self_play_logger.log(
                                    "value_eval", step=step_id, role=role,
                                    score=score, success=decision.task_success,
                                )

                            debug_print(f"Value decision: success={decision.task_success}, exit_reason={decision.exit_reason}")

                            if decision.task_success is True:
                                end_reason = decision.exit_reason or "value_success"
                                task_success = True
                                break

                            if decision.new_prompt:
                                active_prompt = decision.new_prompt
                                policy_runtime.reinit(active_prompt)
                                _record_recovery_prompt(
                                    new_prompt=active_prompt, step_id=step_id, t_s=timestamp,
                                    prompt_switches=prompt_switches,
                                    self_play_logger=self_play_logger, role=role,
                                    reason="value_decision",
                                )
                        else:
                            # Value model failure
                            debug_print(f"VALUE MODEL FAILURE: step={step_id}, role={role}, is_home={is_home}")
                            decision = state_machine.on_value_model_failure(is_home)
                            if self_play_logger:
                                self_play_logger.log("value_model_failure", step=step_id, role=role)

                            debug_print(f"Failure decision: exit_reason={decision.exit_reason}, takeover={decision.request_takeover}")

                            if decision.exit_reason:
                                end_reason = decision.exit_reason
                                task_success = False
                                events["stop_recording"] = True
                                break

                            if decision.request_takeover:
                                force_takeover_required = True
                                force_takeover_reason = decision.takeover_reason
                                inference_paused = True

                            if decision.new_prompt:
                                active_prompt = decision.new_prompt
                                policy_runtime.reinit(active_prompt)
                                _record_recovery_prompt(
                                    new_prompt=active_prompt, step_id=step_id, t_s=timestamp,
                                    prompt_switches=prompt_switches,
                                    self_play_logger=self_play_logger, role=role,
                                    reason="value_model_failure",
                                )
                            # Set n2_recovery_start_ts for timeout tracking
                            if state_machine.state.n2_recovery_start_ts == 0.0:
                                state_machine.state.n2_recovery_start_ts = timestamp

            # ---- Home dwell end: stayed at home too long without value success ----
            if (
                not current_intervention
                and not inference_paused
            ):
                dwell_decision = state_machine.check_home_dwell_end(is_home_pose, timestamp)
                if dwell_decision is not None:
                    debug_print(f"Home dwell end: step={step_id}, t={timestamp:.1f}s")
                    if self_play_logger:
                        self_play_logger.log(
                            "home_dwell_timeout", step=step_id, role=role,
                        )
                    if dwell_decision.announce and log_say_fn:
                        log_say_fn(dwell_decision.announce, blocking=True)
                    end_reason = dwell_decision.exit_reason or "home_dwell_timeout"
                    task_success = dwell_decision.task_success
                    break

            # ---- Track recording time (only when inference is active) ----
            if not current_intervention and not inference_paused:
                now_ts = time.perf_counter()
                if last_recording_tick_ts is not None:
                    recording_time_s += now_ts - last_recording_tick_ts
                last_recording_tick_ts = now_ts
            else:
                last_recording_tick_ts = None  # stop counting

            # ---- Timeout check (uses recording time, not wall clock) ----
            if not current_intervention and not inference_paused:
                timeout_decision = state_machine.check_timeout(recording_time_s, is_home)
                if timeout_decision is not None:
                    debug_print(f"Timeout: step={step_id}, t={timestamp:.1f}s, is_home={is_home}, action={timeout_decision.action_mode.value}, exit={timeout_decision.exit_reason}")
                    if self_play_logger:
                        self_play_logger.log(
                            "timeout", step=step_id, role=role,
                            is_home=is_home, action=timeout_decision.action_mode.value,
                            exit_reason=timeout_decision.exit_reason,
                        )

                    if timeout_decision.action_mode.value == "break":
                        end_reason = timeout_decision.exit_reason or "role_time_limit_exceeded"
                        task_success = timeout_decision.task_success
                        break

                    # Non-break timeout: keep loop alive for recovery to home
                    post_timeout_recovery = True

                    if timeout_decision.request_takeover:
                        force_takeover_required = True
                        force_takeover_reason = timeout_decision.takeover_reason
                        inference_paused = True
                        # Enter gravity comp so the human can move the arm immediately
                        if hasattr(self.robot, "set_gravity_compensation_mode"):
                            self.robot.set_gravity_compensation_mode()
                        if log_say_fn:
                            log_say_fn(
                                "超时，机械臂已进入重力补偿模式。请手动将机械臂带回初始位，然后按控制加空格确认",
                                blocking=True,
                            )

                    if timeout_decision.new_prompt:
                        active_prompt = timeout_decision.new_prompt
                        policy_runtime.reinit(active_prompt)
                        _record_recovery_prompt(
                            new_prompt=active_prompt, step_id=step_id, t_s=timestamp,
                            prompt_switches=prompt_switches,
                            self_play_logger=self_play_logger, role=role,
                            reason="timeout_recovery",
                        )
                        if log_say_fn:
                            log_say_fn("超时，机器人正在自动返回初始位")
                        debug_print(f"Timeout recovery: switching to N2 prompt: {active_prompt[:60]}")

                    if timeout_decision.announce and log_say_fn and not timeout_decision.request_takeover:
                        # Don't double-announce for takeover (already announced above)
                        log_say_fn(timeout_decision.announce)

            # ---- Get action ----
            action = None
            if current_intervention:
                # Human is in control
                if intervention.teleop is not None:
                    teleop_action = intervention.teleop.get_action()
                    action = self.robot.send_action(teleop_action)
                else:
                    # No teleop — hold position
                    action = self.robot.get_joint_positions()
            elif inference_paused or force_takeover_required:
                # Hold current position
                action = self.robot.get_joint_positions()
            else:
                # Policy inference
                policy_runtime.lang_prompt = active_prompt
                action = policy_runtime.get_action(observation, step_id)

                # Apply transition weight (smooth resume after intervention)
                if action is not None and policy_runtime.transition_weight > 0:
                    joint_obs = self.robot.get_joint_positions()
                    for key in self.robot.action_features:
                        if key in action and key in joint_obs:
                            w = policy_runtime.transition_weight
                            action[key] = (1 - w) * action[key] + w * joint_obs[key]
                    policy_runtime.transition_weight = max(
                        0, policy_runtime.transition_weight - 1.0 / policy_runtime.transition_steps
                    )

                # Apply role action mask (e.g., builder=right_hand_only → hold left arm)
                if action is not None:
                    role_spec = state_machine._spec.roles.get(role)
                    action_mask_cfg = role_spec.action_mask if role_spec else None
                    if action_mask_cfg:
                        action = _apply_action_mask(action, observation, action_mask_cfg)

                if action is not None:
                    self.robot.send_action(action)

            # ---- Write frame to dataset (downsample to base_fps) ----
            # Always use the role prompt as the task label, not recovery prompts
            if self.dataset is not None and action is not None and self._should_write_frame(step_id):
                self._write_frame(observation, action, lang_prompt, current_intervention)
                if self._write_frame_fatal:
                    events["stop_recording"] = True

            # ---- Visualize in rerun ----
            if self._log_rerun and action is not None:
                self._log_rerun(observation, action, step_id, self._image_decimate)

            step_id += 1
            dt_s = time.perf_counter() - loop_start
            wait_time = 1.0 / self.fps - dt_s
            if wait_time > 0:
                _busy_wait(wait_time)

            timestamp = time.perf_counter() - start_time

        # Finalize home tracking
        if home_start_ts is not None:
            home_duration_s += time.perf_counter() - home_start_ts

        # If episode ended by time limit without explicit success/failure, mark as timeout
        if end_reason == "completed" and task_success is None:
            end_reason = "exceed_time_limit"
            task_success = False

        # Finalize intervention tracking (get_stats handles active sessions)
        if intervention:
            stats = intervention.get_stats()
            final_intervention_count = stats["count"]
            final_intervention_duration = stats["duration_s"]
        else:
            final_intervention_count = 0
            final_intervention_duration = 0.0

        return EpisodeResult(
            step_count=step_id,
            duration_s=timestamp,
            end_reason=end_reason,
            task_success=task_success,
            last_value_score=last_value_score,
            last_is_home=last_is_home,
            intervention_count=final_intervention_count,
            intervention_duration_s=final_intervention_duration,
            first_home_time_s=first_home_time_s,
            home_duration_s=home_duration_s,
            collision_count=len(collision_events),
            collision_max_recovery=collision_max_recovery,
            collision_events=collision_events,
            prompt_switches=prompt_switches,
        )

    def run_indefinite(
        self,
        action_source: Any,
        events: dict,
        prompt_switcher: Any | None = None,
        initial_prompt: str = "",
    ) -> EpisodeResult:
        """Run indefinitely until stop_recording. No dataset, no time limit.

        When ``prompt_switcher`` is provided, Ctrl+<key> presses swap the
        policy's prompt on the next chunk and the change is recorded in the
        returned ``EpisodeResult.prompt_switches`` timeline.
        """
        start_time = time.perf_counter()
        step_id = 0
        inference_paused = False
        current_prompt = initial_prompt or getattr(action_source, "lang_prompt", "")
        prompt_switches: list = [
            {"step": 0, "t_s": 0.0, "prompt": current_prompt, "source": "initial"}
        ]

        while True:
            loop_start = time.perf_counter()

            if events.get("stop_recording", False):
                break
            if events.get("switch_infer_mode", False):
                events["switch_infer_mode"] = False
                inference_paused = True
            if events.get("resume_inference", False):
                events["resume_inference"] = False
                _, inference_paused = self._handle_resume_infer_mode(
                    current_intervention=False,
                    inference_paused=inference_paused,
                    intervention=None,
                    events=events,
                    action_source=action_source,
                    lang_prompt=getattr(action_source, "lang_prompt", ""),
                )

            elapsed = time.perf_counter() - start_time
            current_prompt, _ = _handle_prompt_switch(
                events=events, prompt_switcher=prompt_switcher,
                action_source=action_source, current_prompt=current_prompt,
                step_id=step_id, t_s=elapsed, prompt_switches=prompt_switches,
                logger=self.logger,
            )

            observation = self.robot.get_observation()
            if inference_paused:
                action = self.robot.get_joint_positions()
            else:
                action = action_source.get_action(observation, step_id)

            if action is not None:
                self.robot.send_action(action)

            if self._log_rerun and action is not None:
                self._log_rerun(observation, action, step_id, self._image_decimate)

            step_id += 1
            dt_s = time.perf_counter() - loop_start
            wait_time = 1.0 / self.fps - dt_s
            if wait_time > 0:
                _busy_wait(wait_time)

        duration_s = time.perf_counter() - start_time
        return EpisodeResult(
            step_count=step_id, duration_s=duration_s, prompt_switches=prompt_switches,
        )

    def _write_frame(
        self,
        observation: dict,
        action: dict,
        lang_prompt: str,
        is_human_intervention: bool = False,
        subtask_index: int = -1,
    ) -> None:
        """Build and write a dataset frame."""
        try:
            from lerobot.datasets.utils import build_dataset_frame
            obs_frame = build_dataset_frame(self.dataset.features, observation, prefix="observation")
            act_frame = build_dataset_frame(self.dataset.features, action, prefix="action")
            frame = {**obs_frame, **act_frame}
            frame["is_human_intervention"] = np.array(is_human_intervention).reshape(1)
            if "subtask_index" in self.dataset.features:
                frame["subtask_index"] = np.array(subtask_index).reshape(1)
            self.dataset.add_frame(frame, task=lang_prompt)
            self._write_fail_count = 0
            self._write_frame_fatal = False
        except (
            OSError, RuntimeError, KeyError, ValueError, IndexError,
            TypeError, AttributeError,
        ):
            # Whitelist runtime failures we intentionally tolerate (recording
            # should never crash on a single bad frame from a flaky sensor):
            #   - OSError: disk full, closed file handle, codec errors
            #   - RuntimeError: HDF5 / dataset backend failures
            #   - KeyError: obs/action field missing (e.g. camera dropped a frame)
            #   - ValueError: dtype / shape mismatch in np.array construction
            #   - IndexError: feature-length mismatches
            #   - TypeError: dtype coercion failures from camera/sensor backends
            #     occasionally returning None instead of np.ndarray
            #   - AttributeError: optional sensor fields that occasionally come
            #     back as None
            # NameError / ImportError still propagate — those are unambiguous
            # code bugs that should crash loud in dev.
            self._write_fail_count += 1
            if self._write_fail_count == 1:
                logging.error("_write_frame failed — data may be lost", exc_info=True)
            elif self._write_fail_count >= 10:
                logging.error(f"_write_frame: {self._write_fail_count} consecutive failures, stopping recording")
                self._write_frame_fatal = True
            else:
                logging.warning(f"_write_frame failed ({self._write_fail_count} consecutive)", exc_info=True)
