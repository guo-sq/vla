"""State machine for self-play flow phase transitions.

Given evaluator results and safety events, produces Decisions that tell
the control loop what to do: continue, hold, break, change prompt, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from lerobot.recording.runtime.safety_runtime import EscalationAction
from lerobot.recording.task.task_spec import TaskSpec


class FlowPhase(Enum):
    MAIN = "main"
    RECOVERY_N1 = "recovery_n1"
    RECOVERY_N2 = "recovery_n2"


class ActionMode(Enum):
    CONTINUE = "continue"     # proceed normally
    HOLD = "hold"             # hold current position, skip policy
    BREAK = "break"           # end episode


@dataclass
class Decision:
    action_mode: ActionMode = ActionMode.CONTINUE
    new_prompt: str | None = None
    request_takeover: bool = False
    takeover_reason: str | None = None
    exit_reason: str | None = None
    task_success: bool | None = None
    announce: str | None = None


@dataclass
class EpisodeState:
    phase: FlowPhase = FlowPhase.MAIN
    role: str = ""
    active_prompt: str = ""
    role_prompt: str = ""
    # Value evaluation
    value_eval_pending: bool = False
    value_eval_home_latched: bool = False
    home_detected_since_ts: float | None = None
    last_value_score: float | None = None
    # Collision
    collision_count: int = 0
    # N1 recovery
    n1_recovery_start_ts: float | None = None
    n1_last_collision_ts: float | None = None
    n1_speed_stable_since_ts: float | None = None  # when speed first went low during N1
    # N2 recovery (from collision)
    n2_recovery_start_ts: float | None = None
    n2_speed_stable_since_ts: float | None = None  # when speed first went low during N2 at home
    # Role timeout
    role_limit_soft_exceeded: bool = False
    role_limit_recovery_start_ts: float | None = None
    # Value model failure
    value_model_failure_recovery: bool = False
    # Home tracking
    first_home_time_s: float | None = None
    home_duration_s: float = 0.0
    # Home dwell end: tracks continuous home time after value eval was tried
    home_dwell_start_ts: float | None = None
    # Whether robot has left home at least once this episode
    has_left_home: bool = False


class StateMachine:
    """Drives flow phase transitions for self-play episodes.

    Methods are called by the control loop at appropriate points.
    Each returns a Decision (or None if no action needed).
    """

    def __init__(self, task_spec: TaskSpec):
        self._spec = task_spec
        self.state = EpisodeState()

    def reset_for_episode(self, role: str, prompt: str) -> None:
        self.state = EpisodeState()
        self.state.role = role
        self.state.active_prompt = prompt
        self.state.role_prompt = prompt

    # ------------------------------------------------------------------
    # Timeout
    # ------------------------------------------------------------------

    def check_timeout(self, elapsed_s: float, is_home: bool) -> Decision | None:
        """Check role time limit. Returns Decision if timed out, None otherwise."""
        if self.state.role_limit_soft_exceeded:
            return None  # already handled

        role_spec = self._spec.roles.get(self.state.role)
        if role_spec is None:
            return None
        max_time = role_spec.max_time_s
        if max_time is None:
            return None  # no time limit
        if elapsed_s < max_time:
            return None

        self.state.role_limit_soft_exceeded = True

        if is_home:
            return Decision(
                action_mode=ActionMode.BREAK,
                exit_reason="role_time_limit_exceeded",
                task_success=False,
                announce="超时，已在初始位，结束当前回合",
            )

        # Not at home: check per-role timeout_recovery config
        timeout_recovery = role_spec.timeout_recovery

        if timeout_recovery == "recovery_prompt" and self._spec.has_safety and "n2" in self._spec.safety.recovery:
            # Use N2 prompt to go home (e.g., builder)
            self.state.phase = FlowPhase.RECOVERY_N2
            n2_prompt = self._spec.safety.recovery["n2"].prompt
            self.state.active_prompt = n2_prompt or self.state.role_prompt
            self.state.role_limit_recovery_start_ts = elapsed_s
            self.state.n2_recovery_start_ts = elapsed_s
            return Decision(
                action_mode=ActionMode.CONTINUE,
                new_prompt=self.state.active_prompt,
                announce="超时，机器人正在自动返回初始位",
            )
        else:
            # force_human_takeover, or no config specified, or no N2 prompt
            return Decision(
                action_mode=ActionMode.HOLD,
                request_takeover=True,
                takeover_reason="role_time_limit_not_home",
                announce="超时，机械臂已进入重力补偿模式。请手动将机械臂带回初始位，然后按控制加空格确认",
            )

    # ------------------------------------------------------------------
    # Value evaluation
    # ------------------------------------------------------------------

    def on_value_score(self, score: float, is_home: bool) -> Decision:
        """Handle a value model score result."""
        self.state.last_value_score = score
        self.state.value_eval_pending = False
        self.state.value_eval_home_latched = True

        role_spec = self._spec.roles.get(self.state.role)
        if role_spec is None:
            return Decision()

        success_cond = role_spec.success_when
        is_success = False
        if success_cond.value_gte is not None and score >= success_cond.value_gte:
            is_success = True
        if success_cond.value_lte is not None and score <= success_cond.value_lte:
            is_success = True

        if is_success:
            return Decision(
                action_mode=ActionMode.BREAK,
                exit_reason="value_success",
                task_success=True,
            )

        # Not successful — return to main phase if in recovery
        if self.state.phase != FlowPhase.MAIN:
            self.state.phase = FlowPhase.MAIN
            self.state.active_prompt = self.state.role_prompt
            return Decision(
                action_mode=ActionMode.CONTINUE,
                new_prompt=self.state.role_prompt,
            )

        return Decision(action_mode=ActionMode.CONTINUE)

    def on_value_model_failure(self, is_home: bool) -> Decision:
        """Handle value model unavailable/error."""
        self.state.value_model_failure_recovery = True

        if is_home:
            return Decision(
                action_mode=ActionMode.BREAK,
                exit_reason="value_model_unavailable_home",
                task_success=False,
            )

        if self.state.role == "builder" and self._spec.has_safety:
            self.state.phase = FlowPhase.RECOVERY_N2
            self.state.n2_recovery_start_ts = 0.0  # will be set properly by control loop
            n2_prompt = self._spec.safety.recovery["n2"].prompt if "n2" in self._spec.safety.recovery else None
            self.state.active_prompt = n2_prompt or self.state.role_prompt
            return Decision(
                action_mode=ActionMode.CONTINUE,
                new_prompt=self.state.active_prompt,
                announce="Value model不可用，正在尝试返回初始位",
            )

        return Decision(
            action_mode=ActionMode.HOLD,
            request_takeover=True,
            takeover_reason="value_model_unavailable_not_home",
        )

    # ------------------------------------------------------------------
    # Home detection
    # ------------------------------------------------------------------

    def update_home_state(self, is_home_pose: bool, timestamp: float) -> None:
        """Update home detection tracking. Arms value_eval_pending when stable at home."""
        if not is_home_pose:
            self.state.has_left_home = True
            self.state.value_eval_home_latched = False
            self.state.home_detected_since_ts = None
            self.state.value_eval_pending = False
            return

        if not self.state.has_left_home:
            return

        if self.state.phase not in (FlowPhase.MAIN, FlowPhase.RECOVERY_N2):
            return
        if self.state.value_eval_home_latched:
            return

        if self._spec.reset is None:
            return
        home_wait = self._spec.reset.home_wait_s
        if home_wait <= 0:
            self.state.value_eval_pending = True
            return

        if self.state.home_detected_since_ts is None:
            self.state.home_detected_since_ts = timestamp
        elif (timestamp - self.state.home_detected_since_ts) >= home_wait:
            self.state.value_eval_pending = True

    def check_home_dwell_end(self, is_home_pose: bool, timestamp: float) -> Decision | None:
        """End episode if robot stayed at home too long without value model success.

        Only fires when:
        - value_eval_home_latched is True (value was already tried and didn't pass)
        - robot has been continuously at home for home_wait_before_end_s
        """
        if not self.state.has_left_home:
            return None

        if not self.state.value_eval_home_latched:
            self.state.home_dwell_start_ts = None
            return None

        role_spec = self._spec.roles.get(self.state.role)
        if role_spec is None:
            return None
        home_end_s = role_spec.home_wait_before_end_s
        if home_end_s is None:
            return None

        if not is_home_pose:
            self.state.home_dwell_start_ts = None
            return None

        if self.state.home_dwell_start_ts is None:
            self.state.home_dwell_start_ts = timestamp
        elif (timestamp - self.state.home_dwell_start_ts) >= home_end_s:
            return Decision(
                action_mode=ActionMode.BREAK,
                exit_reason="home_dwell_timeout",
                task_success=False,
                announce="在初始位等待超时，结束当前回合",
            )
        return None

    # ------------------------------------------------------------------
    # Collision escalation
    # ------------------------------------------------------------------

    def on_collision_escalation(self, action: EscalationAction, timestamp: float) -> Decision:
        """Handle a collision escalation from SafetyRuntime."""
        if action == EscalationAction.PAUSE:
            return Decision(action_mode=ActionMode.HOLD)

        if action == EscalationAction.N1_SAFE_PROMPT:
            self.state.phase = FlowPhase.RECOVERY_N1
            self.state.n1_recovery_start_ts = timestamp
            self.state.n1_last_collision_ts = timestamp
            n1_prompt = self._spec.safety.recovery["n1"].prompt if self._spec.safety and "n1" in self._spec.safety.recovery else None
            self.state.active_prompt = n1_prompt or self.state.role_prompt
            return Decision(
                action_mode=ActionMode.CONTINUE,
                new_prompt=self.state.active_prompt,
            )

        if action == EscalationAction.N2_HOME_PROMPT:
            self.state.phase = FlowPhase.RECOVERY_N2
            self.state.n1_recovery_start_ts = None
            self.state.n2_recovery_start_ts = timestamp
            n2_prompt = self._spec.safety.recovery["n2"].prompt if self._spec.safety and "n2" in self._spec.safety.recovery else None
            self.state.active_prompt = n2_prompt or self.state.role_prompt
            return Decision(
                action_mode=ActionMode.CONTINUE,
                new_prompt=self.state.active_prompt,
            )

        if action in (EscalationAction.N3_FORCE_TAKEOVER, EscalationAction.DESTROYER_FORCE_TAKEOVER):
            reason = "collision_n3" if action == EscalationAction.N3_FORCE_TAKEOVER else "collision_destroyer"
            return Decision(
                action_mode=ActionMode.HOLD,
                request_takeover=True,
                takeover_reason=reason,
            )

        return Decision(action_mode=ActionMode.HOLD)

    # ------------------------------------------------------------------
    # N1 recovery check
    # ------------------------------------------------------------------

    def check_n1_recovery(self, timestamp: float, max_abs_velocity: float) -> Decision | None:
        """Check if N1 recovery should transition.

        Flow:
        1. Execute: run N1 prompt for execute_time_s (let robot reach safe position)
        2. Stable check: max velocity below per-level speed_threshold for stable_time_s → back to main
        3. Timeout: if timeout_s exceeded → escalate to N2
        """
        if self.state.phase != FlowPhase.RECOVERY_N1:
            return None
        if self.state.n1_recovery_start_ts is None:
            return None

        n1_level = self._spec.safety.recovery.get("n1") if self._spec.safety else None
        if n1_level is None:
            return None

        elapsed = timestamp - self.state.n1_recovery_start_ts
        execute_time_s = n1_level.execute_time_s
        stable_time_s = n1_level.stable_time_s
        speed_thresh = n1_level.speed_threshold
        timeout_s = n1_level.timeout_s

        # Phase 1: execute recovery prompt for execute_time_s
        if elapsed < execute_time_s:
            return None

        # Phase 2: check speed stability against per-level threshold
        is_speed_low = max_abs_velocity <= speed_thresh
        if is_speed_low:
            if self.state.n1_speed_stable_since_ts is None:
                self.state.n1_speed_stable_since_ts = timestamp
            elif (timestamp - self.state.n1_speed_stable_since_ts) >= stable_time_s:
                # Stable → back to main
                self.state.phase = FlowPhase.MAIN
                self.state.n1_recovery_start_ts = None
                self.state.n1_speed_stable_since_ts = None
                self.state.active_prompt = self.state.role_prompt
                return Decision(
                    action_mode=ActionMode.CONTINUE,
                    new_prompt=self.state.role_prompt,
                    announce="恢复稳定，切换回正常提示词",
                )
        else:
            self.state.n1_speed_stable_since_ts = None

        # Phase 3: hard timeout → escalate to N2
        if timeout_s and elapsed >= timeout_s:
            self.state.phase = FlowPhase.RECOVERY_N2
            self.state.n1_recovery_start_ts = None
            self.state.n1_speed_stable_since_ts = None
            self.state.n2_recovery_start_ts = timestamp
            n2_prompt = self._spec.safety.recovery["n2"].prompt if self._spec.safety and "n2" in self._spec.safety.recovery else None
            self.state.active_prompt = n2_prompt or self.state.role_prompt
            return Decision(
                action_mode=ActionMode.CONTINUE,
                new_prompt=self.state.active_prompt,
                announce="恢复超时，切换到回初始位提示词",
            )

        return None

    def check_n2_recovery_home(self, is_home: bool, max_abs_velocity: float, timestamp: float) -> Decision | None:
        """During N2 recovery (from collision or value-model failure), check if robot reached home.

        Flow:
        1. If home + speed stable for stable_time_s → success (back to main or stop program)
        2. If N2 timeout_s exceeded → force human takeover
        """
        if self.state.phase != FlowPhase.RECOVERY_N2:
            return None

        n2_level = self._spec.safety.recovery.get("n2") if self._spec.safety else None
        stable_time_s = n2_level.stable_time_s if n2_level else 1.0
        speed_thresh = n2_level.speed_threshold if n2_level else 0.1
        n2_timeout_s = n2_level.timeout_s if n2_level else None

        is_speed_low = max_abs_velocity <= speed_thresh
        if is_home and is_speed_low:
            if self.state.n2_speed_stable_since_ts is None:
                self.state.n2_speed_stable_since_ts = timestamp
            elif (timestamp - self.state.n2_speed_stable_since_ts) >= stable_time_s:
                # Reached home stable
                if self.state.value_model_failure_recovery:
                    # Value model failure recovery: stop program on reaching home
                    return Decision(
                        action_mode=ActionMode.BREAK,
                        exit_reason="value_model_unavailable_home",
                        task_success=False,
                        announce="已返回初始位，value model不可用，停止程序",
                    )
                if self.state.role_limit_soft_exceeded:
                    # Timeout recovery: end episode on reaching home
                    return Decision(
                        action_mode=ActionMode.BREAK,
                        exit_reason="role_time_limit_recovered_home",
                        task_success=False,
                        announce="已返回初始位，结束当前回合",
                    )
                # Collision recovery: back to main
                self.state.phase = FlowPhase.MAIN
                self.state.n2_speed_stable_since_ts = None
                self.state.n2_recovery_start_ts = None
                self.state.active_prompt = self.state.role_prompt
                return Decision(
                    action_mode=ActionMode.CONTINUE,
                    new_prompt=self.state.role_prompt,
                    announce="已返回初始位，恢复正常推理",
                )
        else:
            self.state.n2_speed_stable_since_ts = None

        # N2 timeout: force human takeover if not reaching home in time
        if n2_timeout_s and self.state.n2_recovery_start_ts is not None:
            n2_elapsed = timestamp - self.state.n2_recovery_start_ts
            if n2_elapsed >= n2_timeout_s:
                reason = "n2_timeout_not_home"
                if self.state.role_limit_soft_exceeded:
                    reason = "timeout_n2_not_home"
                elif self.state.value_model_failure_recovery:
                    reason = "value_model_failure_n2_not_home"
                return Decision(
                    action_mode=ActionMode.HOLD,
                    request_takeover=True,
                    takeover_reason=reason,
                    announce="回初始位超时，请人工接管",
                )

        return None
