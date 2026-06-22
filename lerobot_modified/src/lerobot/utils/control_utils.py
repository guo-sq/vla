# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

########################################################################################
# Utilities
########################################################################################


import atexit
import logging
import os
import sys
import threading
import traceback
from contextlib import nullcontext
from copy import copy
from dataclasses import dataclass, field
from functools import cache

import numpy as np
import torch
from deepdiff import DeepDiff
from termcolor import colored

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.robots import Robot


def log_control_info(robot: Robot, dt_s, episode_index=None, frame_index=None, fps=None):
    log_items = []
    if episode_index is not None:
        log_items.append(f"ep:{episode_index}")
    if frame_index is not None:
        log_items.append(f"frame:{frame_index}")

    def log_dt(shortname, dt_val_s):
        nonlocal log_items, fps
        info_str = f"{shortname}:{dt_val_s * 1000:5.2f} ({1 / dt_val_s:3.1f}hz)"
        if fps is not None:
            actual_fps = 1 / dt_val_s
            if actual_fps < fps - 1:
                info_str = colored(info_str, "yellow")
        log_items.append(info_str)

    # total step time displayed in milliseconds and its frequency
    log_dt("dt", dt_s)

    # TODO(aliberts): move robot-specific logs logic in robot.print_logs()
    if not robot.robot_type.startswith("stretch"):
        for name in robot.leader_arms:
            key = f"read_leader_{name}_pos_dt_s"
            if key in robot.logs:
                log_dt("dtRlead", robot.logs[key])

        for name in robot.follower_arms:
            key = f"write_follower_{name}_goal_pos_dt_s"
            if key in robot.logs:
                log_dt("dtWfoll", robot.logs[key])

            key = f"read_follower_{name}_pos_dt_s"
            if key in robot.logs:
                log_dt("dtRfoll", robot.logs[key])

        for name in robot.cameras:
            key = f"read_camera_{name}_dt_s"
            if key in robot.logs:
                log_dt(f"dtR{name}", robot.logs[key])

    info_str = " ".join(log_items)
    logging.info(info_str)


@cache
def is_headless():
    """Detects if python is running without a monitor."""
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print()
        return True


def predict_action(
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    use_amp: bool,
    task: str | None = None,
    robot_type: str | None = None,
):
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        for name in observation:
            observation[name] = torch.from_numpy(observation[name])
            if "image" in name:
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)

        observation["task"] = task if task else ""
        observation["robot_type"] = robot_type if robot_type else ""

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)

        # Remove batch dimension
        action = action.squeeze(0)

        # Move to cpu, if not already the case
        action = action.to("cpu")

    return action


@dataclass
class PromptSwitcher:
    """Holds the Ctrl+<key> → prompt-string mapping consumed by the keyboard
    listener and the ControlLoop. Built from ``TaskSpec.prompts``.

    Duck-typed against the interface ``init_keyboard_listener`` already expects:
    ``enabled: bool`` and ``variants: dict[str, str]`` (single-char key → prompt
    string). Multi-char keys are filtered out — the listener matches via
    ``key.char.lower()`` so they would never fire.
    """

    enabled: bool = False
    variants: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_task_spec(cls, task_spec) -> "PromptSwitcher":
        raw = getattr(task_spec, "prompts", None) or {}
        norm = {str(k).lower(): str(v) for k, v in raw.items() if len(str(k)) == 1}
        return cls(enabled=bool(norm), variants=norm)

    def consume(self, events: dict) -> str | None:
        """Read events['lang_prompt_key'], clear it, return the mapped prompt or None."""
        key = events.get("lang_prompt_key")
        if key is None:
            return None
        events["lang_prompt_key"] = None
        return self.variants.get(key)

    def describe(self) -> str:
        if not self.enabled:
            return ""
        lines = [f"  Ctrl+{k} → {v}" for k, v in self.variants.items()]
        return "Prompt switching enabled:\n" + "\n".join(lines)


def init_keyboard_listener(prompt_switcher=None, play_sounds: bool = True, enable_log_say: bool = True):
    """初始化键盘监听器。

    使用 ``pynput.keyboard.GlobalHotKeys`` 注册显式的快捷键组合，
    由 pynput 在原生层维护修饰键状态，避免旧实现里 ``ctrl_pressed``
    粘连导致的误触发（如 Ctrl release 事件丢失或窗口失焦后单按方向键被
    误判为 Ctrl+方向键）。

    Args:
        prompt_switcher: 可选的 prompt 切换器对象，用于通过 Ctrl+字母 切换 lang_prompt。
            需要具有 ``enabled: bool`` 属性和 ``variants: dict[str, str]`` 属性
            （键为单个字母，值为对应的 prompt 文本）。
            如果提供，则 events["lang_prompt_key"] 会在切换时更新。
            **注意**：variants 的内容在监听器启动时被快照绑定；
            会话期间增删 variants 不会影响已注册的快捷键。

    Returns:
        (listener, events) 元组

    events 字典包含：
        - exit_early: bool
        - rerecord_episode: bool
        - stop_recording: bool
        - switch_infer_mode: bool
        - resume_inference: bool
        - lang_prompt_key: str | None  (最新按下的 prompt 切换键，主循环读取后清除)
        - task_success: bool | None  (None=未标记, True=成功(Ctrl+Right), False=失败(Ctrl+Down)，
            每次 episode 保存或重录后由主循环重置为 None)
    """
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
        "switch_infer_mode": False,
        "resume_inference": False,
        "lang_prompt_key": None,
        "task_success": None,
    }

    # SSH / no DISPLAY: pynput's GlobalHotKeys can't reach an X server (and
    # may even fail to import on Linux without DISPLAY), so try the
    # terminal-based listener FIRST. Operators can also force this path via
    # env var when they prefer typing single keys over chord shortcuts.
    use_terminal = (
        os.environ.get("LEROBOT_TERMINAL_KEYS") == "1"
        or not os.environ.get("DISPLAY")
    )
    if use_terminal:
        listener = _init_terminal_keyboard_listener(events, prompt_switcher, play_sounds, enable_log_say)
        if listener is not None:
            return listener, events
        # stdin not a TTY (CI, piped) — fall through to pynput attempt below.

    if is_headless():
        # pynput unimportable AND terminal listener wasn't viable: no input.
        logging.warning(
            "Headless environment detected and stdin is not a TTY. "
            "On-screen cameras display and keyboard inputs will not be available."
        )
        return None, events

    return _init_pynput_keyboard_listener(events, prompt_switcher, play_sounds, enable_log_say), events


def _keyboard_feedback(text: str, *, play_sounds: bool = True, enable_log_say: bool = True):
    if not enable_log_say:
        print(text, flush=True)
        return
    try:
        from lerobot.recording.utils.tts import log_say
        log_say(text, play_sounds=play_sounds, blocking=False, enabled=True)
    except Exception:
        print(text, flush=True)


def _init_pynput_keyboard_listener(events, prompt_switcher, play_sounds=True, enable_log_say=True):
    """Original GlobalHotKeys path for local / X-forwarded sessions."""
    from pynput import keyboard

    def _on_success():
        _keyboard_feedback("任务成功，结束当前录制", play_sounds=play_sounds, enable_log_say=enable_log_say)
        events["task_success"] = True
        events["exit_early"] = True

    def _on_fail():
        _keyboard_feedback("任务失败，结束当前录制", play_sounds=play_sounds, enable_log_say=enable_log_say)
        events["task_success"] = False
        events["exit_early"] = True

    def _on_rerecord():
        _keyboard_feedback("重新录制当前数据", play_sounds=play_sounds, enable_log_say=enable_log_say)
        events["rerecord_episode"] = True
        events["exit_early"] = True

    def _on_stop():
        _keyboard_feedback("停止采集任务", play_sounds=play_sounds, enable_log_say=enable_log_say)
        events["stop_recording"] = True
        events["exit_early"] = True

    def _on_switch_infer():
        _keyboard_feedback("切换到人工接管", play_sounds=play_sounds, enable_log_say=enable_log_say)
        events["switch_infer_mode"] = True

    def _on_resume_infer():
        _keyboard_feedback("恢复推理", play_sounds=play_sounds, enable_log_say=enable_log_say)
        events["resume_inference"] = True

    hotkeys = {
        "<ctrl>+<right>": _on_success,
        "<ctrl>+<down>": _on_fail,
        "<ctrl>+<left>": _on_rerecord,
        "<ctrl>+<esc>": _on_stop,
        "<ctrl>+<space>": _on_switch_infer,
        "<ctrl>+<enter>": _on_resume_infer,
    }

    if prompt_switcher is not None and getattr(prompt_switcher, "enabled", False):
        variants_snapshot = dict(prompt_switcher.variants)

        def _make_prompt_cb(letter, name):
            def _cb():
                events["lang_prompt_key"] = letter
                print(f"Ctrl + {letter} pressed. Switching prompt → {name}")
            return _cb

        for raw_letter, variant_name in variants_snapshot.items():
            letter = raw_letter.lower()
            if len(letter) != 1 or not letter.isalpha():
                logging.warning(
                    f"init_keyboard_listener: skipping invalid prompt_switcher key {raw_letter!r}"
                )
                continue
            hotkeys[f"<ctrl>+{letter}"] = _make_prompt_cb(letter, variant_name)

    listener = keyboard.GlobalHotKeys(hotkeys)
    listener.start()
    return listener


# ---------------------------------------------------------------------------
# Terminal keyboard listener (SSH-friendly, no X11 required)
# ---------------------------------------------------------------------------
#
# pynput.GlobalHotKeys needs an X server (Linux) or accessibility (macOS) to
# capture global hotkeys, so it doesn't work over a plain SSH session — VS
# Code's remote terminal, tmux through ssh, etc. To support those operators
# we read ANSI key codes from stdin in cbreak mode in a daemon thread.
#
# Single-key bindings (no modifiers required, easier to type over SSH):
#
#   y       mark task success and exit episode
#   n       mark task failure and exit episode
#   r       re-record current episode (skip save)
#   q       stop session (end recording)
#   space   toggle takeover / pause inference
#   enter   resume inference after takeover
#
# Plus, when prompt_switcher is enabled, each variant letter (bare, no Ctrl)
# triggers the corresponding lang_prompt switch. Conflicts with y/n/r/q/s/e
# resolve in favor of the operator's prompt_switcher; document accordingly.

_TERMINAL_KEYBOARD_HELP = (
    "=" * 60 + "\n"
    "Terminal keyboard mode (SSH / no DISPLAY). Single-key bindings:\n"
    "  y      mark success + exit episode\n"
    "  n      mark failure + exit episode\n"
    "  r      re-record current episode\n"
    "  q      stop session\n"
    "  space  toggle takeover / pause inference\n"
    "  enter  resume inference\n"
    + "=" * 60
)


class _TerminalKeyboardListener:
    """Daemon-thread stdin reader. Public API matches pynput's listener
    enough for callers (``stop()`` is the only method record.py invokes)."""

    def __init__(self, events, prompt_switcher, play_sounds=True, enable_log_say=True):
        self._events = events
        self._prompt_switcher = prompt_switcher
        self._play_sounds = play_sounds
        self._enable_log_say = enable_log_say
        self._prompt_keys = (
            {k.lower() for k in prompt_switcher.variants}
            if prompt_switcher is not None and getattr(prompt_switcher, "enabled", False)
            else set()
        )
        self._stop = threading.Event()
        # Resolve stdin fd lazily in start() so unit tests can exercise
        # _dispatch without needing a real TTY.
        self._fd = None
        self._old_termios = None
        self._thread = None

    def start(self):
        # Late-import to avoid pulling termios on Windows (this listener is
        # POSIX-only; Windows operators use the pynput path).
        import termios
        import tty

        self._fd = sys.stdin.fileno()
        self._old_termios = termios.tcgetattr(self._fd)
        # Restore terminal settings even on hard exits.
        atexit.register(self._restore)
        try:
            tty.setcbreak(self._fd)
        except Exception as e:
            logging.warning(f"setcbreak failed: {e}")
            self._restore()
            return False

        print(_TERMINAL_KEYBOARD_HELP, flush=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._stop.set()
        self._restore()

    def _restore(self):
        if self._old_termios is None:
            return
        try:
            import termios
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_termios)
        except Exception:
            pass
        self._old_termios = None

    def _run(self):
        import select
        try:
            while not self._stop.is_set():
                # 100 ms poll so the thread can notice stop() promptly.
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not r:
                    continue
                ch = sys.stdin.read(1)
                if not ch:
                    continue
                self._dispatch(ch)
        finally:
            self._restore()

    def _dispatch(self, ch):
        ev = self._events
        # Prompt switcher takes precedence over the y/n/r/q/etc. bindings —
        # operators who use prompt_switcher should pick non-overlapping
        # letters, but if there's a conflict, the prompt switch wins.
        if ch in self._prompt_keys:
            ev["lang_prompt_key"] = ch
            if self._prompt_switcher is not None:
                variant = self._prompt_switcher.variants.get(ch, ch)
                print(f"[term-keys] prompt → {variant}", flush=True)
            return

        if ch == "y":
            ev["task_success"] = True
            ev["exit_early"] = True
            _keyboard_feedback("任务成功，结束当前录制", play_sounds=self._play_sounds, enable_log_say=self._enable_log_say)
        elif ch == "n":
            ev["task_success"] = False
            ev["exit_early"] = True
            _keyboard_feedback("任务失败，结束当前录制", play_sounds=self._play_sounds, enable_log_say=self._enable_log_say)
        elif ch == "r":
            ev["rerecord_episode"] = True
            ev["exit_early"] = True
            _keyboard_feedback("重新录制当前数据", play_sounds=self._play_sounds, enable_log_say=self._enable_log_say)
        elif ch == "q":
            ev["stop_recording"] = True
            ev["exit_early"] = True
            _keyboard_feedback("停止采集任务", play_sounds=self._play_sounds, enable_log_say=self._enable_log_say)
        elif ch == " ":
            ev["switch_infer_mode"] = True
            _keyboard_feedback("切换到人工接管", play_sounds=self._play_sounds, enable_log_say=self._enable_log_say)
        elif ch in ("\r", "\n"):
            ev["resume_inference"] = True
            _keyboard_feedback("恢复推理", play_sounds=self._play_sounds, enable_log_say=self._enable_log_say)


def _init_terminal_keyboard_listener(events, prompt_switcher, play_sounds=True, enable_log_say=True):
    """Build a terminal-mode listener. Returns None if stdin isn't a TTY
    (e.g. piped input, CI runner) so the caller can fall through to pynput
    or just bail out cleanly."""
    if not sys.stdin.isatty():
        logging.warning(
            "stdin is not a TTY; terminal keyboard listener disabled. "
            "Set LEROBOT_TERMINAL_KEYS=0 or run interactively to enable."
        )
        return None
    listener = _TerminalKeyboardListener(events, prompt_switcher, play_sounds, enable_log_say)
    if not listener.start():
        return None
    return listener


def sanity_check_dataset_name(repo_id, policy_cfg):
    _, dataset_name = repo_id.split("/")
    # either repo_id doesnt start with "eval_" and there is no policy
    # or repo_id starts with "eval_" and there is a policy

    # Check if dataset_name starts with "eval_" but policy is missing
    if dataset_name.startswith("eval_") and policy_cfg is None:
        raise ValueError(
            f"Your dataset name begins with 'eval_' ({dataset_name}), but no policy is provided ({policy_cfg.type})."
        )

    # Check if dataset_name does not start with "eval_" but policy is provided
    if not dataset_name.startswith("eval_") and policy_cfg is not None:
        raise ValueError(
            f"Your dataset name does not begin with 'eval_' ({dataset_name}), but a policy is provided ({policy_cfg.type})."
        )


def sanity_check_dataset_robot_compatibility(
    dataset: LeRobotDataset, robot: Robot, fps: int, features: dict
) -> None:
    fields = [
        ("robot_type", dataset.meta.robot_type, robot.robot_type),
        ("fps", dataset.fps, fps),
        ("features", dataset.features, {**features, **DEFAULT_FEATURES}),
    ]

    mismatches = []
    for field, dataset_value, present_value in fields:
        diff = DeepDiff(dataset_value, present_value, exclude_regex_paths=[r".*\['info'\]$"])
        if diff:
            mismatches.append(f"{field}: expected {present_value}, got {dataset_value}")

    if mismatches:
        raise ValueError(
            "Dataset metadata compatibility check failed with mismatches:\n" + "\n".join(mismatches)
        )
