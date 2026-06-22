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

"""测试 control_utils.init_keyboard_listener 基于 GlobalHotKeys 的实现"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True, scope="module")
def mock_pynput_module():
    """Mock pynput.keyboard.GlobalHotKeys，避免 X server 依赖"""
    mock_keyboard = MagicMock()

    captured = {}

    def make_global_hotkeys(hotkeys_dict):
        captured["hotkeys"] = hotkeys_dict
        listener = MagicMock()
        listener.start = MagicMock()
        listener.stop = MagicMock()
        return listener

    mock_keyboard.GlobalHotKeys = MagicMock(side_effect=make_global_hotkeys)
    mock_keyboard._captured = captured

    mock_pynput = MagicMock()
    mock_pynput.keyboard = mock_keyboard

    with patch.dict(sys.modules, {"pynput": mock_pynput, "pynput.keyboard": mock_keyboard}):
        yield mock_keyboard


@pytest.fixture(autouse=True)
def mock_robot_modules():
    """Mock robot 依赖模块，避免导入 pinocchio 等本地依赖"""
    mock_modules = {
        "piper_sdk": MagicMock(),
        "pinocchio": MagicMock(),
        "lerobot.robots.piper": MagicMock(),
        "lerobot.robots.piper.piper": MagicMock(),
        "lerobot.robots.piper.piper_sdk_interface": MagicMock(),
        "lerobot.robots.bi_piper_follower": MagicMock(),
        "lerobot.robots.bi_piper_follower.bi_piper_follower": MagicMock(),
    }
    with patch.dict(sys.modules, mock_modules):
        yield


def _init_and_capture(mock_keyboard, prompt_switcher=None):
    """Init listener with pynput mocked, return (events, hotkeys_dict)."""
    from lerobot.utils.control_utils import init_keyboard_listener
    with patch("lerobot.utils.control_utils.is_headless", return_value=False):
        listener, events = init_keyboard_listener(prompt_switcher=prompt_switcher)
    return events, mock_keyboard._captured["hotkeys"]


class TestEventsDict:
    def test_initial_event_keys(self, mock_pynput_module):
        events, _ = _init_and_capture(mock_pynput_module)
        for key in ("exit_early", "rerecord_episode", "stop_recording",
                    "switch_infer_mode", "resume_inference",
                    "lang_prompt_key", "task_success"):
            assert key in events
        assert events["task_success"] is None
        assert events["lang_prompt_key"] is None


class TestHotkeysRegistered:
    def test_all_static_chords_present(self, mock_pynput_module):
        _, hotkeys = _init_and_capture(mock_pynput_module)
        for chord in (
            "<ctrl>+<right>", "<ctrl>+<down>", "<ctrl>+<left>",
            "<ctrl>+<esc>", "<ctrl>+<space>", "<ctrl>+<enter>",
        ):
            assert chord in hotkeys


class TestStaticChordCallbacks:
    """The whole point of GlobalHotKeys: a single chord callback runs only when
    BOTH keys are down — no sticky modifier state to misfire later."""

    def test_ctrl_right_marks_success(self, mock_pynput_module):
        events, hotkeys = _init_and_capture(mock_pynput_module)
        with patch("lerobot.recording.utils.tts.log_say") as log_say:
            hotkeys["<ctrl>+<right>"]()
        assert events["task_success"] is True
        assert events["exit_early"] is True
        log_say.assert_called_once_with(
            "任务成功，结束当前录制", play_sounds=True, blocking=False, enabled=True
        )

    def test_ctrl_down_marks_failure(self, mock_pynput_module):
        events, hotkeys = _init_and_capture(mock_pynput_module)
        with patch("lerobot.recording.utils.tts.log_say") as log_say:
            hotkeys["<ctrl>+<down>"]()
        assert events["task_success"] is False
        assert events["exit_early"] is True
        log_say.assert_called_once_with(
            "任务失败，结束当前录制", play_sounds=True, blocking=False, enabled=True
        )

    def test_ctrl_left_rerecord(self, mock_pynput_module):
        events, hotkeys = _init_and_capture(mock_pynput_module)
        with patch("lerobot.recording.utils.tts.log_say") as log_say:
            hotkeys["<ctrl>+<left>"]()
        assert events["rerecord_episode"] is True
        assert events["exit_early"] is True
        log_say.assert_called_once_with(
            "重新录制当前数据", play_sounds=True, blocking=False, enabled=True
        )

    def test_ctrl_esc_stop(self, mock_pynput_module):
        events, hotkeys = _init_and_capture(mock_pynput_module)
        hotkeys["<ctrl>+<esc>"]()
        assert events["stop_recording"] is True
        assert events["exit_early"] is True

    def test_ctrl_space_switch_infer(self, mock_pynput_module):
        events, hotkeys = _init_and_capture(mock_pynput_module)
        hotkeys["<ctrl>+<space>"]()
        assert events["switch_infer_mode"] is True

    def test_ctrl_enter_resume_infer(self, mock_pynput_module):
        events, hotkeys = _init_and_capture(mock_pynput_module)
        hotkeys["<ctrl>+<enter>"]()
        assert events["resume_inference"] is True


class TestPromptSwitcher:
    def test_letters_registered_when_switcher_enabled(self, mock_pynput_module):
        switcher = MagicMock()
        switcher.enabled = True
        switcher.variants = {"a": "alpha", "B": "bravo"}
        _, hotkeys = _init_and_capture(mock_pynput_module, prompt_switcher=switcher)
        assert "<ctrl>+a" in hotkeys
        assert "<ctrl>+b" in hotkeys

    def test_callback_sets_lang_prompt_key(self, mock_pynput_module):
        switcher = MagicMock()
        switcher.enabled = True
        switcher.variants = {"a": "alpha"}
        events, hotkeys = _init_and_capture(mock_pynput_module, prompt_switcher=switcher)
        hotkeys["<ctrl>+a"]()
        assert events["lang_prompt_key"] == "a"

    def test_disabled_switcher_registers_no_letter_chords(self, mock_pynput_module):
        switcher = MagicMock()
        switcher.enabled = False
        switcher.variants = {"a": "alpha"}
        _, hotkeys = _init_and_capture(mock_pynput_module, prompt_switcher=switcher)
        assert "<ctrl>+a" not in hotkeys

    def test_no_switcher_registers_no_letter_chords(self, mock_pynput_module):
        _, hotkeys = _init_and_capture(mock_pynput_module)
        # Only the static chords, nothing else.
        assert all(k.startswith("<ctrl>+<") for k in hotkeys)


class TestNoStickyState:
    """Regression: with the old implementation, calling a non-Ctrl callback
    after a missed Ctrl release would fire combos. With GlobalHotKeys there
    is no Python-side modifier state to leak — the only callable surface is
    the chord dict, and each entry requires both keys held."""

    def test_no_modifier_state_attribute_exposed(self, mock_pynput_module):
        # The new implementation must not expose any sticky boolean —
        # there is nothing for callers to inspect or accidentally read stale.
        from lerobot.utils import control_utils
        assert not hasattr(control_utils, "ctrl_pressed")

    def test_callbacks_are_independent(self, mock_pynput_module):
        events, hotkeys = _init_and_capture(mock_pynput_module)
        # Firing the success chord must not bleed into other chords' state.
        hotkeys["<ctrl>+<right>"]()
        assert events["task_success"] is True
        # Firing a different chord overwrites only its own keys.
        hotkeys["<ctrl>+<space>"]()
        assert events["switch_infer_mode"] is True
        assert events["task_success"] is True  # unchanged


# ---------------------------------------------------------------------------
# Terminal-mode listener (no-DISPLAY, SSH-friendly)
# ---------------------------------------------------------------------------

class TestTerminalKeyboardListener:
    """When DISPLAY is unset (typical SSH session), the listener reads
    single-key bindings from stdin instead of hooking pynput's GlobalHotKeys."""

    def _build_listener_with_fake_stdin(self):
        """Construct a _TerminalKeyboardListener whose ``_dispatch`` we can
        invoke directly (skipping the actual stdin reader thread + termios)."""
        from lerobot.utils.control_utils import _TerminalKeyboardListener
        events = {
            "exit_early": False, "rerecord_episode": False,
            "stop_recording": False, "switch_infer_mode": False,
            "resume_inference": False, "lang_prompt_key": None,
            "task_success": None,
        }
        # __init__ touches sys.stdin.fileno() — fine in pytest (it's a real FD)
        listener = _TerminalKeyboardListener(events, prompt_switcher=None)
        return events, listener

    def test_y_marks_success(self, capsys):
        events, listener = self._build_listener_with_fake_stdin()
        with patch("lerobot.recording.utils.tts.log_say") as log_say:
            listener._dispatch("y")
        assert events["task_success"] is True
        assert events["exit_early"] is True
        log_say.assert_called_once_with(
            "任务成功，结束当前录制", play_sounds=True, blocking=False, enabled=True
        )

    def test_n_marks_failure(self, capsys):
        events, listener = self._build_listener_with_fake_stdin()
        listener._dispatch("n")
        assert events["task_success"] is False
        assert events["exit_early"] is True

    def test_r_rerecords(self, capsys):
        events, listener = self._build_listener_with_fake_stdin()
        listener._dispatch("r")
        assert events["rerecord_episode"] is True
        assert events["exit_early"] is True

    def test_q_stops_session(self, capsys):
        events, listener = self._build_listener_with_fake_stdin()
        listener._dispatch("q")
        assert events["stop_recording"] is True
        assert events["exit_early"] is True

    def test_space_switches_infer(self, capsys):
        events, listener = self._build_listener_with_fake_stdin()
        listener._dispatch(" ")
        assert events["switch_infer_mode"] is True

    def test_enter_resumes_infer(self, capsys):
        events, listener = self._build_listener_with_fake_stdin()
        for ch in ("\r", "\n"):
            events, listener = self._build_listener_with_fake_stdin()
            listener._dispatch(ch)
            assert events["resume_inference"] is True

    def test_unknown_key_ignored(self, capsys):
        events, listener = self._build_listener_with_fake_stdin()
        listener._dispatch("z")
        # Nothing was set.
        assert events["task_success"] is None
        assert events["exit_early"] is False

    def test_prompt_switcher_letter_takes_precedence(self):
        from lerobot.utils.control_utils import _TerminalKeyboardListener
        switcher = MagicMock()
        switcher.enabled = True
        switcher.variants = {"y": "alpha"}  # collides with built-in 'y'
        events = {
            "exit_early": False, "rerecord_episode": False,
            "stop_recording": False, "switch_infer_mode": False,
            "resume_inference": False, "lang_prompt_key": None,
            "task_success": None,
        }
        listener = _TerminalKeyboardListener(events, prompt_switcher=switcher)
        listener._dispatch("y")
        # Prompt switch wins; success NOT marked.
        assert events["lang_prompt_key"] == "y"
        assert events["task_success"] is None
        assert events["exit_early"] is False

    def test_init_returns_none_when_stdin_not_tty(self):
        from lerobot.utils.control_utils import _init_terminal_keyboard_listener
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            assert _init_terminal_keyboard_listener({}, None) is None


class TestDispatchSelection:
    """init_keyboard_listener picks terminal vs pynput based on DISPLAY +
    LEROBOT_TERMINAL_KEYS env vars."""

    def test_no_display_routes_to_terminal(self, mock_pynput_module):
        from lerobot.utils.control_utils import init_keyboard_listener
        with patch.dict("os.environ", {"DISPLAY": "", "LEROBOT_TERMINAL_KEYS": ""}, clear=False), \
             patch("lerobot.utils.control_utils.is_headless", return_value=False), \
             patch("lerobot.utils.control_utils._init_terminal_keyboard_listener") as term_init, \
             patch("lerobot.utils.control_utils._init_pynput_keyboard_listener") as pynput_init:
            os.environ.pop("DISPLAY", None)
            term_init.return_value = MagicMock()
            init_keyboard_listener()
            term_init.assert_called_once()
            pynput_init.assert_not_called()

    def test_display_set_routes_to_pynput(self, mock_pynput_module):
        from lerobot.utils.control_utils import init_keyboard_listener
        with patch.dict("os.environ", {"DISPLAY": ":0"}, clear=False), \
             patch("lerobot.utils.control_utils.is_headless", return_value=False), \
             patch("lerobot.utils.control_utils._init_terminal_keyboard_listener") as term_init, \
             patch("lerobot.utils.control_utils._init_pynput_keyboard_listener") as pynput_init:
            pynput_init.return_value = MagicMock()
            init_keyboard_listener()
            term_init.assert_not_called()
            pynput_init.assert_called_once()

    def test_terminal_force_env_var_overrides_display(self, mock_pynput_module):
        from lerobot.utils.control_utils import init_keyboard_listener
        with patch.dict("os.environ", {"DISPLAY": ":0", "LEROBOT_TERMINAL_KEYS": "1"}, clear=False), \
             patch("lerobot.utils.control_utils.is_headless", return_value=False), \
             patch("lerobot.utils.control_utils._init_terminal_keyboard_listener") as term_init, \
             patch("lerobot.utils.control_utils._init_pynput_keyboard_listener") as pynput_init:
            term_init.return_value = MagicMock()
            init_keyboard_listener()
            term_init.assert_called_once()
            pynput_init.assert_not_called()

    def test_terminal_init_failure_falls_back_to_pynput(self, mock_pynput_module):
        from lerobot.utils.control_utils import init_keyboard_listener
        with patch.dict("os.environ", {"DISPLAY": "", "LEROBOT_TERMINAL_KEYS": ""}, clear=False), \
             patch("lerobot.utils.control_utils.is_headless", return_value=False), \
             patch("lerobot.utils.control_utils._init_terminal_keyboard_listener", return_value=None), \
             patch("lerobot.utils.control_utils._init_pynput_keyboard_listener") as pynput_init:
            os.environ.pop("DISPLAY", None)
            pynput_init.return_value = MagicMock()
            init_keyboard_listener()
            pynput_init.assert_called_once()

    def test_no_display_with_pynput_unimportable_uses_terminal(self, mock_pynput_module):
        """Regression: SSH'd Linux box where pynput fails to import (no
        DISPLAY → ImportError raised by pynput's X-connection probe) used
        to short-circuit on ``is_headless()`` and disable all keyboard
        input. Now the terminal listener gets a chance first — operators
        running through VS Code Remote-SSH still get y/n/r/q bindings."""
        from lerobot.utils.control_utils import init_keyboard_listener
        with patch.dict("os.environ", {"DISPLAY": ""}, clear=False), \
             patch("lerobot.utils.control_utils.is_headless", return_value=True), \
             patch("lerobot.utils.control_utils._init_terminal_keyboard_listener") as term_init, \
             patch("lerobot.utils.control_utils._init_pynput_keyboard_listener") as pynput_init:
            os.environ.pop("DISPLAY", None)
            term_init.return_value = MagicMock()
            listener, _ = init_keyboard_listener()
            term_init.assert_called_once()
            pynput_init.assert_not_called()
            assert listener is not None
