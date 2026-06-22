"""Tests for recording.utils.tts — TTSService (thread-safe singleton) and log_say."""

import threading
import unittest.mock as mock

import pytest

from lerobot.recording.utils.tts import TTSService, log_say, make_log_say


class TestTTSServiceSingleton:
    def setup_method(self):
        TTSService._instance = None

    def test_get_instance_returns_same_object(self):
        a = TTSService.get_instance()
        b = TTSService.get_instance()
        assert a is b

    def test_concurrent_get_instance_returns_same_object(self):
        results = []
        barrier = threading.Barrier(10)

        def get():
            barrier.wait()
            results.append(TTSService.get_instance())

        threads = [threading.Thread(target=get) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        assert all(r is results[0] for r in results)


class TestLogSay:
    def test_log_say_prints_message_when_play_sounds_false(self, capsys):
        log_say("test message", play_sounds=False)
        captured = capsys.readouterr()
        assert "test message" in captured.out

    def test_log_say_does_not_crash_when_play_sounds_true(self):
        # Should not raise even without audio hardware
        log_say("test message", play_sounds=True, blocking=False)

    def test_log_say_skips_when_disabled(self, capsys):
        log_say("test", play_sounds=False, enabled=False)
        captured = capsys.readouterr()
        assert "test" not in captured.out

    def test_log_say_prints_when_enabled(self, capsys):
        log_say("test message", play_sounds=False, enabled=True)
        captured = capsys.readouterr()
        assert "test message" in captured.out


class TestNonBlockingReap:
    """Non-blocking _os_say must spawn a thread that waits on the child so
    the kernel reaps it — otherwise a long session accumulates zombies."""

    def test_non_blocking_runs_in_daemon_thread(self):
        captured = {}

        def fake_say(text, blocking):
            # Record what the inner say sees.
            captured["text"] = text
            captured["blocking"] = blocking

        with mock.patch("lerobot.utils.utils.say", side_effect=fake_say):
            TTSService._os_say("hello", blocking=False)
            # Thread is daemon and the inner work is trivial; give it a
            # moment to run.
            import time as _t
            for _ in range(20):
                if "text" in captured:
                    break
                _t.sleep(0.01)
        assert captured.get("text") == "hello"
        # The wrapping thread always passes blocking=True to the inner
        # ``say`` so ``subprocess.run`` (which waits + reaps) is used.
        assert captured.get("blocking") is True

    def test_blocking_runs_inline(self):
        captured = {}
        with mock.patch("lerobot.utils.utils.say",
                        side_effect=lambda t, blocking: captured.setdefault("text", t)):
            TTSService._os_say("hello", blocking=True)
        # Synchronous: by the time _os_say returns, ``say`` has been called.
        assert captured["text"] == "hello"


class TestCleanupHook:
    """``cleanup`` is registered via atexit and tears down the pygame mixer."""

    def setup_method(self):
        TTSService._instance = None

    def test_cleanup_idempotent(self):
        svc = TTSService.__new__(TTSService)
        svc._mixer_lock = threading.Lock()
        svc._mixer_initialized = False
        svc._tts_url = None
        svc._available = False
        # Calling cleanup with mixer already-uninitialized is a no-op.
        svc.cleanup()
        svc.cleanup()  # still fine

    def test_cleanup_stops_initialized_mixer(self):
        svc = TTSService.__new__(TTSService)
        svc._mixer_lock = threading.Lock()
        svc._mixer_initialized = True
        svc._tts_url = None
        svc._available = False
        # Patch the module-level pygame symbol on the class's module.
        import lerobot.recording.utils.tts as tts_mod
        with mock.patch.object(tts_mod, "_HAS_PYGAME", True), \
             mock.patch.object(tts_mod, "pygame", create=True) as pg:
            svc.cleanup()
        pg.mixer.stop.assert_called_once()
        pg.mixer.quit.assert_called_once()
        assert svc._mixer_initialized is False


class TestSpeakFallback:
    """speak() should fall through to subprocess espeak-ng/say/PowerShell when
    the HTTP TTS server isn't available — operators without the server
    running still get audio cues."""

    def setup_method(self):
        TTSService._instance = None

    def _service_with_no_http(self):
        svc = TTSService.__new__(TTSService)
        svc._mixer_lock = mock.MagicMock()
        svc._mixer_initialized = False
        svc._tts_url = None
        svc._available = False
        return svc

    def test_falls_back_to_os_say_when_http_unavailable(self):
        svc = self._service_with_no_http()
        with mock.patch("lerobot.utils.utils.say") as mock_say:
            svc.speak("hello", blocking=False)
            # Non-blocking path runs subprocess in a daemon thread; give
            # the thread a moment to invoke its inner say().
            import time as _t
            for _ in range(20):
                if mock_say.call_count:
                    break
                _t.sleep(0.01)
        # The daemon thread always passes blocking=True to ``say`` so
        # subprocess.run waits + reaps the child (no zombie).
        mock_say.assert_called_once_with("hello", blocking=True)

    def test_falls_back_when_http_post_raises(self):
        svc = TTSService.__new__(TTSService)
        svc._mixer_lock = mock.MagicMock()
        svc._mixer_initialized = False
        svc._tts_url = "http://localhost:5050/tts"
        svc._available = True
        with mock.patch("lerobot.recording.utils.tts.requests.post",
                        side_effect=Exception("network down")), \
             mock.patch("lerobot.utils.utils.say") as mock_say:
            svc.speak("hello", blocking=True)
        mock_say.assert_called_once_with("hello", blocking=True)

    def test_os_say_swallows_filenotfounderror(self):
        # espeak-ng missing → FileNotFoundError → silent no-op (text already
        # printed by log_say upstream).
        with mock.patch("lerobot.utils.utils.say",
                        side_effect=FileNotFoundError("espeak-ng")):
            TTSService._os_say("hello", blocking=False)


class TestMakeLogSay:
    def test_factory_binds_play_sounds_and_enabled(self, capsys):
        cfg = mock.Mock(play_sounds=False, enable_log_say=True)
        say = make_log_say(cfg)
        say("hello")
        assert "hello" in capsys.readouterr().out

    def test_factory_skips_when_cfg_disables(self, capsys):
        cfg = mock.Mock(play_sounds=False, enable_log_say=False)
        say = make_log_say(cfg)
        say("hidden")
        assert "hidden" not in capsys.readouterr().out
