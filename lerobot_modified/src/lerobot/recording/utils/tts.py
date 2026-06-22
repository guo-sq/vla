"""Thread-safe TTS service and the single canonical log_say."""

import atexit
import io
import logging
import threading
import time
from typing import Callable, Optional

# Optional dependency — gracefully degrade without pygame/requests
try:
    import pygame
    _HAS_PYGAME = True
except ImportError:
    _HAS_PYGAME = False

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


class TTSService:
    """Thread-safe singleton TTS service.

    Uses an external TTS HTTP server if available, otherwise falls back
    to print-only mode.  All audio playback is serialized via a lock
    to avoid pygame.mixer thread-safety issues.
    """

    _instance: Optional["TTSService"] = None
    _init_lock = threading.Lock()

    def __init__(self):
        self._mixer_lock = threading.Lock()
        self._mixer_initialized = False
        self._tts_url: str | None = None
        self._available = False
        self._detect_tts_server()

    @classmethod
    def get_instance(cls) -> "TTSService":
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = cls()
                atexit.register(cls._instance.cleanup)
        return cls._instance

    def cleanup(self):
        """Stop all mixer channels and release pygame resources. Idempotent.

        Long sessions can accumulate ``pygame.mixer.Sound`` references inside
        active channels; this hook gives pygame a chance to release its
        internal buffers at process exit. Wired via ``atexit`` from
        ``get_instance`` so any callers that touched the singleton get
        cleaned up regardless of whether they call this directly.
        """
        if self._mixer_initialized and _HAS_PYGAME:
            try:
                with self._mixer_lock:
                    pygame.mixer.stop()
                    pygame.mixer.quit()
            except Exception as e:
                logging.warning(f"TTSService: cleanup failed: {e}")
            self._mixer_initialized = False

    def _detect_tts_server(self):
        if not _HAS_REQUESTS:
            return
        for port in [5050, 5000]:
            url = f"http://localhost:{port}/tts"
            try:
                resp = requests.get(url.replace("/tts", "/health"), timeout=1)
                if resp.status_code == 200:
                    self._tts_url = url
                    self._available = True
                    return
            except Exception:
                continue

    def _ensure_mixer(self):
        if not _HAS_PYGAME or self._mixer_initialized:
            return
        try:
            pygame.mixer.init()
            self._mixer_initialized = True
        except Exception as e:
            logging.warning(f"TTSService: pygame.mixer.init() failed: {e}")

    def speak(self, text: str, blocking: bool = True):
        """Speak the text via the HTTP TTS server when available, else fall
        back to OS-level TTS (``espeak-ng`` on Linux, ``say`` on macOS,
        PowerShell on Windows). Operators without the HTTP TTS server
        running still get audio cues."""
        # Fast path — HTTP TTS server (preferred: nicer voice, network audio).
        if self._available and self._tts_url and _HAS_REQUESTS:
            try:
                resp = requests.post(self._tts_url, json={"text": text}, timeout=10)
                if resp.status_code == 200 and _HAS_PYGAME:
                    self._play_audio_bytes(resp.content, blocking=blocking)
                    return
            except Exception as e:
                logging.warning(f"TTSService: HTTP speak failed, falling back: {e}")

        # Fallback — subprocess to the OS TTS binary.
        self._os_say(text, blocking=blocking)

    @staticmethod
    def _os_say(text: str, blocking: bool = True):
        """Subprocess fallback to OS-level TTS. Silent no-op when the OS
        binary (e.g. espeak-ng on Linux) is not installed.

        Non-blocking calls run the legacy ``say`` (which uses
        ``subprocess.Popen``) inside a daemon thread that ``wait()``s on the
        child — this guarantees the OS reaps the espeak-ng process and
        prevents zombie accumulation over long sessions. Without this, every
        non-blocking ``say(...)`` left a ``<defunct>`` entry in the process
        table; over a 50-minute session that adds up.
        """
        def _do():
            try:
                from lerobot.utils.utils import say as _os_say_fn
                # Force blocking=True inside the helper: subprocess.run
                # waits + reaps internally. Our own thread is the
                # "non-blocking" caller; it dies once the child exits.
                _os_say_fn(text, blocking=True)
            except FileNotFoundError:
                # espeak-ng / say / PowerShell missing — no audio is fine,
                # the text was already printed by ``log_say``.
                pass
            except Exception as e:
                logging.warning(f"TTSService: OS-level speak failed: {e}")

        if blocking:
            _do()
        else:
            threading.Thread(target=_do, daemon=True).start()

    def _play_audio_bytes(self, wav_bytes: bytes, blocking: bool = True):
        self._ensure_mixer()
        if not self._mixer_initialized:
            return
        try:
            with self._mixer_lock:
                sound = pygame.mixer.Sound(io.BytesIO(wav_bytes))
                sound.set_volume(10.0)
                sound.play()
                if blocking:
                    time.sleep(sound.get_length())
        except Exception as e:
            logging.warning(f"TTSService: audio playback failed: {e}")


def log_say(
    text: str,
    *,
    play_sounds: bool = True,
    blocking: bool = False,
    enabled: bool = True,
):
    """Print a message and optionally speak it via TTS.

    Set ``enabled=False`` to skip both the print and the speak — replaces the
    old ``maybe_log_say`` helper. Set ``play_sounds=False`` to print without
    speaking.
    """
    if not enabled:
        return
    print(text)
    if play_sounds:
        try:
            tts = TTSService.get_instance()
            tts.speak(text, blocking=blocking)
        except Exception:
            pass


def make_log_say(cfg) -> Callable[..., None]:
    """Bind ``play_sounds`` and ``enabled`` from a RecordConfig once.

    Returns a closure ``say(text, *, blocking=False)`` so call sites stay short.
    """
    play_sounds = cfg.play_sounds
    enabled = cfg.enable_log_say

    def say(text: str, *, blocking: bool = False):
        log_say(text, play_sounds=play_sounds, blocking=blocking, enabled=enabled)

    return say
