"""Thread-safe async logging utilities for recording sessions."""

import json
import os
import queue
import threading
import time
from datetime import datetime

# Module-level debug flag — set once at startup via set_debug_enabled()
_DEBUG_ENABLED = False


def set_debug_enabled(enabled: bool) -> None:
    """Enable or disable debug prints for the recording package."""
    global _DEBUG_ENABLED
    _DEBUG_ENABLED = enabled


def is_debug_enabled() -> bool:
    return _DEBUG_ENABLED


def debug_print(msg: str) -> None:
    """Print a [DEBUG] message if debug mode is enabled."""
    if _DEBUG_ENABLED:
        print(f"[DEBUG] {msg}")


class AsyncLogger:
    """Async file logger with background writer thread.

    All log() calls are non-blocking — messages are queued and written
    by a background thread.  Thread-safe for concurrent callers.
    """

    def __init__(self, log_dir: str, enabled: bool = True):
        self._enabled = enabled
        self._closed = False
        if not enabled:
            return

        os.makedirs(log_dir, exist_ok=True)
        self._log_path = os.path.join(log_dir, "log.txt")
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._thread.start()

    def _writer(self):
        with open(self._log_path, "a") as f:
            while True:
                msg = self._queue.get()
                if msg is None:
                    break
                f.write(msg)
                f.flush()

    def log(self, message: str) -> None:
        if not self._enabled or self._closed:
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._queue.put(f"[{ts}] {message}\n")

    def close(self) -> None:
        if not self._enabled or self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join(timeout=5.0)


class ActionChunkLogger:
    """Logs raw action chunks from policy inference to a Parquet file.

    Each row contains the step IDs, timestamp, prompt, and the flattened
    chunk so that action chunks can be replayed for open-loop evaluation.
    The chunk shape (horizon, action_dim) is stored as separate columns.
    """

    _FLUSH_EVERY = 100  # write a row group every N records

    def __init__(self, log_dir: str):
        import pyarrow as pa
        import pyarrow.parquet as pq

        os.makedirs(log_dir, exist_ok=True)
        self._path = os.path.join(log_dir, "action_chunks.parquet")
        self._pa = pa
        self._pq = pq
        self._schema = pa.schema([
            ("episode", pa.int32()),
            ("start_step", pa.int32()),
            ("receive_step", pa.int32()),
            ("ts", pa.float64()),
            ("prompt", pa.string()),
            ("chunk_rows", pa.int32()),
            ("chunk_cols", pa.int32()),
            ("chunk", pa.list_(pa.float32())),
        ])
        self._writer = pq.ParquetWriter(self._path, self._schema)
        self._buffer: list[dict] = []
        self._lock = threading.Lock()
        self._episode = 0

    def set_episode(self, episode: int) -> None:
        self._episode = episode

    def log_chunk(
        self,
        chunk,
        start_step: int,
        receive_step: int,
        prompt: str,
    ) -> None:
        import numpy as np

        arr = np.asarray(chunk, dtype=np.float32)
        record = {
            "episode": self._episode,
            "start_step": start_step,
            "receive_step": receive_step,
            "ts": time.time(),
            "prompt": prompt,
            "chunk_rows": arr.shape[0],
            "chunk_cols": arr.shape[1] if arr.ndim > 1 else 1,
            "chunk": arr.ravel().tolist(),
        }
        with self._lock:
            self._buffer.append(record)
            if len(self._buffer) >= self._FLUSH_EVERY:
                self._flush_locked()

    def _flush_locked(self) -> None:
        """Write buffered records as a row group. Caller must hold _lock."""
        if not self._buffer:
            return
        table = self._pa.Table.from_pydict(
            {k: [r[k] for r in self._buffer] for k in self._schema.names},
            schema=self._schema,
        )
        self._writer.write_table(table)
        self._buffer.clear()

    def close(self) -> None:
        with self._lock:
            self._flush_locked()
            self._writer.close()


class SelfPlayLogger:
    """Structured JSONL event logger for self-play sessions.

    Each event is a single JSON line with a timestamp, event name,
    and arbitrary keyword data.  Thread-safe.
    """

    def __init__(self, log_dir: str, enabled: bool = True):
        self._enabled = enabled
        self._closed = False
        self._lock = threading.Lock()
        if not enabled:
            return

        os.makedirs(log_dir, exist_ok=True)
        self._log_path = os.path.join(log_dir, "self_play_events.jsonl")
        self._file = open(self._log_path, "a")

    def log(self, event: str, **kwargs) -> None:
        if not self._enabled or self._closed:
            return
        record = {"ts": time.time(), "event": event, **kwargs}
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            self._file.write(line + "\n")
            self._file.flush()

    def close(self) -> None:
        if not self._enabled or self._closed:
            return
        self._closed = True
        with self._lock:
            self._file.close()
