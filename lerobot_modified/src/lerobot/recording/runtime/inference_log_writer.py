"""Per-inference debug log written alongside a recorded dataset.

When ``inference.persist_inference_log`` is set in the session config, every
inference call PolicyRuntime makes is appended here. Stored as a parquet
side-table at ``<dataset_root>/meta/inference_log.parquet`` so it doesn't
touch the main frame parquet schema — downstream tooling that doesn't care
(compute_stats, transforms, the training loader) keeps ignoring it.

Use cases
---------
- Latency tail analysis: ``df['latency_ms'].quantile(0.99)`` per episode.
- Fusion debugging: compare ``action_chunk_raw`` (this log) against the
  executed action sequence in the main dataset.
- "Policy intent" for downstream RL: train on the raw pre-fusion chunk rather
  than the smoothed output the robot actually ran.

The writer buffers in memory across the whole session and writes once at
session end. Rerecord drops the in-flight episode's rows via
``discard_episode`` so the on-disk file matches the on-disk dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np


class InferenceLogWriter:
    """Collects per-inference rows in memory; flushes to parquet at session end."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        dataset_root: Path | str | None,
        action_horizon: int,
        action_dim: int,
    ):
        self.dataset_root: Path | None = Path(dataset_root) if dataset_root else None
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self._rows: list[dict] = []
        # Logged when the writer is constructed but flush() finds nothing to
        # do; lets us distinguish "no inference happened" from "persistence
        # was disabled".
        self._writes_attempted = False

    # ------------------------------------------------------------------ append
    def append(
        self,
        *,
        episode_index: int,
        step_id: int,
        t_submit: float,
        t_complete: float,
        action_chunk_raw: Sequence | np.ndarray,
        prompt: str,
        delay: int,
        role: str = "",
    ) -> None:
        """Record one inference call.

        ``t_submit`` / ``t_complete`` are seconds (``time.perf_counter``).
        ``action_chunk_raw`` is the pre-fusion chunk; shape ``(H, D)`` or any
        flatten-able equivalent — stored as a 1D float32 list of length
        ``H * D`` so parquet doesn't need a nested-array column type.
        """
        chunk = np.asarray(action_chunk_raw, dtype=np.float32).reshape(-1)
        latency_ms = float((t_complete - t_submit) * 1000.0)
        self._rows.append(
            {
                "episode_index": int(episode_index),
                "step_id": int(step_id),
                "t_submit": float(t_submit),
                "t_complete": float(t_complete),
                "latency_ms": latency_ms,
                "action_chunk_raw": chunk.tolist(),
                "action_horizon": self.action_horizon,
                "action_dim": self.action_dim,
                "prompt": str(prompt),
                "delay": int(delay),
                "role": str(role),
            }
        )

    # --------------------------------------------------------- episode control
    def discard_episode(self, episode_index: int) -> None:
        """Drop all rows tagged with ``episode_index``.

        Called on rerecord so the persisted log doesn't reference frames that
        were never saved to the main dataset.
        """
        before = len(self._rows)
        self._rows = [r for r in self._rows if r["episode_index"] != episode_index]
        dropped = before - len(self._rows)
        if dropped:
            logging.info(
                "InferenceLogWriter: discarded %d rows from episode %d (rerecord)",
                dropped,
                episode_index,
            )

    # --------------------------------------------------------------- introspect
    def __len__(self) -> int:
        return len(self._rows)

    @property
    def row_count(self) -> int:
        return len(self._rows)

    # -------------------------------------------------------------------- flush
    def flush(self) -> Path | None:
        """Write everything to ``<dataset_root>/meta/inference_log.parquet``.

        No-op if ``dataset_root`` is None (e.g. pure ``infer`` mode where there
        is no dataset to colocate with) or the buffer is empty. Returns the
        path written, or None.
        """
        self._writes_attempted = True
        if self.dataset_root is None:
            if self._rows:
                logging.warning(
                    "InferenceLogWriter: persist_inference_log=True but no "
                    "dataset_root (probably pure infer mode) — %d buffered "
                    "inference rows will be dropped.",
                    len(self._rows),
                )
            return None
        if not self._rows:
            return None

        try:
            import pandas as pd
        except ImportError as e:  # pragma: no cover - pandas is a project dep
            logging.error(
                "InferenceLogWriter.flush failed: pandas import (%s); skipping write.",
                e,
            )
            return None

        out = self.dataset_root / "meta" / "inference_log.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self._rows)
        # Header attrs we'd like to ship even if rows is empty in a future
        # call: schema version + action shape. Pandas drops parquet
        # custom_metadata silently on most engines; embed in columns instead
        # by pinning action_horizon/action_dim on every row (already done in
        # append()), so readers can recover the shape.
        df.to_parquet(out, index=False)
        logging.info(
            "InferenceLogWriter: wrote %d inference rows to %s",
            len(df),
            out,
        )
        return out
