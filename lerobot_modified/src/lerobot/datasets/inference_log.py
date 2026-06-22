"""Reader helper for the per-inference debug log written alongside a
recorded dataset by :class:`lerobot.recording.runtime.inference_log_writer.InferenceLogWriter`.

The on-disk file lives at ``<dataset_root>/meta/inference_log.parquet`` and
is sparse — one row per inference call, not per frame. Each row stores the
raw pre-fusion action chunk as a flat ``float32`` list of length
``action_horizon * action_dim``; columns ``action_horizon`` and
``action_dim`` carry the shape so readers can reshape.

Typical use::

    from lerobot.datasets.inference_log import read_inference_log
    df = read_inference_log(dataset_root)
    if df is None:
        print("no inference log for this dataset")
    else:
        print(df["latency_ms"].describe())
        # chunks reshaped to (N, H, D)
        chunks = df.attrs["action_chunks"]
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

INFERENCE_LOG_RELPATH = "meta/inference_log.parquet"


def read_inference_log(dataset_root: Path | str, *, reshape_chunks: bool = True):
    """Return a pandas DataFrame for the dataset's inference log, or None.

    ``reshape_chunks=True`` (default) also stashes a reshaped ``(N, H, D)``
    float32 numpy array under ``df.attrs["action_chunks"]`` for convenience.
    The raw column ``action_chunk_raw`` remains as the source of truth.
    """
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover - pandas is a project dep
        raise RuntimeError(
            "read_inference_log requires pandas; install pandas to use this helper"
        ) from e

    path = Path(dataset_root) / INFERENCE_LOG_RELPATH
    if not path.exists():
        return None

    df = pd.read_parquet(path)
    if df.empty:
        return df

    if reshape_chunks:
        # action_horizon / action_dim are pinned per-row, but constant within
        # a session in practice. Use the first row's values; fall back to a
        # plain object array if rows disagree (shouldn't happen).
        try:
            H = int(df["action_horizon"].iloc[0])
            D = int(df["action_dim"].iloc[0])
            if (df["action_horizon"] == H).all() and (df["action_dim"] == D).all():
                arr = np.stack(
                    [np.asarray(c, dtype=np.float32).reshape(H, D) for c in df["action_chunk_raw"]]
                )
                df.attrs["action_chunks"] = arr
        except Exception as e:  # pragma: no cover - reshape is best-effort
            logging.warning("read_inference_log: chunk reshape skipped: %s", e)

    return df
