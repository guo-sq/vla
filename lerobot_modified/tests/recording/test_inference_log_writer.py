"""Unit tests for InferenceLogWriter + read_inference_log round-trip."""

from __future__ import annotations

import numpy as np
import pytest

from lerobot.recording.runtime.inference_log_writer import InferenceLogWriter
from lerobot.datasets.inference_log import read_inference_log


def _sample_chunk(H: int = 4, D: int = 3) -> np.ndarray:
    return np.arange(H * D, dtype=np.float32).reshape(H, D)


class TestInferenceLogWriter:
    def test_append_and_flush_writes_parquet(self, tmp_path):
        w = InferenceLogWriter(dataset_root=tmp_path, action_horizon=4, action_dim=3)
        w.append(
            episode_index=0, step_id=0,
            t_submit=0.0, t_complete=0.05,
            action_chunk_raw=_sample_chunk(4, 3),
            prompt="hang seatbelt", delay=2, role="operator",
        )
        w.append(
            episode_index=0, step_id=10,
            t_submit=1.0, t_complete=1.012,
            action_chunk_raw=_sample_chunk(4, 3),
            prompt="hang seatbelt", delay=2, role="operator",
        )
        out = w.flush()
        assert out is not None
        assert out.exists()
        df = read_inference_log(tmp_path)
        assert df is not None
        assert len(df) == 2
        # Latency derived from timestamps
        np.testing.assert_allclose(df["latency_ms"].values, [50.0, 12.0], atol=1e-6)
        # Reshape attached for convenience
        assert "action_chunks" in df.attrs
        assert df.attrs["action_chunks"].shape == (2, 4, 3)

    def test_discard_episode_drops_only_that_episode(self, tmp_path):
        w = InferenceLogWriter(dataset_root=tmp_path, action_horizon=2, action_dim=2)
        for ep in range(3):
            w.append(
                episode_index=ep, step_id=ep * 10,
                t_submit=0.0, t_complete=0.01,
                action_chunk_raw=_sample_chunk(2, 2),
                prompt="x", delay=0,
            )
        assert w.row_count == 3
        w.discard_episode(1)
        assert w.row_count == 2
        w.flush()
        df = read_inference_log(tmp_path)
        assert sorted(df["episode_index"].tolist()) == [0, 2]

    def test_flush_with_no_rows_returns_none(self, tmp_path):
        w = InferenceLogWriter(dataset_root=tmp_path, action_horizon=1, action_dim=1)
        assert w.flush() is None
        assert not (tmp_path / "meta" / "inference_log.parquet").exists()

    def test_flush_without_dataset_root_logs_warning_and_drops(self, tmp_path, caplog):
        import logging
        w = InferenceLogWriter(dataset_root=None, action_horizon=1, action_dim=1)
        w.append(
            episode_index=0, step_id=0, t_submit=0.0, t_complete=0.001,
            action_chunk_raw=_sample_chunk(1, 1), prompt="x", delay=0,
        )
        with caplog.at_level(logging.WARNING):
            assert w.flush() is None
        assert any("no dataset_root" in rec.message for rec in caplog.records)

    def test_read_inference_log_missing_returns_none(self, tmp_path):
        assert read_inference_log(tmp_path) is None
