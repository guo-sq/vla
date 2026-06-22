"""Regression tests for compute_episode_metrics with short (sparse-mode) sequences.

Step 7.5 of the sparse inference optimization: when ``run_inference_on_repo`` is
called with ``sparse_mode=True``, each episode result contains only 2 frames
(head + tail) or fewer. The original ``compute_episode_metrics`` assumed dense
sequences and would IndexError on empty arrays and produce meaningless
mae/rmse/pearson on 1-2 frames. This module pins down the behavior we want:

- len == 0 -> every metric is NaN (defensive guard).
- len == 1 -> head_mse computed, tail_mse NaN, dense metrics NaN.
- len == 2 -> head_mse + tail_mse computed, mae/rmse/pearson/r² NaN.
- len >= 3 -> unchanged (existing dense behavior).

NaN is used (instead of ``None``) to preserve compatibility with
``compute_quadrant_summary``, which uses ``np.nanmean`` / ``np.nanstd`` to
aggregate across mixed dense + sparse episode lists.

See plan Step 7.5 at /root/.claude/plans/playful-marinating-summit.md.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# scripts.benchmark.metrics is introduced by companion MR !7 (benchmark v4
# framework). Until that MR merges to gitlab/dev/anyverse, the module does
# not exist in this branch and the tests must be skipped gracefully.
metrics_module = pytest.importorskip(
    "scripts.benchmark.metrics",
    reason="scripts.benchmark.metrics requires companion MR !7 (benchmark v4) to merge first",
)
compute_episode_metrics = metrics_module.compute_episode_metrics

_DENSE_ONLY_KEYS = ("mae", "rmse", "pearson", "r_squared")


def test_metrics_with_two_frames():
    """head + tail only -> head_mse/tail_mse exact, dense metrics NaN."""
    pred = np.array([0.0, -0.85], dtype=np.float32)
    gt = np.array([0.0, -1.0], dtype=np.float32)

    m = compute_episode_metrics(pred, gt)

    assert math.isclose(m["head_mse"], 0.0, abs_tol=1e-9)
    assert math.isclose(m["tail_mse"], 0.0225, abs_tol=1e-6)

    for key in _DENSE_ONLY_KEYS:
        assert math.isnan(m[key]), f"{key} should be NaN for len=2 sparse episode"

    # Head/tail pred/gt must still be captured - the 2D classifier reads them.
    assert math.isclose(m["head_pred"], 0.0, abs_tol=1e-6)
    assert math.isclose(m["tail_pred"], -0.85, abs_tol=1e-6)
    assert math.isclose(m["head_gt"], 0.0, abs_tol=1e-6)
    assert math.isclose(m["tail_gt"], -1.0, abs_tol=1e-6)


def test_metrics_with_one_frame():
    """Single-frame episode (head == tail dedup) -> head_mse only, tail_mse NaN."""
    pred = np.array([0.5], dtype=np.float32)
    gt = np.array([0.3], dtype=np.float32)

    m = compute_episode_metrics(pred, gt)

    assert math.isclose(m["head_mse"], 0.04, abs_tol=1e-6)
    assert math.isnan(m["tail_mse"])

    for key in _DENSE_ONLY_KEYS:
        assert math.isnan(m[key]), f"{key} should be NaN for len=1 episode"

    assert math.isclose(m["head_pred"], 0.5, abs_tol=1e-6)
    assert math.isclose(m["head_gt"], 0.3, abs_tol=1e-6)


def test_metrics_with_dense_frames_unchanged():
    """len >= 3 path retains full dense metrics (regression guard)."""
    pred = np.linspace(0, -1, 100, dtype=np.float32)
    gt = (np.linspace(0, -1, 100) + 0.01).astype(np.float32)

    m = compute_episode_metrics(pred, gt)

    # All metrics should be finite (non-NaN) for dense mode.
    for key in ("head_mse", "tail_mse", *_DENSE_ONLY_KEYS):
        assert not math.isnan(m[key]), f"{key} must be finite for dense len=100"

    # pearson should be very high (near 1) for two nearly-identical ramps.
    assert m["pearson"] > 0.99


def test_metrics_with_empty_pred():
    """Empty pred/gt -> every metric is NaN; no IndexError."""
    pred = np.array([], dtype=np.float32)
    gt = np.array([], dtype=np.float32)

    m = compute_episode_metrics(pred, gt)

    for key in ("head_mse", "tail_mse", *_DENSE_ONLY_KEYS):
        assert math.isnan(m[key]), f"{key} should be NaN for empty pred"
