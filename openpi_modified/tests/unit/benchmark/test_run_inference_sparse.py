"""Unit tests for sparse/dense inference loops used by run_benchmark.

Step 7.4 extracts the per-repo inference loops from run_benchmark.py into a pure
module ``scripts.benchmark.inference_loops`` so we can exercise them without JAX
or real data. These tests drive the design of three functions:

- ``_build_frame_to_episode_map`` — ``{(ep_idx, per_ep_frame_idx): role}`` lookup
  keyed by the composite ``(episode_index, frame_index_in_episode)`` tuple that
  LeRobot returns in each sample, NOT by the global raw frame index.
- ``sparse_inference_loop`` — consumes 4-tuple batches
  ``(pred, gt, frame_index, episode_index)`` and collects head + tail preds per
  episode.
- ``dense_inference_loop`` — preserves the existing FindGap-based behavior
  (also on 4-tuples for consistency, though the episode_index is unused).

See plan Step 7.4 at /root/.claude/plans/playful-marinating-summit.md
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.benchmark.inference_loops import _build_frame_to_episode_map
from scripts.benchmark.inference_loops import dense_inference_loop
from scripts.benchmark.inference_loops import sparse_inference_loop

# ---------------------------------------------------------------------------
# _build_frame_to_episode_map
# ---------------------------------------------------------------------------


def test_build_frame_to_episode_map_two_episodes():
    """Composite keys: each episode's head = (ep_idx, 0), tail = (ep_idx, ep_length-1).

    For ``episode_mapping = {0: (0, 99), 100: (100, 199)}`` — two episodes, each
    of length 100 — the per-episode tail index is 99 for both (raw_end - raw_start).
    """
    ep_map = {0: (0, 99), 100: (100, 199)}
    result = _build_frame_to_episode_map(ep_map)

    assert result[(0, 0)] == "head"
    assert result[(0, 99)] == "tail"
    assert result[(1, 0)] == "head"
    assert result[(1, 99)] == "tail"
    assert len(result) == 4


def test_build_frame_to_episode_map_variable_length_episodes():
    """Different-length episodes get different per-episode tail indices."""
    # ep 0: global [0..49] → length 50 → per_ep_tail = 49
    # ep 1: global [50..249] → length 200 → per_ep_tail = 199
    ep_map = {0: (0, 49), 50: (50, 249)}
    result = _build_frame_to_episode_map(ep_map)

    assert result[(0, 0)] == "head"
    assert result[(0, 49)] == "tail"
    assert result[(1, 0)] == "head"
    assert result[(1, 199)] == "tail"


def test_build_frame_to_episode_map_single_frame_episode_dedups():
    """Episode with head==tail stores only head entry (matches sampler dedup)."""
    # Single frame at global index 5 → per_ep_tail = 0, so head == tail → only head
    ep_map = {0: (5, 5)}
    result = _build_frame_to_episode_map(ep_map)

    assert result == {(0, 0): "head"}


# ---------------------------------------------------------------------------
# sparse_inference_loop
# ---------------------------------------------------------------------------


def test_sparse_inference_two_frames_per_episode():
    """Sparse loop yields exactly head + tail preds per episode, in frame order."""
    ep_map = {0: (0, 99), 100: (100, 199)}
    frame_to_ep = _build_frame_to_episode_map(ep_map)

    # Each batch yields (pred, gt, frame_index, episode_index) — episode_index
    # distinguishes each episode since per-episode frame indices repeat across episodes.
    batches = [
        (
            np.array([0.0, -0.85], dtype=np.float32),  # preds for ep 0 head + tail
            np.array([0.0, -1.0], dtype=np.float32),
            np.array([0, 99]),  # per-episode frame indices
            np.array([0, 0]),  # both rows belong to ep 0
        ),
        (
            np.array([0.05, -0.90], dtype=np.float32),  # preds for ep 1 head + tail
            np.array([0.0, -1.0], dtype=np.float32),
            np.array([0, 99]),  # per-episode frame indices (restart at 0!)
            np.array([1, 1]),  # ep 1
        ),
    ]

    results = sparse_inference_loop(iter(batches), frame_to_ep)

    assert set(results.keys()) == {0, 1}
    for ep_idx in (0, 1):
        assert len(results[ep_idx]["pred"]) == 2, f"episode {ep_idx}: expected head+tail (2 preds)"
        assert len(results[ep_idx]["gt"]) == 2

    assert results[0]["pred"][0] == pytest.approx(0.0)
    assert results[0]["pred"][-1] == pytest.approx(-0.85)
    assert results[1]["pred"][0] == pytest.approx(0.05)
    assert results[1]["pred"][-1] == pytest.approx(-0.90)


def test_sparse_loop_multi_episode_single_batch_regression():
    """Production regression: multi-episode repos with per-episode frame indices.

    In v0402.batch.2 (7 episodes), LeRobot returned ``frame_index`` as per-episode
    (0..ep_length-1), not global raw. A lookup keyed only on ``frame_index`` sent
    every episode's head (frame_idx=0) to episode 0's head slot, and every
    non-zero-ep tail missed the map entirely → ``seen_frames`` never caught up to
    ``expected_frames`` → early-exit never fired → TorchRLDataLoader's re-iter
    loop consumed thousands of duplicate batches for 10+ minutes per repo.

    The composite key ``(episode_index, frame_index)`` fixes dispatch so each
    real frame reaches its true episode.
    """
    # 2 episodes of length 3 each (global [0..2] and [3..5]).
    ep_map = {0: (0, 2), 3: (3, 5)}
    frame_to_ep = _build_frame_to_episode_map(ep_map)

    # Single batch containing head+tail for BOTH episodes, as sparse would.
    # frame_index is per-episode: ep 0 head=0 tail=2, ep 1 head=0 tail=2 (reused!).
    batches = [
        (
            np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32),
            np.array([0.0, -0.5, 0.0, -0.5], dtype=np.float32),
            np.array([0, 2, 0, 2]),
            np.array([0, 0, 1, 1]),
        ),
    ]

    results = sparse_inference_loop(iter(batches), frame_to_ep)

    assert set(results.keys()) == {0, 1}, (
        "both episodes must be collected; earlier bug dropped ep 1 because both "
        "heads had per-episode frame_index=0 and mapped onto ep 0's head slot"
    )
    for ep_idx in (0, 1):
        assert len(results[ep_idx]["pred"]) == 2, f"ep {ep_idx} needs head+tail"

    assert results[0]["pred"][0] == pytest.approx(0.1)
    assert results[0]["pred"][-1] == pytest.approx(-0.2)
    assert results[1]["pred"][0] == pytest.approx(0.3)
    assert results[1]["pred"][-1] == pytest.approx(-0.4)


def test_sparse_loop_exits_early_on_full_coverage():
    """Regression: TorchRLDataLoader re-iters on StopIteration, so a naive loop
    over ``batch_iter`` would consume the same frames forever. sparse_inference_loop
    must exit after every expected frame has been observed once.

    This test supplies a generator that would yield infinitely many copies of
    the same batch — if sparse_inference_loop didn't break early, the test
    would hang (pytest timeout).
    """
    ep_map = {0: (0, 99)}
    frame_to_ep = _build_frame_to_episode_map(ep_map)

    def infinite_duplicate_batches():
        while True:
            yield (
                np.array([0.0, -0.85], dtype=np.float32),
                np.array([0.0, -1.0], dtype=np.float32),
                np.array([0, 99]),
                np.array([0, 0]),
            )

    results = sparse_inference_loop(infinite_duplicate_batches(), frame_to_ep)

    assert len(results) == 1
    assert len(results[0]["pred"]) == 2


def test_sparse_loop_dedups_within_pass():
    """Padded sentinel rows sharing the same (ep_idx, frame_idx) should not duplicate."""
    ep_map = {0: (0, 99)}
    frame_to_ep = _build_frame_to_episode_map(ep_map)

    # head + tail + three padded copies of the tail row (ep 0, frame 99).
    batches = [
        (
            np.array([0.0, -0.85, -0.85, -0.85, -0.85], dtype=np.float32),
            np.array([0.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            np.array([0, 99, 99, 99, 99]),
            np.array([0, 0, 0, 0, 0]),
        ),
    ]

    results = sparse_inference_loop(iter(batches), frame_to_ep)
    assert len(results[0]["pred"]) == 2
    assert results[0]["pred"][0] == pytest.approx(0.0)
    assert results[0]["pred"][-1] == pytest.approx(-0.85)


def test_sparse_loop_skips_unknown_frames():
    """Frames not in frame_to_ep_map (e.g., padded sentinels with frame=-1)
    are silently ignored."""
    ep_map = {0: (0, 99)}
    frame_to_ep = _build_frame_to_episode_map(ep_map)

    batches = [
        (
            np.array([0.0, 0.5, -0.85], dtype=np.float32),  # middle row is padded sentinel
            np.array([0.0, -0.5, -1.0], dtype=np.float32),
            np.array([0, -1, 99]),  # frame=-1 marks padded row
            np.array([0, 0, 0]),
        ),
    ]

    results = sparse_inference_loop(iter(batches), frame_to_ep)

    assert 0 in results
    assert len(results[0]["pred"]) == 2  # head + tail, sentinel skipped


# ---------------------------------------------------------------------------
# dense_inference_loop (regression coverage — ensure legacy path still works)
# ---------------------------------------------------------------------------


def test_dense_mode_still_uses_findgap():
    """Dense loop collects all frames per episode, splitting at FindGap boundaries.

    Dense mode ignores the episode_index column (4-tuple) since FindGap already
    detects boundaries from frame_index jumps. This preserves the legacy behavior.
    """
    dense_batches = [
        (
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
            np.array([0.0, -0.5, -1.0], dtype=np.float32),
            np.array([0, 1, 2]),
            np.array([0, 0, 0]),
        ),
        (
            np.array([0.4, 0.5, 0.6], dtype=np.float32),
            np.array([0.0, -0.5, -1.0], dtype=np.float32),
            np.array([10, 11, 12]),
            np.array([1, 1, 1]),
        ),
    ]

    result = dense_inference_loop(iter(dense_batches), total_episodes=2)

    assert 0 in result
    assert 1 in result
    assert len(result[0]["pred"]) == 3
    assert len(result[1]["pred"]) == 3
    np.testing.assert_allclose(result[0]["pred"], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(result[1]["pred"], [0.4, 0.5, 0.6])


# ---------------------------------------------------------------------------
# Cross-mode consistency — head_pred[0] and tail_pred[-1] must match
# ---------------------------------------------------------------------------


def test_sparse_matches_dense_at_head_tail():
    """For frames both modes observe, head_pred[0] and tail_pred[-1] agree exactly."""
    # Dense: 1 episode, 5 frames (all in one batch).
    dense_batches = [
        (
            np.array([0.0, -0.2, -0.4, -0.6, -0.85], dtype=np.float32),
            np.array([0.0, -0.25, -0.5, -0.75, -1.0], dtype=np.float32),
            np.array([0, 1, 2, 3, 4]),
            np.array([0, 0, 0, 0, 0]),
        ),
    ]
    dense_result = dense_inference_loop(iter(dense_batches), total_episodes=1)

    # Sparse: same episode, only per-episode frames 0 and 4.
    ep_map = {0: (0, 4)}
    frame_to_ep = _build_frame_to_episode_map(ep_map)
    sparse_batches = [
        (
            np.array([0.0, -0.85], dtype=np.float32),
            np.array([0.0, -1.0], dtype=np.float32),
            np.array([0, 4]),
            np.array([0, 0]),
        ),
    ]
    sparse_result = sparse_inference_loop(iter(sparse_batches), frame_to_ep)

    assert dense_result[0]["pred"][0] == pytest.approx(sparse_result[0]["pred"][0], abs=1e-6)
    assert dense_result[0]["pred"][-1] == pytest.approx(sparse_result[0]["pred"][-1], abs=1e-6)
