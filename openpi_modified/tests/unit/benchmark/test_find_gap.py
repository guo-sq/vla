"""Tests for the FindGap episode boundary detector and the phantom-episode guard.

`FindGap` is the gap-detection primitive that drives episode boundary
finalization in `run_inference_on_repo`. The phantom-episode bug (commit
d16a2dc3) was a boundary condition that ate the last episode whenever
total_episodes was reached after a frame-index reset; without tests it is
trivial to reintroduce.
"""

import numpy as np
import pytest

pytestmark = [pytest.mark.unit]

from scripts.benchmark.run_benchmark import FindGap


class TestFindGapNoBoundary:
    def test_monotonic_within_threshold_returns_none(self):
        gap = FindGap()
        assert gap(np.array([0, 1, 2, 3, 4])) is None
        assert gap(np.array([5, 6, 7, 8, 9])) is None

    def test_first_call_no_history_returns_none(self):
        gap = FindGap()
        # No previous batch to compare against → no inter-batch gap signal
        assert gap(np.array([0, 1, 2])) is None


class TestFindGapWithinBatch:
    def test_jump_inside_batch_returns_jump_index(self):
        gap = FindGap()
        # 0,1,2,3,4 then a >threshold jump back at index 5: |0-4|=4 > 2
        assert gap(np.array([0, 1, 2, 3, 4, 0, 1])) == 5

    def test_jump_in_second_position_returned(self):
        gap = FindGap()
        gap(np.array([0, 1, 2]))
        # Next batch: starts cleanly at 3, then resets at index 2
        assert gap(np.array([3, 4, 0, 1])) == 2


class TestFindGapAcrossBatches:
    def test_inter_batch_jump_returns_zero(self):
        gap = FindGap()
        gap(np.array([0, 1, 2, 3]))
        # Next batch starts at 0 instead of 4 → boundary at index 0
        assert gap(np.array([0, 1, 2, 3])) == 0

    def test_inter_batch_continuous_returns_none(self):
        gap = FindGap()
        gap(np.array([0, 1, 2, 3]))
        assert gap(np.array([4, 5, 6, 7])) is None

    def test_threshold_respected(self):
        # Default threshold is 2, so a one-frame skip is not a gap
        gap = FindGap()
        gap(np.array([0, 1, 2]))
        assert gap(np.array([4, 5, 6])) is None  # |4-2|=2 not > 2
        gap2 = FindGap()
        gap2(np.array([0, 1, 2]))
        assert gap2(np.array([5, 6, 7])) == 0  # |5-2|=3 > 2


class TestFindGapMultipleEpisodes:
    def test_three_episodes_in_sequence(self):
        """Simulate three episodes back-to-back across multiple batches."""
        gap = FindGap()
        # Episode 0: frames 0..7
        assert gap(np.array([0, 1, 2, 3])) is None
        # Episode 0 still: 4..7, then episode 1 starts at 0 mid-batch
        assert gap(np.array([4, 5, 6, 7])) is None
        # Boundary at the next batch: ep1 starts cleanly
        assert gap(np.array([0, 1, 2, 3])) == 0
        # Continues
        assert gap(np.array([4, 5, 6, 7])) is None
        # Episode 2 starts mid-batch
        assert gap(np.array([8, 0, 1, 2])) == 1


class TestPhantomEpisodeGuard:
    """Replays the inner loop's gap-handling logic for the boundary case where
    `current_episode` reaches `total_episodes`. We do not import the JAX-coupled
    inference function — we re-run only the buffer-finalize state machine.
    """

    @staticmethod
    def _simulate(frame_batches: list[np.ndarray], total_episodes: int) -> list[int]:
        """Mirror the frame-buffer state machine in `run_inference_on_repo`.

        Returns the list of finalized episode indices."""
        gap = FindGap()
        finalized: list[int] = []
        buf_frames: list[np.ndarray] = []
        current_episode = 0
        exhausted_naturally = False

        for batch in frame_batches:
            g = gap(batch)
            if g is not None:
                if buf_frames:
                    finalized.append(current_episode)
                current_episode += 1
                if current_episode >= total_episodes:
                    buf_frames = []
                    exhausted_naturally = True
                    break
                buf_frames = [batch[g:]]
            else:
                buf_frames.append(batch)
        else:
            exhausted_naturally = True

        if buf_frames:
            finalized.append(current_episode)

        if not exhausted_naturally:
            raise RuntimeError("loop did not drain")
        return finalized

    def test_two_episodes_clean_finalize(self):
        # ep0: 0..7, ep1: 0..7
        batches = [
            np.array([0, 1, 2, 3]),
            np.array([4, 5, 6, 7]),
            np.array([0, 1, 2, 3]),
            np.array([4, 5, 6, 7]),
        ]
        finalized = self._simulate(batches, total_episodes=2)
        assert finalized == [0, 1]

    def test_truncated_last_episode(self):
        # ep0 fully (0..7), ep1 truncated (0..3)
        batches = [
            np.array([0, 1, 2, 3]),
            np.array([4, 5, 6, 7]),
            np.array([0, 1, 2, 3]),
        ]
        finalized = self._simulate(batches, total_episodes=2)
        assert finalized == [0, 1]

    def test_boundary_at_total_episodes_does_not_emit_phantom(self):
        """Regression: previously the loop would finalize a phantom n+1 episode
        whenever the gap that reset `current_episode` to `total_episodes` left
        post-tail data in the buffer."""
        # ep0: 0..3, ep1: 0..3, then a stray batch that would have produced a phantom
        batches = [
            np.array([0, 1, 2, 3]),
            np.array([0, 1, 2, 3]),
            np.array([0, 1, 2, 3]),  # would be ep2 = phantom
        ]
        finalized = self._simulate(batches, total_episodes=2)
        assert finalized == [0, 1]
        assert 2 not in finalized

    def test_single_episode_runs_to_completion(self):
        batches = [
            np.array([0, 1, 2, 3]),
            np.array([4, 5, 6, 7]),
        ]
        finalized = self._simulate(batches, total_episodes=1)
        assert finalized == [0]


class TestFindGapKnownLimitation:
    """Documents an actual current behavior that callers should be aware of."""

    def test_first_batch_starts_with_high_index(self):
        """When the data loader's very first batch begins at a high frame index
        (e.g. resuming after a checkpoint), there is no prior history so no gap
        is signaled — current_episode stays 0 even though those frames belong
        to a later episode. The benchmark guards against this elsewhere by
        always processing repos from frame 0."""
        gap = FindGap()
        assert gap(np.array([100, 101, 102])) is None
