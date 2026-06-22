"""Tests for compute_episode_returns pure function and related bug fixes.

TDD RED phase: all tests should FAIL before implementation.

Tests 1-6: compute_episode_returns pure function
Test 7: Bug 1 regression - episode_prompt_map key should use actual episode_index
Test 8: Bug 2 regression - preprocessor should NOT output returns
Test 9: Numerical equivalence with original formula
"""

from __future__ import annotations

from pathlib import Path
import textwrap
from unittest.mock import Mock

import torch

from openpi.training.frame_attributes_preprocessors.base import EXTRA_EPISODE_PROMPT_MAP
from openpi.training.frame_attributes_preprocessors.base import DatasetContext
from openpi.training.frame_attributes_preprocessors.base import EpisodeBoundary
from openpi.training.frame_attributes_preprocessors.base import FrameAttributes
from openpi.training.frame_attributes_preprocessors.value_returns_preprocessor import ValueReturnsPreprocessor
from openpi.training.rl_dataset import compute_episode_returns

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hf_dataset(num_frames: int) -> Mock:
    ds = Mock()
    ds.__len__ = Mock(return_value=num_frames)
    return ds


def _make_meta(episodes: dict[int, dict]) -> Mock:
    meta = Mock()
    meta.episodes = episodes
    return meta


def _make_ctx(
    episode_lengths: list[int],
    episodes_meta: dict[int, dict],
    repo_id: str = "test_repo",
    root: str | None = None,
) -> DatasetContext:
    total = sum(episode_lengths)
    from_indices: list[int] = []
    to_indices: list[int] = []
    offset = 0
    for length in episode_lengths:
        from_indices.append(offset)
        to_indices.append(offset + length)
        offset += length

    return DatasetContext(
        repo_id=repo_id,
        hf_dataset=_make_hf_dataset(total),
        episode_data_index={"from": from_indices, "to": to_indices},
        meta=_make_meta(episodes_meta),
        delta_indices=None,
        root=root,
    )


def _write_yaml(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content))


# ===========================================================================
# Tests 1-6: compute_episode_returns pure function
# ===========================================================================


class TestComputeEpisodeReturnsBothConfirmed:
    """EpisodeBoundary.BOTH_CONFIRMED: both start and end GT known."""

    def test_both_confirmed_positive_returns(self):
        """Test 1: positive episode, BOTH_CONFIRMED, 100 frames.
        Returns should go from ~-1 to 0 linearly.
        BOTH_CONFIRMED forces norm_length = total_steps (ignores external norm).
        """
        n = 100
        valid_start, valid_end = 0, n
        frame_indices = torch.arange(valid_start, valid_end)

        returns = compute_episode_returns(
            frame_indices=frame_indices,
            valid_start=valid_start,
            valid_end=valid_end,
            is_negative=False,
            episode_boundary=EpisodeBoundary.BOTH_CONFIRMED,
            norm_length=9999,  # should be ignored for BOTH_CONFIRMED
        )

        assert returns.shape == (n,)
        # First frame: -(remaining-1)/total = -(100-1)/100 = -0.99
        expected_start = -(n - 1) / n
        torch.testing.assert_close(returns[0], torch.tensor(expected_start), atol=1e-5, rtol=1e-5)
        # Last frame: -(1-1)/total = 0.0
        torch.testing.assert_close(returns[-1], torch.tensor(0.0), atol=1e-5, rtol=1e-5)
        # Monotonically non-decreasing
        diffs = returns[1:] - returns[:-1]
        assert torch.all(diffs >= -1e-7), "should be monotonically non-decreasing"

    def test_both_confirmed_negative_returns(self):
        """Test 2: negative episode, BOTH_CONFIRMED, 100 frames.
        Returns should go from ~0 to -1 linearly.
        """
        n = 100
        valid_start, valid_end = 0, n
        frame_indices = torch.arange(valid_start, valid_end)

        returns = compute_episode_returns(
            frame_indices=frame_indices,
            valid_start=valid_start,
            valid_end=valid_end,
            is_negative=True,
            episode_boundary=EpisodeBoundary.BOTH_CONFIRMED,
            norm_length=9999,
        )

        assert returns.shape == (n,)
        # First frame: -1 + (remaining-1)/total = -1 + (100-1)/100 = -0.01
        expected_start = -1.0 + (n - 1) / n
        torch.testing.assert_close(returns[0], torch.tensor(expected_start), atol=1e-5, rtol=1e-5)
        # Last frame: -1 + (1-1)/total = -1.0
        torch.testing.assert_close(returns[-1], torch.tensor(-1.0), atol=1e-5, rtol=1e-5)
        # Monotonically non-increasing
        diffs = returns[1:] - returns[:-1]
        assert torch.all(diffs <= 1e-7), "should be monotonically non-increasing"


class TestComputeEpisodeReturnsEndConfirmed:
    """EpisodeBoundary.END_CONFIRMED: only end GT known."""

    def test_end_only_positive_returns(self):
        """Test 3: positive, END_CONFIRMED, 100 frames, norm_length=200.
        End frame = 0, start depends on norm.
        """
        n = 100
        norm_length = 200
        valid_start, valid_end = 0, n
        frame_indices = torch.arange(valid_start, valid_end)

        returns = compute_episode_returns(
            frame_indices=frame_indices,
            valid_start=valid_start,
            valid_end=valid_end,
            is_negative=False,
            episode_boundary=EpisodeBoundary.END_CONFIRMED,
            norm_length=norm_length,
        )

        assert returns.shape == (n,)
        # End = 0
        torch.testing.assert_close(returns[-1], torch.tensor(0.0), atol=1e-5, rtol=1e-5)
        # Start = -(99)/200 = -0.495
        expected_start = -(n - 1) / norm_length
        torch.testing.assert_close(returns[0], torch.tensor(expected_start), atol=1e-5, rtol=1e-5)
        # Monotonically non-decreasing
        diffs = returns[1:] - returns[:-1]
        assert torch.all(diffs >= -1e-7)

    def test_end_only_negative_returns(self):
        """Test 4: negative, END_CONFIRMED, 100 frames, norm_length=200.
        End frame = -1, start floats above -1.
        """
        n = 100
        norm_length = 200
        valid_start, valid_end = 0, n
        frame_indices = torch.arange(valid_start, valid_end)

        returns = compute_episode_returns(
            frame_indices=frame_indices,
            valid_start=valid_start,
            valid_end=valid_end,
            is_negative=True,
            episode_boundary=EpisodeBoundary.END_CONFIRMED,
            norm_length=norm_length,
        )

        assert returns.shape == (n,)
        # End = -1
        torch.testing.assert_close(returns[-1], torch.tensor(-1.0), atol=1e-5, rtol=1e-5)
        # Start = -1 + (99)/200 = -0.505
        expected_start = -1.0 + (n - 1) / norm_length
        torch.testing.assert_close(returns[0], torch.tensor(expected_start), atol=1e-5, rtol=1e-5)
        # Monotonically non-increasing
        diffs = returns[1:] - returns[:-1]
        assert torch.all(diffs <= 1e-7)


class TestComputeEpisodeReturnsFailure:
    """UNCONFIRMED_NEGATIVE_END: FAILURE_FP → constant -1 (heuristic)."""

    def test_failure_returns(self):
        """FP (builder-failure): all frames get -1 via UNCONFIRMED_NEGATIVE_END."""
        n = 80
        valid_start, valid_end = 0, n
        frame_indices = torch.arange(valid_start, valid_end)

        returns = compute_episode_returns(
            frame_indices=frame_indices,
            valid_start=valid_start,
            valid_end=valid_end,
            is_negative=False,
            episode_boundary=EpisodeBoundary.UNCONFIRMED_NEGATIVE_END,
            norm_length=200,
        )
        assert returns.shape == (n,)
        torch.testing.assert_close(
            returns,
            torch.full((n,), -1.0),
            atol=1e-5,
            rtol=1e-5,
            msg="FAILURE_FP (UNCONFIRMED_NEGATIVE_END) should yield all -1",
        )


class TestComputeEpisodeReturnsBothConfirmedIgnoresNorm:
    """BOTH_CONFIRMED should ignore external norm_length."""

    def test_both_confirmed_ignores_norm_length(self):
        """Test 6: BOTH_CONFIRMED with absurd norm_length=9999.
        Result should match total_steps as norm.
        """
        n = 100
        valid_start, valid_end = 0, n
        frame_indices = torch.arange(valid_start, valid_end)

        returns_big_norm = compute_episode_returns(
            frame_indices=frame_indices,
            valid_start=valid_start,
            valid_end=valid_end,
            is_negative=False,
            episode_boundary=EpisodeBoundary.BOTH_CONFIRMED,
            norm_length=9999,
        )

        returns_exact_norm = compute_episode_returns(
            frame_indices=frame_indices,
            valid_start=valid_start,
            valid_end=valid_end,
            is_negative=False,
            episode_boundary=EpisodeBoundary.BOTH_CONFIRMED,
            norm_length=n,  # total_steps
        )

        torch.testing.assert_close(
            returns_big_norm,
            returns_exact_norm,
            atol=1e-7,
            rtol=1e-7,
            msg="BOTH_CONFIRMED should ignore external norm_length",
        )


# ===========================================================================
# Test 7: Bug 1 regression - episode_prompt_map key
# ===========================================================================


class TestEpisodePromptMapKeyBug:
    """Bug 1: episode_prompt_map key must be actual episode_index, not enumerate index."""

    def test_episode_prompt_map_uses_actual_episode_index(self):
        """Test 7: episodes with indices {5, 10} should produce prompt_map keys {5, 10}."""
        ep5_len = 30
        ep10_len = 40
        total = ep5_len + ep10_len

        # Construct context with episodes keyed by 5 and 10 (not 0 and 1)
        episodes_meta = {
            5: {"tasks": ["hang_clothes"], "role": "builder", "success": True},
            10: {"tasks": ["take_off_clothes"], "role": "destroyer", "success": True},
        }

        ctx = DatasetContext(
            repo_id="test_repo",
            hf_dataset=_make_hf_dataset(total),
            episode_data_index={"from": [0, ep5_len], "to": [ep5_len, total]},
            meta=_make_meta(episodes_meta),
            delta_indices=None,
        )
        attrs = FrameAttributes()

        proc = ValueReturnsPreprocessor(
            classification_mode="episode",
            state_confirmation="end_only",
        )
        proc(ctx, attrs)

        prompt_map = ctx.extras.get(EXTRA_EPISODE_PROMPT_MAP)
        assert prompt_map is not None
        # Keys should be actual episode indices {5, 10}, NOT enumerate indices {0, 1}
        assert set(prompt_map.keys()) == {5, 10}, (
            f"prompt_map keys should be actual episode indices {{5, 10}}, " f"got {set(prompt_map.keys())}"
        )
        assert prompt_map[5] == proc.positive_prompt  # builder -> positive
        assert prompt_map[10] == proc.negative_prompt  # destroyer -> negative


# ===========================================================================
# Test 9: Numerical equivalence with original formula
# ===========================================================================


class TestNumericalEquivalence:
    """Regression: compute_episode_returns must match original _precompute_returns formula."""

    def test_numerical_equivalence_with_original(self):
        """Test 9: BOTH_CONFIRMED + positive + per_episode norm should exactly match
        the original formula: -(remaining-1)/total_steps.
        """
        n = 100
        valid_start, valid_end = 0, n
        frame_indices = torch.arange(valid_start, valid_end)

        returns = compute_episode_returns(
            frame_indices=frame_indices,
            valid_start=valid_start,
            valid_end=valid_end,
            is_negative=False,
            episode_boundary=EpisodeBoundary.BOTH_CONFIRMED,
            norm_length=n,
        )

        # Original formula from rl_dataset._precompute_returns:
        # remaining = (valid_end - frame_indices).float()
        # returns = clamp(-(remaining - 1) / norm_length, -1, 0)
        remaining = (valid_end - frame_indices).float()
        expected = torch.clamp(-(remaining - 1) / n, min=-1.0, max=0.0)

        torch.testing.assert_close(
            returns,
            expected,
            atol=1e-7,
            rtol=1e-7,
            msg="compute_episode_returns must be numerically identical to original formula",
        )


# ===========================================================================
# Part 2: UNCONFIRMED_NEGATIVE_END / UNCONFIRMED_POSITIVE_END split
# ===========================================================================


class TestFailureBoundarySplit:
    """Part 2: UNCONFIRMED split into NEGATIVE_END (GT=-1) / POSITIVE_END (GT=0).

    Heuristic approximation (NOT correct label — see Phase D state-based GT):
    - FAILURE_FP = builder-failure = typically ends at initial-state (far from
      target) → -1 is a reasonable constant proxy.
    - FAILURE_FN = destroyer-failure = typically ends at initial-state (still
      AT target because 'pull failed') → 0 is a reasonable constant proxy.
    """

    def test_unconfirmed_negative_end_returns_all_minus_one(self):
        """UNCONFIRMED_NEGATIVE_END: all frames get constant -1."""
        import pytest

        pytest.importorskip("openpi.training.frame_attributes_preprocessors.base")
        from openpi.training.frame_attributes_preprocessors.base import EpisodeBoundary as EB  # noqa: N817

        if not hasattr(EB, "UNCONFIRMED_NEGATIVE_END"):
            pytest.fail("EpisodeBoundary.UNCONFIRMED_NEGATIVE_END not yet defined (Red phase)")

        n = 80
        valid_start, valid_end = 0, n
        frame_indices = torch.arange(valid_start, valid_end)

        returns = compute_episode_returns(
            frame_indices=frame_indices,
            valid_start=valid_start,
            valid_end=valid_end,
            is_negative=False,  # FP ≡ builder failure ≡ positive-prompt side
            episode_boundary=EB.UNCONFIRMED_NEGATIVE_END,
            norm_length=200,
        )
        assert returns.shape == (n,)
        torch.testing.assert_close(returns, torch.full((n,), -1.0), atol=1e-5, rtol=1e-5)

    def test_unconfirmed_positive_end_returns_all_zero(self):
        """UNCONFIRMED_POSITIVE_END: all frames get constant 0."""
        import pytest

        from openpi.training.frame_attributes_preprocessors.base import EpisodeBoundary as EB  # noqa: N817

        if not hasattr(EB, "UNCONFIRMED_POSITIVE_END"):
            pytest.fail("EpisodeBoundary.UNCONFIRMED_POSITIVE_END not yet defined (Red phase)")

        n = 80
        valid_start, valid_end = 0, n
        frame_indices = torch.arange(valid_start, valid_end)

        returns = compute_episode_returns(
            frame_indices=frame_indices,
            valid_start=valid_start,
            valid_end=valid_end,
            is_negative=True,  # FN ≡ destroyer failure ≡ negative-prompt side
            episode_boundary=EB.UNCONFIRMED_POSITIVE_END,
            norm_length=200,
        )
        assert returns.shape == (n,)
        torch.testing.assert_close(returns, torch.full((n,), 0.0), atol=1e-5, rtol=1e-5)

    def test_unconfirmed_negative_end_rejects_is_negative_true(self):
        """C4 assert: NEGATIVE_END requires is_negative=False (FP class)."""
        import pytest

        from openpi.training.frame_attributes_preprocessors.base import EpisodeBoundary as EB  # noqa: N817

        if not hasattr(EB, "UNCONFIRMED_NEGATIVE_END"):
            pytest.fail("EpisodeBoundary.UNCONFIRMED_NEGATIVE_END not yet defined (Red phase)")

        n = 40
        with pytest.raises(AssertionError):
            compute_episode_returns(
                frame_indices=torch.arange(n),
                valid_start=0,
                valid_end=n,
                is_negative=True,  # wrong combination
                episode_boundary=EB.UNCONFIRMED_NEGATIVE_END,
                norm_length=100,
            )

    def test_unconfirmed_positive_end_rejects_is_negative_false(self):
        """C4 assert: POSITIVE_END requires is_negative=True (FN class)."""
        import pytest

        from openpi.training.frame_attributes_preprocessors.base import EpisodeBoundary as EB  # noqa: N817

        if not hasattr(EB, "UNCONFIRMED_POSITIVE_END"):
            pytest.fail("EpisodeBoundary.UNCONFIRMED_POSITIVE_END not yet defined (Red phase)")

        n = 40
        with pytest.raises(AssertionError):
            compute_episode_returns(
                frame_indices=torch.arange(n),
                valid_start=0,
                valid_end=n,
                is_negative=False,  # wrong combination
                episode_boundary=EB.UNCONFIRMED_POSITIVE_END,
                norm_length=100,
            )
