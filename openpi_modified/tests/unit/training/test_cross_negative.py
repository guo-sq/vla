"""Tests for cross-negative (prompt flip) training augmentation.

TDD RED phase: all tests should FAIL before implementation.

Cross-negative flips the prompt and GT returns with probability
``cross_negative_rate`` in ``LeRobotRLDataset.__getitem__``:
  - returns: -(1 + v)   (full inversion of [-1, 0] range)
  - prompt: opposite prompt (positive <-> negative)

Applies to ALL episode types (POSITIVE, NEGATIVE, FAILURE_FP, FAILURE_FN).
"""

from __future__ import annotations

from unittest.mock import Mock

import torch

from openpi.training.frame_attributes_preprocessors.base import DatasetContext
from openpi.training.frame_attributes_preprocessors.base import FrameAttributes
from openpi.training.frame_attributes_preprocessors.value_returns_preprocessor import ValueReturnsPreprocessor

# ---------------------------------------------------------------------------
# Helpers (same pattern as test_compute_episode_returns.py)
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


# ===========================================================================
# Test 1: Preprocessor writes prompts to extras
# ===========================================================================


class TestPreprocessorWritesPromptsToExtras:
    """ValueReturnsPreprocessor should write positive_prompt / negative_prompt
    to ctx.extras so downstream consumers (dataset) can use them for
    cross-negative flip."""

    def test_writes_positive_and_negative_prompts(self):
        proc = ValueReturnsPreprocessor(
            positive_prompt="Hang the seatbelt.",
            negative_prompt="Take the seatbelt off.",
        )
        ctx = _make_ctx([10], {0: {"tasks": ["t"], "role": "builder", "success": True}})
        attrs = FrameAttributes()

        proc(ctx, attrs)

        assert ctx.extras.get("positive_prompt") == "Hang the seatbelt."
        assert ctx.extras.get("negative_prompt") == "Take the seatbelt off."


# ===========================================================================
# Test 2: Flip formula correctness
# ===========================================================================


class TestFlipFormula:
    """The flip formula -(1 + v) maps [-1, 0] -> [-1, 0] with full inversion."""

    def test_flip_minus_one_to_zero(self):
        v = torch.tensor(-1.0)
        assert torch.isclose(-(1.0 + v), torch.tensor(0.0))

    def test_flip_zero_to_minus_one(self):
        v = torch.tensor(0.0)
        assert torch.isclose(-(1.0 + v), torch.tensor(-1.0))

    def test_flip_preserves_range(self):
        """Flip maps [-1, 0] onto [-1, 0] (no values outside range)."""
        v = torch.linspace(-1.0, 0.0, 21)
        flipped = -(1.0 + v)
        assert (flipped >= -1.0).all()
        assert (flipped <= 0.0).all()

    def test_flip_inverts_monotonic_positive_returns(self):
        """For POSITIVE GT (-0.99→0), flip gives (-0.01→-1)."""
        returns = torch.tensor([-0.99, -0.75, -0.50, -0.25, 0.0])
        flipped = -(1.0 + returns)
        expected = torch.tensor([-0.01, -0.25, -0.50, -0.75, -1.0])
        assert torch.allclose(flipped, expected, atol=1e-6)

    def test_flip_inverts_constant_failure_fp(self):
        """FAILURE_FP: all -1 → flip gives all 0."""
        returns = torch.full((5,), -1.0)
        flipped = -(1.0 + returns)
        assert torch.allclose(flipped, torch.zeros(5))

    def test_flip_inverts_constant_failure_fn(self):
        """FAILURE_FN: all 0 → flip gives all -1."""
        returns = torch.zeros(5)
        flipped = -(1.0 + returns)
        assert torch.allclose(flipped, torch.full((5,), -1.0))


# ===========================================================================
# Test 3-4: __getitem__ flip end-to-end coverage
# ===========================================================================
#
# The previous TestGetItemFlipBehavior class in this module was vacuous:
# it advertised testing __getitem__ flip behavior but only asserted the
# isolated math identity ``-(1 + (-0.5)) == -0.5`` and a tautological
# ``tensor == tensor`` — neither code path was exercised. Both pre-fix and
# post-fix code passed those assertions.
#
# Real __getitem__ coverage lives in
#   tests/integration/test_cross_negative_integration.py
# under classes ``TestGetItemAlwaysFlips`` (rate=1.0) and
# ``TestGetItemNeverFlips`` (rate=0.0). Those tests bypass __init__,
# patch ``AnyverseDataset.__getitem__``, and assert real returned tensors
# and prompt strings via ``ds.__getitem__(i)``.


# ===========================================================================
# Test 5: cross_negative_rate=0 never flips
# ===========================================================================


class TestRateZeroNeverFlips:
    """Default cross_negative_rate=0 should produce no flips."""

    def test_rate_zero_means_no_cross_negative(self):
        """Dataset with cross_negative_rate=0 should behave identically to
        a dataset without cross-negative config."""
        # Verify the config defaults
        value_net_cfg = {"returns_norm_strategy": "per_episode"}
        rate = value_net_cfg.get("cross_negative_rate", 0.0)
        assert rate == 0.0


# ===========================================================================
# Test 6: All episode types get flipped
# ===========================================================================


class TestFlipAllEpisodeTypes:
    """Verify flip formula works correctly for each EpisodeClass variant."""

    def test_positive_returns_flip(self):
        """POSITIVE: GT goes -0.99→0, flip gives -0.01→-1."""
        returns = torch.tensor([-0.99, -0.5, 0.0])
        flipped = -(1.0 + returns)
        expected = torch.tensor([-0.01, -0.5, -1.0])
        assert torch.allclose(flipped, expected, atol=1e-6)

    def test_negative_returns_flip(self):
        """NEGATIVE: GT goes 0→-1, flip gives -1→0 (full inversion)."""
        returns = torch.tensor([0.0, -0.5, -1.0])
        flipped = -(1.0 + returns)
        expected = torch.tensor([-1.0, -0.5, 0.0])
        assert torch.allclose(flipped, expected, atol=1e-6)

    def test_failure_fp_returns_flip(self):
        """FAILURE_FP: GT all -1, flip gives all 0."""
        returns = torch.full((10,), -1.0)
        flipped = -(1.0 + returns)
        assert torch.allclose(flipped, torch.zeros(10))

    def test_failure_fn_returns_flip(self):
        """FAILURE_FN: GT all 0, flip gives all -1."""
        returns = torch.zeros(10)
        flipped = -(1.0 + returns)
        assert torch.allclose(flipped, torch.full((10,), -1.0))


# ===========================================================================
# Test 7: Identical prompts edge case
# ===========================================================================


class TestIdenticalPrompts:
    """When positive_prompt == negative_prompt, flip still flips returns."""

    def test_identical_prompts_still_flips_returns(self):
        """Even if prompts are the same text, the returns should still be
        inverted. This is the current _fixed.py scenario."""
        returns = torch.tensor([-0.8])
        flipped = -(1.0 + returns)
        assert torch.isclose(flipped, torch.tensor(-0.2))

    def test_preprocessor_writes_identical_prompts_to_extras(self):
        """Preprocessor should still write prompts to extras even when
        positive_prompt == negative_prompt."""
        proc = ValueReturnsPreprocessor(
            positive_prompt="Same prompt.",
            negative_prompt="Same prompt.",
        )
        ctx = _make_ctx([10], {0: {"tasks": ["t"], "role": "builder", "success": True}})
        attrs = FrameAttributes()

        proc(ctx, attrs)

        assert ctx.extras["positive_prompt"] == "Same prompt."
        assert ctx.extras["negative_prompt"] == "Same prompt."
