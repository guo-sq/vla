"""Integration tests for cross-negative (prompt flip) training augmentation.

Tests the full pipeline from preprocessor through to ``__getitem__`` output:
  ValueReturnsPreprocessor -> extras -> dataset attributes -> __getitem__ flip

Unlike unit tests (test_cross_negative.py) which test the flip formula in
isolation, these integration tests exercise the real
``LeRobotRLDataset.__getitem__`` code path with minimal mocking of the parent
class ``AnyverseDataset.__getitem__``.
"""

from __future__ import annotations

import random
from unittest.mock import patch

import torch

from openpi.training.anyverse_dataset import AnyverseDataset
from openpi.training.frame_attributes_preprocessors.base import DatasetContext
from openpi.training.frame_attributes_preprocessors.base import FrameAttributes
from openpi.training.frame_attributes_preprocessors.value_returns_preprocessor import ValueReturnsPreprocessor
from openpi.training.rl_dataset import LeRobotRLDataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

POS_PROMPT = "Hang the seatbelt with right hand under 20 seconds."
NEG_PROMPT = "Take the seatbelt off under 20 seconds."


def _make_ctx(episode_lengths: list[int], episodes_meta: dict) -> DatasetContext:
    """Build a minimal DatasetContext (same helper as unit tests)."""
    from unittest.mock import Mock

    total = sum(episode_lengths)
    from_indices: list[int] = []
    to_indices: list[int] = []
    offset = 0
    for length in episode_lengths:
        from_indices.append(offset)
        to_indices.append(offset + length)
        offset += length

    meta = Mock()
    meta.episodes = episodes_meta
    hf_ds = Mock()
    hf_ds.__len__ = Mock(return_value=total)

    return DatasetContext(
        repo_id="test_repo",
        hf_dataset=hf_ds,
        episode_data_index={"from": from_indices, "to": to_indices},
        meta=meta,
        delta_indices=None,
        root=None,
    )


def _make_rl_dataset(
    num_frames: int = 20,
    *,
    cross_negative_rate: float = 0.0,
    positive_prompt: str = POS_PROMPT,
    negative_prompt: str = NEG_PROMPT,
    precomputed_returns: torch.Tensor | None = None,
    is_negative: list[bool] | None = None,
    episode_prompt_map: dict[int, str] | None = None,
) -> LeRobotRLDataset:
    """Create a LeRobotRLDataset bypassing __init__, for __getitem__ testing.

    Sets up just enough internal state for ``__getitem__`` to work end-to-end.
    ``_valid_frame_indices`` is identity-mapped (sample idx == raw frame idx).
    """
    ds = LeRobotRLDataset.__new__(LeRobotRLDataset)

    if precomputed_returns is not None:
        ds._precomputed_returns = precomputed_returns
    else:
        ds._precomputed_returns = torch.linspace(-1.0, 0.0, num_frames)

    ds._valid_frame_indices = torch.arange(num_frames)

    if is_negative is not None:
        ds.is_negative_episode_tensor = torch.tensor(is_negative, dtype=torch.bool)
    else:
        ds.is_negative_episode_tensor = torch.zeros(num_frames, dtype=torch.bool)

    if episode_prompt_map is not None:
        ds.episode_prompt_map = episode_prompt_map
    else:
        # frames 0..9 → ep 0 (positive prompt), frames 10..19 → ep 1 (negative prompt)
        ds.episode_prompt_map = {
            0: positive_prompt,
            1: negative_prompt,
        }

    ds.cross_negative_rate = cross_negative_rate
    ds.positive_prompt = positive_prompt
    ds.negative_prompt = negative_prompt

    return ds


def _parent_getitem(episode_index: int = 0, task: str = POS_PROMPT):
    """Return a callable that mocks AnyverseDataset.__getitem__."""
    return {"episode_index": torch.tensor(episode_index), "task": task}


# ===========================================================================
# Pipeline: Preprocessor -> extras
# ===========================================================================


class TestPreprocessorToExtrasPipeline:
    """Verify ValueReturnsPreprocessor writes correct extras that a downstream
    consumer can read."""

    def test_prompts_flow_through_extras(self):
        proc = ValueReturnsPreprocessor(
            positive_prompt=POS_PROMPT,
            negative_prompt=NEG_PROMPT,
        )
        ctx = _make_ctx(
            [10, 10],
            {
                0: {"tasks": ["t"], "role": "builder", "success": True},
                1: {"tasks": ["t"], "role": "destroyer", "success": True},
            },
        )
        attrs = FrameAttributes()
        proc(ctx, attrs)

        # Verify extras contain prompts readable by downstream dataset
        assert ctx.extras["positive_prompt"] == POS_PROMPT
        assert ctx.extras["negative_prompt"] == NEG_PROMPT
        # Verify episode_prompt_map is also populated
        assert ctx.extras.get("episode_prompt_map") is not None
        ep_map = ctx.extras["episode_prompt_map"]
        assert ep_map[0] == POS_PROMPT  # builder → positive
        assert ep_map[1] == NEG_PROMPT  # destroyer → negative

    def test_classification_drives_is_negative_tensor(self):
        proc = ValueReturnsPreprocessor(
            negative_roles=["destroyer"],
        )
        ctx = _make_ctx(
            [5, 5, 5],
            {
                0: {"tasks": ["t"], "role": "builder", "success": True},
                1: {"tasks": ["t"], "role": "destroyer", "success": True},
                2: {"tasks": ["t"], "role": "builder", "success": False},
            },
        )
        attrs = FrameAttributes()
        proc(ctx, attrs)

        is_neg = attrs.is_negative_episode
        assert is_neg is not None
        assert not is_neg[0]  # builder → positive
        assert is_neg[5]  # destroyer → negative
        assert not is_neg[10]  # builder failure → FAILURE_FP, not is_negative


# ===========================================================================
# __getitem__ end-to-end: rate=1.0 always flips
# ===========================================================================


class TestGetItemRateOneAlwaysFlips:
    """With cross_negative_rate=1.0, every sample should be flipped."""

    def test_all_returns_flipped(self):
        num = 20
        original = torch.linspace(-1.0, 0.0, num)
        ds = _make_rl_dataset(
            num,
            cross_negative_rate=1.0,
            precomputed_returns=original.clone(),
        )

        with patch.object(AnyverseDataset, "__getitem__", return_value=_parent_getitem()):
            for i in range(num):
                result = ds.__getitem__(i)
                expected = -(1.0 + original[i])
                assert torch.isclose(
                    result["returns"], expected
                ), f"Frame {i}: expected flip {expected.item():.4f}, got {result['returns'].item():.4f}"

    def test_positive_frames_get_negative_prompt(self):
        ds = _make_rl_dataset(
            10,
            cross_negative_rate=1.0,
            is_negative=[False] * 10,
        )
        mock_item = _parent_getitem(episode_index=0, task=POS_PROMPT)

        with patch.object(AnyverseDataset, "__getitem__", return_value=mock_item):
            for i in range(10):
                result = ds.__getitem__(i)
                assert result["task"] == NEG_PROMPT, f"Frame {i}: positive episode should flip to negative prompt"

    def test_negative_frames_get_positive_prompt(self):
        ds = _make_rl_dataset(
            10,
            cross_negative_rate=1.0,
            is_negative=[True] * 10,
        )
        mock_item = _parent_getitem(episode_index=1, task=NEG_PROMPT)

        with patch.object(AnyverseDataset, "__getitem__", return_value=mock_item):
            for i in range(10):
                result = ds.__getitem__(i)
                assert result["task"] == POS_PROMPT, f"Frame {i}: negative episode should flip to positive prompt"


# ===========================================================================
# __getitem__ end-to-end: rate=0.0 never flips
# ===========================================================================


class TestGetItemRateZeroNeverFlips:
    """With cross_negative_rate=0.0 (default), no sample should be flipped."""

    def test_no_returns_flipped(self):
        num = 20
        original = torch.linspace(-1.0, 0.0, num)
        ds = _make_rl_dataset(
            num,
            cross_negative_rate=0.0,
            precomputed_returns=original.clone(),
        )

        with patch.object(AnyverseDataset, "__getitem__", return_value=_parent_getitem()):
            for i in range(num):
                result = ds.__getitem__(i)
                assert torch.isclose(
                    result["returns"], original[i]
                ), f"Frame {i}: rate=0 should not flip, but got {result['returns'].item():.4f} vs {original[i].item():.4f}"

    def test_original_prompt_preserved(self):
        ds = _make_rl_dataset(
            10,
            cross_negative_rate=0.0,
            is_negative=[False] * 10,
        )
        mock_item = _parent_getitem(episode_index=0, task=POS_PROMPT)

        with patch.object(AnyverseDataset, "__getitem__", return_value=mock_item):
            for i in range(10):
                result = ds.__getitem__(i)
                # episode_prompt_map[0] = POS_PROMPT, so task stays POS_PROMPT
                assert result["task"] == POS_PROMPT


# ===========================================================================
# __getitem__ end-to-end: rate=0.5 statistical
# ===========================================================================


class TestGetItemRateHalfStatistical:
    """With rate=0.5, approximately 50% of samples should be flipped."""

    def test_approximately_half_flipped(self):
        num = 200
        original = torch.linspace(-1.0, 0.0, num).repeat(1)  # just use linspace
        ds = _make_rl_dataset(
            num,
            cross_negative_rate=0.5,
            precomputed_returns=original.clone(),
        )

        random.seed(42)  # deterministic for test
        flipped = 0
        with patch.object(AnyverseDataset, "__getitem__", return_value=_parent_getitem()):
            for i in range(num):
                result = ds.__getitem__(i)
                if not torch.isclose(result["returns"], original[i]):
                    flipped += 1

        ratio = flipped / num
        assert 0.35 < ratio < 0.65, f"Expected ~50% flip rate with rate=0.5, got {ratio:.1%} ({flipped}/{num})"

    def test_flipped_values_match_formula(self):
        """When flip occurs, verify -(1+v) is applied exactly."""
        num = 100
        original = torch.linspace(-1.0, 0.0, num)
        ds = _make_rl_dataset(
            num,
            cross_negative_rate=1.0,  # force flip to test formula
            precomputed_returns=original.clone(),
        )

        with patch.object(AnyverseDataset, "__getitem__", return_value=_parent_getitem()):
            for i in range(num):
                result = ds.__getitem__(i)
                expected = -(1.0 + original[i])
                assert torch.isclose(
                    result["returns"], expected, atol=1e-6
                ), f"Frame {i}: flip formula mismatch: {result['returns'].item():.6f} vs {expected.item():.6f}"


# ===========================================================================
# Mixed episode types in single dataset
# ===========================================================================


class TestMixedEpisodeTypesFlip:
    """Verify flip behavior when a single dataset has both positive and
    negative episodes."""

    def test_mixed_episodes_get_correct_prompts(self):
        # First 10 frames: positive, next 10: negative
        is_neg = [False] * 10 + [True] * 10
        ds = _make_rl_dataset(
            20,
            cross_negative_rate=1.0,
            is_negative=is_neg,
        )

        random.seed(123)
        with patch.object(AnyverseDataset, "__getitem__", return_value=_parent_getitem(episode_index=0)):
            for i in range(10):
                result = ds.__getitem__(i)
                assert result["task"] == NEG_PROMPT, f"Positive frame {i} should flip to NEG"

        with patch.object(AnyverseDataset, "__getitem__", return_value=_parent_getitem(episode_index=1)):
            for i in range(10, 20):
                result = ds.__getitem__(i)
                assert result["task"] == POS_PROMPT, f"Negative frame {i} should flip to POS"


# ===========================================================================
# getattr resilience (hot-code-update defense)
# ===========================================================================


class TestGetattrResilience:
    """Verify __getitem__ survives missing attributes (stale pickle scenario)."""

    def test_missing_cross_negative_rate_defaults_to_zero(self):
        """Object without cross_negative_rate should not crash, just skip flip."""
        ds = _make_rl_dataset(5, cross_negative_rate=1.0)
        # Simulate stale pickle: delete the attribute
        del ds.cross_negative_rate

        original = ds._precomputed_returns[0].clone()
        with patch.object(AnyverseDataset, "__getitem__", return_value=_parent_getitem()):
            result = ds.__getitem__(0)
            # getattr returns 0.0 → no flip → returns unchanged
            assert torch.isclose(
                result["returns"], original
            ), "Missing cross_negative_rate should default to 0.0 (no flip)"

    def test_missing_prompts_falls_back_to_current_task(self):
        """Object without positive/negative_prompt should keep current task."""
        ds = _make_rl_dataset(5, cross_negative_rate=1.0)
        del ds.positive_prompt
        del ds.negative_prompt

        mock_item = _parent_getitem(task=POS_PROMPT)
        with patch.object(AnyverseDataset, "__getitem__", return_value=mock_item):
            result = ds.__getitem__(0)
            # Returns are still flipped (rate=1.0), but task falls back to current
            assert result["task"] == POS_PROMPT, "Missing prompts should fall back to current task"
            # But returns ARE flipped (rate check passes, getattr gives 1.0)
            original = ds._precomputed_returns[0]
            expected = -(1.0 + original)
            assert torch.isclose(result["returns"], expected), "Returns should still be flipped"

    def test_completely_bare_object_no_crash(self):
        """Object with no cross-negative attributes at all should not crash."""
        ds = LeRobotRLDataset.__new__(LeRobotRLDataset)
        ds._precomputed_returns = torch.tensor([-0.5])
        ds._valid_frame_indices = torch.tensor([0])
        ds.episode_prompt_map = None
        ds.is_negative_episode_tensor = None

        with patch.object(AnyverseDataset, "__getitem__", return_value=_parent_getitem()):
            result = ds.__getitem__(0)
            assert torch.isclose(
                result["returns"], torch.tensor(-0.5)
            ), "Bare object should pass through returns unchanged"
