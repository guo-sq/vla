"""Unit tests for SparseFrameSampler.

TDD RED phase: these tests define the contract of SparseFrameSampler, which samples only
head + tail (+ optional middle frames) from each episode via `dataset.episode_mapping`.
Before implementation, every test in this module must FAIL with ImportError.

See plan Step 7.2 at /root/.claude/plans/playful-marinating-summit.md
"""

from __future__ import annotations

import torch

from openpi.training.data_loader_rl import SparseFrameSampler

# ---------------------------------------------------------------------------
# Test fixtures - minimal dataset mocks exposing only the attrs SparseFrameSampler needs
# ---------------------------------------------------------------------------


class FakeDataset:
    """Mock dataset exposing `episode_mapping` and `__len__`.

    `episode_mapping`: {ep_from: (valid_start, valid_end)} - same shape as
    `LeRobotRLDataset.episode_mapping`, but values are treated as position indices
    directly (no `_valid_frame_indices` -> identity mapping).
    """

    def __init__(self, episode_mapping: dict[int, tuple[int, int]]):
        self.episode_mapping = episode_mapping

    def __len__(self) -> int:
        if not self.episode_mapping:
            return 0
        return max(end for _, end in self.episode_mapping.values()) + 1


class FakeMultiDataset:
    """Mock MultiRLAnyverseDataset exposing `_datasets` and `cum_sizes`."""

    def __init__(self, sub_datasets: list[FakeDataset]):
        self._datasets = sub_datasets
        self.cum_sizes = []
        total = 0
        for ds in sub_datasets:
            total += len(ds)
            self.cum_sizes.append(total)

    def __len__(self) -> int:
        return self.cum_sizes[-1] if self.cum_sizes else 0


class FakeTransformedDataset:
    """Mock the ``TransformedDataset`` wrapper used in ``transform_rl_dataset``.

    Holds a reference to the inner dataset via ``_dataset`` but does NOT forward
    ``episode_mapping`` or ``_datasets``. ``SparseFrameSampler`` must drill through
    this wrapper to find the concrete RL dataset underneath.
    """

    def __init__(self, inner):
        self._dataset = inner

    def __len__(self) -> int:
        return len(self._dataset)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_single_dataset_head_tail():
    """2 episodes -> 4 indices (head + tail each), no dedup needed."""
    ds = FakeDataset({0: (0, 99), 100: (100, 199)})
    sampler = SparseFrameSampler(ds)
    assert list(sampler) == [0, 99, 100, 199]
    assert len(sampler) == 4


def test_multi_dataset():
    """Multi-dataset wraps sub-dataset indices with cumulative offsets."""
    sub1 = FakeDataset({0: (0, 49)})  # len=50, contributes positions 0 and 49
    sub2 = FakeDataset({0: (0, 29)})  # len=30, contributes positions 50+0 and 50+29
    multi = FakeMultiDataset([sub1, sub2])

    sampler = SparseFrameSampler(multi)
    indices = list(sampler)

    assert len(indices) == 4, f"expected 4 indices (2 sub-datasets x 2 frames), got {indices}"
    # sub1 frames should be in [0, 49], sub2 frames should be in [50, 79]
    assert 0 in indices, f"sub1 head missing: {indices}"
    assert 49 in indices, f"sub1 tail missing: {indices}"
    assert 50 in indices, f"sub2 head (offset 50) missing: {indices}"
    assert 79 in indices, f"sub2 tail (offset 50) missing: {indices}"


def test_single_frame_episode():
    """Episode with head==tail (single-frame episode) dedups to 1 index."""
    ds = FakeDataset({0: (5, 5)})
    sampler = SparseFrameSampler(ds)
    assert list(sampler) == [5]
    assert len(sampler) == 1


def test_empty_dataset():
    """Empty episode_mapping -> empty sampler."""
    ds = FakeDataset({})
    sampler = SparseFrameSampler(ds)
    assert list(sampler) == []
    assert len(sampler) == 0


def test_deduplication():
    """Duplicates across head/tail are removed and result is sorted ascending."""
    # Two single-frame episodes - head == tail for each
    ds = FakeDataset({0: (10, 10), 1: (20, 20)})
    sampler = SparseFrameSampler(ds)
    indices = list(sampler)

    assert indices == sorted(set(indices)), f"indices must be deduped and sorted: {indices}"
    assert indices == [10, 20]


def test_unwraps_transformed_dataset_wrapper():
    """The real pipeline wraps MultiRLAnyverseDataset in a TransformedDataset.

    Regression for a bug where ``SparseFrameSampler`` received the outer
    ``TransformedDataset``, which has neither ``episode_mapping`` nor
    ``_datasets``, and silently produced 0 sparse indices - causing the data
    loader to infinite-loop on an empty iterator for ~10+ minutes before manual
    kill. ``_unwrap`` must follow ``_dataset`` attrs until it finds a concrete
    RL dataset with the expected metadata.
    """
    inner = FakeDataset({0: (0, 49), 50: (50, 99)})
    wrapped = FakeTransformedDataset(inner)

    sampler = SparseFrameSampler(wrapped)

    assert list(sampler) == [0, 49, 50, 99]
    assert len(sampler) == 4


def test_unwraps_transformed_dataset_wrapping_multi_dataset():
    """Stack the wrappers exactly like the real pipeline: Transformed -> MultiDataset -> sub-datasets."""
    sub1 = FakeDataset({0: (0, 49)})
    sub2 = FakeDataset({0: (0, 29)})
    multi = FakeMultiDataset([sub1, sub2])
    wrapped = FakeTransformedDataset(multi)

    sampler = SparseFrameSampler(wrapped)
    indices = list(sampler)

    assert len(indices) == 4
    assert 0 in indices
    assert 49 in indices
    assert 50 in indices
    assert 79 in indices


def test_sampler_interface():
    """SparseFrameSampler must conform to torch.utils.data.Sampler protocol."""
    ds = FakeDataset({0: (0, 99)})
    sampler = SparseFrameSampler(ds)

    assert isinstance(sampler, torch.utils.data.Sampler)

    indices_from_iter = list(iter(sampler))
    assert len(indices_from_iter) == len(
        sampler
    ), f"__iter__ count {len(indices_from_iter)} must match __len__ {len(sampler)}"
