"""Unit tests for the RL sampler selection helper used by create_torch_data_loader.

Step 7.3 of the sparse inference optimization plan wires a new
``sparse_frame_sampling`` parameter through ``create_rl_data_loader`` ->
``create_torch_data_loader`` -> ``_build_rl_sampler``. This module tests the
decision helper directly so we do not have to monkey-patch the heavy LeRobot
data pipeline to exercise the sampler priority logic.

Priority order enforced by ``_build_rl_sampler``:
1. sparse_frame_sampling=True -> SparseFrameSampler (forces shuffle off)
2. existing_sampler (e.g., DistributedSampler) -> keep as-is
3. data_config.repo_sampling_weights -> RepoWeightedRandomSampler
4. neither -> ``None`` (caller keeps its original shuffle setting)

See plan Step 7.3 at /root/.claude/plans/playful-marinating-summit.md
"""

from __future__ import annotations

from types import SimpleNamespace

from openpi.training.data_loader_rl import SparseFrameSampler
from openpi.training.data_loader_rl import _build_rl_sampler


class FakeDataset:
    """Mirror of the stub in test_sparse_frame_sampler.py so tests are self-contained."""

    def __init__(self, episode_mapping: dict[int, tuple[int, int]]):
        self.episode_mapping = episode_mapping

    def __len__(self) -> int:
        if not self.episode_mapping:
            return 0
        return max(end for _, end in self.episode_mapping.values()) + 1


def _empty_data_config() -> SimpleNamespace:
    """Minimal data_config shaped for ``_build_repo_weighted_sampler`` to short-circuit."""
    return SimpleNamespace(repo_sampling_weights=None)


def test_sparse_mode_uses_sparse_sampler():
    """sparse_frame_sampling=True returns a SparseFrameSampler regardless of weights."""
    ds = FakeDataset({0: (0, 49), 50: (50, 99)})
    sampler, force_shuffle_off = _build_rl_sampler(
        ds,
        data_config=_empty_data_config(),
        seed=0,
        existing_sampler=None,
        sparse_frame_sampling=True,
    )
    assert isinstance(sampler, SparseFrameSampler)
    assert force_shuffle_off is True
    # 2 episodes x (head + tail) = 4 indices
    assert len(sampler) == 4


def test_dense_mode_preserves_default_sampler():
    """sparse_frame_sampling=False with no weights and no existing sampler -> (None, False)."""
    ds = FakeDataset({0: (0, 49)})
    sampler, force_shuffle_off = _build_rl_sampler(
        ds,
        data_config=_empty_data_config(),
        seed=0,
        existing_sampler=None,
        sparse_frame_sampling=False,
    )
    assert sampler is None
    assert force_shuffle_off is False


def test_sparse_mode_disables_shuffle():
    """Sparse mode MUST force shuffle off - order matters because indices are pre-sorted."""
    ds = FakeDataset({0: (0, 99)})
    _, force_shuffle_off = _build_rl_sampler(
        ds,
        data_config=_empty_data_config(),
        seed=0,
        existing_sampler=None,
        sparse_frame_sampling=True,
    )
    assert force_shuffle_off is True, (
        "sparse indices are sorted by construction; shuffling them would break"
        " downstream frame-to-episode mapping in run_inference_on_repo"
    )
