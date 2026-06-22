"""Pure inference-loop helpers for run_benchmark - kept JAX-free for testability.

``run_benchmark.py`` orchestrates model loading, batch scoring, and JAX/device
plumbing. The per-repo aggregation logic (episode boundary detection, head/tail
collection) used to live inline there, which tangled it with JAX. This module
extracts three pure functions operating on ``(pred_np, gt_np, frame_np)`` numpy
tuples so they can be unit-tested without a GPU:

- ``FindGap`` - stateful gap detector, split detection between episodes via the
  frame index jump threshold (moved verbatim from run_benchmark.py).
- ``dense_inference_loop`` - legacy path, accumulates every frame and splits at
  ``FindGap``-detected boundaries.
- ``_build_frame_to_episode_map`` - lookup table for sparse dispatch.
- ``sparse_inference_loop`` - collects head/tail preds per episode by direct
  frame-index lookup (no gap detection needed, because the sampler already
  knows episode boundaries).

See plan Step 7.4 at /root/.claude/plans/playful-marinating-summit.md.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeAlias

import numpy as np

# A single sparse/dense inference batch, post-JAX-to-numpy conversion:
# ``(pred, gt, frame_index_per_episode, episode_index)``. ``frame_index`` is
# LeRobot's per-episode frame index (0..ep_length-1), NOT a global raw index.
# ``episode_index`` is the per-repo episode index (0..n_episodes-1). The sparse
# aggregator needs both to unambiguously dispatch each frame to its true episode.
BatchTuple: TypeAlias = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
EpisodeResults: TypeAlias = dict[int, dict[str, np.ndarray]]


class FindGap:
    """Detect episode boundaries in a stream of frame indices.

    Returns the index of the first frame in the batch that starts a new episode,
    or ``None`` if the entire batch belongs to the current episode.
    """

    def __init__(self, threshold: int = 2):
        self.last_frame_index: int | None = None
        self.threshold = threshold

    def __call__(self, frame_index_np: np.ndarray) -> int | None:
        if self.last_frame_index is not None and abs(int(frame_index_np[0]) - self.last_frame_index) > self.threshold:
            self.last_frame_index = int(frame_index_np[-1])
            return 0
        self.last_frame_index = int(frame_index_np[-1])
        for i in range(1, len(frame_index_np)):
            if abs(int(frame_index_np[i]) - int(frame_index_np[i - 1])) > self.threshold:
                return i
        return None


def _build_frame_to_episode_map(
    episode_mapping: dict[int, tuple[int, int]],
) -> dict[tuple[int, int], str]:
    """Build a ``{(episode_idx, per_episode_frame_idx): role}`` lookup.

    LeRobot stores each sample's ``frame_index`` as a **per-episode** counter
    that restarts at 0 for every episode - it is not a global dataset index.
    That means the lookup key must be the composite ``(episode_index,
    frame_index)``; keying only on ``frame_index`` would collapse every
    episode's head (all with ``frame_index == 0``) onto episode 0, which was
    the root cause of the multi-episode infinite-loop bug in smoke testing.

    The per-episode tail index is computed from the ``episode_mapping`` values:
    each episode's global frame range ``(raw_start, raw_end)`` implies a length
    of ``raw_end - raw_start + 1`` and a per-episode tail index of ``raw_end -
    raw_start``. Single-frame episodes (``raw_start == raw_end``) store only a
    head entry to match :class:`SparseFrameSampler` dedup behavior.

    Args:
        episode_mapping: ``{ep_from: (raw_head_frame, raw_tail_frame)}`` as
            produced by ``LeRobotRLDataset.calc_episode``, where ``raw_*`` are
            global frame indices into the dataset.

    Returns:
        ``{(episode_idx, per_episode_frame_idx): "head" | "tail"}``.
    """
    frame_to_ep: dict[tuple[int, int], str] = {}
    for ep_idx, (_ep_from, (raw_head, raw_tail)) in enumerate(sorted(episode_mapping.items())):
        frame_to_ep[(ep_idx, 0)] = "head"
        per_episode_tail = int(raw_tail) - int(raw_head)
        if per_episode_tail > 0:
            frame_to_ep[(ep_idx, per_episode_tail)] = "tail"
    return frame_to_ep


def sparse_inference_loop(
    batch_iter: Iterable[BatchTuple],
    frame_to_ep_map: dict[tuple[int, int], str],
) -> EpisodeResults:
    """Collect per-episode head/tail preds from a stream of sparse batches.

    Each batch is ``(pred_np, gt_np, frame_np, episode_np)`` - the numpy
    projection of one DataLoader output after JAX scoring. The lookup key into
    ``frame_to_ep_map`` is ``(episode_index, frame_index)`` because LeRobot's
    ``frame_index`` is per-episode (restarts at 0 each episode). Rows not
    present in the map (e.g. padded sentinels with ``frame_index == -1``) are
    silently skipped.

    Exits early once every key in ``frame_to_ep_map`` has been observed -
    this matters because ``TorchRLDataLoader`` re-wraps its iterator on
    StopIteration (a training-oriented behavior), so a naive ``for`` loop over
    ``batch_iter`` would otherwise consume duplicates forever when benchmarks
    do not pin ``num_batches``. Duplicates within a single batch (e.g. padded
    sentinel rows that share a key) are filtered by ``setdefault`` dedup.

    Preds and GTs are stored per-episode in **ascending frame-index order**, so
    ``results[ep]["pred"][0]`` is always the head prediction and
    ``results[ep]["pred"][-1]`` is the tail.

    Args:
        batch_iter: Iterable yielding
            ``(pred_np, gt_np, frame_np, episode_np)`` tuples.
        frame_to_ep_map: Output of :func:`_build_frame_to_episode_map`; keyed
            by ``(episode_idx, per_episode_frame_idx)``.

    Returns:
        ``{episode_idx: {"pred": np.ndarray, "gt": np.ndarray}}``.
    """
    ep_entries: dict[int, dict[int, tuple[float, float]]] = {}
    expected_keys = set(frame_to_ep_map.keys())
    seen_keys: set[tuple[int, int]] = set()

    for pred_np, gt_np, frame_np, episode_np in batch_iter:
        for i in range(len(frame_np)):
            frame_idx = int(frame_np[i])
            ep_idx = int(episode_np[i])
            key = (ep_idx, frame_idx)
            if key not in frame_to_ep_map:
                continue
            ep_entries.setdefault(ep_idx, {}).setdefault(frame_idx, (float(pred_np[i]), float(gt_np[i])))
            seen_keys.add(key)
        if expected_keys and seen_keys >= expected_keys:
            break

    results: EpisodeResults = {}
    for ep_idx, frame_map in ep_entries.items():
        sorted_frames = sorted(frame_map.items())
        results[ep_idx] = {
            "pred": np.array([pred for _, (pred, _) in sorted_frames], dtype=np.float32),
            "gt": np.array([gt for _, (_, gt) in sorted_frames], dtype=np.float32),
        }
    return results


def dense_inference_loop(
    batch_iter: Iterable[BatchTuple],
    total_episodes: int,
) -> EpisodeResults:
    """Dense FindGap-based aggregator - the legacy code path, extracted verbatim.

    Accepts 4-tuple batches for signature parity with ``sparse_inference_loop``
    (the ``episode_np`` column is unused - FindGap detects boundaries from
    frame-index jumps alone).

    Behavioral quirk preserved: if a gap is detected in the very first batch
    before any frames have been buffered, the leading portion of that batch is
    discarded (``if buf_preds:`` guard). This only matters when episode lengths
    are shorter than a single batch, which is rare for the clothes dataset.

    Args:
        batch_iter: Iterable yielding
            ``(pred_np, gt_np, frame_np, episode_np)`` tuples.
        total_episodes: Upper bound on episodes to collect; the loop exits early
            once ``current_episode >= total_episodes``.

    Returns:
        ``{episode_idx: {"pred": np.ndarray, "gt": np.ndarray}}``.
    """
    find_gap = FindGap()
    current_episode = 0
    episode_results: EpisodeResults = {}
    buf_preds: list[np.ndarray] = []
    buf_gts: list[np.ndarray] = []

    for pred_np, gt_np, frame_np, _episode_np in batch_iter:
        gap = find_gap(frame_np)
        if gap is not None:
            if buf_preds:
                all_pred = np.concatenate([*buf_preds, pred_np[:gap]])
                all_gt = np.concatenate([*buf_gts, gt_np[:gap]])
                episode_results[current_episode] = {"pred": all_pred, "gt": all_gt}

            current_episode += 1
            if current_episode >= total_episodes:
                break

            buf_preds = [pred_np[gap:]]
            buf_gts = [gt_np[gap:]]
        else:
            buf_preds.append(pred_np)
            buf_gts.append(gt_np)

    if buf_preds:
        episode_results[current_episode] = {
            "pred": np.concatenate(buf_preds),
            "gt": np.concatenate(buf_gts),
        }
    return episode_results
