import bisect
from collections.abc import Callable
import json
import logging
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from openpi.training.anyverse_dataset import AnyverseDataset
import openpi.training.config as _config
from openpi.training.frame_attributes_preprocessors.base import EpisodeBoundary
from openpi.training.frame_attributes_preprocessors.base import get_episode_task

logger = logging.getLogger(__name__)


def compute_episode_returns(
    frame_indices: torch.Tensor,
    valid_start: int,
    valid_end: int,
    *,
    is_negative: bool,
    episode_boundary: int,
    norm_length: int,
) -> torch.Tensor:
    """Compute GT returns for a single episode. Pure function, no self dependency.

    episode_boundary semantics:
        - UNCONFIRMED_NEGATIVE_END (0): FAILURE_FP, all frames get -1 (heuristic)
        - START_CONFIRMED (1): start GT known, uses norm_length
        - END_CONFIRMED (2): end GT known, uses norm_length
        - BOTH_CONFIRMED (3): both known, forces norm_length = total_steps
        - UNCONFIRMED_POSITIVE_END (4): FAILURE_FN, all frames get 0 (heuristic)

    The UNCONFIRMED_* branches are heuristic approximations, not correct
    labels. See the PR description C1/C5 discussion and Phase D for the
    state-based GT plan.
    """
    total_steps = valid_end - valid_start
    remaining = (valid_end - frame_indices).float()

    if episode_boundary == EpisodeBoundary.UNCONFIRMED_NEGATIVE_END:
        assert not is_negative, (
            "UNCONFIRMED_NEGATIVE_END (FAILURE_FP) requires is_negative=False; "
            f"got is_negative={is_negative}. Preprocessor boundary assignment is out of sync."
        )
        return torch.full_like(remaining, -1.0)

    if episode_boundary == EpisodeBoundary.UNCONFIRMED_POSITIVE_END:
        assert is_negative, (
            "UNCONFIRMED_POSITIVE_END (FAILURE_FN) requires is_negative=True; "
            f"got is_negative={is_negative}. Preprocessor boundary assignment is out of sync."
        )
        return torch.full_like(remaining, 0.0)

    if episode_boundary == EpisodeBoundary.BOTH_CONFIRMED:
        norm_length = total_steps

    if is_negative:
        return torch.clamp(-1.0 + (remaining - 1) / norm_length, -1.0, 0.0)
    return torch.clamp(-(remaining - 1) / norm_length, -1.0, 0.0)


def _resolve_per_task_norm_length(
    value_net_cfg: dict[str, Any],
    *,
    task_to_raw_lengths: dict[str, list[int]],
    returns_norm_percentile: float,
) -> dict[str, int]:
    """Decide the per-task norm_length dict inside a single LeRobotRLDataset.

    Priority:
    1. ``pinned_task_to_norm_length`` in cfg → use as-is (returning a copy).
    2. ``strict_rl_norm_stats`` default True → raise, pointing at the script.
    3. Explicit ``strict_rl_norm_stats=False`` → legacy percentile fallback with
       a warning so callers notice the drift-prone code path.
    """
    pinned = value_net_cfg.get("pinned_task_to_norm_length")
    if pinned is not None:
        return dict(pinned)

    strict = value_net_cfg.get("strict_rl_norm_stats", True)
    if strict:
        raise ValueError(
            "per_task strategy requires pinned_task_to_norm_length in value_net_cfg, "
            "but none was provided. Run: "
            "python scripts/compute_rl_norm_stats.py --config <name>  "
            "to precompute the per-repo raw lengths."
        )

    logger.warning(
        "strict_rl_norm_stats=False: falling back to legacy per-dataset percentile "
        "computation. This path is known to cause train/val drift — use "
        "compute_rl_norm_stats to pin stats for training."
    )
    result: dict[str, int] = {}
    for task, lengths in task_to_raw_lengths.items():
        if not lengths:
            continue
        if returns_norm_percentile >= 1.0:
            result[task] = int(max(lengths))
        else:
            result[task] = int(np.percentile(lengths, returns_norm_percentile * 100))
    return result


def _maybe_short_circuit_pinned(
    *,
    value_net_cfg: dict[str, Any] | None,
    datasets: list,
) -> dict[str, int] | None:
    """Return the pinned task_to_norm_length dict when present, else None.

    B4 hardening: the pinned dict MUST cover every task surfaced by any
    sub-dataset's ``_task_to_raw_lengths``. If a task is missing we raise
    eagerly so the user knows to re-run the precompute script.
    """
    if not value_net_cfg:
        return None
    pinned = value_net_cfg.get("pinned_task_to_norm_length")
    if pinned is None:
        return None

    required_tasks: set[str] = set()
    for ds in datasets:
        raw = getattr(ds, "_task_to_raw_lengths", None)
        if isinstance(raw, dict):
            required_tasks.update(raw.keys())

    missing = sorted(required_tasks - set(pinned.keys()))
    if missing:
        raise ValueError(
            f"pinned_task_to_norm_length is missing tasks required by sub-datasets: {missing}. "
            f"Re-run: python scripts/compute_rl_norm_stats.py --config <name> --force"
        )
    return dict(pinned)


class LeRobotRLDataset(AnyverseDataset):
    """A thin wrapper around `LeRobotDataset` that returns the current frame and
    the next-frame (t+1) data in a single item. This is useful for RL training
    where transitions (s, a, s') are needed.

    Behavior:
    - If the requested index is the last frame of an episode/chunk, the next-frame
      fields will be duplicated from the current frame (i.e., s' == s). This keeps
      indexing simple and avoids raising IndexError inside DataLoader.
    - The returned dict merges the original sample with keys prefixed by
      `next_` for the t+1 values (for example, `next_observation.images...`,
      `next_actions`, `next_timestamp`).

    This class intentionally makes minimal changes to the base class and does not
    change any I/O or download behavior.
    """

    def __init__(self, *args, **kwargs):
        # Returns normalization configuration
        self.value_net_cfg = kwargs.pop("value_net_cfg", None)
        assert self.value_net_cfg is not None, "value_net_cfg must be provided, should be configured in the config file"

        self.returns_norm_strategy = self.value_net_cfg.get("returns_norm_strategy", "fixed")
        self.returns_norm_percentile = self.value_net_cfg.get("returns_norm_percentile", 1.0)
        self.returns_norm_length = self.value_net_cfg.get("returns_norm_length", None)
        self.failure_decrease_threshold = self.value_net_cfg.get("failure_decrease_threshold", None)
        self.cross_negative_rate = self.value_net_cfg.get("cross_negative_rate", 0.0)

        # Segmented strategy parameters
        self.segment_values = self.value_net_cfg.get("segment_values", None)
        self.segment_values_file = self.value_net_cfg.get("segment_values_file", "segment_values.json")
        self._segment_boundaries: dict[int, list[int]] | None = None

        assert self.returns_norm_strategy in (
            "per_episode",
            "per_task",
            "fixed",
            "segmented",
        ), f"returns_norm_strategy must be 'per_episode', 'per_task', 'fixed', or 'segmented', got '{self.returns_norm_strategy}'"
        if self.returns_norm_strategy == "fixed":
            assert self.returns_norm_length is not None, "returns_norm_length is required when strategy is 'fixed'"
        if self.returns_norm_strategy == "per_task":
            assert (
                0.0 < self.returns_norm_percentile <= 1.0
            ), f"returns_norm_percentile must be in (0, 1], got {self.returns_norm_percentile}"
        if self.returns_norm_strategy == "segmented":
            assert self.segment_values is not None, "segment_values is required when strategy is 'segmented'"
            assert len(self.segment_values) >= 2, "segment_values must have at least 2 segments"

        logger.info(
            f"returns_norm_strategy={self.returns_norm_strategy}, "
            f"returns_norm_percentile={self.returns_norm_percentile}, "
            f"returns_norm_length={self.returns_norm_length}, "
            f"failure_decrease_threshold={self.failure_decrease_threshold}"
        )

        super().__init__(*args, **kwargs)

        # Initialize task-specific tensors (used by parent class)
        if "subtask_index" in self.hf_dataset.features:
            self.subtask_index_tensor = torch.tensor(self.hf_dataset["subtask_index"], dtype=torch.int64)
        else:
            self.subtask_index_tensor = torch.full((len(self.hf_dataset),), -1, dtype=torch.int64)
        logger.debug(f"self.subtask_index_tensor = {self.subtask_index_tensor}")

        if "frame_state" in self.hf_dataset.features:
            self.frame_state_tensor = torch.tensor(self.hf_dataset["frame_state"], dtype=torch.int64)
        else:
            self.frame_state_tensor = torch.full((len(self.hf_dataset),), 3, dtype=torch.int64)
        logger.debug(f"self.frame_state_tensor = {self.frame_state_tensor}")
        logger.debug(f"self.repo_id = {self.repo_id}")

        # Load segment boundaries for segmented strategy
        self.require_segment_file = self.value_net_cfg.get("require_segment_file", True)
        if self.returns_norm_strategy == "segmented":
            seg_file = Path(self.root) / "meta" / self.segment_values_file
            if seg_file.exists():
                with open(seg_file) as f:
                    seg_data = json.load(f)
                self._segment_boundaries = {int(k): v for k, v in seg_data["boundaries"].items()}
                logger.info(f"Loaded segment boundaries for {len(self._segment_boundaries)} episodes from {seg_file}")
            else:
                if self.require_segment_file:
                    raise FileNotFoundError(f"segment_values_file not found: {seg_file}")
                self._segment_boundaries = None
                logger.warning(f"segment_values_file not found: {seg_file}, will fallback to per_episode returns")

        self._valid_frame_indices = self._sampler_indices
        self.episode_mapping = self.calc_episode()

        # Compute per-task raw lengths once (MultiRL merge + pinned consistency
        # check both read _task_to_raw_lengths). Resolution of task_to_norm_length
        # itself goes through _resolve_per_task_norm_length so pinned stats from
        # DataConfig are picked up and strict mode gates legacy fallback.
        self.task_to_norm_length: dict[str, int] = {}
        self._task_to_raw_lengths: dict[str, list[int]] = {}
        if self.returns_norm_strategy == "per_task":
            for ep_from, (valid_start, valid_end) in self.episode_mapping.items():
                episode_length = valid_end - valid_start
                meta_key = self._episode_from_to_meta_key[ep_from]
                task = get_episode_task(self.meta, meta_key)
                self._task_to_raw_lengths.setdefault(task, []).append(episode_length)

            self.task_to_norm_length = _resolve_per_task_norm_length(
                self.value_net_cfg,
                task_to_raw_lengths=self._task_to_raw_lengths,
                returns_norm_percentile=self.returns_norm_percentile,
            )
            logger.info(
                f"Per-task norm lengths (p{self.returns_norm_percentile * 100:.0f}): {self.task_to_norm_length}"
            )

        # Always compute returns here using classification tensors from preprocessor
        self._precomputed_returns = self._precompute_returns()

    def __len__(self) -> int:
        return len(self._valid_frame_indices)

    def get_data_file_path(self, ep_index: int, refactor_path: str | None = None) -> Path:
        ep_chunk = self.meta.get_episode_chunk(ep_index)
        data_path = self.meta.info["data_path"]
        if refactor_path is not None and os.path.exists(refactor_path):
            data_path = data_path.replace("data/", "data_refractor/")
        fpath = data_path.format(episode_chunk=ep_chunk, episode_index=ep_index)
        return Path(fpath)

    def calc_episode(self):
        episode_mapping = {}
        # ep_from → real meta.episodes key. Required because exclude_failures drops
        # entries here, so enumerate(episode_mapping) no longer lines up with
        # self.meta.episodes keys — and real meta keys may be non-contiguous.
        self._episode_from_to_meta_key: dict[int, Any] = {}
        # Positionally-aligned snapshot of meta.episodes keys (matches the i-th
        # entry of episode_data_index["from"]). None when meta is a list or missing.
        meta_episode_keys: list | None = None
        if self.meta is not None and hasattr(self.meta, "episodes"):
            meta_eps = self.meta.episodes
            if isinstance(meta_eps, dict):
                meta_episode_keys = list(meta_eps.keys())
        sorted_indices = self._valid_frame_indices
        skipped = 0
        for i in range(len(self.episode_data_index["from"])):
            ep_from = self.episode_data_index["from"][i].item()
            ep_to = self.episode_data_index["to"][i].item()
            left = torch.searchsorted(sorted_indices, ep_from).item()
            right = torch.searchsorted(sorted_indices, ep_to).item() - 1
            if left > right:
                # Episode has no valid frames (e.g. excluded by exclude_failures)
                skipped += 1
                continue
            episode_mapping[ep_from] = (sorted_indices[left].item(), sorted_indices[right].item())
            if meta_episode_keys is not None and i < len(meta_episode_keys):
                self._episode_from_to_meta_key[ep_from] = meta_episode_keys[i]
            else:
                self._episode_from_to_meta_key[ep_from] = i
        if skipped:
            logger.info(f"Skipped {skipped} episodes with no valid frames (excluded by preprocessor)")
        logger.info(f"episode_from_to: {episode_mapping}")
        return episode_mapping

    def _get_episode_negative(self, frame_idx: int) -> bool:
        """Check if a frame belongs to a negative episode."""
        if self.is_negative_episode_tensor is None:
            return False  # no preprocessor → all positive
        return bool(self.is_negative_episode_tensor[frame_idx])

    def _get_episode_boundary(self, frame_idx: int) -> int:
        """Get episode boundary type for a frame."""
        if self.episode_boundary_tensor is None:
            return EpisodeBoundary.BOTH_CONFIRMED  # no preprocessor → both confirmed
        return int(self.episode_boundary_tensor[frame_idx])

    def _precompute_returns(self) -> torch.Tensor:
        """Compute returns using classification tensors and compute_episode_returns pure function."""
        n = len(self._valid_frame_indices)
        returns = torch.zeros(n, dtype=torch.float32)

        for ep_i, (ep_from, (valid_start, valid_end)) in enumerate(self.episode_mapping.items()):
            left = torch.searchsorted(self._valid_frame_indices, valid_start).item()
            right = torch.searchsorted(self._valid_frame_indices, valid_end, right=True).item()

            total_steps = valid_end - valid_start

            if self.returns_norm_strategy == "per_episode":
                norm_length = total_steps
            elif self.returns_norm_strategy == "per_task":
                meta_key = self._episode_from_to_meta_key[ep_from]
                task = get_episode_task(self.meta, meta_key)
                assert task in self.task_to_norm_length, (
                    f"per_task lookup miss for task={task!r}, known keys={list(self.task_to_norm_length)}. "
                    f"Pinned stats must cover every task surfaced by the dataset — re-run compute_rl_norm_stats."
                )
                task_norm = self.task_to_norm_length[task]
                norm_length = max(task_norm, total_steps)
            elif self.returns_norm_strategy == "segmented":
                # Segmented value computation
                seg_frame_indices = self._valid_frame_indices[left:right]
                boundaries = self._segment_boundaries.get(ep_i) if self._segment_boundaries else None
                if boundaries is None:
                    # Fallback to per_episode linear returns when segment info is missing
                    offsets = (seg_frame_indices - valid_start).float()
                    returns[left:right] = torch.clamp(offsets / total_steps - 1.0, min=-1.0, max=0.0)
                    continue

                seg_values = self.segment_values
                total_value = sum(seg_values)
                cum_values = [0.0] + [sum(seg_values[: i + 1]) / total_value for i in range(len(seg_values))]
                cuts = [0, *boundaries, total_steps]

                offsets = (seg_frame_indices - valid_start).float()
                ep_returns = torch.zeros(right - left, dtype=torch.float32)
                for k in range(len(cuts) - 1):
                    seg_start, seg_end = cuts[k], cuts[k + 1]
                    seg_len = seg_end - seg_start
                    if seg_len <= 0:
                        continue
                    if k == len(cuts) - 2:  # last segment: include endpoint
                        mask = (offsets >= seg_start) & (offsets <= seg_end)
                    else:
                        mask = (offsets >= seg_start) & (offsets < seg_end)
                    progress = (offsets[mask] - seg_start) / seg_len
                    ep_returns[mask] = cum_values[k] + (cum_values[k + 1] - cum_values[k]) * progress

                returns[left:right] = torch.clamp(ep_returns - 1.0, min=-1.0, max=0.0)
                continue
            else:  # fixed
                norm_length = max(self.returns_norm_length, total_steps)

            frame_indices = self._valid_frame_indices[left:right]
            is_negative = self._get_episode_negative(valid_start)
            ep_boundary = self._get_episode_boundary(valid_start)

            returns[left:right] = compute_episode_returns(
                frame_indices,
                valid_start,
                valid_end,
                is_negative=is_negative,
                episode_boundary=ep_boundary,
                norm_length=norm_length,
            )

        # pred_value_tensor override (set once at init, immutable during training)
        if self.pred_value_tensor is not None:
            returns = torch.clamp(self.pred_value_tensor[:n] - self.failure_decrease_threshold, min=-1.0)

        return returns

    def __getitem__(self, idx: int) -> dict[str, Any]:
        cur = super().__getitem__(idx)
        cur["returns"] = self._precomputed_returns[idx].clone()

        # Preprocessor-determined prompt overrides episode/default prompt
        if self.episode_prompt_map is not None:
            ep_idx = int(cur["episode_index"])
            cur["task"] = self.episode_prompt_map[ep_idx]

        # Cross-negative: random prompt flip + GT inversion for prompt conditioning.
        # getattr guards against DataLoader worker stale-pickle mismatches when
        # code is updated mid-run (workers re-import the module from disk).
        cross_rate = getattr(self, "cross_negative_rate", 0.0)
        if cross_rate > 0 and random.random() < cross_rate:
            cur["returns"] = -(1.0 + cur["returns"])
            frame_idx = self._valid_frame_indices[idx]
            is_neg = self._get_episode_negative(frame_idx)
            pos_prompt = getattr(self, "positive_prompt", None)
            neg_prompt = getattr(self, "negative_prompt", None)
            if is_neg:
                cur["task"] = pos_prompt or cur["task"]
            else:
                cur["task"] = neg_prompt or cur["task"]

        return cur


class MultiRLAnyverseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_config: _config.DataConfig,
        action_horizon: int,
        image_transforms: Callable | None = None,
        download_videos: bool = True,  # noqa: FBT001, FBT002
        video_backend: str | None = None,
    ):
        super().__init__()

        self.root = Path(data_config.root_dir)
        self.repo_ids = data_config.repo_id if isinstance(data_config.repo_id, list) else [data_config.repo_id]

        self.robot_align_info = data_config.robot_align_info
        self.align_dim = data_config.align_dim
        self.unify_action_space = data_config.unify_action_space

        frame_attributes_preprocessors = getattr(data_config, "frame_attributes_preprocessors", None)

        # Handle different data config types - some may not have action_sequence_keys
        # Gr00tLerobotDataConfig stores these in base_config
        if hasattr(data_config, "base_config") and data_config.base_config is not None:
            base = data_config.base_config
            action_keys = getattr(base, "action_sequence_keys", ["action"])
            value_net_cfg = getattr(base, "value_net_cfg", None)
        else:
            action_keys = getattr(data_config, "action_sequence_keys", ["action"])
            value_net_cfg = getattr(data_config, "value_net_cfg", None)

        norm_strategy = value_net_cfg.get("returns_norm_strategy", "fixed") if value_net_cfg else "fixed"
        norm_percentile = value_net_cfg.get("returns_norm_percentile", 1.0) if value_net_cfg else 1.0

        # Handle Mock objects in tests - ensure action_keys is iterable
        if not isinstance(action_keys, list | tuple):
            action_keys = ["action"]

        self.delta_indices = {key: list(range(action_horizon)) for key in action_keys}

        episodes = data_config.episode
        if data_config.episode_fail and data_config.dataset_length:
            episodes = {}
            for repo_id, episode_f, data_len in zip(
                self.repo_ids, data_config.episode_fail, data_config.dataset_length, strict=False
            ):
                episodes[repo_id] = [i for i in range(data_len) if i not in episode_f]

        self._datasets = [
            LeRobotRLDataset(
                repo_id,
                root=self.root / repo_id,
                delta_indices=self.delta_indices,
                episodes=episodes[repo_id] if episodes else None,
                image_transforms=image_transforms,
                download_videos=download_videos,
                video_backend=video_backend,
                robot_align_info=self.robot_align_info,
                align_dim=self.align_dim,
                unify_action_space=self.unify_action_space,
                value_net_cfg=value_net_cfg,  # 已包含所有必需配置
                frame_attributes_preprocessors=frame_attributes_preprocessors,
            )
            for repo_id in self.repo_ids
        ]

        self.cum_sizes = []
        total = 0
        for ds in self._datasets:
            total += len(ds)
            self.cum_sizes.append(total)

        # Prealloc dataset_index tensors to avoid per-sample allocation
        self._dataset_index_tensors = [torch.tensor([i], dtype=torch.int64) for i in range(len(self._datasets))]

        # For per_task strategy: prefer the pinned dict from DataConfig (written
        # by _load_rl_norm_stats at config build time), otherwise fall back to
        # the legacy merge of sub-dataset raw lengths. Pinned short-circuit keeps
        # train/val identical because both splits read the same precomputed file.
        if norm_strategy == "per_task":
            merged_task_to_norm = _maybe_short_circuit_pinned(value_net_cfg=value_net_cfg, datasets=self._datasets)
            if merged_task_to_norm is None:
                merged_task_to_lengths: dict[str, list[int]] = {}
                for ds in self._datasets:
                    raw_lengths = getattr(ds, "_task_to_raw_lengths", None)
                    if raw_lengths and isinstance(raw_lengths, dict):
                        for task, lengths in raw_lengths.items():
                            merged_task_to_lengths.setdefault(task, []).extend(lengths)
                    else:
                        sub_meta_keys: list | None = None
                        sub_eps = getattr(ds.meta, "episodes", None) if hasattr(ds, "meta") else None
                        if isinstance(sub_eps, dict):
                            sub_meta_keys = list(sub_eps.keys())
                        for i, (_start_idx, (valid_start, valid_end)) in enumerate(ds.episode_mapping.items()):
                            episode_length = valid_end - valid_start
                            meta_key = sub_meta_keys[i] if sub_meta_keys is not None and i < len(sub_meta_keys) else i
                            task = get_episode_task(ds.meta, meta_key)
                            merged_task_to_lengths.setdefault(task, []).append(episode_length)

                merged_task_to_norm = {}
                for task, lengths in merged_task_to_lengths.items():
                    if norm_percentile >= 1.0:
                        merged_task_to_norm[task] = max(lengths)
                    else:
                        merged_task_to_norm[task] = int(np.percentile(lengths, norm_percentile * 100))

            # Distribute merged norm lengths and recompute precomputed returns
            for ds in self._datasets:
                ds.task_to_norm_length = merged_task_to_norm
                if hasattr(ds, "_precompute_returns") and callable(ds._precompute_returns):  # noqa: SLF001
                    ds._precomputed_returns = ds._precompute_returns()  # noqa: SLF001

            self.task_to_norm_length = merged_task_to_norm
            logger.info(f"Multi-dataset per-task norm lengths (p{norm_percentile * 100:.0f}): {merged_task_to_norm}")

    def __len__(self):
        return self.cum_sizes[-1] if self.cum_sizes else 0

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self)}.")

        ds_idx = bisect.bisect_right(self.cum_sizes, idx)
        sub_idx = idx - (self.cum_sizes[ds_idx - 1] if ds_idx > 0 else 0)
        try:
            sample = self._datasets[ds_idx][sub_idx]
        except Exception as e:
            raise RuntimeError(
                f"Failed to get sample {sub_idx} from sub-dataset {ds_idx} "
                f"(repo_id={self.repo_ids[ds_idx] if hasattr(self, 'repo_ids') else 'unknown'}): {e}"
            ) from e

        if sample is None:
            raise ValueError(
                f"Sub-dataset {ds_idx} returned None for index {sub_idx}. "
                "This indicates a bug in the underlying dataset implementation."
            )

        if not isinstance(sample, dict):
            raise TypeError(f"Sub-dataset {ds_idx} returned non-dict type {type(sample)} for index {sub_idx}.")

        sample["dataset_index"] = self._dataset_index_tensors[ds_idx]
        return sample

    @property
    def num_frames(self) -> int:
        """Number of samples/frames."""
        return sum(d.num_frames for d in self._datasets)

    @property
    def num_episodes(self) -> int:
        """Number of episodes."""
        return sum(d.num_episodes for d in self._datasets)

    @property
    def tolerance_s(self) -> float:
        """Tolerance in seconds used to discard loaded frames when their timestamps
        are not close enough from the requested frames. It is only used when `delta_timestamps`
        is provided or when loading video frames from mp4 files.
        """
        # 1e-4 to account for possible numerical error
        return 1 / self.fps - 1e-4


__all__ = ["LeRobotRLDataset"]
