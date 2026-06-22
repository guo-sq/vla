"""Preprocessor that classifies episodes for value model training.

Writes ``is_negative_episode``, ``episode_boundary``, and ``episode_prompt_map``
based on per-episode role/success metadata or a YAML config. Classification
modes:

- ``"episode"``: uses per-episode metadata (role/success) from LeRobotDatasetMeta.
- ``"dataset"``: uses a YAML config with fnmatch patterns against repo_id.
- ``"auto"``: picks ``"episode"`` if meta has role metadata, else ``"dataset"``.

Does NOT compute returns or ``task_to_norm_length``. The [-1, 0] returns are
produced by ``compute_episode_returns()`` in ``rl_dataset.py`` using these
classification outputs plus the ``task_to_norm_length`` maintained by
``LeRobotRLDataset``; the preprocessor is deliberately single-purpose to avoid
two independent copies of per-task normalisation state drifting apart.
"""

from __future__ import annotations

import dataclasses
from enum import IntEnum
import fnmatch
import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from openpi.training.frame_attributes_preprocessors.base import EXTRA_EPISODE_PROMPT_MAP
from openpi.training.frame_attributes_preprocessors.base import DatasetContext
from openpi.training.frame_attributes_preprocessors.base import EpisodeBoundary
from openpi.training.frame_attributes_preprocessors.base import FrameAttributeProcessor
from openpi.training.frame_attributes_preprocessors.base import FrameAttributes

logger = logging.getLogger(__name__)


class EpisodeClass(IntEnum):
    """Episode-level classification.

    POSITIVE / NEGATIVE align with dataset_types_*.yaml keys
    (positive_datasets, negative_datasets). FAILURE_FP / FAILURE_FN are
    episode-mode only (success=False); dataset-mode cannot produce failures.

    - FAILURE_FP: builder-failure ("false positive"). Role is NOT in
      negative_roles → GT heuristic: constant -1 (episode ends far from
      the success terminal state).
    - FAILURE_FN: destroyer-failure ("false negative"). Role IS in
      negative_roles → GT heuristic: constant 0 (episode ends AT target
      because the destroyer action failed to move it).

    Both heuristics are approximations — see PR description C1/C5.
    """

    POSITIVE = 0
    NEGATIVE = 1
    FAILURE_FP = 2
    FAILURE_FN = 3


# Maps the string returned by `_match_dataset_type()` (derived from yaml keys
# `positive_datasets` / `negative_datasets`) to the enum member.
_DATASET_TYPE_TO_CLASS: dict[str, EpisodeClass] = {
    "positive": EpisodeClass.POSITIVE,
    "negative": EpisodeClass.NEGATIVE,
}


@dataclasses.dataclass
class ValueReturnsPreprocessor(FrameAttributeProcessor):
    """Classify episodes and emit is_negative_episode, episode_boundary, episode_prompt_map.

    Per-task normalisation lengths are owned by ``LeRobotRLDataset`` — this
    class is classification-only and must not duplicate that state.
    """

    # --- classification ---
    classification_mode: str = "episode"  # "auto" | "episode" | "dataset"
    dataset_type_config_path: str | None = None  # required when mode == "dataset"

    # --- state confirmation ---
    state_confirmation: str = "auto"  # "auto" | "both" | "end_only"
    state_confirmation_by_role: dict[str, str] | None = None  # per-role override

    # --- classification (role → positive/negative) ---
    negative_roles: list[str] = dataclasses.field(default_factory=lambda: ["destroyer"])

    # --- prompts ---
    positive_prompt: str = "Hang the seatbelt with right hand under 20 seconds."
    negative_prompt: str = "Take the seatbelt off under 20 seconds."

    # --- filtering ---
    exclude_failures: bool = False  # If True, set valid_mask=False for failure episodes

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def __call__(self, ctx: DatasetContext, attrs: FrameAttributes) -> None:
        n = len(ctx.hf_dataset)
        num_episodes = len(ctx.episode_data_index["from"])

        # 1. Resolve effective classification mode (for auto) -------------------
        effective_cls_mode = self._resolve_classification_mode(ctx)

        # 2. Classify episodes ------------------------------------------------
        ep_classes = self._classify_episodes(ctx, num_episodes)

        # 3. Resolve effective state_confirmation per episode ------------------
        effective_sc = self._resolve_state_confirmation(effective_cls_mode)

        # 4. Compute is_negative_episode, episode_boundary, prompt_map --------
        is_neg_arr = np.zeros(n, dtype=bool)
        episode_boundary_arr = np.zeros(n, dtype=np.int8)

        episode_prompt_map: dict[int, str] = {}

        # Get actual episode indices from metadata (for correct prompt_map keys)
        actual_ep_indices = self._get_actual_episode_indices(ctx, num_episodes)

        for i in range(num_episodes):
            ep_from = int(ctx.episode_data_index["from"][i])
            ep_to = int(ctx.episode_data_index["to"][i])
            cls = ep_classes[i]

            is_fp = cls == EpisodeClass.FAILURE_FP
            is_fn = cls == EpisodeClass.FAILURE_FN
            is_fail = is_fp or is_fn
            # FN episodes must carry is_negative=True so the GT=0 branch fires
            # (and the C4 assert in compute_episode_returns stays consistent).
            is_neg = cls == EpisodeClass.NEGATIVE or is_fn

            # --- episode_boundary ---
            if is_fp:
                eb = EpisodeBoundary.UNCONFIRMED_NEGATIVE_END
            elif is_fn:
                eb = EpisodeBoundary.UNCONFIRMED_POSITIVE_END
            elif self.state_confirmation_by_role is not None:
                role = (
                    ctx.meta.episodes.get(actual_ep_indices[i], {}).get("role", "builder")
                    if ctx.meta and hasattr(ctx.meta, "episodes")
                    else "builder"
                )
                sc = self.state_confirmation_by_role.get(role, effective_sc)
                if sc not in ("both", "end_only"):
                    raise ValueError(
                        f"Invalid state_confirmation {sc!r} for role={role!r}; expected 'both' or 'end_only'. "
                        f"state_confirmation_by_role={self.state_confirmation_by_role}, "
                        f"fallback effective_sc={effective_sc!r}"
                    )
                eb = EpisodeBoundary.BOTH_CONFIRMED if sc == "both" else EpisodeBoundary.END_CONFIRMED
            elif effective_sc == "both":
                eb = EpisodeBoundary.BOTH_CONFIRMED
            elif effective_sc == "auto_per_role":
                role = (
                    ctx.meta.episodes.get(actual_ep_indices[i], {}).get("role", "builder")
                    if ctx.meta and hasattr(ctx.meta, "episodes")
                    else "builder"
                )
                eb = EpisodeBoundary.BOTH_CONFIRMED if role == "builder" else EpisodeBoundary.END_CONFIRMED
            else:
                eb = EpisodeBoundary.END_CONFIRMED

            episode_boundary_arr[ep_from:ep_to] = eb
            is_neg_arr[ep_from:ep_to] = is_neg

            # --- exclude failures via valid_mask ---
            if is_fail and self.exclude_failures:
                if attrs.valid_mask is None:
                    attrs.valid_mask = np.ones(n, dtype=bool)
                attrs.valid_mask[ep_from:ep_to] = False

            # --- prompt map (use actual episode_index, not enumerate index) ---
            actual_idx = actual_ep_indices[i] if i < len(actual_ep_indices) else i
            if is_neg:
                episode_prompt_map[actual_idx] = self.negative_prompt
            else:
                episode_prompt_map[actual_idx] = self.positive_prompt

        # 5. Write results (NO returns — computed by rl_dataset) ---------------
        attrs.is_negative_episode = is_neg_arr
        attrs.episode_boundary = episode_boundary_arr
        ctx.extras[EXTRA_EPISODE_PROMPT_MAP] = episode_prompt_map
        # Expose prompts for cross-negative flip in downstream dataset
        ctx.extras["positive_prompt"] = self.positive_prompt
        ctx.extras["negative_prompt"] = self.negative_prompt

        # Logging summary
        n_pos = sum(1 for c in ep_classes if c == EpisodeClass.POSITIVE)
        n_neg = sum(1 for c in ep_classes if c == EpisodeClass.NEGATIVE)
        n_fp = sum(1 for c in ep_classes if c == EpisodeClass.FAILURE_FP)
        n_fn = sum(1 for c in ep_classes if c == EpisodeClass.FAILURE_FN)
        n_both = int((episode_boundary_arr == EpisodeBoundary.BOTH_CONFIRMED).sum() > 0)
        n_end = int((episode_boundary_arr == EpisodeBoundary.END_CONFIRMED).sum() > 0)
        logger.info(
            "[%s] ValueReturns: %d episodes (pos=%d, neg=%d, fp=%d, fn=%d), " "sc=%s, boundary_types=[BOTH=%s, END=%s]",
            ctx.repo_id,
            num_episodes,
            n_pos,
            n_neg,
            n_fp,
            n_fn,
            effective_sc,
            "yes" if n_both else "no",
            "yes" if n_end else "no",
        )

    # ------------------------------------------------------------------
    # Classification helpers
    # ------------------------------------------------------------------

    def _classify_episodes(self, ctx: DatasetContext, num_episodes: int) -> list[EpisodeClass]:
        effective_mode = self._resolve_classification_mode(ctx)
        if effective_mode == "episode":
            return self._classify_episodes_by_role(ctx, num_episodes)
        if effective_mode == "dataset":
            return self._classify_episodes_by_dataset(ctx, num_episodes)
        raise ValueError(f"Unknown classification_mode: {self.classification_mode!r}")

    def _resolve_classification_mode(self, ctx: DatasetContext) -> str:
        """Resolve effective classification mode, auto-detecting when mode is 'auto'.

        Auto-detection checks whether episodes have 'role' metadata:
        - If any episode has 'role' -> episode mode (self-play data)
        - Otherwise -> dataset mode (scripted hang/take_off data)
        """
        if self.classification_mode != "auto":
            return self.classification_mode

        if ctx.meta and hasattr(ctx.meta, "episodes"):
            first_ep = next(iter(ctx.meta.episodes.values()), {})
            if "role" in first_ep:
                logger.info(
                    "[%s] auto classification_mode -> 'episode' (role metadata found)",
                    ctx.repo_id,
                )
                return "episode"

        logger.info(
            "[%s] auto classification_mode -> 'dataset' (no role metadata)",
            ctx.repo_id,
        )
        return "dataset"

    def _classify_episodes_by_role(
        self,
        ctx: DatasetContext,
        num_episodes: int,
    ) -> list[EpisodeClass]:
        """Classify using per-episode metadata (role / success)."""
        actual_indices = self._get_actual_episode_indices(ctx, num_episodes)
        classes: list[EpisodeClass] = []
        for ep_idx in actual_indices:
            ep_meta = ctx.meta.episodes.get(ep_idx, {}) if ctx.meta and hasattr(ctx.meta, "episodes") else {}
            role = ep_meta.get("role", "")
            success = ep_meta.get("success", True)

            if success is False:
                if role in self.negative_roles:
                    classes.append(EpisodeClass.FAILURE_FN)
                else:
                    classes.append(EpisodeClass.FAILURE_FP)
            elif role in self.negative_roles:
                classes.append(EpisodeClass.NEGATIVE)
            else:
                classes.append(EpisodeClass.POSITIVE)
        return classes

    def _classify_episodes_by_dataset(
        self,
        ctx: DatasetContext,
        num_episodes: int,
    ) -> list[EpisodeClass]:
        """Classify all episodes of this dataset by repo_id matching a YAML config."""
        if self.dataset_type_config_path is None:
            raise ValueError("dataset_type_config_path is required when classification_mode='dataset'")
        config = _load_dataset_type_config(self.dataset_type_config_path)
        dtype = _match_dataset_type(ctx.repo_id, config)
        cls = _DATASET_TYPE_TO_CLASS.get(dtype, EpisodeClass.POSITIVE)
        return [cls] * num_episodes

    # ------------------------------------------------------------------
    # State confirmation
    # ------------------------------------------------------------------

    def _resolve_state_confirmation(self, effective_classification_mode: str) -> str:
        if self.state_confirmation != "auto":
            return self.state_confirmation
        if effective_classification_mode == "episode":
            if self.state_confirmation_by_role is None:
                return "auto_per_role"  # per-role: builder→BOTH, destroyer→END
            return "both"  # by_role handles per-role; fallback is "both"
        return "both"  # dataset mode → all BOTH_CONFIRMED

    # ------------------------------------------------------------------
    # Episode index extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _get_actual_episode_indices(ctx: DatasetContext, num_episodes: int) -> list[int]:
        """Get actual episode indices from metadata, falling back to range(num_episodes)."""
        if ctx.meta and hasattr(ctx.meta, "episodes"):
            indices = list(ctx.meta.episodes.keys())
            if len(indices) == num_episodes:
                return indices
        return list(range(num_episodes))


# ---------------------------------------------------------------------------
# YAML config loading (for dataset classification mode)
# ---------------------------------------------------------------------------


def _load_dataset_type_config(path: str) -> dict[str, Any]:
    """Load and validate a dataset type YAML config.

    Expected format:
        positive_datasets:
          - "pattern1*"
          - "exact_name"
        negative_datasets:
          - "neg_pattern*"
        default_type: "positive"
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset type config not found: {path}")
    with config_path.open("r") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Dataset type config must be a YAML mapping, got {type(config)}")
    return config


def _match_dataset_type(repo_id: str, config: dict[str, Any]) -> str:
    """Match repo_id against positive/negative patterns, return type string."""
    for pattern in config.get("positive_datasets", []):
        if fnmatch.fnmatch(repo_id, pattern):
            return "positive"
    for pattern in config.get("negative_datasets", []):
        if fnmatch.fnmatch(repo_id, pattern):
            return "negative"
    return config.get("default_type", "positive")
