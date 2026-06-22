"""Precompute per-repo RL norm stats (task_to_raw_lengths) for pinned training.

Part 1 of PR #100 follow-up — root-fix for train/val ``task_to_norm_length``
drift under ``per_task`` strategy. Mirrors the ``scripts/compute_norm_stats_fast.py``
offline-precompute pattern: load a TrainConfig → for each repo, assemble
``LeRobotRLDataset`` in non-strict mode so its existing ``_task_to_raw_lengths``
computation runs → write the raw lengths (with a fingerprint) to
``{assets_dirs}/{asset_id}/rl_norm_stats.json``. Both train and val data
loaders then read from the same precomputed file, eliminating drift.

Design choices:
- **Store raw lengths, not percentiles**: ``MultiRLAnyverseDataset`` merge
  semantics (`rl_dataset.py:379-384`) aggregate raw lengths across repos
  *then* compute the percentile; storing per-repo percentiles would break
  that equivalence.
- **Per-repo files, not global**: matches the existing ``_load_norm_stats``
  per-asset_id pattern and lets single-repo edits avoid a full rebuild.
- **Fingerprint**: hashes ``value_net_cfg`` fields that affect the raw
  lengths (``exclude_failures``, ``failure_decrease_threshold``, …) plus the
  ``repo_id``. Loader compares and hard-fails on mismatch so stale files
  surface immediately.
- **DRY**: the only place that iterates episodes to collect task lengths is
  ``LeRobotRLDataset.__init__`` (which already tracks ``_task_to_raw_lengths``
  regardless of the strict gate). This script is a thin wrapper.

Usage:
    python scripts/compute_rl_norm_stats.py --config pi06_seatbelt_value_selfplay_fixed
    python scripts/compute_rl_norm_stats.py --config <name> --force
    python scripts/compute_rl_norm_stats.py --config <name> --repo-id seatbelt/xyz
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import tyro

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RL_NORM_STATS_FILENAME = "rl_norm_stats.json"

#: ``value_net_cfg`` fields that materially affect the per-repo raw lengths.
#: Keep this list in sync with `_load_norm_stats` on the loader side —
#: fingerprint mismatches mean the precomputed file no longer matches the
#: config the user is training with and we must force a re-run.
RL_NORM_STATS_FINGERPRINT_FIELDS: tuple[str, ...] = (
    "returns_norm_strategy",
    "returns_norm_percentile",
    "returns_norm_length",
    "exclude_failures",
    "failure_decrease_threshold",
)


# ---------------------------------------------------------------------------
# Fingerprint / serialization helpers (pure)
# ---------------------------------------------------------------------------


def _compute_rl_norm_stats_fingerprint(*, repo_id: str, value_net_cfg: dict[str, Any]) -> str:
    """Return a deterministic SHA-256 fingerprint of the per-repo config inputs.

    Uses ``sorted(dict.items())`` so insertion order does not affect the
    hash. Only fields in ``RL_NORM_STATS_FINGERPRINT_FIELDS`` are hashed —
    adding new fields to ``value_net_cfg`` that don't affect raw lengths
    (e.g. ``pinned_task_to_norm_length``) must NOT flip the fingerprint.
    """
    relevant = {k: value_net_cfg.get(k) for k in RL_NORM_STATS_FINGERPRINT_FIELDS}
    payload = {"repo_id": repo_id, "value_net_cfg": sorted(relevant.items())}
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_rl_norm_stats_file(
    path: Path,
    *,
    task_to_raw_lengths: dict[str, list[int]],
    fingerprint: str,
) -> None:
    """Atomically write the per-repo raw-length file with a fingerprint header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fingerprint": fingerprint,
        "task_to_raw_lengths": {task: list(lengths) for task, lengths in task_to_raw_lengths.items()},
    }
    fd, tmp_name = tempfile.mkstemp(suffix=".json", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def load_rl_norm_stats_file(path: Path) -> tuple[dict[str, list[int]], str]:
    """Load the per-repo file, returning ``(task_to_raw_lengths, fingerprint)``.

    Raises ``FileNotFoundError`` if the file is missing and ``ValueError``
    if the JSON payload is malformed — the caller is responsible for
    fingerprint comparison.
    """
    with path.open("r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or "fingerprint" not in payload or "task_to_raw_lengths" not in payload:
        raise ValueError(f"Malformed rl_norm_stats file at {path}: missing required keys")
    raw = payload["task_to_raw_lengths"]
    if not isinstance(raw, dict):
        raise ValueError(f"Malformed rl_norm_stats.task_to_raw_lengths at {path}: expected dict, got {type(raw)}")
    task_to_raw_lengths = {task: [int(x) for x in lengths] for task, lengths in raw.items()}
    return task_to_raw_lengths, str(payload["fingerprint"])


# ---------------------------------------------------------------------------
# Build: per-repo raw_lengths via LeRobotRLDataset reuse
# ---------------------------------------------------------------------------


def _build_rl_norm_stats_for_repo(
    config,  # type: _config.TrainConfig (imported lazily to keep script import light)
    repo_id: str,
) -> dict[str, list[int]]:
    """Instantiate ``LeRobotRLDataset`` in non-strict mode and read its raw lengths.

    ``_task_to_raw_lengths`` is populated during ``__init__`` regardless of
    whether ``pinned_task_to_norm_length`` is present, so passing
    ``strict_rl_norm_stats=False`` in the cfg lets us skip the pinning gate
    without running the obs/action data loader at all (we only touch the
    attribute after init returns).
    """
    # Imports are deferred so tests that only exercise the pure helpers
    # (fingerprint, IO) don't pay the JAX / lerobot import cost.
    import openpi.training.config as _config  # noqa: F401
    from openpi.training.rl_dataset import LeRobotRLDataset

    data_cfg = config.data.create(config.assets_dirs, config.model)

    # Tell the dataset to fall back to legacy per-repo computation so the
    # strict gate doesn't fire — we only want `_task_to_raw_lengths`.
    vn_cfg = dict(data_cfg.value_net_cfg or {})
    vn_cfg["strict_rl_norm_stats"] = False
    vn_cfg.pop("pinned_task_to_norm_length", None)

    repo_root = Path(data_cfg.root_dir) / repo_id
    ds = LeRobotRLDataset(
        repo_id=repo_id,
        root=repo_root,
        delta_indices={k: list(range(config.model.action_horizon)) for k in data_cfg.action_sequence_keys},
        download_videos=False,
        robot_align_info=data_cfg.robot_align_info,
        align_dim=data_cfg.align_dim,
        unify_action_space=data_cfg.unify_action_space,
        value_net_cfg=vn_cfg,
        frame_attributes_preprocessors=data_cfg.frame_attributes_preprocessors,
    )
    return dict(ds._task_to_raw_lengths)  # noqa: SLF001


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Args:
    """CLI args for compute_rl_norm_stats."""

    #: TrainConfig name (same convention as scripts/train_rl.py --config).
    config: str
    #: Overwrite existing rl_norm_stats.json (default: skip if fingerprint matches).
    force: bool = False
    #: If set, only (re)compute for this repo_id (must be part of the config).
    repo_id: str | None = None


def _resolve_target_repos(data_cfg, override_repo_id: str | None) -> list[str]:
    """Return the list of repo_ids this script should process."""
    all_repos = data_cfg.repo_id
    if isinstance(all_repos, str):
        all_repos = [all_repos]
    elif not isinstance(all_repos, list):
        raise ValueError(f"data.repo_id must be str or list, got {type(all_repos)}")

    if override_repo_id is None:
        return list(all_repos)
    if override_repo_id not in all_repos:
        raise ValueError(f"--repo-id {override_repo_id!r} is not in the config's data.repo_id list: {all_repos}")
    return [override_repo_id]


def _output_path_for_repo(assets_dirs: Path, repo_id: str) -> Path:
    """Always write to ``{assets_dirs}/{repo_id}/rl_norm_stats.json`` — per-repo.

    obs/action norm_stats.json lives under ``{assets_dirs}/{asset_id}`` (one
    shared file for the whole config), but rl_norm_stats.json is per-repo,
    so we key by repo_id regardless of whether data_cfg.asset_id is a
    single shared string or a list. base_cfg._load_rl_norm_stats reads from
    this same location.
    """
    return Path(assets_dirs) / repo_id / RL_NORM_STATS_FILENAME


def main(args: _Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import openpi.training.config as _config

    config = _config.get_config(args.config)
    data_cfg = config.data.create(config.assets_dirs, config.model)

    value_net_cfg = data_cfg.value_net_cfg or {}
    strategy = value_net_cfg.get("returns_norm_strategy")
    if strategy != "per_task":
        logger.info(
            "returns_norm_strategy=%r; nothing to precompute (only per_task needs pinned stats). Exiting.",
            strategy,
        )
        return

    repos = _resolve_target_repos(data_cfg, args.repo_id)
    assets_dirs = Path(config.assets_dirs)

    logger.info("Computing rl_norm_stats for %d repo(s): %s", len(repos), repos)
    for repo_id in repos:
        out_path = _output_path_for_repo(assets_dirs, repo_id)

        fingerprint = _compute_rl_norm_stats_fingerprint(repo_id=repo_id, value_net_cfg=value_net_cfg)

        if out_path.exists() and not args.force:
            try:
                _, existing_fp = load_rl_norm_stats_file(out_path)
            except (ValueError, json.JSONDecodeError):
                existing_fp = ""
            if existing_fp == fingerprint:
                logger.info("Skipping %s — fingerprint matches (use --force to rebuild)", out_path)
                continue
            logger.info("Fingerprint mismatch at %s — rebuilding", out_path)

        t0 = time.time()
        logger.info("Building raw lengths for repo_id=%s", repo_id)
        task_to_raw_lengths = _build_rl_norm_stats_for_repo(config, repo_id)
        logger.info(
            "Built raw lengths for %s in %.1fs: %d tasks, %d total episodes",
            repo_id,
            time.time() - t0,
            len(task_to_raw_lengths),
            sum(len(v) for v in task_to_raw_lengths.values()),
        )

        write_rl_norm_stats_file(out_path, task_to_raw_lengths=task_to_raw_lengths, fingerprint=fingerprint)
        logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main(tyro.cli(_Args))
