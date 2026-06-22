"""Seatbelt cleaned self-play data splits (train / val / test).

Temporal split (方案 E):
- Train: 0325-0401 (34 repos, 2011 eps, 66%)
- Val:   0402.batch.1-6 (6 repos, 331 eps, 11%)
- Test:  0402.batch.7-13 + 0403.batch.8-12 (12 repos, 726 eps, 24%)

Excludes 0403.batch.1-7 (330 eps, labels not human-corrected).
Total usable: 52 repos, 3068 eps.
"""

from __future__ import annotations

from pathlib import Path
import re

_PREFIX = "seatbelt.single.self_play_record_cleaned.0205_0312_self_play_recovery."

# Date + batch boundaries
_TRAIN_END_DATE = "20260401"  # inclusive
_VAL_DATE = "20260402"
_VAL_MAX_BATCH = 6  # inclusive
# Test: 0402.batch.7-13 + 0403.batch.8-12

# Uncorrected batches to exclude
_EXCLUDED = {f"{_PREFIX}20260403.batch.{i}" for i in range(1, 8)}


def _parse_repo(name: str) -> tuple[str, int] | None:
    """Extract (date, batch_num) from a cleaned repo name."""
    m = re.search(r"\.(\d{8})\.batch\.(\d+)$", name)
    if m is None:
        return None
    return m.group(1), int(m.group(2))


def classify_repo(repo_name: str) -> str | None:
    """Classify a cleaned seatbelt repo into train/val/test/excluded.

    Returns 'train', 'val', 'test', or None (excluded/unrecognized).
    """
    if repo_name in _EXCLUDED:
        return None
    parsed = _parse_repo(repo_name)
    if parsed is None:
        return None
    date, batch = parsed
    if date <= _TRAIN_END_DATE:
        return "train"
    if date == _VAL_DATE:
        return "val" if batch <= _VAL_MAX_BATCH else "test"
    if date == "20260403":
        return "test"  # batch.8-12 (batch.1-7 already excluded)
    return None


def get_split_repos(
    split: str,
    dataset_root: str | Path = "/mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/seatbelt",
) -> list[str]:
    """Return repo_id list for a given split.

    Args:
        split: One of 'train', 'val', 'test', 'all' (train+val+test).
        dataset_root: Path to the seatbelt data directory.

    Returns:
        Sorted list of repo directory names.
    """
    root = Path(dataset_root)
    repos = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or "cleaned" not in d.name:
            continue
        assigned = classify_repo(d.name)
        if assigned is None:
            continue
        if split in ("all", assigned):
            repos.append(d.name)
    return repos


def get_split_summary(
    dataset_root: str | Path = "/mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/seatbelt",
) -> dict[str, dict]:
    """Return a summary of the split: repos and episode counts per split."""

    root = Path(dataset_root)
    summary: dict[str, dict] = {s: {"repos": [], "episodes": 0} for s in ("train", "val", "test", "excluded")}
    for d in sorted(root.iterdir()):
        if not d.is_dir() or "cleaned" not in d.name:
            continue
        assigned = classify_repo(d.name)
        if assigned is None:
            assigned = "excluded"
        ep_file = d / "meta" / "episodes.jsonl"
        n_eps = sum(1 for _ in open(ep_file)) if ep_file.exists() else 0
        summary[assigned]["repos"].append(d.name)
        summary[assigned]["episodes"] += n_eps
    return summary
