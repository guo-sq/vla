"""Concatenate per-batch ``episode_details.json`` files from a chunked
sparse benchmark run into a single top-level file.

``wait_gpu_then_run.sh`` runs the 306-repo clothes benchmark in chunks of
60 so a single crash only costs the current batch. This script combines
the resulting ``batch_NNN/metrics/episode_details.json`` shards into one
``<output_dir>/metrics/episode_details.json`` so ``fill_head_pred_ranges``
and downstream analysis have a single canonical input.

Batches missing a ``metrics/episode_details.json`` (crashed before write)
are skipped silently - the top-level launcher already logs which batches
failed. Non-batch subdirectories (e.g. ``visualization/``) are ignored.

Usage:

    python scripts/benchmark/merge_batches.py \\
      --output-dir test_results/benchmark/clothes_v0409_sparse/fast_mode_max3600
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

_BATCH_DIR_PATTERN = re.compile(r"^batch_\d+$")


def _sorted_batch_dirs(output_dir: Path) -> list[Path]:
    return sorted(p for p in output_dir.iterdir() if p.is_dir() and _BATCH_DIR_PATTERN.match(p.name))


def merge_batches(output_dir: Path, merged_path: Path) -> int:
    """Concatenate per-batch episode_details.json files into ``merged_path``.

    Args:
        output_dir: Directory containing ``batch_NNN`` subdirectories.
        merged_path: Where to write the merged JSON list.

    Returns:
        Number of episodes in the merged output (post-dedup).
    """
    seen_keys: set[str] = set()
    merged: list[dict] = []
    for batch_dir in _sorted_batch_dirs(output_dir):
        details_path = batch_dir / "metrics" / "episode_details.json"
        if not details_path.exists():
            continue
        entries = json.loads(details_path.read_text())
        for entry in entries:
            key = entry.get("episode_key")
            if key and key in seen_keys:
                continue
            if key:
                seen_keys.add(key)
            merged.append(entry)

    merged_path.parent.mkdir(parents=True, exist_ok=True)
    merged_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    return len(merged)


def merge_skipped_repos(output_dir: Path, merged_path: Path) -> int:
    """Concatenate per-batch skipped_repos.json files into ``merged_path``.

    ``run_benchmark.py`` writes a ``skipped_repos.json`` list at each batch's
    root whenever its per-repo try/except catches a failure (typically corrupt
    LeRobot/torchcodec metadata). This function aggregates them across all
    batches so the failing repos can be triaged or added to the exclusion
    list in one place.

    Args:
        output_dir: Directory containing ``batch_NNN`` subdirectories.
        merged_path: Where to write the merged JSON list of failure entries.

    Returns:
        Number of unique failed repos in the merged output.
    """
    seen_repos: set[str] = set()
    merged: list[dict] = []
    for batch_dir in _sorted_batch_dirs(output_dir):
        skip_path = batch_dir / "skipped_repos.json"
        if not skip_path.exists():
            continue
        entries = json.loads(skip_path.read_text())
        for entry in entries:
            repo_id = entry.get("repo_id")
            if repo_id and repo_id in seen_repos:
                continue
            if repo_id:
                seen_repos.add(repo_id)
            merged.append(entry)

    merged_path.parent.mkdir(parents=True, exist_ok=True)
    merged_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    return len(merged)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Sparse benchmark output root containing batch_NNN subdirectories",
    )
    parser.add_argument(
        "--merged-path",
        type=Path,
        help="Where to write the merged episode_details; defaults to " "<output-dir>/metrics/episode_details.json",
    )
    parser.add_argument(
        "--skipped-path",
        type=Path,
        help="Where to write the merged skipped_repos; defaults to " "<output-dir>/skipped_repos.json",
    )
    args = parser.parse_args()

    merged_path = args.merged_path or (args.output_dir / "metrics" / "episode_details.json")
    skipped_path = args.skipped_path or (args.output_dir / "skipped_repos.json")

    count = merge_batches(args.output_dir, merged_path)
    print(f"[merge_batches] Merged {count} episodes -> {merged_path}")

    skipped_count = merge_skipped_repos(args.output_dir, skipped_path)
    print(f"[merge_batches] Merged {skipped_count} skipped repos -> {skipped_path}")


if __name__ == "__main__":
    main()
