#!/usr/bin/env python3
"""Re-walk the on-disk batches referenced by ``batch_registry.json`` and
overwrite their ``total_size_gb`` field with the current actual size.

``BatchTracker.record_batch_info`` snapshots ``total_size_gb`` once at
record time and the daemon / web dashboard read that snapshot — they
never re-walk the directory. After offline maintenance shrinks the
on-disk content (e.g. ``upload_system/repair_dataset_batches.py``
re-encoding bloated mp4s), the registry is stale and the dashboard
keeps showing the old size. This script reconciles them.

Usage::

    # Update every batch in the registry
    python upload_system/refresh_batch_sizes.py

    # Inspect what would change without rewriting the registry
    python upload_system/refresh_batch_sizes.py --dry-run

    # Only refresh batches under a specific subtree
    python upload_system/refresh_batch_sizes.py --only-prefix \\
        ~/lerobot_data_collection/20260428/pack_socks

    # Point at a non-default config
    python upload_system/refresh_batch_sizes.py --config path/to/upload_config.yaml

Safe to run while the upload daemon is running — uses the same
``_atomic_update`` lock the daemon uses for writes.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

# Make ``lerobot.common.data_tracker`` importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lerobot.common.data_tracker import BatchTracker  # noqa: E402

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("refresh-sizes")


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"upload config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    data_section = cfg.get("data") or {}
    raw_root = data_section.get("root")
    if not raw_root:
        raise ValueError(f"data.root missing from {path}")
    # Match the daemon's expansion semantics so we hit the same registry it does.
    data_section["root"] = os.path.expandvars(os.path.expanduser(str(raw_root)))
    cfg["data"] = data_section
    return cfg


def refresh(
    tracker: BatchTracker,
    only_prefix: Path | None,
    dry_run: bool,
) -> tuple[int, int, float]:
    """Return ``(updated, skipped_missing, total_gb_delta)``."""
    updated = 0
    skipped_missing = 0
    delta_gb = 0.0
    only_prefix_resolved = only_prefix.resolve() if only_prefix else None

    def _is_under(child: Path, ancestor: Path) -> bool:
        try:
            child.relative_to(ancestor)
            return True
        except ValueError:
            return False

    def _walk_and_update(data: dict) -> dict:
        nonlocal updated, skipped_missing, delta_gb

        def walk(node: dict) -> None:
            nonlocal updated, skipped_missing, delta_gb
            for key, val in node.items():
                if key == "_metadata":
                    continue
                if not isinstance(val, dict):
                    continue
                if "status" in val:
                    # Leaf: a single batch entry.
                    local_path_str = val.get("local_path") or ""
                    local_path = Path(local_path_str)
                    if only_prefix_resolved is not None and not _is_under(
                        local_path.resolve() if local_path.is_absolute() else local_path,
                        only_prefix_resolved,
                    ):
                        continue
                    if not local_path.exists():
                        skipped_missing += 1
                        log.warning(
                            "  %s: local_path missing (%s) — leaving entry alone",
                            val.get("batch_id", "?"), local_path,
                        )
                        continue
                    new_gb = round(tracker._get_directory_size(local_path), 2)
                    old_gb = float(val.get("total_size_gb", 0.0))
                    if abs(new_gb - old_gb) < 0.005:
                        continue
                    log.info(
                        "  %s: %.2f GB → %.2f GB  (Δ %+.2f GB)",
                        val.get("batch_id", "?"), old_gb, new_gb, new_gb - old_gb,
                    )
                    delta_gb += new_gb - old_gb
                    updated += 1
                    if not dry_run:
                        val["total_size_gb"] = new_gb
                else:
                    walk(val)

        walk(data)
        return data

    if dry_run:
        # Read-only: don't touch the registry, just walk a copy in memory.
        snapshot = tracker._read_json()
        _walk_and_update(snapshot)
    else:
        tracker._atomic_update(_walk_and_update)

    return updated, skipped_missing, delta_gb


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=Path("upload_system/upload_config.yaml"),
        help="path to upload_config.yaml (default: upload_system/upload_config.yaml)",
    )
    p.add_argument("--dry-run", action="store_true", help="report only, change nothing")
    p.add_argument(
        "--only-prefix",
        type=Path,
        default=None,
        help="only refresh batches whose local_path is under this directory",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    data_root = cfg["data"]["root"]
    registry_filename = cfg["data"].get("registry_filename", "batch_registry.json")
    log.info("data_root: %s", data_root)
    log.info("registry : %s/%s", data_root, registry_filename)
    if args.only_prefix:
        log.info("filter  : only batches under %s", args.only_prefix)

    tracker = BatchTracker(data_root, registry_filename)
    if not tracker.registry_path.exists():
        log.error("registry file does not exist: %s", tracker.registry_path)
        return 1

    updated, missing, delta_gb = refresh(tracker, args.only_prefix, args.dry_run)
    log.info("=" * 60)
    log.info("Summary:")
    log.info("  batches updated     : %d", updated)
    log.info("  batches skipped     : %d (local_path missing)", missing)
    log.info("  net size change     : %+.2f GB", delta_gb)
    if args.dry_run:
        log.info("  (dry-run — registry not modified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
