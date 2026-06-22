#!/usr/bin/env python3
"""Prepare minitest dataset for CI openloop evaluation.

This script copies a minimal set of episodes from OSS to CPFS to enable
fast openloop evaluation (target: <10 minutes, 10 robot types).

Usage:
    python scripts/prepare_minitest_dataset.py
    python scripts/prepare_minitest_dataset.py --dry-run
    python scripts/prepare_minitest_dataset.py --cpfs-root /mnt/cpfs/openpi_minitest

The script will:
1. Copy 1 episode per robot type from OSS to CPFS
2. Maintain the same directory structure for compatibility
3. Skip already copied data (idempotent)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shutil
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Source (OSS) and destination (CPFS) mapping
# Each entry: (robot_type, oss_path, cpfs_relative_path, max_episodes)
MINITEST_DATASET_SPECS: list[dict[str, Any]] = [
    # 1. Agilex (from robomind_2)
    {
        "robot_type": "agilex",
        "oss_path": "/mnt/oss_data/X-Humanoid/RoboMIND2.0-Agilex_lerobot/agilex/fold_clothes/success_episodes",
        "cpfs_path": "robomind_2/agilex/fold_clothes/success_episodes",
        "max_episodes": 3,
    },
    # 2. Tianyi (from robomind_2)
    {
        "robot_type": "tianyi",
        "oss_path": "/mnt/oss_data/X-Humanoid/RoboMIND2.0-Tianyi_lerobot/tienyi/close_drawer_under_combined_cabinet/success_episodes",
        "cpfs_path": "robomind_2/tianyi/close_drawer_under_combined_cabinet/success_episodes",
        "max_episodes": 3,
    },
    # 3. Galaxea (from OpenGalaxea)
    {
        "robot_type": "galaxea",
        "oss_path": "/mnt/oss_data/OpenGalaxea/Galaxea-Open-World-Dataset/lerobot_unzip/Adjust_The_Air_Conditioner_Temperature_20250711_006",
        "cpfs_path": "galaxea/Adjust_The_Air_Conditioner_Temperature_20250711_006",
        "max_episodes": 3,
    },
    # 4. ALOHA static (from aloha)
    {
        "robot_type": "aloha_static",
        "oss_path": "/mnt/oss_data/lerobot/aloha_static_coffee",
        "cpfs_path": "aloha/aloha_static_coffee",
        "max_episodes": 3,
    },
    # 5. Intern A1 (from intern_a1_real)
    {
        "robot_type": "intern_a1",
        "oss_path": "/mnt/oss_data/InternRobotics/InternData-A1/real/genie1/pickup_a_bag_of_bread_into_the_basket/set_0",
        "cpfs_path": "intern_a1/pickup_a_bag_of_bread_into_the_basket/set_0",
        "max_episodes": 3,
    },
    # 6. RDT (from rdt-ft-data)
    {
        "robot_type": "rdt",
        "oss_path": "/mnt/oss_data/robotics-diffusion-transformer/rdt-ft-data/lerobot_data/pick_place_water_bottle",
        "cpfs_path": "rdt/pick_place_water_bottle",
        "max_episodes": 3,
    },
    # 7. AgiBot G1 (from robocoin)
    {
        "robot_type": "agibot_g1",
        "oss_path": "/mnt/oss_data/robocoin/RoboCOIN/AgiBot-g1_box_storage_a",
        "cpfs_path": "robocoin/AgiBot-g1_box_storage_a",
        "max_episodes": 3,
    },
    # 8. Cobot Magic (from robocoin)
    {
        "robot_type": "cobot_magic",
        "oss_path": "/mnt/oss_data/robocoin/RoboCOIN/Cobot_Magic_move_the_cup",
        "cpfs_path": "robocoin/Cobot_Magic_move_the_cup",
        "max_episodes": 3,
    },
    # 9. AIRBOT MMK2 (from robocoin)
    {
        "robot_type": "airbot_mmk2",
        "oss_path": "/mnt/oss_data/robocoin/RoboCOIN/AIRBOT_MMK2_mobile_car",
        "cpfs_path": "robocoin/AIRBOT_MMK2_mobile_car",
        "max_episodes": 3,
    },
    # 10. Galbot G1 (from robocoin)
    {
        "robot_type": "galbot_g1",
        "oss_path": "/mnt/oss_data/robocoin/RoboCOIN/Galbot_g1_steamer_storage_baozi_a",
        "cpfs_path": "robocoin/Galbot_g1_steamer_storage_baozi_a",
        "max_episodes": 3,
    },
]


def get_episode_dirs(dataset_path: Path) -> list[Path]:
    """Get episode directories in a dataset."""
    if not dataset_path.exists():
        return []

    # Common episode directory patterns
    episode_dirs = []
    for pattern in ["episode_*", "ep_*", "*"]:
        matches = sorted(dataset_path.glob(pattern))
        if matches:
            episode_dirs.extend(
                m for m in matches if m.is_dir() and ((m / "meta" / "info.json").exists() or (m / "data").exists())
            )
            if episode_dirs:
                break

    # Fallback: check if the dataset itself has episode structure
    if not episode_dirs and (dataset_path / "meta" / "info.json").exists():
        episode_dirs = [dataset_path]

    return episode_dirs


def copy_episode(src: Path, dst: Path, *, dry_run: bool = False) -> bool:
    """Copy a single episode directory."""
    if dst.exists():
        logger.info(f"  [SKIP] Already exists: {dst}")
        return True

    if dry_run:
        logger.info(f"  [DRY-RUN] Would copy: {src} -> {dst}")
        return True

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
        logger.info(f"  [COPY] {src} -> {dst}")
        return True
    except Exception as e:
        logger.error(f"  [ERROR] Failed to copy {src}: {e}")
        return False


def resolve_episode_destination(base_dst: Path, episode_dir: Path, source_root: Path) -> Path:
    if episode_dir == source_root:
        return base_dst
    return base_dst / episode_dir.name


def prepare_dataset(
    spec: dict[str, Any],
    cpfs_root: Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Prepare a single dataset for minitest."""
    robot_type = spec["robot_type"]
    oss_path = Path(spec["oss_path"])
    cpfs_path = cpfs_root / spec["cpfs_path"]
    max_episodes = spec.get("max_episodes", 3)

    result = {
        "robot_type": robot_type,
        "oss_path": str(oss_path),
        "cpfs_path": str(cpfs_path),
        "status": "pending",
        "episodes_copied": 0,
        "error": None,
    }

    logger.info(f"Processing {robot_type}...")

    if not oss_path.exists():
        result["status"] = "source_not_found"
        result["error"] = f"Source path does not exist: {oss_path}"
        logger.warning(f"  Source not found: {oss_path}")
        return result

    # Get episode directories
    episode_dirs = get_episode_dirs(oss_path)
    if not episode_dirs:
        result["status"] = "no_episodes"
        result["error"] = "No episode directories found"
        logger.warning(f"  No episodes found in {oss_path}")
        return result

    # Copy limited episodes
    episodes_to_copy = episode_dirs[:max_episodes]
    copied = 0

    for episode_dir in episodes_to_copy:
        episode_dst = resolve_episode_destination(cpfs_path, episode_dir, oss_path)
        if copy_episode(episode_dir, episode_dst, dry_run=dry_run):
            copied += 1

    result["status"] = "success" if copied > 0 else "partial"
    result["episodes_copied"] = copied
    result["total_episodes_found"] = len(episode_dirs)

    logger.info(f"  Copied {copied}/{len(episodes_to_copy)} episodes")
    return result


def prepare_all_datasets(
    cpfs_root: Path,
    *,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Prepare all minitest datasets."""
    results = []

    logger.info(f"Preparing minitest datasets to {cpfs_root}")
    logger.info(f"Dry run: {dry_run}")
    logger.info(f"Total datasets to prepare: {len(MINITEST_DATASET_SPECS)}")

    if not dry_run:
        cpfs_root.mkdir(parents=True, exist_ok=True)

    for spec in MINITEST_DATASET_SPECS:
        result = prepare_dataset(spec, cpfs_root, dry_run=dry_run)
        results.append(result)

    # Write manifest
    manifest_path = cpfs_root / "minitest_manifest.json"
    manifest = {
        "version": "1.0",
        "created_at": __import__("datetime").datetime.now().isoformat(),
        "total_robot_types": len(MINITEST_DATASET_SPECS),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "datasets": results,
    }

    if not dry_run:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info(f"Manifest written to {manifest_path}")
    else:
        logger.info(f"[DRY-RUN] Would write manifest to {manifest_path}")
        print(json.dumps(manifest, indent=2))

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare minitest dataset for CI openloop evaluation")
    parser.add_argument(
        "--cpfs-root",
        type=str,
        default="/mnt/workspace/openpi_minitest",
        help="CPFS root directory for minitest data (default: /mnt/cpfs/openpi_minitest)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually copying",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="",
        help="Prepare only a specific robot type (default: all)",
    )
    args = parser.parse_args()

    cpfs_root = Path(args.cpfs_root)

    if args.robot_type:
        specs = [s for s in MINITEST_DATASET_SPECS if s["robot_type"] == args.robot_type]
        if not specs:
            logger.error(f"Unknown robot type: {args.robot_type}")
            logger.info(f"Available types: {[s['robot_type'] for s in MINITEST_DATASET_SPECS]}")
            return 1
        results = [prepare_dataset(specs[0], cpfs_root, dry_run=args.dry_run)]
    else:
        results = prepare_all_datasets(cpfs_root, dry_run=args.dry_run)

    # Summary
    successful = sum(1 for r in results if r["status"] == "success")
    total = len(results)
    logger.info(f"\nSummary: {successful}/{total} datasets prepared successfully")

    if successful < total:
        logger.warning("Some datasets failed to prepare. Check logs above.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
