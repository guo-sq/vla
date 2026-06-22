"""Generate segment_values.json for each dataset by detecting the segment boundary
from observation.state dim7 (the 8th dimension, 0-indexed as 7).

Detection logic:
  dim7 exhibits a pattern of: min → first ascent → plateau → second ascent → peak → descent.
  The boundary we want is the peak of the second ascent.

  Step 1: In frame range [700, 1300], find the minimum value of dim7.
          The minimum should be < -0.6 (typically around -0.6 to -0.8).

  Step 2: From the minimum, scan forward to find the first plateau using a sliding
          window (20 frames, value range < 0.01). The plateau value is typically
          in [-0.6, -0.2], representing the end of the first ascent.

  Step 3: From the first plateau, search within the next 300 frames for the global
          maximum. This is the peak of the second ascent (typically around -0.2 to 0),
          and is recorded as the segment boundary.

Output: meta/segment_values.json per dataset, e.g.:
  {"boundaries": {"0": [1193], "1": [1071]}}
"""

import json
import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Detection parameters
DIM_INDEX = 7  # observation.state dimension to analyze
SEARCH_START = 700
SEARCH_END = 1800
MIN_THRESHOLD = -0.5  # minimum value must be below this
MIN_SEARCH_END = 1300  # minimum must be found within this range
WINDOW_SIZE = 20  # sliding window for plateau detection
PLATEAU_TOLERANCE = 0.01
PLATEAU_VALUE_LOW = -0.6  # first plateau value range
PLATEAU_VALUE_HIGH = -0.2


def detect_boundary(dim7: np.ndarray, episode_idx: int, repo_id: str) -> int | None:
    """Detect the boundary: the peak of the second ascent.

    Pattern: min(-0.6~-0.8) → first ascent plateau(-0.6~-0.3) → second ascent peak(~-0.2~0) → descent.
    Steps:
      1. Find min in [700, 1300], verify < -0.6
      2. Find first plateau (sliding window, value in [-0.6, -0.2])
      3. From first plateau onward, find global max = second ascent peak

    Returns the frame index of the second peak, or None if detection fails.
    """
    n = len(dim7)
    min_search_end = min(MIN_SEARCH_END, n)
    peak_search_end = min(SEARCH_END, n)

    if min_search_end <= SEARCH_START:
        logger.warning(f"[{repo_id}] ep {episode_idx}: episode too short ({n} frames), skipping")
        return None

    # Step 1: Find minimum in [700, 1300]
    segment = dim7[SEARCH_START:min_search_end]
    min_local_idx = segment.argmin()
    min_idx = SEARCH_START + min_local_idx
    min_val = dim7[min_idx]

    if min_val >= MIN_THRESHOLD:
        logger.warning(
            f"[{repo_id}] ep {episode_idx}: min value {min_val:.4f} at index {min_idx} "
            f"is not below {MIN_THRESHOLD}, skipping"
        )
        return None

    # Step 2: From min_idx, find the first plateau (value in [-0.6, -0.2])
    plateau_idx = None
    for i in range(min_idx, n - WINDOW_SIZE):
        window = dim7[i : i + WINDOW_SIZE]
        value_range = window.max() - window.min()
        if value_range < PLATEAU_TOLERANCE:
            stop_val = dim7[i]
            if stop_val < PLATEAU_VALUE_LOW:
                continue  # still too low, keep scanning
            if PLATEAU_VALUE_LOW <= stop_val <= PLATEAU_VALUE_HIGH:
                plateau_idx = i
                break

    if plateau_idx is None:
        logger.warning(f"[{repo_id}] ep {episode_idx}: no first plateau found after min at {min_idx}")
        return None

    plateau_val = dim7[plateau_idx]

    # Step 3: From first plateau, search within 500 frames for global max = second ascent peak
    peak_end = min(plateau_idx + 500, peak_search_end)
    search_segment = dim7[plateau_idx:peak_end]
    if len(search_segment) == 0:
        logger.warning(f"[{repo_id}] ep {episode_idx}: no data after plateau at {plateau_idx}")
        return None

    peak_local_idx = search_segment.argmax()
    peak_idx = plateau_idx + peak_local_idx
    peak_val = dim7[peak_idx]

    logger.info(
        f"[{repo_id}] ep {episode_idx}: min={min_val:.4f}@{min_idx}, "
        f"plateau={plateau_val:.4f}@{plateau_idx}, "
        f"boundary(peak)={peak_idx}, value={peak_val:.4f}"
    )
    return peak_idx


def process_repo(root_dir: str, repo_id: str) -> dict[str, list[int]] | None:
    """Process all episodes in a repo and return boundaries dict."""
    repo_path = Path(root_dir) / repo_id
    parquet_pattern = str(repo_path / "data" / "chunk-*" / "episode_*.parquet")
    parquet_files = sorted(glob.glob(parquet_pattern))

    if not parquet_files:
        logger.error(f"[{repo_id}] No parquet files found")
        return None

    boundaries = {}
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        states = np.stack(df["observation.state"].values)
        dim7 = states[:, DIM_INDEX]

        # Extract episode index from filename
        ep_idx = int(Path(pf).stem.split("_")[-1])

        boundary = detect_boundary(dim7, ep_idx, repo_id)
        if boundary is not None:
            boundaries[str(ep_idx)] = [int(boundary)]
        else:
            logger.error(f"[{repo_id}] ep {ep_idx}: FAILED to detect boundary")
            return None

    return boundaries


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate segment_values.json for each dataset by detecting segment boundaries from observation.state dim7.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  # 直接指定 repo_ids
  python tools/fold_box/generate_segment_boundaries.py \\
      --root-dir /mnt/oss_data/.../close_the_flap/ \\
      --repo-ids "total_steps/batch.1" "total_steps/batch.2"

  # 从 config 文件自动提取 repo_ids
  python tools/fold_box/generate_segment_boundaries.py \\
      --config src/openpi/configs/cfg_pi05_base_finetune_box_value_0407_close.py
""",
    )
    parser.add_argument("--root-dir", type=str, help="数据集根目录")
    parser.add_argument("--repo-ids", nargs="+", help="要处理的 repo_id 列表")
    parser.add_argument("--config", type=str, help="从 config 文件中提取 ROOT_DIR 和 REPO_ID")
    parser.add_argument("--skip-existing", action="store_true", help="跳过已有 segment_values.json 的数据集")
    args = parser.parse_args()

    if args.config:
        root_dir, repo_ids = _parse_config(args.config)
        if args.root_dir:
            root_dir = args.root_dir
        if args.repo_ids:
            repo_ids = args.repo_ids
    else:
        if not args.root_dir or not args.repo_ids:
            parser.error("必须指定 --config 或同时指定 --root-dir 和 --repo-ids")
        root_dir = args.root_dir
        repo_ids = args.repo_ids

    success_count = 0
    fail_count = 0
    skip_count = 0

    for repo_id in repo_ids:
        if args.skip_existing:
            seg_file = Path(root_dir) / repo_id / "meta" / "segment_values.json"
            if seg_file.exists():
                logger.info(f"[{repo_id}] SKIPPED (segment_values.json already exists)")
                skip_count += 1
                continue

        boundaries = process_repo(root_dir, repo_id)
        if boundaries is None:
            logger.error(f"[{repo_id}] SKIPPED due to detection failure")
            fail_count += 1
            continue

        # Write segment_values.json
        output_path = Path(root_dir) / repo_id / "meta" / "segment_values.json"
        output_data = {"boundaries": boundaries}
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=4)

        logger.info(f"[{repo_id}] Written {output_path} with {len(boundaries)} episodes")
        success_count += 1

    logger.info(f"Done: {success_count} success, {fail_count} failed, {skip_count} skipped")


def _parse_config(config_path: str) -> tuple[str, list[str]]:
    """从 config 文件中提取 ROOT_DIR 和 REPO_ID。"""
    import re

    text = Path(config_path).read_text()

    m = re.search(r'ROOT_DIR\s*=\s*["\'](.+?)["\']', text)
    if not m:
        raise ValueError(f"无法从 config 中提取 ROOT_DIR: {config_path}")
    root_dir = m.group(1)

    repo_ids = []
    in_repo_list = False
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"REPO_ID\s*=\s*\[", stripped):
            in_repo_list = True
            continue
        if in_repo_list:
            if stripped.startswith("]"):
                break
            if stripped.startswith("#"):
                continue
            m = re.search(r'["\'](.+?)["\']', stripped)
            if m:
                repo_ids.append(m.group(1))

    if not repo_ids:
        raise ValueError(f"无法从 config 中提取 REPO_ID: {config_path}")

    return root_dir, repo_ids


if __name__ == "__main__":
    main()
