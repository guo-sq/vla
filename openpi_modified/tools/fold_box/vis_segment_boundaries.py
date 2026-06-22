"""Visualize segment boundary frames by extracting the frame at boundary index
from head camera videos and saving as images."""

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "/mnt/workspace/zengqi/openpi_modified/vis_critical_segment"


def extract_frame(video_path: str, frame_index: int):
    """Extract a single frame from a video file."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="可视化 segment boundary 帧，从 head camera 视频中提取边界帧保存为图片。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  # 直接指定
  python tools/fold_box/vis_segment_boundaries.py \\
      --root-dir /mnt/oss_data/.../close_the_flap/ \\
      --repo-ids "total_steps/batch.1" "total_steps/batch.2"

  # 从 config 文件提取
  python tools/fold_box/vis_segment_boundaries.py \\
      --config src/openpi/configs/cfg_pi05_base_finetune_box_value_0407_close.py
""",
    )
    parser.add_argument("--root-dir", type=str, help="数据集根目录")
    parser.add_argument("--repo-ids", nargs="+", help="要处理的 repo_id 列表")
    parser.add_argument("--config", type=str, help="从 config 文件中提取 ROOT_DIR 和 REPO_ID")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="输出目录")
    args = parser.parse_args()

    if args.config:
        from generate_segment_boundaries import _parse_config

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

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    total = 0
    for repo_id in repo_ids:
        repo_path = Path(root_dir) / repo_id
        seg_file = repo_path / "meta" / "segment_values.json"

        if not seg_file.exists():
            logger.warning(f"[{repo_id}] segment_values.json not found, skipping")
            continue

        with open(seg_file) as f:
            seg_data = json.load(f)

        # Extract batch name from repo_id (last part)
        batch_name = repo_id.split("/")[-1]

        for ep_str, boundaries in seg_data["boundaries"].items():
            ep_idx = int(ep_str)
            video_path = repo_path / "videos" / "chunk-000" / "observation.images.head" / f"episode_{ep_idx:06d}.mp4"
            parquet_path = repo_path / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"

            if not video_path.exists():
                logger.warning(f"[{batch_name}] ep {ep_idx}: video not found at {video_path}")
                continue

            # Read observation.state dim7 values
            dim7 = None
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
                states = np.stack(df["observation.state"].values)
                dim7 = states[:, 7]

            for seg_i, boundary in enumerate(boundaries):
                frame = extract_frame(str(video_path), boundary)
                if frame is None:
                    logger.error(f"[{batch_name}] ep {ep_idx}: failed to read frame {boundary}")
                    continue

                val_str = f"_val{dim7[boundary]:.4f}" if dim7 is not None and boundary < len(dim7) else ""
                out_name = f"{batch_name}_ep{ep_idx}_boundary{seg_i}_frame{boundary}{val_str}.jpg"
                out_path = output_root / out_name
                cv2.imwrite(str(out_path), frame)
                logger.info(f"Saved {out_path}")
                total += 1

    logger.info(f"Done: saved {total} images to {output_root}")


if __name__ == "__main__":
    main()
