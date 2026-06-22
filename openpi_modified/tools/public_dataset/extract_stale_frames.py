#!/usr/bin/env python3
"""
抽取某个 batch 数据集里指定 episode 的前 N 帧为 PNG,用于肉眼核验陈旧帧(stale frame)问题。
背景见 lerobot_modified/docs/mp4_stale_first_frame.md。

用法示例:
    python tools/public_dataset/extract_stale_frames.py \
        --dataset-dir /mnt/oss_data/.../close_the_flap.Mold.85s.20260407.batch.6

    python tools/public_dataset/extract_stale_frames.py \
        --dataset-dir .../batch.6 --episode 3 --num-frames 15 \
        --cameras observation.images.head observation.images.left_wrist
"""

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import tyro


@dataclass
class Args:
    dataset_dir: Path
    """batch 数据集根目录,里面需有 videos/chunk-000/<camera>/episode_XXXXXX.mp4。"""

    episode: int = 0
    """episode 索引,默认 0(对应 episode_000000.mp4)。"""

    num_frames: int = 30
    """抽前多少帧,默认 30。"""

    cameras: list[str] = field(default_factory=list)
    """相机视角名(videos/chunk-000 下的子目录名)。留空则自动抽所有可见相机。"""

    output_dir: Path | None = None
    """输出目录。默认 <project_root>/stale_frame_out/<batch_name>/ep<episode>/。"""


def discover_cameras(videos_root: Path) -> list[str]:
    chunk_dir = videos_root / "chunk-000"
    if not chunk_dir.is_dir():
        raise FileNotFoundError(f"videos/chunk-000 not found under {videos_root}")
    return sorted(p.name for p in chunk_dir.iterdir() if p.is_dir())


def extract_one_camera(mp4_path: Path, num_frames: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        print(f"[WARN] cannot open {mp4_path}, skip")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[{mp4_path.parent.name}] total_frames={total}, fps={fps:.2f} → extract {min(num_frames, total)}")

    written = 0
    for i in range(num_frames):
        ok, frame = cap.read()
        if not ok:
            print(f"  [WARN] reached EOF at frame {i} (requested {num_frames})")
            break
        ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        out_path = out_dir / f"frame_{i:03d}.png"
        cv2.imwrite(str(out_path), frame)
        if i == 0 or i == num_frames - 1:
            print(f"  frame {i:03d} pos_msec={ts_ms:.1f} → {out_path.name}")
        written += 1

    cap.release()
    print(f"  wrote {written} frames to {out_dir}")


def main(args: Args) -> None:
    dataset_dir = args.dataset_dir.resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset_dir not a directory: {dataset_dir}")

    videos_root = dataset_dir / "videos"
    cameras = args.cameras or discover_cameras(videos_root)
    if not cameras:
        raise RuntimeError(f"no cameras found under {videos_root}/chunk-000")

    if args.output_dir is None:
        project_root = Path(__file__).resolve().parents[2]
        out_root = project_root / "stale_frame_out" / dataset_dir.name / f"ep{args.episode:06d}"
    else:
        out_root = args.output_dir.resolve()

    print(f"dataset_dir: {dataset_dir}")
    print(f"episode:     {args.episode}")
    print(f"num_frames:  {args.num_frames}")
    print(f"cameras:     {cameras}")
    print(f"output_dir:  {out_root}")
    print("-" * 60)

    episode_file = f"episode_{args.episode:06d}.mp4"
    for cam in cameras:
        mp4_path = videos_root / "chunk-000" / cam / episode_file
        if not mp4_path.is_file():
            print(f"[WARN] missing {mp4_path}, skip")
            continue
        extract_one_camera(mp4_path, args.num_frames, out_root / cam)


if __name__ == "__main__":
    main(tyro.cli(Args))
