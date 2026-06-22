"""Slice a segment from an existing LeRobot dataset to create a new standalone dataset.

The segment duration (in integer seconds) is auto-injected before the final
dotted suffix of --dst's basename. The suffix is preserved as-is — common
labels are .good/.bad (or .rise/.flat). Caller passes e.g. `...ep4.good`; the
script writes to `...ep4.<N>s.good`.

Single-row usage:
    python tools/fold_box/slice_dataset.py \
        --src /path/to/source_dataset \
        --dst /path/to/dest_dataset.epX.rise \
        --episode X \
        --start-idx <start> \
        --end-idx <end>

Single-row example:
    python tools/fold_box/slice_dataset.py \
        --src /mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/fold_box/close_the_flap/infer/raw_infer/close_the_box_infer.origin.pi05_base_finetune_box_recap_pt_0419_close.1w9.6000s.20260420.batch.3 \
        --dst /mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/fold_box/close_the_flap/infer/infer_value_labeled_good/close_the_box_infer.origin.pi05_base_finetune_box_recap_pt_0419_close.1w9.6000s.20260420.batch.3.ep4.rise \
        --episode 4 \
        --start-idx 270 \
        --end-idx 390
    # actual output dir: ...batch.3.ep4.4s.rise  (4s = (390-270)/30)

Batch usage:
    python tools/fold_box/slice_dataset.py --batch-file slices.csv

    When --batch-file is given, the other args (--src/--dst/--episode/
    --start-idx/--end-idx/--model-path) are ignored.

    CSV columns (header required):
        src         - source dataset path relative to data_path (required)
        dst_dir     - destination DIRECTORY relative to data_path (required).
                      The dataset name is auto-composed as
                        <batch_name>.ep<episode>.<Ns>.<label>
                      where batch_name is the basename of src.
        episode     - episode index (required)
        start_sec   - start time in seconds (required)
        end_sec     - end time in seconds (required)
        label       - 'good' or 'bad' (preferred). If the column is absent
                      or blank, the label is inferred from dst_dir's name:
                      contains 'bad' -> bad, contains 'good' -> good.
        data_path   - optional prefix for src/dst_dir. Defaults to
                      /mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/
                      fold_box/close_the_flap/infer/
                      Leave blank to use default.
        model_path  - optional, checkpoint path to record in slice_info.json

    src/dst_dir are always joined under data_path. A leading '/' on
    src/dst_dir is stripped before joining (treated as a visual separator,
    not absolute path).

    Lines starting with '#' are treated as comments and skipped. Rows with an
    empty src are skipped. Row failures are logged but do not abort the batch.
"""

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def compute_feature_stats(series, feature_info):
    """Compute min/max/mean/std/count stats for a single feature column."""
    dtype = feature_info["dtype"]

    if dtype == "video":
        # Video features cannot be recomputed without decoding frames.
        # Return placeholder stats.
        shape = feature_info["shape"]  # e.g. [480, 640, 3]
        channels = shape[-1]
        zeros = [[[0.0]] for _ in range(channels)]
        ones = [[[1.0]] for _ in range(channels)]
        halves = [[[0.5]] for _ in range(channels)]
        quarter = [[[0.25]] for _ in range(channels)]
        return {
            "min": zeros,
            "max": ones,
            "mean": halves,
            "std": quarter,
            "count": [len(series)],
        }

    if dtype == "bool":
        vals = series.values.astype(float)
        return {
            "min": [bool(vals.min() > 0.5)],
            "max": [bool(vals.max() > 0.5)],
            "mean": [float(vals.mean())],
            "std": [float(vals.std())],
            "count": [len(vals)],
        }

    # Numeric features: float32, int64, float32 arrays
    vals = np.stack(series.values)
    if vals.ndim == 1:
        vals = vals.reshape(-1, 1)

    result = {
        "min": vals.min(axis=0).tolist(),
        "max": vals.max(axis=0).tolist(),
        "mean": vals.mean(axis=0).tolist(),
        "std": vals.std(axis=0).tolist(),
        "count": [len(vals)],
    }

    # Convert numpy types for JSON serialization
    for key in ["min", "max", "mean", "std"]:
        result[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in result[key]]

    return result


def slice_parquet(src_parquet: Path, dst_parquet: Path, start_idx: int, end_idx: int, fps: int):
    """Read source parquet, slice by frame_index range, reset indices, and save."""
    df = pd.read_parquet(src_parquet)

    # Slice by frame_index
    mask = (df["frame_index"] >= start_idx) & (df["frame_index"] < end_idx)
    sliced = df[mask].copy()

    if len(sliced) == 0:
        raise ValueError(f"No frames found in range [{start_idx}, {end_idx}). "
                         f"Episode has frame_index range [{df['frame_index'].min()}, {df['frame_index'].max()}]")

    n_frames = len(sliced)

    # Reset fields
    sliced["frame_index"] = np.arange(n_frames, dtype=np.int64)
    sliced["episode_index"] = np.zeros(n_frames, dtype=np.int64)
    sliced["index"] = np.arange(n_frames, dtype=np.int64)
    sliced["timestamp"] = np.arange(n_frames, dtype=np.float32) / fps
    sliced["task_index"] = np.zeros(n_frames, dtype=np.int64)

    dst_parquet.parent.mkdir(parents=True, exist_ok=True)
    sliced.to_parquet(dst_parquet, index=False)

    return sliced, n_frames


def slice_video(src_video: Path, dst_video: Path, start_idx: int, end_idx: int, fps: int):
    """Cut video segment using ffmpeg with re-encoding for frame-accurate cuts."""
    import tempfile

    start_sec = start_idx / fps
    end_sec = end_idx / fps

    dst_video.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temp file first, then copy to destination.
    # This avoids issues with network filesystems that don't support
    # the seek operations required by movflags +faststart.
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(src_video),
            "-ss", f"{start_sec:.6f}",
            "-to", f"{end_sec:.6f}",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",  # no audio
            tmp_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {src_video}:\n{result.stderr}")

        shutil.copy2(tmp_path, dst_video)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def generate_meta(
    src_dir: Path,
    dst_dir: Path,
    sliced_df: pd.DataFrame,
    n_frames: int,
    episode_idx: int,
    start_idx: int,
    end_idx: int,
    fps: int,
    model_path: str | None = None,
):
    """Generate all meta files for the new dataset."""
    meta_dir = dst_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # 1. info.json
    with open(src_dir / "meta" / "info.json") as f:
        info = json.load(f)

    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    info["total_episodes"] = 1
    info["total_frames"] = n_frames
    info["total_videos"] = len(video_keys)
    info["total_chunks"] = 1
    info["splits"] = {"train": "0:1"}

    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    # 2. tasks.jsonl - copy from source
    shutil.copy2(src_dir / "meta" / "tasks.jsonl", meta_dir / "tasks.jsonl")

    # 3. episodes.jsonl
    # Read source tasks for this episode
    with open(src_dir / "meta" / "episodes.jsonl") as f:
        for line in f:
            ep = json.loads(line)
            if ep["episode_index"] == episode_idx:
                tasks = ep.get("tasks", [])
                break
        else:
            tasks = []

    episode_entry = {"episode_index": 0, "tasks": tasks, "length": n_frames}
    with open(meta_dir / "episodes.jsonl", "w") as f:
        f.write(json.dumps(episode_entry) + "\n")

    # 4. episodes_stats.jsonl
    stats = {}
    for feature_name, feature_info in info["features"].items():
        if feature_name in sliced_df.columns:
            stats[feature_name] = compute_feature_stats(sliced_df[feature_name], feature_info)
        elif feature_info.get("dtype") == "video":
            stats[feature_name] = compute_feature_stats(
                pd.Series([None] * n_frames), feature_info
            )

    stats_entry = {"episode_index": 0, "stats": stats}
    with open(meta_dir / "episodes_stats.jsonl", "w") as f:
        f.write(json.dumps(stats_entry) + "\n")

    # 5. slice_info.json
    slice_info = {
        "source_dataset": str(src_dir),
        "source_episode_index": episode_idx,
        "source_start_frame_index": start_idx,
        "source_end_frame_index": end_idx,
        "fps": fps,
        "sliced_length": n_frames,
        "model_path": model_path,
        "sliced_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(meta_dir / "slice_info.json", "w") as f:
        json.dump(slice_info, f, indent=4)


def slice_one(
    src: str,
    dst: str,
    episode: int,
    start_idx: int | None,
    end_idx: int | None,
    start_sec: float | None = None,
    end_sec: float | None = None,
    model_path: str | None = None,
):
    """Perform one slice. Either (start_idx, end_idx) or (start_sec, end_sec) must be set."""
    src_dir = Path(src)
    dst_dir_arg = Path(dst)

    if not src_dir.exists():
        raise FileNotFoundError(f"Source dataset not found: {src_dir}")

    with open(src_dir / "meta" / "info.json") as f:
        info = json.load(f)
    fps = info["fps"]

    if start_idx is None:
        if start_sec is None:
            raise ValueError("Either start_idx or start_sec must be provided")
        start_idx = int(round(start_sec * fps))
    if end_idx is None:
        if end_sec is None:
            raise ValueError("Either end_idx or end_sec must be provided")
        end_idx = int(round(end_sec * fps))

    duration_label = f"{int(round((end_idx - start_idx) / fps))}s"
    prefix, _, type_suffix = dst_dir_arg.name.rpartition(".")
    dst_dir = dst_dir_arg.parent / f"{prefix}.{duration_label}.{type_suffix}"

    print(f"Slicing episode {episode}, frames [{start_idx}, {end_idx}) from:")
    print(f"  {src_dir}")
    print(f"  fps={fps}, expected frames={end_idx - start_idx}")
    print(f"  dst (with duration): {dst_dir}")

    src_parquet = src_dir / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
    dst_parquet = dst_dir / "data" / "chunk-000" / "episode_000000.parquet"
    print(f"\n[1/4] Slicing parquet: {src_parquet.name}")
    sliced_df, n_frames = slice_parquet(src_parquet, dst_parquet, start_idx, end_idx, fps)
    print(f"  -> {n_frames} frames saved")

    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    print(f"\n[2/4] Slicing {len(video_keys)} videos")
    for vk in video_keys:
        src_video = src_dir / "videos" / "chunk-000" / vk / f"episode_{episode:06d}.mp4"
        dst_video = dst_dir / "videos" / "chunk-000" / vk / "episode_000000.mp4"
        if src_video.exists():
            print(f"  Cutting {vk}...")
            slice_video(src_video, dst_video, start_idx, end_idx, fps)
        else:
            print(f"  WARNING: video not found: {src_video}")

    print("\n[3/4] Generating meta files")
    generate_meta(
        src_dir, dst_dir, sliced_df, n_frames, episode,
        start_idx, end_idx, fps,
        model_path=model_path,
    )

    print(f"\n[4/4] Done! New dataset at:")
    print(f"  {dst_dir}")
    print(f"  Episodes: 1, Frames: {n_frames}, Duration: {n_frames/fps:.1f}s")
    return dst_dir


def _parse_optional_int(val) -> int | None:
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return None
    return int(float(s))


def _parse_optional_float(val) -> float | None:
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return None
    return float(s)


def _parse_optional_str(val) -> str | None:
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return None
    return s


DEFAULT_DATA_PATH = "/mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/fold_box/close_the_flap/infer/"


def _join_under(data_path: str, rel: str) -> str:
    """Join rel onto data_path, treating a leading '/' on rel as a separator."""
    return data_path.rstrip("/") + "/" + rel.lstrip("/")


def run_batch(batch_file: str):
    """Read batch CSV/TSV and run slice_one for each row. Continue on failure."""
    batch_path = Path(batch_file)
    if not batch_path.exists():
        raise FileNotFoundError(f"Batch file not found: {batch_path}")

    sep = "\t" if batch_path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(batch_path, sep=sep, comment="#", skip_blank_lines=True,
                     dtype=str, keep_default_na=False, index_col=False)
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]

    required = {"src", "dst_dir", "episode", "start_sec", "end_sec"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Batch file missing required columns: {sorted(missing)}")

    total = len(df)
    successes, failures = [], []
    for i, row in df.iterrows():
        src_rel = _parse_optional_str(row.get("src"))
        if src_rel is None:
            continue
        dst_dir_rel = _parse_optional_str(row.get("dst_dir"))
        ep = _parse_optional_int(row.get("episode"))
        start_sec = _parse_optional_float(row.get("start_sec"))
        end_sec = _parse_optional_float(row.get("end_sec"))
        label = _parse_optional_str(row.get("label")) if "label" in df.columns else None
        data_path = _parse_optional_str(row.get("data_path")) if "data_path" in df.columns else None
        model_path = _parse_optional_str(row.get("model_path")) if "model_path" in df.columns else None

        if data_path is None:
            data_path = DEFAULT_DATA_PATH

        print(f"\n{'='*80}\n[row {i+1}/{total}] episode={ep}")
        try:
            if dst_dir_rel is None:
                raise ValueError("dst_dir is empty")
            if ep is None:
                raise ValueError("episode is empty")

            if label is None:
                dir_lower = dst_dir_rel.strip("/").lower()
                if "bad" in dir_lower:
                    label = "bad"
                elif "good" in dir_lower:
                    label = "good"
                else:
                    raise ValueError(
                        f"label column is empty and dst_dir '{dst_dir_rel}' "
                        f"contains neither 'good' nor 'bad' — set label explicitly"
                    )
            if label not in ("good", "bad"):
                raise ValueError(f"label must be 'good' or 'bad', got {label!r}")

            batch_name = Path(src_rel).name
            composed_name = f"{batch_name}.ep{ep}.{label}"
            full_src = _join_under(data_path, src_rel)
            full_dst = _join_under(data_path, dst_dir_rel.rstrip("/")) + "/" + composed_name

            dst_dir = slice_one(
                src=full_src, dst=full_dst, episode=ep,
                start_idx=None, end_idx=None,
                start_sec=start_sec, end_sec=end_sec,
                model_path=model_path,
            )
            successes.append((i + 1, dst_dir))
        except Exception as e:
            print(f"  ERROR on row {i+1}: {e}")
            failures.append((i + 1, str(e)))

    print(f"\n{'='*80}\nBatch summary: {len(successes)} succeeded, {len(failures)} failed")
    if failures:
        print("Failed rows:")
        for idx, err in failures:
            print(f"  row {idx}: {err}")


def main():
    parser = argparse.ArgumentParser(description="Slice a segment from a LeRobot dataset")
    parser.add_argument("--batch-file", type=str, default=None,
                        help="CSV/TSV of slice specs. When set, other single-row args are ignored.")
    parser.add_argument("--src", type=str, help="Source dataset path")
    parser.add_argument("--dst", type=str, help="Destination dataset path")
    parser.add_argument("--episode", type=int, help="Episode index to slice from")
    parser.add_argument("--start-idx", type=int, help="Start frame_index (inclusive)")
    parser.add_argument("--end-idx", type=int, help="End frame_index (exclusive)")
    parser.add_argument("--model-path", type=str, default=None, help="Model checkpoint path used for inference")
    args = parser.parse_args()

    if args.batch_file:
        run_batch(args.batch_file)
        return

    missing = [name for name, val in [
        ("--src", args.src), ("--dst", args.dst), ("--episode", args.episode),
        ("--start-idx", args.start_idx), ("--end-idx", args.end_idx),
    ] if val is None]
    if missing:
        parser.error(f"Missing required args (unless --batch-file is used): {missing}")

    slice_one(
        src=args.src, dst=args.dst, episode=args.episode,
        start_idx=args.start_idx, end_idx=args.end_idx,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()
