#!/usr/bin/env python3
"""Maintenance utility for LeRobot recording batches.

Two passes over every dataset under a root path:

  1. **Cleanup.** Delete leftover ``images/<image_key>/episode_NNNNNN/``
     directories whose corresponding
     ``videos/chunk-XXX/<image_key>/episode_NNNNNN.mp4`` already exists.
     `LeRobotDataset.encode_episode_videos` is supposed to ``rmtree``
     these after a successful encode, but a crash mid-encode (NVENC
     refusal, ENOSPC, KeyboardInterrupt) leaves the PNGs on disk.

  2. **Re-encode.** For each ``episode_NNNNNN.mp4`` whose bitrate is
     above ``--bitrate-threshold-mbps`` (default 30 Mbps), transcode
     with ``h264_nvenc`` to ~6 Mbps. The CPU fallback path used to
     produce lossless RGB at 200–500 Mbps; this pass shrinks those
     ~50× back to NVENC sizes. Lossy → lossy transcode loses a tiny
     amount of quality vs the original PNGs, but is what the operator
     would have gotten from a working GPU at record time.

Usage::

    conda activate lerobot_dev_anyverse  # need ffmpeg with NVENC
    python upload_system/repair_dataset_batches.py \\
        /home/wujie/playground/lerobot_data_collection/20260428/pack_socks

    # Inspect what would happen, no changes
    python upload_system/repair_dataset_batches.py <root> --dry-run

    # Only cleanup, no re-encode
    python upload_system/repair_dataset_batches.py <root> --no-reencode

    # Only re-encode, leave image dirs alone
    python upload_system/repair_dataset_batches.py <root> --no-cleanup

    # Tighter bitrate threshold (re-encode anything > 15 Mbps)
    python upload_system/repair_dataset_batches.py <root> --bitrate-threshold-mbps 15

A dataset is any directory containing ``meta/info.json``. The script
recurses, so passing a parent directory like
``/home/wujie/playground/lerobot_data_collection/20260428`` walks every
task / batch under it.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("repair")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def find_dataset_roots(root: Path) -> list[Path]:
    """Return every directory under ``root`` that looks like a LeRobot dataset
    (has ``meta/info.json``).

    Datasets won't be nested inside other datasets, so we prune the walk
    when we find one — saves time on large trees.
    """
    out: list[Path] = []
    stack = [root]
    while stack:
        d = stack.pop()
        if not d.is_dir():
            continue
        if (d / "meta" / "info.json").is_file():
            out.append(d)
            continue  # don't descend into a dataset
        for child in d.iterdir():
            if child.is_dir() and not child.name.startswith("."):
                stack.append(child)
    out.sort()
    return out


# ---------------------------------------------------------------------------
# Pass 1: leftover-image cleanup
# ---------------------------------------------------------------------------

@dataclass
class CleanupReport:
    deleted_dirs: list[Path] = field(default_factory=list)
    skipped_no_video: list[Path] = field(default_factory=list)
    bytes_freed: int = 0


def _dir_bytes(p: Path) -> int:
    total = 0
    for f in p.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total


def cleanup_orphan_images(dataset: Path, dry_run: bool) -> CleanupReport:
    """Delete ``images/<key>/episode_NNNNNN/`` dirs whose video exists."""
    rep = CleanupReport()
    images_root = dataset / "images"
    if not images_root.is_dir():
        return rep

    for image_key_dir in sorted(images_root.iterdir()):
        if not image_key_dir.is_dir():
            continue
        image_key = image_key_dir.name
        for ep_dir in sorted(image_key_dir.iterdir()):
            if not ep_dir.is_dir() or not ep_dir.name.startswith("episode_"):
                continue
            ep_idx = ep_dir.name.removeprefix("episode_")
            # videos/<chunk>/<image_key>/episode_NNNNNN.mp4 — the chunk depth
            # may vary, so glob.
            mp4_glob = list(
                dataset.glob(f"videos/*/{image_key}/episode_{ep_idx}.mp4")
            )
            if not mp4_glob:
                rep.skipped_no_video.append(ep_dir)
                continue
            size = _dir_bytes(ep_dir)
            rep.deleted_dirs.append(ep_dir)
            rep.bytes_freed += size
            if dry_run:
                log.info("  [dry-run] would rmtree %s (%s)", ep_dir, _fmt_bytes(size))
            else:
                shutil.rmtree(ep_dir)
                log.info("  removed %s (%s)", ep_dir, _fmt_bytes(size))

    # Tidy up any now-empty ``images/<key>/`` and ``images/`` shells.
    if not dry_run:
        for image_key_dir in list(images_root.iterdir()):
            if image_key_dir.is_dir() and not any(image_key_dir.iterdir()):
                image_key_dir.rmdir()
        if images_root.is_dir() and not any(images_root.iterdir()):
            images_root.rmdir()

    return rep


# ---------------------------------------------------------------------------
# Pass 2: re-encode oversized videos
# ---------------------------------------------------------------------------

@dataclass
class ReencodeReport:
    reencoded: list[tuple[Path, int, int]] = field(default_factory=list)  # (path, before, after)
    skipped_under_threshold: int = 0
    failed: list[tuple[Path, str]] = field(default_factory=list)
    bytes_saved: int = 0


def probe_bitrate_bps(mp4: Path) -> int | None:
    """Return the video stream's bitrate in bits/s, or None if unknown."""
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=bit_rate",
                "-of", "default=nokey=1:noprint_wrappers=1",
                str(mp4),
            ],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        log.error("ffprobe not on PATH — activate the env with ffmpeg installed")
        sys.exit(1)
    s = (out.stdout or "").strip()
    if not s or s == "N/A":
        # Some muxes don't carry stream bitrate in the header (NVENC at
        # constant QP is one of them). Fall back to file_size * 8 / duration
        # which approximates the same number.
        return _bitrate_from_size_and_duration(mp4)
    try:
        return int(s)
    except ValueError:
        return _bitrate_from_size_and_duration(mp4)


def _bitrate_from_size_and_duration(mp4: Path) -> int | None:
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=nokey=1:noprint_wrappers=1",
                str(mp4),
            ],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        return None
    s = (out.stdout or "").strip()
    try:
        dur = float(s)
        if dur <= 0:
            return None
    except ValueError:
        return None
    try:
        size_bytes = mp4.stat().st_size
    except OSError:
        return None
    return int(size_bytes * 8 / dur)


def reencode_with_nvenc(mp4: Path, dry_run: bool) -> tuple[bool, str]:
    """Transcode ``mp4`` in place using h264_nvenc.

    Strategy:
      1. write to ``<mp4>.reencoded.mp4`` in the same dir
      2. if ffmpeg succeeds AND new file is smaller, atomic-rename over the
         original
      3. otherwise leave the original untouched and remove the temp file
    """
    if dry_run:
        return True, "would re-encode"

    tmp = mp4.with_suffix(".reencoded.mp4")
    if tmp.exists():
        tmp.unlink()

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(mp4),
        "-c:v", "h264_nvenc",
        "-preset", "p4", "-tune", "hq",
        "-b:v", "6M", "-bf", "0",
        "-pix_fmt", "yuv420p",
        "-g", "2",
        "-rc", "constqp", "-qp", "30",
        "-an",  # the recorder writes silent video; no audio stream to keep
        str(tmp),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        if tmp.exists():
            tmp.unlink()
        return False, (res.stderr or "ffmpeg failed").strip().splitlines()[-1]

    try:
        new_size = tmp.stat().st_size
        old_size = mp4.stat().st_size
    except OSError as e:
        if tmp.exists():
            tmp.unlink()
        return False, f"stat failed: {e}"

    if new_size >= old_size:
        # NVENC at constqp 30 should always shrink a lossless RGB blob; if
        # it didn't, the source was already small. Bail without modifying.
        tmp.unlink()
        return False, f"re-encoded file not smaller ({new_size} ≥ {old_size}); skipped"

    tmp.replace(mp4)
    return True, "re-encoded"


def reencode_oversized_videos(
    dataset: Path,
    threshold_bps: int,
    dry_run: bool,
) -> ReencodeReport:
    rep = ReencodeReport()
    videos_root = dataset / "videos"
    if not videos_root.is_dir():
        return rep
    for mp4 in sorted(videos_root.rglob("episode_*.mp4")):
        bitrate = probe_bitrate_bps(mp4)
        if bitrate is None:
            log.warning("  could not determine bitrate of %s; skipping", mp4)
            continue
        if bitrate < threshold_bps:
            rep.skipped_under_threshold += 1
            continue
        before_size = mp4.stat().st_size
        log.info(
            "  %s — bitrate=%.1f Mbps, size=%s — re-encoding",
            mp4.relative_to(dataset),
            bitrate / 1e6,
            _fmt_bytes(before_size),
        )
        ok, msg = reencode_with_nvenc(mp4, dry_run)
        if not ok:
            log.error("    failed: %s", msg)
            rep.failed.append((mp4, msg))
            continue
        if dry_run:
            continue
        after_size = mp4.stat().st_size
        rep.reencoded.append((mp4, before_size, after_size))
        rep.bytes_saved += before_size - after_size
        log.info(
            "    -> %s (saved %s, %.1f%% reduction)",
            _fmt_bytes(after_size),
            _fmt_bytes(before_size - after_size),
            100.0 * (1 - after_size / before_size) if before_size else 0.0,
        )
    return rep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Clean leftover image dirs and re-encode oversized videos under "
            "every LeRobot dataset found beneath a given path."
        ),
    )
    p.add_argument("root", type=Path, help="path to walk (e.g. .../20260428/pack_socks)")
    p.add_argument("--dry-run", action="store_true", help="report only, change nothing")
    p.add_argument("--no-cleanup", action="store_true", help="skip image-cleanup pass")
    p.add_argument("--no-reencode", action="store_true", help="skip re-encode pass")
    p.add_argument(
        "--bitrate-threshold-mbps",
        type=float,
        default=30.0,
        help=(
            "re-encode any episode mp4 whose video bitrate exceeds this "
            "(NVENC produces ~6 Mbps; the lossless CPU fallback was "
            "200–500 Mbps; default 30 Mbps catches the bloat without "
            "touching legitimate NVENC files)"
        ),
    )
    args = p.parse_args()

    if not args.root.is_dir():
        log.error("not a directory: %s", args.root)
        return 1

    threshold_bps = int(args.bitrate_threshold_mbps * 1e6)

    datasets = find_dataset_roots(args.root)
    if not datasets:
        log.error("no LeRobot datasets (meta/info.json) found under %s", args.root)
        return 1
    log.info("found %d dataset(s) under %s", len(datasets), args.root)

    total_bytes_freed = 0
    total_bytes_saved = 0
    total_reencoded = 0
    total_cleanup_dirs = 0
    total_failures: list[tuple[Path, str]] = []

    for ds in datasets:
        log.info("dataset: %s", ds)
        if not args.no_cleanup:
            log.info(" pass 1/2: cleaning leftover image dirs")
            cr = cleanup_orphan_images(ds, dry_run=args.dry_run)
            total_cleanup_dirs += len(cr.deleted_dirs)
            total_bytes_freed += cr.bytes_freed
            if cr.skipped_no_video:
                log.info(
                    "   note: %d image dir(s) have no matching mp4 (encode "
                    "may have failed) — left in place",
                    len(cr.skipped_no_video),
                )
        if not args.no_reencode:
            log.info(" pass 2/2: re-encoding oversized videos (>%g Mbps)", args.bitrate_threshold_mbps)
            rr = reencode_oversized_videos(ds, threshold_bps, dry_run=args.dry_run)
            total_reencoded += len(rr.reencoded)
            total_bytes_saved += rr.bytes_saved
            total_failures.extend(rr.failed)

    log.info("=" * 60)
    log.info("Summary:")
    log.info("  datasets processed     : %d", len(datasets))
    log.info("  image dirs removed     : %d (%s freed)", total_cleanup_dirs, _fmt_bytes(total_bytes_freed))
    log.info("  videos re-encoded      : %d (%s saved)", total_reencoded, _fmt_bytes(total_bytes_saved))
    if total_failures:
        log.warning("  re-encode failures: %d", len(total_failures))
        for path, msg in total_failures:
            log.warning("    %s — %s", path, msg)
        return 2
    if args.dry_run:
        log.info("  (dry-run — nothing was actually changed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
