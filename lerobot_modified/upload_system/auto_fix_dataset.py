import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# 全局日志文件对象
_log_file = None


def _log(message: str) -> None:
    """同时输出到stdout和日志文件"""
    print(message)
    if _log_file:
        _log_file.write(message + "\n")
        _log_file.flush()


def _parse_episode_idx(path: Path) -> int | None:
    # 期望文件名: episode_000123.parquet / episode_000123.mp4
    if not path.stem.startswith("episode_"):
        return None
    parts = path.stem.split("_")
    if len(parts) != 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _backup_file(path: Path) -> Path:
    base = path.with_suffix(path.suffix + ".bak")
    candidate = base
    idx = 1
    while candidate.exists():
        candidate = path.with_suffix(path.suffix + f".bak.{idx}")
        idx += 1
    shutil.copy2(path, candidate)
    return candidate


def _atomic_write_text(path: Path, content: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)
    tmp_path.replace(path)


def _print_action(apply: bool, message: str) -> None:
    prefix = "[Do]" if apply else "[DryRun]"
    _log(f"  {prefix} {message}")


def _collect_video_keys(info: dict) -> list[str]:
    features = info.get("features", {})
    if not isinstance(features, dict):
        return []
    video_keys = []
    for key, ft in features.items():
        if isinstance(ft, dict) and ft.get("dtype") == "video":
            video_keys.append(key)
    return sorted(video_keys)


def fix_single_dataset(dataset_path: Path, apply: bool, backup: bool) -> None:
    """
    对单个数据集执行 9 条修复规则（安全版）：
    - 默认 dry-run，仅输出计划动作
    - 仅 --apply 时才会真正修改文件
    """
    dataset_path = Path(dataset_path)
    _log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    _log(f"处理数据集: {dataset_path.name}")

    info_path = dataset_path / "meta/info.json"
    episodes_path = dataset_path / "meta/episodes.jsonl"
    data_dir = dataset_path / "data"
    videos_dir = dataset_path / "videos"
    images_dir = dataset_path / "images"

    info_modified = False
    jsonl_modified = False
    has_fix_actions = False
    unsafe_for_total_frames = False

    # ---------------------------------------------------------
    # [规则 1] 如果存在 images 目录，删除它
    # ---------------------------------------------------------
    if images_dir.exists():
        has_fix_actions = True
        _print_action(apply, f"[Fix-1] 删除 images 目录: {images_dir}")
        if apply:
            shutil.rmtree(images_dir)

    # ---------------------------------------------------------
    # [规则 2] 检查必要文件是否存在
    # ---------------------------------------------------------
    missing = []
    if not info_path.exists():
        missing.append(str(info_path))
    if not episodes_path.exists():
        missing.append(str(episodes_path))
    if missing:
        _log("  [Error] 缺少必要元数据文件，跳过此数据集:")
        for p in missing:
            _log(f"    - {p}")
        return

    try:
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
    except Exception as e:
        _log(f"  [Error] 读取 info.json 失败: {e}")
        return

    jsonl_lines: list[dict] = []
    try:
        with open(episodes_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                jsonl_lines.append(json.loads(line))
    except Exception as e:
        _log(f"  [Error] 读取 episodes.jsonl 失败: {e}")
        return

    # 获取目标 total_episodes (必须显式存在，避免默认 0 导致误删)
    total_episodes = info.get("total_episodes", None)
    if not isinstance(total_episodes, int) or total_episodes < 0:
        _log("  [Error] info['total_episodes'] 缺失或非法，跳过此数据集以避免误删。")
        return
    target_episodes = total_episodes

    video_keys = _collect_video_keys(info)
    _log(f"  [Info] 目标 episodes: {target_episodes}, video_keys: {video_keys}")

    # ---------------------------------------------------------
    # [规则 10] 如果所有相机视频数、parquet 数、episodes.jsonl 行数都相等为 N，
    #           且 N != target_episodes，则修正 info.json 中的元数据
    # ---------------------------------------------------------
    count_jsonl_early = len(jsonl_lines)

    # 统计 parquet 文件数
    count_parquet_early = sum(
        1 for p in data_dir.glob("chunk-*/episode_*.parquet")
        if _parse_episode_idx(p) is not None
    )

    # 统计每个相机的视频数
    cam_video_counts: list[int] = []
    for cam in video_keys:
        cam_count = sum(
            1 for v in videos_dir.glob(f"chunk-*/{cam}/episode_*.mp4")
            if _parse_episode_idx(v) is not None
        )
        cam_video_counts.append(cam_count)

    all_counts = [count_jsonl_early, count_parquet_early] + cam_video_counts
    # 所有数据源数量一致、大于 0、且与 info.total_episodes 不同 → 信任实际数据
    if (
        len(all_counts) >= 2
        and len(set(all_counts)) == 1
        and all_counts[0] > 0
        and all_counts[0] != target_episodes
    ):
        N = all_counts[0]
        cam_desc = ", ".join(f"{cam}={c}" for cam, c in zip(video_keys, cam_video_counts))
        _log(
            f"  [Info] [Fix-10] 所有数据源一致 (N={N}): "
            f"episodes.jsonl={count_jsonl_early}, parquet={count_parquet_early}"
            + (f", {cam_desc}" if cam_desc else "")
            + f"，但 info.total_episodes={target_episodes}"
        )

        total_frames_from_jsonl = sum(ep.get("length", 0) for ep in jsonl_lines)
        new_total_videos = len(video_keys) * N

        has_fix_actions = True
        _print_action(apply, f"[Fix-10] 修正 total_episodes: {target_episodes} -> {N}")
        _print_action(apply, f"[Fix-10] 修正 total_videos: {info.get('total_videos', 0)} -> {new_total_videos}")
        _print_action(
            apply,
            f"[Fix-10] 修正 splits.train: {info.get('splits', {}).get('train', '0:0')} -> 0:{N}",
        )
        _print_action(apply, f"[Fix-10] 修正 total_frames: {info.get('total_frames', 0)} -> {total_frames_from_jsonl}")

        if apply:
            info["total_episodes"] = N
            info["total_videos"] = new_total_videos
            info.setdefault("splits", {})
            info["splits"]["train"] = f"0:{N}"
            info["total_frames"] = total_frames_from_jsonl
        target_episodes = N
        info_modified = True
        _log(f"  [Info] 目标 episodes 已更新为: {target_episodes}")

    # ---------------------------------------------------------
    # [规则 5] 修正 episodes.jsonl 行数（仅允许截断，不反向改 total_episodes）
    # ---------------------------------------------------------
    count_jsonl = len(jsonl_lines)
    if count_jsonl != target_episodes:
        if count_jsonl > target_episodes:
            has_fix_actions = True
            _print_action(
                apply,
                f"[Fix-5] episodes.jsonl 行数 {count_jsonl} > {target_episodes}，截断最后 {count_jsonl - target_episodes} 行",
            )
            if apply:
                jsonl_lines = jsonl_lines[:target_episodes]
            jsonl_modified = True
        else:
            _log(
                f"  [Error] [Fix-5] episodes.jsonl 行数 {count_jsonl} < total_episodes {target_episodes}，疑似数据缺失，仅报告不修改。"
            )

    # ---------------------------------------------------------
    # [规则 6] parquet 数量与 total_episodes 不一致时，删除最后的 parquet 直到一致
    # ---------------------------------------------------------
    parquet_pairs: list[tuple[int, Path]] = []
    for p in data_dir.glob("chunk-*/episode_*.parquet"):
        ep_idx = _parse_episode_idx(p)
        if ep_idx is None:
            _log(f"  [Warn] 跳过无法解析 episode 索引的 parquet: {p}")
            continue
        parquet_pairs.append((ep_idx, p))
    parquet_pairs.sort(key=lambda x: (x[0], str(x[1])))

    count_parquet = len(parquet_pairs)
    if count_parquet != target_episodes:
        if count_parquet > target_episodes:
            to_delete = parquet_pairs[target_episodes:]
            has_fix_actions = True
            _print_action(
                apply,
                f"[Fix-6] parquet 数量 {count_parquet} > {target_episodes}，删除最后 {len(to_delete)} 个 parquet",
            )
            for _, path in to_delete:
                _print_action(apply, f"删除 parquet: {path}")
                if apply:
                    path.unlink()
        else:
            _log(
                f"  [Error] [Fix-6] parquet 数量 {count_parquet} < total_episodes {target_episodes}，疑似数据缺失，仅报告不删除。"
            )

    # ---------------------------------------------------------
    # [规则 7] 每个相机视频数量与 total_episodes 不一致时，删除最后视频直到一致
    # ---------------------------------------------------------
    for cam in video_keys:
        video_pairs: list[tuple[int, Path]] = []
        for v in videos_dir.glob(f"chunk-*/{cam}/episode_*.mp4"):
            ep_idx = _parse_episode_idx(v)
            if ep_idx is None:
                _log(f"  [Warn] 跳过无法解析 episode 索引的视频: {v}")
                continue
            video_pairs.append((ep_idx, v))
        video_pairs.sort(key=lambda x: (x[0], str(x[1])))

        video_counts = len(video_pairs)
        if video_counts != target_episodes:
            if video_counts > target_episodes:
                to_delete = video_pairs[target_episodes:]
                has_fix_actions = True
                _print_action(
                    apply,
                    f"[Fix-7] 相机 {cam} 视频数 {video_counts} > {target_episodes}，删除最后 {len(to_delete)} 个视频",
                )
                for _, path in to_delete:
                    _print_action(apply, f"删除视频: {path}")
                    if apply:
                        path.unlink()
            else:
                _log(
                    f"  [Error] [Fix-7] 相机 {cam} 视频数 {video_counts} < total_episodes {target_episodes}，疑似数据缺失，仅报告。"
                )

    # ---------------------------------------------------------
    # [规则 9] 若 parquet 行数与 episodes.jsonl 的 length 不一致，修正 jsonl
    # [规则 8] 统计 parquet 总行数，修正 info.total_frames（若存在缺失则不落盘）
    # ---------------------------------------------------------
    chunk_size = info.get("chunks_size", 1000)
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        _log(f"  [Warn] chunks_size 非法({chunk_size})，回退使用 1000")
        chunk_size = 1000

    real_total_frames = 0
    for i, ep_meta in enumerate(jsonl_lines):
        ep_idx = ep_meta.get("episode_index", None)
        expected_len = ep_meta.get("length", None)
        if not isinstance(ep_idx, int) or ep_idx < 0:
            _log(f"  [Error] episodes.jsonl 第 {i + 1} 行 episode_index 非法: {ep_idx}")
            unsafe_for_total_frames = True
            continue
        if not isinstance(expected_len, int) or expected_len < 0:
            _log(f"  [Warn] episodes.jsonl 第 {i + 1} 行 length 非法: {expected_len}，将按 parquet 真实长度修正")

        chunk_id = ep_idx // chunk_size
        parquet_path = data_dir / f"chunk-{chunk_id:03d}" / f"episode_{ep_idx:06d}.parquet"
        if not parquet_path.exists():
            _log(f"  [Error] Ep {ep_idx} 缺失 parquet: {parquet_path}")
            unsafe_for_total_frames = True
            continue

        try:
            df = pd.read_parquet(parquet_path)
            actual_len = len(df)
        except Exception as e:
            _log(f"  [Error] 读取 parquet 失败 ({parquet_path}): {e}")
            unsafe_for_total_frames = True
            continue

        if expected_len != actual_len:
            _print_action(
                apply,
                f"[Fix-9] Ep {ep_idx} length 修正: {expected_len} -> {actual_len}",
            )
            if apply:
                jsonl_lines[i]["length"] = actual_len
            jsonl_modified = True

        real_total_frames += actual_len

    if unsafe_for_total_frames:
        _log("  [Warn] [Fix-8] 存在缺失/读取失败的 parquet，跳过 total_frames 自动修正。")
    else:
        old_total_frames = info.get("total_frames", 0)
        if old_total_frames != real_total_frames:
            _print_action(
                apply,
                f"[Fix-8] 修正 total_frames: {old_total_frames} -> {real_total_frames}",
            )
            if apply:
                info["total_frames"] = real_total_frames
            info_modified = True

    # ---------------------------------------------------------
    # [规则 3] 修正 train split
    # ---------------------------------------------------------
    expected_split = f"0:{target_episodes}"
    splits = info.get("splits", {})
    if not isinstance(splits, dict):
        splits = {}
    current_split = splits.get("train", "0:0")
    if current_split != expected_split:
        _print_action(apply, f"[Fix-3] 修正 splits.train: {current_split} -> {expected_split}")
        if apply:
            info.setdefault("splits", {})
            info["splits"]["train"] = expected_split
        info_modified = True

    # ---------------------------------------------------------
    # [规则 4] 修正 total_videos（按动态 video_keys 计算）
    # ---------------------------------------------------------
    expected_videos = target_episodes * len(video_keys)
    current_videos = info.get("total_videos", 0)
    if current_videos != expected_videos:
        _print_action(
            apply,
            f"[Fix-4] 修正 total_videos: {current_videos} -> {expected_videos}",
        )
        if apply:
            info["total_videos"] = expected_videos
        info_modified = True

    # ---------------------------------------------------------
    # 保存更改（带备份 + 原子写）
    # ---------------------------------------------------------
    if apply:
        if jsonl_modified:
            if backup:
                bak = _backup_file(episodes_path)
                _log(f"  [Backup] episodes 备份到: {bak}")
            content = "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in jsonl_lines)
            _atomic_write_text(episodes_path, content)
            _log("  [Write] 已保存 meta/episodes.jsonl")

        if info_modified:
            if backup:
                bak = _backup_file(info_path)
                _log(f"  [Backup] info 备份到: {bak}")
            content = json.dumps(info, indent=4, ensure_ascii=False) + "\n"
            _atomic_write_text(info_path, content)
            _log("  [Write] 已保存 meta/info.json")
    else:
        if has_fix_actions or jsonl_modified or info_modified:
            _log("  [DryRun] 检查完成，存在可修复项；使用 --apply 以执行修复。")
        else:
            _log("  [DryRun] 检查完成，未发现需要修复的项。")

    _log("  -> 完成。")


def main() -> None:
    parser = argparse.ArgumentParser(description="自动检测并修复 Dataset 完整性问题（默认 dry-run）")
    parser.add_argument("root_dir", type=str, help="包含多个 Dataset 子文件夹的根目录路径")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="执行真实修改（默认仅 dry-run）",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="执行修改时不生成 .bak 备份（默认会备份）",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="保存日志到指定文件（可选）",
    )
    args = parser.parse_args()
    
    # 初始化日志文件
    global _log_file
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _log_file = open(log_path, 'w', encoding='utf-8')
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _log_file.write(f"=== Auto Fix Dataset Log ===\n")
        _log_file.write(f"Time: {timestamp}\n")
        _log_file.write(f"Command: {' '.join(sys.argv)}\n")
        _log_file.write(f"=" * 60 + "\n\n")

    root_path = Path(args.root_dir)
    if not root_path.exists():
        _log(f"错误: 路径 '{root_path}' 不存在")
        if _log_file:
            _log_file.close()
        sys.exit(1)
    if not root_path.is_dir():
        _log(f"错误: 路径 '{root_path}' 不是目录")
        if _log_file:
            _log_file.close()
        sys.exit(1)

    mode = "APPLY(真实写入)" if args.apply else "DRY-RUN(仅检查)"
    backup = not args.no_backup

    # 检查是否是单个数据集目录（包含 meta/info.json）
    meta_info = root_path / "meta" / "info.json"
    if meta_info.exists():
        # 这是一个单独的数据集目录
        _log(f"检测到单个数据集目录: {root_path.name}")
        _log(f"模式: {mode}, backup: {backup}")
        fix_single_dataset(root_path, apply=args.apply, backup=backup)
    else:
        # 这是包含多个数据集的父目录
        subdirs = sorted([d for d in root_path.iterdir() if d.is_dir()])
        if not subdirs:
            _log(f"在 '{root_path}' 下没有找到子目录。")
            if _log_file:
                _log_file.close()
            sys.exit(0)

        _log(f"找到 {len(subdirs)} 个潜在数据集，模式: {mode}, backup: {backup}")

        for d in subdirs:
            fix_single_dataset(d, apply=args.apply, backup=backup)

    _log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    _log("所有任务执行完毕。")
    
    # 关闭日志文件
    if _log_file:
        _log_file.write(f"\n{'=' * 60}\n")
        _log_file.write(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        _log_file.close()


if __name__ == "__main__":
    main()