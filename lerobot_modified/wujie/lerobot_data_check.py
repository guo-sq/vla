import json
import os
import sys
from pathlib import Path

import pandas as pd


def check_dataset(dataset_root):
    dataset_path = Path(dataset_root)
    print(f"\r正在检查数据集: {dataset_path.name} ...", end="", flush=True)

    # 用于收集当前数据集的所有错误信息
    error_logs = []

    # 路径定义
    info_path = dataset_path / "meta/info.json"
    episodes_path = dataset_path / "meta/episodes.jsonl"
    data_dir = dataset_path / "data"
    videos_dir = dataset_path / "videos"

    # 0. 基础文件检查
    if not info_path.exists() or not episodes_path.exists():
        msg = f"  [Error] 缺少 meta 信息文件 (info.json 或 episodes.jsonl)"
        print(msg)
        error_logs.append(msg)
        return error_logs  # 缺少关键文件无法继续检查，直接返回

    try:
        with open(info_path, "r") as f:
            info = json.load(f)
    except Exception as e:
        msg = f"  [Error] 读取 info.json 失败: {e}"
        print(msg)
        error_logs.append(msg)
        return error_logs

    # ================= 功能一：检查 Episode 和 Video 数量一致性 =================

    # 1.1 检查 splits 与 total_episodes
    try:
        train_split = info.get("splits", {}).get("train", "0:0")
        start_idx, end_idx = map(int, train_split.split(":"))

        if info["total_episodes"] != end_idx:
            msg = f"  [Fail] info.json total_episodes ({info['total_episodes']}) != splits 结束索引 ({end_idx})"
            # print(msg)
            error_logs.append(msg)
    except Exception as e:
        msg = f"  [Fail] 解析 splits 字段失败: {e}"
        print(msg)
        error_logs.append(msg)

    # 1.2 检查 total_videos (应为 3 倍)
    expected_videos = info.get("total_episodes", 0) * 3
    if info.get("total_videos", 0) != expected_videos:
        msg = f"  [Fail] info.json total_videos ({info.get('total_videos')}) 不是 total_episodes 的3倍"
        # print(msg)
        error_logs.append(msg)

    # 1.3 统计各处实际数量
    try:
        with open(episodes_path, "r") as f:
            ep_lines = [json.loads(line) for line in f]
        count_jsonl = len(ep_lines)
    except Exception as e:
        msg = f"  [Error] 读取 episodes.jsonl 失败: {e}"
        # print(msg)
        error_logs.append(msg)
        return error_logs  # 无法继续后续检查

    count_parquet = len(list(data_dir.glob("chunk-*/episode_*.parquet")))

    cameras = [
        "observation.images.head",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    ]
    video_counts = {}
    for cam in cameras:
        video_counts[cam] = len(list(videos_dir.glob(f"chunk-*/{cam}/episode_*.mp4")))

    # 1.4 统一比对
    target = info.get("total_episodes", 0)

    checks = {
        "episodes.jsonl 行数": count_jsonl,
        "Parquet 文件数量": count_parquet,
    }
    checks.update({f"视频流 {k}": v for k, v in video_counts.items()})

    for name, val in checks.items():
        if val != target:
            msg = f"  [Fail] {name} ({val}) 与 total_episodes ({target}) 不一致"
            # print(msg)
            error_logs.append(msg)

    # ================= 功能二：检查 Frame 数量一致性 =================

    total_frames_accumulated = 0
    chunk_size = info.get("chunks_size", 1000)

    # 简单提示，不需要append到error_logs
    # print("  正在校验 Frame 数量 (读取 Parquet)...")

    for ep in ep_lines:
        ep_idx = ep["episode_index"]
        expected_len = ep["length"]

        chunk_id = ep_idx // chunk_size
        parquet_file = (
            data_dir / f"chunk-{chunk_id:03d}" / f"episode_{ep_idx:06d}.parquet"
        )

        if not parquet_file.exists():
            msg = f"  [Error] Ep {ep_idx}: 找不到文件 {parquet_file}"
            # print(msg)
            error_logs.append(msg)
            continue

        try:
            df = pd.read_parquet(parquet_file, columns=None)
            actual_len = len(df)

            if actual_len != expected_len:
                msg = f"  [Fail] Ep {ep_idx}: Jsonl长度 ({expected_len}) != Parquet行数 ({actual_len})"
                # print(msg)
                error_logs.append(msg)

            total_frames_accumulated += expected_len

        except Exception as e:
            msg = f"  [Error] 无法读取 {parquet_file.name}: {e}"
            print(msg)
            error_logs.append(msg)

    # 检查总帧数
    if total_frames_accumulated != info.get("total_frames", 0):
        msg = f"  [Fail] 累加总帧数 ({total_frames_accumulated}) != info.json total_frames ({info.get('total_frames')})"
        # print(msg)
        error_logs.append(msg)

    # 如果没有错误，打印Pass
    # if not error_logs:
    #     print(f"  [Pass] 检查通过")

    # print("-" * 50)

    # 返回收集到的错误列表
    return error_logs


def run_checks(root_path: str = ".") -> bool:
    """
    扫描指定目录下的所有数据集并进行检查。

    Args:
        root_path: 如果传入路径，则从该路径开始递归查找。

    Returns:
        bool: 如果所有数据集检查通过返回 True，否则（或未找到数据集）返回 False。
    """
    root_dir = Path(root_path)

    if not root_dir.exists():
        print(f"目录不存在: {root_dir}")
        return False

    # 递归查找所有包含 meta/info.json 的目录作为数据集根目录
    datasets = []
    # 使用 rglob 查找所有的 info.json
    for info_file in root_dir.rglob("meta/info.json"):
        # info.json 的上级目录是 meta，再上级是数据集根目录
        dataset_path = info_file.parent.parent
        if dataset_path.is_dir():
            datasets.append(dataset_path)

    # 去重并排序
    datasets = sorted(list(set(datasets)))

    if not datasets:
        print(f"在 {root_dir} 下未找到任何 LeRobot 数据集 (未发现 meta/info.json)")
        return False

    # 用于存储所有数据集的检查结果报告 {dataset_name: [error_list]}
    summary_report = {}

    print(f"找到 {len(datasets)} 个数据集，开始检查...")

    for d in datasets:
        errors = check_dataset(d)
        if errors:
            # 尝试使用相对于 root_dir 的路径以区分同名文件夹
            try:
                name = str(d.relative_to(root_dir))
            except ValueError:
                name = d.name
            summary_report[name] = errors

    # ================= 打印最终汇总报告 =================

    if not summary_report:
        print("\n\n恭喜！所有数据集检查均通过，未发现异常。")
        return True
    else:
        print("\n\n" + "=" * 20 + " 异常数据汇总报告 " + "=" * 20)
        print(f"\n共发现 {len(summary_report)} 个数据集存在问题：\n")
        for name, errs in summary_report.items():
            print(f"DATASET: {name}")
            for e in errs:
                # 为了排版整齐，去掉之前print时的缩进空格
                print(f"  {e.strip()}")
            print("-" * 30)

        print("\n检查结束。")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        target_dir = "."
    else:
        target_dir = sys.argv[1]

    success = run_checks(target_dir)

    if success:
        print("True")
        sys.exit(0)
    else:
        print("False")
        sys.exit(1)
