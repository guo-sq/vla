#!/usr/bin/env python3
"""
根据 config 文件中实际使用的数据(REPO_ID_WITH_WEIGHT 中未被注释的条目)统计数据集各类的数据条数和总时长。

使用方法:
    python3 scripts/analyze_config_dataset_stats.py <config_file>

示例:
    python3 scripts/analyze_config_dataset_stats.py src/openpi/configs/cfg_pi05_base_finetune_box_recap_pt_0322_hc.py
"""

from collections import OrderedDict
from collections import defaultdict
import json
from pathlib import Path
import re
import sys


def parse_active_repos(config_path):
    """
    从 config 文件中解析出未被注释的 REPO_ID_WITH_WEIGHT 条目。

    Returns:
        list[tuple]: [(repo_path, weight_list_str), ...]
    """
    with open(config_path) as f:
        content = f.read()

    # 提取 ROOT_DIR
    root_dir_match = re.search(r'ROOT_DIR\s*=\s*["\'](.+?)["\']', content)
    if not root_dir_match:
        print("ERROR: 无法从配置文件中解析 ROOT_DIR")
        sys.exit(1)
    root_dir = root_dir_match.group(1)

    # 提取 REPO_ID_WITH_WEIGHT 列表中所有未被注释的条目
    # 匹配形如: ("fold_box_from_scratch/...", [...]),
    active_repos = []
    in_list = False
    for line in content.split("\n"):
        stripped = line.strip()
        if "REPO_ID_WITH_WEIGHT" in stripped and "=" in stripped:
            in_list = True
            continue
        if in_list:
            if stripped == "]":
                break
            # 跳过注释行
            if stripped.startswith("#"):
                continue
            # 匹配 ("path", [...]),
            match = re.search(r'\(\s*["\'](.+?)["\']\s*,', stripped)
            if match:
                active_repos.append(match.group(1))

    return root_dir, active_repos


COLOR_KEYWORDS = ["green", "purple", "silver", "yellow"]


def is_colored(repo_path):
    """
    判断 repo 路径是否为彩色数据集。
    彩色数据集路径中包含颜色关键词,如 green, purple, silver, yellow。
    """
    filename = repo_path.split("/")[-1].lower()
    return any(color in filename for color in COLOR_KEYWORDS)


def classify_repo(repo_path):
    """
    根据 repo 路径判断所属类别。
    repo_path 形如 "fold_box_from_scratch/total_steps/fold_box_scratch.all.102s.20260303.batch.1"
    """
    parts = repo_path.split("/")
    if len(parts) >= 2:
        return parts[1]  # e.g. "total_steps", "recover2-8", etc.
    return "unknown"


def analyze_config_dataset(root_dir, active_repos):
    """
    统计 config 中实际使用的数据集的各类数据条数和总时长。
    """
    # 四大类映射
    four_categories = OrderedDict(
        [
            ("全流程类", ["total_steps", "first_half_steps", "second_half_steps"]),
            ("错误恢复类", ["recover2-8", "recover9", "recover31", "recover37"]),
            ("局部动作强化类", ["step_13_14", "step_24_25"]),
            ("bad的数据类", ["bad"]),
        ]
    )

    # 所有子类别(按顺序)
    all_sub_categories = []
    for subs in four_categories.values():
        all_sub_categories.extend(subs)

    # 按类别统计
    stats = defaultdict(
        lambda: {
            "episode_count": 0,
            "total_frames": 0,
            "batch_count": 0,
            "missing_batches": [],
            "color_episode_count": 0,
            "color_total_frames": 0,
            "color_batch_count": 0,
        }
    )

    for repo_path in active_repos:
        category = classify_repo(repo_path)
        batch_dir = Path(root_dir) / repo_path
        info_file = batch_dir / "meta" / "info.json"
        colored = is_colored(repo_path)

        stats[category]["batch_count"] += 1
        if colored:
            stats[category]["color_batch_count"] += 1

        if info_file.exists():
            with open(info_file) as f:
                info = json.load(f)
                episodes = info.get("total_episodes", 0)
                frames = info.get("total_frames", 0)
                stats[category]["episode_count"] += episodes
                stats[category]["total_frames"] += frames
                if colored:
                    stats[category]["color_episode_count"] += episodes
                    stats[category]["color_total_frames"] += frames
        else:
            stats[category]["missing_batches"].append(repo_path)

    # 计算时长
    fps = 30
    result_stats = {}
    for cat in all_sub_categories:
        s = stats[cat]
        total_duration_s = s["total_frames"] / fps if fps > 0 else 0
        total_duration_min = total_duration_s / 60
        color_duration_s = s["color_total_frames"] / fps if fps > 0 else 0
        color_duration_min = color_duration_s / 60
        result_stats[cat] = {
            "batch_count": s["batch_count"],
            "episode_count": s["episode_count"],
            "total_frames": s["total_frames"],
            "total_duration_s": total_duration_s,
            "total_duration_min": total_duration_min,
            "missing_batches": s["missing_batches"],
            "color_batch_count": s["color_batch_count"],
            "color_episode_count": s["color_episode_count"],
            "color_total_frames": s["color_total_frames"],
            "color_duration_s": color_duration_s,
            "color_duration_min": color_duration_min,
        }

    # 计算四大类统计
    category_stats = OrderedDict()
    for big_cat, sub_cats in four_categories.items():
        episode_count = sum(result_stats[c]["episode_count"] for c in sub_cats)
        total_frames = sum(result_stats[c]["total_frames"] for c in sub_cats)
        batch_count = sum(result_stats[c]["batch_count"] for c in sub_cats)
        total_duration_s = sum(result_stats[c]["total_duration_s"] for c in sub_cats)
        total_duration_min = sum(result_stats[c]["total_duration_min"] for c in sub_cats)
        color_batch_count = sum(result_stats[c]["color_batch_count"] for c in sub_cats)
        color_episode_count = sum(result_stats[c]["color_episode_count"] for c in sub_cats)
        color_total_frames = sum(result_stats[c]["color_total_frames"] for c in sub_cats)
        color_duration_s = sum(result_stats[c]["color_duration_s"] for c in sub_cats)
        color_duration_min = sum(result_stats[c]["color_duration_min"] for c in sub_cats)
        category_stats[big_cat] = {
            "batch_count": batch_count,
            "episode_count": episode_count,
            "total_frames": total_frames,
            "total_duration_s": total_duration_s,
            "total_duration_min": total_duration_min,
            "sub_categories": sub_cats,
            "color_batch_count": color_batch_count,
            "color_episode_count": color_episode_count,
            "color_total_frames": color_total_frames,
            "color_duration_s": color_duration_s,
            "color_duration_min": color_duration_min,
        }

    return result_stats, category_stats


def print_stats(config_name, stats, category_stats):
    """打印统计结果"""
    # 计算总计
    total_batch = 0
    total_episode = 0
    total_frames = 0
    total_duration_s = 0

    for cat_stat in category_stats.values():
        total_batch += cat_stat["batch_count"]
        total_episode += cat_stat["episode_count"]
        total_frames += cat_stat["total_frames"]
        total_duration_s += cat_stat["total_duration_s"]

    total_duration_min = total_duration_s / 60 if total_duration_s > 0 else 0

    # 打印子类别统计
    print("=" * 115)
    print(f"Config 数据集统计(按子类别) - {config_name}")
    print("=" * 115)
    print(f"{'类别':<22}  {'batch数':<8}  {'episode数':<10}  {'总帧数':<12}  {'总时长(秒)':<14}  {'总时长(分钟)':<14}")
    print("-" * 115)

    all_sub_cats = []
    for subs in [
        ["total_steps", "first_half_steps", "second_half_steps"],
        ["recover2-8", "recover9", "recover31", "recover37"],
        ["step_13_14", "step_24_25"],
        ["bad"],
    ]:
        all_sub_cats.extend(subs)

    for cat in all_sub_cats:
        s = stats[cat]
        print(
            f"{cat:<22}  {s['batch_count']:<8}  {s['episode_count']:<10}  {s['total_frames']:<12}  "
            f"{s['total_duration_s']:<14.2f}  {s['total_duration_min']:<14.2f}"
        )

    print("-" * 115)
    print(
        f"{'总计':<22}  {total_batch:<8}  {total_episode:<10}  {total_frames:<12}  "
        f"{total_duration_s:<14.2f}  {total_duration_min:<14.2f}"
    )
    print("=" * 115)
    print()

    # 打印四大类统计和占比
    print("=" * 120)
    print(f"Config 数据集统计(按四大类,以时间为占比) - {config_name}")
    print("=" * 120)
    print(f"{'大类':<18}  {'batch数':<8}  {'episode数':<10}  {'总帧数':<12}  {'总时长(分钟)':<14}  {'时间占比':<10}")
    print("-" * 120)

    for big_cat, cat_stat in category_stats.items():
        duration_ratio = (cat_stat["total_duration_min"] / total_duration_min * 100) if total_duration_min > 0 else 0
        print(
            f"{big_cat:<18}  {cat_stat['batch_count']:<8}  {cat_stat['episode_count']:<10}  "
            f"{cat_stat['total_frames']:<12}  {cat_stat['total_duration_min']:<14.2f}  {duration_ratio:<10.2f}%"
        )

        # 打印子类别
        for sub_cat in cat_stat["sub_categories"]:
            sub_stat = stats[sub_cat]
            sub_ratio = (sub_stat["total_duration_min"] / total_duration_min * 100) if total_duration_min > 0 else 0
            print(
                f"  └─ {sub_cat:<14}  {sub_stat['batch_count']:<8}  {sub_stat['episode_count']:<10}  "
                f"{sub_stat['total_frames']:<12}  {sub_stat['total_duration_min']:<14.2f}  {sub_ratio:<10.2f}%"
            )

    print("-" * 120)
    print(
        f"{'总计':<18}  {total_batch:<8}  {total_episode:<10}  {total_frames:<12}  "
        f"{total_duration_min:<14.2f}  100.00%"
    )
    print("=" * 120)

    # 打印彩色数据统计
    print()
    print("=" * 130)
    print(f"Config 数据集统计(彩色数据占比) - {config_name}")
    print("=" * 130)
    print(
        f"{'大类':<18}  {'总batch':<8}  {'彩色batch':<10}  {'总episode':<10}  {'彩色episode':<12}  "
        f"{'彩色episode占比':<16}  {'总时长(分钟)':<14}  {'彩色时长(分钟)':<14}  {'彩色时长占比':<12}"
    )
    print("-" * 130)

    for big_cat, cat_stat in category_stats.items():
        ep_ratio = (
            (cat_stat["color_episode_count"] / cat_stat["episode_count"] * 100) if cat_stat["episode_count"] > 0 else 0
        )
        dur_ratio = (
            (cat_stat["color_duration_min"] / cat_stat["total_duration_min"] * 100)
            if cat_stat["total_duration_min"] > 0
            else 0
        )
        print(
            f"{big_cat:<18}  {cat_stat['batch_count']:<8}  {cat_stat['color_batch_count']:<10}  "
            f"{cat_stat['episode_count']:<10}  {cat_stat['color_episode_count']:<12}  {ep_ratio:<16.2f}%  "
            f"{cat_stat['total_duration_min']:<14.2f}  {cat_stat['color_duration_min']:<14.2f}  {dur_ratio:<12.2f}%"
        )

        # 打印子类别
        for sub_cat in cat_stat["sub_categories"]:
            sub_stat = stats[sub_cat]
            sub_ep_ratio = (
                (sub_stat["color_episode_count"] / sub_stat["episode_count"] * 100)
                if sub_stat["episode_count"] > 0
                else 0
            )
            sub_dur_ratio = (
                (sub_stat["color_duration_min"] / sub_stat["total_duration_min"] * 100)
                if sub_stat["total_duration_min"] > 0
                else 0
            )
            print(
                f"  └─ {sub_cat:<14}  {sub_stat['batch_count']:<8}  {sub_stat['color_batch_count']:<10}  "
                f"{sub_stat['episode_count']:<10}  {sub_stat['color_episode_count']:<12}  {sub_ep_ratio:<16.2f}%  "
                f"{sub_stat['total_duration_min']:<14.2f}  {sub_stat['color_duration_min']:<14.2f}  {sub_dur_ratio:<12.2f}%"
            )

    print("-" * 130)
    total_color_ep = sum(c["color_episode_count"] for c in category_stats.values())
    total_color_dur = sum(c["color_duration_min"] for c in category_stats.values())
    total_color_batch = sum(c["color_batch_count"] for c in category_stats.values())
    total_ep_ratio = (total_color_ep / total_episode * 100) if total_episode > 0 else 0
    total_dur_ratio = (total_color_dur / total_duration_min * 100) if total_duration_min > 0 else 0
    print(
        f"{'总计':<18}  {total_batch:<8}  {total_color_batch:<10}  "
        f"{total_episode:<10}  {total_color_ep:<12}  {total_ep_ratio:<16.2f}%  "
        f"{total_duration_min:<14.2f}  {total_color_dur:<14.2f}  {total_dur_ratio:<12.2f}%"
    )
    print("=" * 130)

    # 打印缺失的 batch
    missing_any = False
    for cat in all_sub_cats:
        if stats[cat]["missing_batches"]:
            if not missing_any:
                print()
                print("⚠ 以下 batch 的 info.json 不存在(未计入统计):")
                missing_any = True
            for mb in stats[cat]["missing_batches"]:
                print(f"  - {mb}")


def main():
    if len(sys.argv) < 2:
        print("用法: python3 scripts/analyze_config_dataset_stats.py <config_file>")
        print(
            "示例: python3 scripts/analyze_config_dataset_stats.py src/openpi/configs/cfg_pi05_base_finetune_box_recap_pt_0322_hc.py"
        )
        sys.exit(1)

    config_path = sys.argv[1]
    config_name = Path(config_path).stem

    print(f"配置文件: {config_path}")

    root_dir, active_repos = parse_active_repos(config_path)
    print(f"数据根目录: {root_dir}")
    print(f"活跃数据集条目数: {len(active_repos)}")
    print()

    stats, category_stats = analyze_config_dataset(root_dir, active_repos)
    print_stats(config_name, stats, category_stats)


if __name__ == "__main__":
    main()
