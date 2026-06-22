#!/usr/bin/env python3
"""
孤儿批次恢复工具

扫描数据目录，发现未在batch_registry.json中注册的数据集，
提供交互式恢复选项。

使用场景：
1. 录制过程中程序崩溃/断电
2. 手动终止录制（Ctrl+C）但batch未记录
3. 系统异常导致batch注册失败
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.common.data_tracker import BatchInfo, BatchTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def scan_data_directory(data_root: Path) -> List[Path]:
    """
    扫描数据目录，查找所有潜在的dataset目录
    
    Returns:
        List[Path]: dataset目录列表
    """
    datasets = []
    
    # 递归查找包含meta/info.json的目录
    for root, dirs, files in os.walk(data_root):
        root_path = Path(root)
        
        # 检查是否是dataset目录（包含meta/info.json）
        info_file = root_path / "meta" / "info.json"
        if info_file.exists():
            datasets.append(root_path)
    
    return datasets


def load_dataset_info(dataset_path: Path) -> Optional[Dict]:
    """加载dataset的info.json"""
    info_file = dataset_path / "meta" / "info.json"
    try:
        with open(info_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load info.json from {dataset_path}: {e}")
        return None


def get_dataset_creation_time(dataset_path: Path) -> Optional[float]:
    """获取dataset的创建时间"""
    info_file = dataset_path / "meta" / "info.json"
    if info_file.exists():
        return info_file.stat().st_ctime
    return None


def extract_metadata_from_path(dataset_path: Path) -> Dict[str, str]:
    """
    从路径中提取元数据
    
    典型路径结构:
    /data_root/{tag}/{task_name}/{repo_id}/
    """
    parts = dataset_path.parts
    metadata = {
        "task_name": "unknown",
        "tag": "unknown",
        "repo_id": dataset_path.name,
    }
    
    if len(parts) >= 3:
        metadata["tag"] = parts[-3]
        
        if len(parts) >= 2:
            metadata["task_name"] = parts[-2]
    
    return metadata


def find_orphan_batches(data_root: Path, tracker: BatchTracker) -> List[Tuple[Path, Dict]]:
    """
    查找孤儿batch（存在于磁盘但未在registry中注册）
    
    Returns:
        List[Tuple[Path, Dict]]: (dataset_path, info_dict)
    """
    logger.info(f"Scanning data directory: {data_root}")
    
    # 扫描所有dataset
    all_datasets = scan_data_directory(data_root)
    logger.info(f"Found {len(all_datasets)} dataset(s)")
    
    # 获取所有已注册的batch
    registered_batches = tracker.get_all_batches()
    registered_paths = {Path(batch.local_path) for batch in registered_batches}
    
    # 找出未注册的
    orphan_batches = []
    for dataset_path in all_datasets:
        if dataset_path not in registered_paths:
            info = load_dataset_info(dataset_path)
            if info:
                orphan_batches.append((dataset_path, info))
    
    logger.info(f"Found {len(orphan_batches)} orphan batch(es)")
    return orphan_batches


def display_orphan_batch(index: int, dataset_path: Path, info: Dict) -> None:
    """显示孤儿batch的详细信息"""
    print(f"\n{'='*70}")
    print(f"孤儿Batch #{index + 1}")
    print(f"{'='*70}")
    print(f"路径: {dataset_path}")
    print(f"Episodes: {info.get('total_episodes', 'N/A')}")
    print(f"Frames: {info.get('total_frames', 'N/A')}")
    print(f"Videos: {info.get('total_videos', 'N/A')}")
    print(f"数据大小: {get_directory_size(dataset_path):.2f} GB")
    
    # 获取创建时间
    ctime = get_dataset_creation_time(dataset_path)
    if ctime:
        print(f"创建时间: {datetime.fromtimestamp(ctime).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 从路径提取元数据
    metadata = extract_metadata_from_path(dataset_path)
    print(f"推断任务: {metadata['task_name']}")
    print(f"推断Tag: {metadata['tag']}")
    print(f"Repo ID: {metadata['repo_id']}")


def get_directory_size(path: Path) -> float:
    """计算目录大小（GB）"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except Exception as e:
        logger.warning(f"Failed to calculate size for {path}: {e}")
    return total_size / (1024**3)


def interactive_recover(orphan_batches: List[Tuple[Path, Dict]], tracker: BatchTracker) -> None:
    """交互式恢复孤儿batch"""
    if not orphan_batches:
        print("\n✅ 未发现孤儿batch，所有数据已正确注册！")
        return
    
    print(f"\n发现 {len(orphan_batches)} 个孤儿batch（未注册到上传系统）")
    print("\n这些数据存在于磁盘但未被上传系统追踪，可能原因：")
    print("  - 录制过程中程序崩溃或被中断（Ctrl+C）")
    print("  - 断电或系统异常")
    print("  - batch注册失败")
    
    print("\n" + "="*70)
    print("恢复选项：")
    print("="*70)
    print("1. 逐个查看并决定是否恢复")
    print("2. 恢复所有孤儿batch")
    print("3. 退出（不恢复）")
    
    choice = input("\n请选择 (1/2/3): ").strip()
    
    if choice == "1":
        recover_interactive(orphan_batches, tracker)
    elif choice == "2":
        recover_all(orphan_batches, tracker)
    else:
        print("退出，未恢复任何数据")


def recover_interactive(orphan_batches: List[Tuple[Path, Dict]], tracker: BatchTracker) -> None:
    """逐个处理孤儿batch"""
    recovered_count = 0
    skipped_count = 0
    
    for i, (dataset_path, info) in enumerate(orphan_batches):
        display_orphan_batch(i, dataset_path, info)
        
        print("\n操作选项:")
        print("  r - 恢复（注册到上传系统）")
        print("  s - 跳过")
        print("  d - 删除（危险！）")
        print("  q - 退出")
        
        action = input("请选择 (r/s/d/q): ").strip().lower()
        
        if action == "q":
            break
        elif action == "r":
            if recover_batch(dataset_path, info, tracker):
                recovered_count += 1
                print("✅ 已恢复")
            else:
                print("❌ 恢复失败")
        elif action == "d":
            confirm = input("⚠️  确认删除？(yes/no): ").strip().lower()
            if confirm == "yes":
                try:
                    import shutil
                    shutil.rmtree(dataset_path)
                    print("✅ 已删除")
                except Exception as e:
                    print(f"❌ 删除失败: {e}")
            else:
                print("取消删除")
                skipped_count += 1
        else:
            print("跳过")
            skipped_count += 1
    
    print(f"\n处理完成！")
    print(f"  恢复: {recovered_count}")
    print(f"  跳过: {skipped_count}")


def recover_all(orphan_batches: List[Tuple[Path, Dict]], tracker: BatchTracker) -> None:
    """恢复所有孤儿batch"""
    print(f"\n开始恢复所有 {len(orphan_batches)} 个孤儿batch...")
    
    recovered_count = 0
    failed_count = 0
    
    for i, (dataset_path, info) in enumerate(orphan_batches):
        print(f"\n[{i+1}/{len(orphan_batches)}] 恢复: {dataset_path.name}")
        
        if recover_batch(dataset_path, info, tracker):
            recovered_count += 1
            print("  ✅ 成功")
        else:
            failed_count += 1
            print("  ❌ 失败")
    
    print(f"\n恢复完成！")
    print(f"  成功: {recovered_count}")
    print(f"  失败: {failed_count}")


def recover_batch(dataset_path: Path, info: Dict, tracker: BatchTracker) -> bool:
    """
    恢复单个batch
    
    Returns:
        bool: 是否成功
    """
    try:
        # 从路径提取元数据
        metadata = extract_metadata_from_path(dataset_path)
        
        # 从info.json提取信息
        total_episodes = info.get("total_episodes", 0)
        
        # 推断robot信息（从路径或其他来源）
        robot_type = "arxx5_bimanual"  # 默认值，可以改进
        robot_id = "arxx5_bimanual"
        
        # 获取创建时间作为start_time
        ctime = get_dataset_creation_time(dataset_path)
        if not ctime:
            ctime = datetime.now().timestamp()
        
        # 记录batch
        batch_info = tracker.record_batch(
            robot_type=robot_type,
            robot_id=robot_id,
            repo_id=f"{metadata['task_name']}/{metadata['repo_id']}",
            task_name=metadata['task_name'],
            dataset_path=dataset_path,
            num_episodes=total_episodes,
            start_time=ctime,
            end_time=ctime + 60,  # 假设录制了1分钟
        )
        
        logger.info(f"Recovered batch: {batch_info.batch_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to recover batch {dataset_path}: {e}", exc_info=True)
        return False


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        description="孤儿批次恢复工具 - 扫描并恢复未注册的数据集"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="数据根目录（对应 upload_config.yaml 的 data.root）",
    )
    parser.add_argument(
        "--auto-recover",
        action="store_true",
        help="自动恢复所有孤儿batch（不交互）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅扫描不恢复",
    )
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"❌ 数据目录不存在: {data_root}")
        sys.exit(1)
    
    # 初始化追踪器
    tracker = BatchTracker(data_root)
    
    # 扫描孤儿batch
    orphan_batches = find_orphan_batches(data_root, tracker)
    
    if args.dry_run:
        # 仅显示信息
        if orphan_batches:
            print(f"\n发现 {len(orphan_batches)} 个孤儿batch：")
            for i, (dataset_path, info) in enumerate(orphan_batches):
                display_orphan_batch(i, dataset_path, info)
        else:
            print("\n✅ 未发现孤儿batch")
        return
    
    # 恢复
    if args.auto_recover:
        recover_all(orphan_batches, tracker)
    else:
        interactive_recover(orphan_batches, tracker)


if __name__ == "__main__":
    main()
