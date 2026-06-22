#!/usr/bin/env python3
"""
重新计算batch_registry.json中所有batch的时长信息（独立脚本）
"""

import json
import fcntl
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_episode_durations(dataset_path: Path) -> tuple:
    """计算episode的实际时长（基于frame数和fps）"""
    try:
        # 读取fps
        info_path = dataset_path / "meta" / "info.json"
        if not info_path.exists():
            logger.warning(f"info.json not found in {dataset_path}")
            return 0.0, 0.0

        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)

        fps = info.get("fps", 30)
        if not isinstance(fps, (int, float)) or fps <= 0:
            fps = 30

        # 查找所有episode parquet文件
        data_dir = dataset_path / "data"
        if not data_dir.exists():
            return 0.0, 0.0

        episode_files = sorted(data_dir.rglob("episode_*.parquet"))
        if not episode_files:
            return 0.0, 0.0

        # 计算每个episode的时长
        episode_durations = []
        for ep_file in episode_files:
            try:
                df = pd.read_parquet(ep_file)
                num_frames = len(df)
                duration_s = num_frames / fps
                episode_durations.append(duration_s)
            except Exception as e:
                logger.warning(f"Failed to read {ep_file}: {e}")
                continue

        if not episode_durations:
            return 0.0, 0.0

        # 计算统计值
        avg_duration_s = sum(episode_durations) / len(episode_durations)
        total_duration_s = sum(episode_durations)
        total_duration_min = total_duration_s / 60

        return avg_duration_s, total_duration_min

    except Exception as e:
        logger.error(f"Error calculating durations for {dataset_path}: {e}")
        return 0.0, 0.0


def update_registry(registry_path: Path):
    """更新registry文件"""
    with open(registry_path, "r+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            data = json.load(f)
            
            updated_count = 0
            skipped_count = 0
            
            # 遍历所有batch
            def traverse(node, path=""):
                nonlocal updated_count, skipped_count
                
                if isinstance(node, dict):
                    # 检查是否是batch节点（含有local_path字段）
                    if "local_path" in node and "batch_id" in node:
                        batch_id = node["batch_id"]
                        local_path = Path(node["local_path"])
                        
                        if not local_path.exists():
                            logger.warning(f"路径不存在，跳过: {batch_id}")
                            skipped_count += 1
                            return
                        
                        # 计算新的时长
                        avg_duration_s, total_duration_min = calculate_episode_durations(local_path)
                        
                        if total_duration_min == 0.0:
                            logger.warning(f"无法计算时长，跳过: {batch_id}")
                            skipped_count += 1
                            return
                        
                        # 更新
                        old_avg = node.get("avg_duration_s", 0.0)
                        old_total = node.get("total_duration_min", 0.0)
                        
                        if abs(old_avg - avg_duration_s) < 0.01 and abs(old_total - total_duration_min) < 0.01:
                            logger.debug(f"时长未变化，跳过: {batch_id}")
                            skipped_count += 1
                            return
                        
                        node["avg_duration_s"] = round(avg_duration_s, 2)
                        node["total_duration_min"] = round(total_duration_min, 2)
                        logger.info(f"✓ 更新 {batch_id}: {old_total:.2f} -> {total_duration_min:.2f} 分钟")
                        updated_count += 1
                    else:
                        # 递归遍历子节点
                        for key, value in node.items():
                            if key != "_metadata":
                                traverse(value, path + "/" + key)
            
            traverse(data)
            
            # 更新元数据
            data["_metadata"]["last_updated"] = datetime.now().isoformat()
            
            # 写回文件
            f.seek(0)
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.truncate()
            
            logger.info(f"\n完成! 更新了 {updated_count} 个batch，跳过 {skipped_count} 个")
            
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="重新计算所有batch的时长信息（独立版，无需导入BatchTracker）")
    parser.add_argument("--data-root", type=str, required=True,
                        help="数据根目录（对应 upload_config.yaml 的 data.root）")
    parser.add_argument("--registry", type=str, default="batch_registry.json",
                        help="注册表文件名（默认 batch_registry.json）")
    args = parser.parse_args()

    registry_path = Path(args.data_root) / args.registry

    if not registry_path.exists():
        logger.error(f"Registry文件不存在: {registry_path}")
        return

    logger.info(f"开始重新计算时长...")
    logger.info(f"Registry: {registry_path}")

    update_registry(registry_path)


if __name__ == "__main__":
    main()
