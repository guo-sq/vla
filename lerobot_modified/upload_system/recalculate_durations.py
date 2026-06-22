#!/usr/bin/env python3
"""
重新计算batch_registry.json中所有batch的时长信息

该脚本会：
1. 读取batch_registry.json中的所有batch
2. 对每个batch，如果数据集存在，重新计算基于frame的准确时长
3. 更新registry中的avg_duration_s和total_duration_min字段
"""

import sys
from pathlib import Path
import logging

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from lerobot.common.data_tracker import BatchTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description="重新计算所有batch的时长信息")
    parser.add_argument("--data-root", type=str, required=True,
                        help="数据根目录（对应 upload_config.yaml 的 data.root）")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        logger.error(f"数据根目录不存在: {data_root}")
        return
    
    # 初始化追踪器
    tracker = BatchTracker(data_root)
    
    # 获取所有batch
    all_batches = tracker.get_all_batches()
    logger.info(f"找到 {len(all_batches)} 个batch，开始重新计算时长...")
    
    updated_count = 0
    skipped_count = 0
    
    for batch in all_batches:
        dataset_path = Path(batch.local_path)
        
        if not dataset_path.exists():
            logger.warning(f"数据集路径不存在，跳过: {batch.batch_id}")
            skipped_count += 1
            continue
        
        # 计算新的时长
        avg_duration_s, total_duration_min = tracker._calculate_episode_durations(dataset_path)
        
        # 如果时长为0，说明计算失败
        if total_duration_min == 0.0:
            logger.warning(f"无法计算时长，跳过: {batch.batch_id}")
            skipped_count += 1
            continue
        
        # 检查是否需要更新
        if (abs(batch.avg_duration_s - avg_duration_s) < 0.01 and 
            abs(batch.total_duration_min - total_duration_min) < 0.01):
            logger.debug(f"时长未变化，跳过: {batch.batch_id}")
            skipped_count += 1
            continue
        
        # 更新batch的时长信息
        batch_key = tracker.get_batch_key(batch)
        
        def update(data):
            # 导航到batch数据
            keys = batch_key.split('/')
            current = data
            for key in keys[:-1]:
                if key in current:
                    current = current[key]
                else:
                    logger.error(f"无法找到batch路径: {batch_key}")
                    return data
            
            batch_id = keys[-1]
            if batch_id in current:
                current[batch_id]['avg_duration_s'] = round(avg_duration_s, 2)
                current[batch_id]['total_duration_min'] = round(total_duration_min, 2)
                logger.info(f"更新 {batch.batch_id}: {total_duration_min:.2f} 分钟")
            else:
                logger.error(f"无法找到batch: {batch_key}")
            
            return data
        
        tracker._atomic_update(update)
        updated_count += 1
    
    logger.info(f"\n完成! 更新了 {updated_count} 个batch，跳过 {skipped_count} 个")


if __name__ == "__main__":
    main()
