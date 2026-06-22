"""
数据集元数据收集工具

本模块用于从多个LeRobot数据集收集和汇总元数据信息，
包括机器人类型、帧数、时长等统计信息。
"""

import json
import time
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Set, Optional
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import tyro

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from openpi.training.base_cfg import *


@dataclass
class DatasetStats:
    """单个数据集的统计信息"""

    repo_id: str
    robot_type: str
    total_episodes: int
    total_frames: int
    fps: int
    features: dict
    total_seconds: float = field(init=False)
    total_hours: float = field(init=False)

    def __post_init__(self):
        """计算衍生统计量"""
        self.total_seconds = self.total_frames / self.fps if self.fps > 0 else 0
        self.total_hours = self.total_seconds / 3600


@dataclass
class AggregateStats:
    """聚合统计信息"""

    total_datasets: int
    total_episodes: int
    total_frames: int
    total_seconds: float
    total_hours: float
    robot_types: List[str]
    datasets_per_robot_type: Dict[str, int]
    frames_per_robot_type: Dict[str, int]
    hours_per_robot_type: Dict[str, float]


class DatasetMetadataCollector:
    """数据集元数据收集器"""

    def __init__(
        self,
        dataset_paths: Optional[List[str]] = None,
        base_path: str = None,
    ):
        """
        初始化收集器

        Args:
            dataset_paths: 数据集路径列表，可以是完整路径或相对于base_path的路径
            base_path: 基础路径，用于补全相对路径
        """
        self.base_path = Path(base_path)
        self.dataset_paths = self._prepare_paths(dataset_paths or [])
        self.datasets_stats: List[DatasetStats] = []
        self.failed_datasets: List[Dict[str, str]] = []

    def _prepare_paths(self, dataset_paths: List[str]) -> List[Path]:
        """准备数据集路径"""
        prepared_paths = []
        for path in dataset_paths:
            prepared_paths.append(path)
        return prepared_paths

    def add_dataset(
        self,
        repo_id: str,
        robot_type: str,
        total_episodes: int,
        total_frames: int,
        features: dict,
        fps: int,
    ):
        """添加单个数据集统计信息"""
        stats = DatasetStats(
            repo_id=repo_id,
            robot_type=robot_type,
            total_episodes=total_episodes,
            total_frames=total_frames,
            fps=fps,
            features=features,
        )
        self.datasets_stats.append(stats)
        return stats

    def collect_metadata(self, show_progress: bool = True) -> bool:
        """
        收集所有数据集的元数据

        Args:
            show_progress: 是否显示进度条

        Returns:
            bool: 是否所有数据集都成功处理
        """
        if not self.dataset_paths:
            print("警告: 没有数据集路径需要处理")
            return False

        iterator = (
            tqdm(self.dataset_paths, desc="处理数据集")
            if show_progress
            else self.dataset_paths
        )

        for repo_path in iterator:
            try:
                # 加载数据集元数据
                ds_meta = LeRobotDatasetMetadata(
                    repo_path,
                    self.base_path / repo_path,
                )

                # 添加统计信息
                self.add_dataset(
                    repo_id=str(repo_path),
                    robot_type=ds_meta.robot_type,
                    total_episodes=ds_meta.total_episodes,
                    total_frames=ds_meta.total_frames,
                    features=ds_meta.features,
                    fps=ds_meta.fps,
                )

                if show_progress:
                    iterator.set_postfix(
                        {
                            "当前机器人类型": ds_meta.robot_type,
                            "episodes": ds_meta.total_episodes,
                            "frames": ds_meta.total_frames,
                        }
                    )

            except Exception as e:
                error_info = {"path": str(repo_path), "error": str(e)}
                self.failed_datasets.append(error_info)
                print(f"处理 {repo_path} 时出错: {e}")

        return len(self.failed_datasets) == 0

    def compute_aggregate_stats(self) -> AggregateStats:
        """计算聚合统计信息"""
        if not self.datasets_stats:
            return AggregateStats(
                total_datasets=0,
                total_episodes=0,
                total_frames=0,
                total_seconds=0.0,
                total_hours=0.0,
                robot_types=[],
                datasets_per_robot_type={},
                frames_per_robot_type={},
                hours_per_robot_type={},
            )

        # 按机器人类型分组统计
        robot_type_stats = defaultdict(
            lambda: {
                "datasets": 0,
                "episodes": 0,
                "frames": 0,
                "seconds": 0.0,
                "hours": 0.0,
            }
        )

        total_episodes = 0
        total_frames = 0
        total_seconds = 0.0

        for stats in self.datasets_stats:
            robot_type = stats.robot_type
            robot_type_stats[robot_type]["datasets"] += 1
            robot_type_stats[robot_type]["episodes"] += stats.total_episodes
            robot_type_stats[robot_type]["frames"] += stats.total_frames
            robot_type_stats[robot_type]["seconds"] += stats.total_seconds
            robot_type_stats[robot_type]["hours"] += stats.total_hours

            total_episodes += stats.total_episodes
            total_frames += stats.total_frames
            total_seconds += stats.total_seconds

        # 创建聚合统计
        aggregate = AggregateStats(
            total_datasets=len(self.datasets_stats),
            total_episodes=total_episodes,
            total_frames=total_frames,
            total_seconds=total_seconds,
            total_hours=total_seconds / 3600,
            robot_types=sorted(robot_type_stats.keys()),
            datasets_per_robot_type={
                rt: info["datasets"] for rt, info in robot_type_stats.items()
            },
            frames_per_robot_type={
                rt: info["frames"] for rt, info in robot_type_stats.items()
            },
            hours_per_robot_type={
                rt: info["hours"] for rt, info in robot_type_stats.items()
            },
        )

        return aggregate

    def print_summary(self, include_details: bool = False):
        """打印统计摘要"""
        aggregate = self.compute_aggregate_stats()

        print("\n" + "=" * 60)
        print("数据集统计摘要")
        print("=" * 60)
        print(f"处理的数据集数量: {aggregate.total_datasets}")
        print(f"总 episode 数: {aggregate.total_episodes:,}")
        print(f"总 frame 数: {aggregate.total_frames:,}")
        print(
            f"总时长: {aggregate.total_hours:.2f} 小时 ({aggregate.total_seconds:.0f} 秒)"
        )
        print(f"机器人类型: {', '.join(aggregate.robot_types)}")

        if include_details:
            print("\n按机器人类型详细统计:")
            for robot_type in aggregate.robot_types:
                print(f"  {robot_type}:")
                print(
                    f"    - 数据集数: {aggregate.datasets_per_robot_type[robot_type]}"
                )
                print(f"    - Frame数: {aggregate.frames_per_robot_type[robot_type]:,}")
                print(
                    f"    - 时长: {aggregate.hours_per_robot_type[robot_type]:.2f} 小时"
                )

        if self.failed_datasets:
            print(f"\n失败的数据集 ({len(self.failed_datasets)} 个):")
            for failure in self.failed_datasets:
                print(f"  - {failure['path']}: {failure['error']}")

        print("=" * 60)

    def save_to_json(self, output_path: str, include_individual: bool = True):
        """
        保存统计信息到JSON文件

        Args:
            output_path: 输出文件路径
            include_individual: 是否包含每个数据集的详细信息
        """
        data = {
            "metadata": {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_datasets_processed": len(self.dataset_paths),
                "base_path": str(self.base_path),
            },
            "aggregate_stats": asdict(self.compute_aggregate_stats()),
            "failed_datasets": self.failed_datasets,
        }

        if include_individual:
            data["individual_datasets"] = [
                asdict(stats) for stats in self.datasets_stats
            ]

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"统计信息已保存到: {output_path}")

    def save_to_csv(self, output_path: str):
        """保存统计信息到CSV文件（简化版）"""
        import csv

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # 写入标题
            writer.writerow(
                [
                    "repo_id",
                    "robot_type",
                    "total_episodes",
                    "total_frames",
                    "fps",
                    "total_seconds",
                    "total_hours",
                ]
            )

            # 写入数据
            for stats in self.datasets_stats:
                writer.writerow(
                    [
                        stats.repo_id,
                        stats.robot_type,
                        stats.total_episodes,
                        stats.total_frames,
                        stats.fps,
                        stats.total_seconds,
                        stats.total_hours,
                    ]
                )

        print(f"CSV文件已保存到: {output_path}")

    def save_feature_json(self, output_path: str):
        """保存features信息"""
        all_data_info = {}
        for stats in self.datasets_stats:
            all_data_info[stats.repo_id] = {
                "robot_type": stats.robot_type,
                "features": stats.features,
            }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_data_info, f, ensure_ascii=False, indent=4)


def load_dataset_list_from_file(file_path: str) -> List[str]:
    """从文件加载数据集列表"""
    file_path = Path(file_path)

    if not file_path.exists():
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 过滤空行和注释行
    datasets = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):
            datasets.append(line)

    return datasets


def main(config="", save_path="./outputs/episode_info.json"):
    config_file = config
    config = Config.fromfile(config_file).cfg
    repo_id_list = config.data.repo_id
    root_dir = config.data.root_dir
    # 创建收集器
    collector = DatasetMetadataCollector(repo_id_list, base_path=root_dir)

    # 收集元数据
    print("开始收集数据集元数据...")
    success = collector.collect_metadata(show_progress=True)

    if success:
        print("所有数据集处理成功！")
    else:
        print(f"有 {len(collector.failed_datasets)} 个数据集处理失败")

    # 打印统计摘要
    collector.print_summary(include_details=True)
    os.makedirs("./outputs", exist_ok=True)
    collector.save_feature_json(save_path)
    print(f"统计信息已保存到 {save_path}")

    return collector


"""
python tools/public_dataset/statis_episode_info.py --config <config_path> --save_path <save_path>
"""
if __name__ == "__main__":
    tyro.cli(main)
