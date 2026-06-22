#!/usr/bin/env python3
"""Parquet 数据完整性验证工具。

验证 value_pred parquet 文件与原始数据集的一致性。

用法:
    python verify_parquet_integrity.py --repo_path /path/to/repo
    python verify_parquet_integrity.py --repo_path /path/to/repo --fix
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class IntegrityIssue:
    """完整性问题"""
    episode: int
    issue_type: str
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class IntegrityReport:
    """完整性验证报告"""
    repo_path: str
    total_episodes: int = 0
    total_pred_episodes: int = 0
    missing_episodes: list[int] = field(default_factory=list)
    extra_episodes: list[int] = field(default_factory=list)
    length_mismatches: list[dict] = field(default_factory=list)
    invalid_columns: list[dict] = field(default_factory=list)
    value_range_issues: list[dict] = field(default_factory=list)
    valid_frame_issues: list[dict] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return bool(
            self.missing_episodes or
            self.extra_episodes or
            self.length_mismatches or
            self.invalid_columns or
            self.value_range_issues or
            self.valid_frame_issues
        )

    @property
    def is_healthy(self) -> bool:
        return not self.has_issues

    def to_dict(self) -> dict:
        return asdict(self)

    def print_summary(self):
        """打印摘要报告"""
        print("\n" + "=" * 60)
        print("Parquet 数据完整性验证报告")
        print("=" * 60)
        print(f"数据集路径: {self.repo_path}")
        print(f"数据集 episode 数量: {self.total_episodes}")
        print(f"预测值 episode 数量: {self.total_pred_episodes}")
        print()

        if self.is_healthy:
            print("✅ 所有检查通过！数据完整性良好。")
            return

        print("❌ 发现以下问题：")
        print()

        if self.missing_episodes:
            print(f"🔴 缺失 episode ({len(self.missing_episodes)}):")
            print(f"   {self.missing_episodes[:20]}{'...' if len(self.missing_episodes) > 20 else ''}")

        if self.extra_episodes:
            print(f"🟠 多余 episode ({len(self.extra_episodes)}):")
            print(f"   {self.extra_episodes[:20]}{'...' if len(self.extra_episodes) > 20 else ''}")

        if self.length_mismatches:
            print(f"🔴 长度不匹配 ({len(self.length_mismatches)}):")
            for item in self.length_mismatches[:5]:
                print(f"   Episode {item['episode']}: data={item['data_len']}, pred={item['pred_len']}")
            if len(self.length_mismatches) > 5:
                print(f"   ... 还有 {len(self.length_mismatches) - 5} 个")

        if self.invalid_columns:
            print(f"🔴 无效列 ({len(self.invalid_columns)}):")
            for item in self.invalid_columns[:5]:
                print(f"   Episode {item['episode']}: {item['message']}")

        if self.value_range_issues:
            print(f"🟡 值范围异常 ({len(self.value_range_issues)}):")
            for item in self.value_range_issues[:5]:
                print(f"   Episode {item['episode']}: {item['message']}")

        if self.valid_frame_issues:
            print(f"🟡 有效帧比例低 ({len(self.valid_frame_issues)}):")
            for item in self.valid_frame_issues[:5]:
                print(f"   Episode {item['episode']}: {item['valid_ratio']:.2%} valid")

        print()
        print("建议操作:")
        print("  bash tools/value_model_tools/auto_test_value_model.sh label")


def discover_all_parquets(root: Path, subdir: str = "data") -> dict[int, Path]:
    """发现所有数据 parquet 文件"""
    parquet_files = {}

    # 检查所有 chunk 目录
    for chunk_dir in sorted(root.glob(f"{subdir}/chunk-*")):
        for parquet_file in sorted(chunk_dir.glob("episode_*.parquet")):
            try:
                episode_num = int(parquet_file.stem.split("_")[1])
                parquet_files[episode_num] = parquet_file
            except (IndexError, ValueError):
                logger.warning("Invalid parquet filename: %s", parquet_file.name)

    return parquet_files


def verify_parquet_integrity(
    repo_path: Path,
    min_valid_ratio: float = 0.8,
    expected_value_range: tuple[float, float] = (-1.0, 1.0),
) -> IntegrityReport:
    """验证 parquet 文件完整性

    Args:
        repo_path: 数据集根目录
        min_valid_ratio: 最低有效帧比例阈值
        expected_value_range: 期望的值范围

    Returns:
        完整性验证报告
    """
    report = IntegrityReport(repo_path=str(repo_path))

    # 发现数据集 parquet 文件
    data_parquets = discover_all_parquets(repo_path, "data")
    report.total_episodes = len(data_parquets)

    # 发现预测值 parquet 文件
    pred_parquets = discover_all_parquets(repo_path, "value_pred")
    report.total_pred_episodes = len(pred_parquets)

    logger.info("Found %d data episodes, %d pred episodes",
                report.total_episodes, report.total_pred_episodes)

    # 检查 episode 数量
    data_episodes = set(data_parquets.keys())
    pred_episodes = set(pred_parquets.keys())

    report.missing_episodes = sorted(data_episodes - pred_episodes)
    report.extra_episodes = sorted(pred_episodes - data_episodes)

    # 检查每个 episode 的详细内容
    common_episodes = data_episodes & pred_episodes

    for episode in sorted(common_episodes):
        data_file = data_parquets[episode]
        pred_file = pred_parquets[episode]

        try:
            # 读取文件
            data_df = pd.read_parquet(data_file)
            pred_df = pd.read_parquet(pred_file)

            # 检查长度
            data_len = len(data_df)
            pred_len = len(pred_df)

            if data_len != pred_len:
                report.length_mismatches.append({
                    "episode": episode,
                    "data_len": data_len,
                    "pred_len": pred_len,
                })

            # 检查必需列
            required_columns = ["pred_value", "value_is_valid"]
            missing_columns = [col for col in required_columns if col not in pred_df.columns]
            if missing_columns:
                report.invalid_columns.append({
                    "episode": episode,
                    "message": f"Missing columns: {missing_columns}",
                })
                continue

            # 检查值范围
            pred_value = pred_df["pred_value"].values
            value_is_valid = pred_df["value_is_valid"].values

            valid_pred = pred_value[value_is_valid]
            if len(valid_pred) > 0:
                min_val, max_val = expected_value_range
                if valid_pred.min() < min_val or valid_pred.max() > max_val:
                    report.value_range_issues.append({
                        "episode": episode,
                        "message": f"Range [{valid_pred.min():.4f}, {valid_pred.max():.4f}] outside [{min_val}, {max_val}]",
                    })

            # 检查有效帧比例
            valid_ratio = value_is_valid.sum() / len(value_is_valid)
            if valid_ratio < min_valid_ratio:
                report.valid_frame_issues.append({
                    "episode": episode,
                    "valid_ratio": valid_ratio,
                })

        except Exception as e:
            logger.error("Error processing episode %d: %s", episode, e)
            report.invalid_columns.append({
                "episode": episode,
                "message": f"Error: {str(e)}",
            })

    return report


def main():
    parser = argparse.ArgumentParser(
        description="验证 parquet 数据完整性"
    )
    parser.add_argument(
        "--repo_path",
        type=str,
        required=True,
        help="数据集根目录路径"
    )
    parser.add_argument(
        "--min_valid_ratio",
        type=float,
        default=0.8,
        help="最低有效帧比例阈值（默认: 0.8）"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="输出报告文件路径（JSON 格式）"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式，只输出结果"
    )

    args = parser.parse_args()

    repo_path = Path(args.repo_path)
    if not repo_path.exists():
        print(f"错误: 路径不存在: {repo_path}")
        sys.exit(1)

    if not args.quiet:
        logger.info("开始验证: %s", repo_path)

    # 执行验证
    report = verify_parquet_integrity(
        repo_path,
        min_valid_ratio=args.min_valid_ratio,
    )

    # 输出报告
    if not args.quiet:
        report.print_summary()

    # 保存到文件
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info("报告已保存到: %s", output_path)

    # 返回退出码
    sys.exit(0 if report.is_healthy else 1)


if __name__ == "__main__":
    main()
