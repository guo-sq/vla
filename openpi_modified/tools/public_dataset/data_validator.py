#!/usr/bin/env python3
"""
数据有效性检查工具
"""

import os
import json
import torch
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import tyro
from datetime import datetime
import logging
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import subprocess


from datasets import load_dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from tqdm import tqdm
from openpi.training.base_cfg import *
from openpi.configs.robot_cfg.base import *
from lerobot.common.datasets.utils import (
    check_timestamps_sync,
    get_episode_data_index,
    hf_transform_to_torch,
)


# ---------------------------
# 轻量“收集式 logger”，用于子进程
# ---------------------------
@dataclass
class LogRecordLite:
    level: int
    msg: str


class CollectingLogger:
    """子进程中使用：记录日志到内存，返回给主进程统一写文件/打印。"""

    def __init__(self):
        self.records: List[LogRecordLite] = []

    def info(self, msg: str):
        self.records.append(LogRecordLite(logging.INFO, msg))

    def warning(self, msg: str):
        self.records.append(LogRecordLite(logging.WARNING, msg))

    def error(self, msg: str):
        self.records.append(LogRecordLite(logging.ERROR, msg))


# ---------------------------
# 原有 DatasetChecker（尽量少改）
# ---------------------------
class DatasetChecker:

    def __init__(
        self,
        root: str,
        repo_id: str,
        logger,
        robot_align_info: RobotAlignInfo,
        check_ts: bool = False,
        check_video: bool = False,
    ):
        self.repo_id = repo_id
        self.logger = logger
        self.root = Path(root)
        self.check_ts = check_ts
        self.check_video = check_video
        self.robot_align_info = robot_align_info  # 保留接口，方便后续扩展

        # 初始化路径
        self._init_paths()

        # 统计无效长度
        self.invalid_length = 0
        self.frame_num_in_info_jsonl = 0
        self.valid_length = 0
        self.flag = False
        self.invalid_episodes = []
        self.missing_parquet_episodes = []
        self.timestamp_check_failed = []

        if self._check_json_files_exist():
            self.flag = True
            self.meta = LeRobotDatasetMetadata(self.repo_id, self.root)

    def _init_paths(self) -> None:
        """初始化所有文件路径"""

        # Parquet文件
        parquet_dir = self.root / "data" / "chunk-000"
        self.parquet_paths = list(parquet_dir.glob("*")) if parquet_dir.exists() else []

        # 元数据文件
        meta_dir = self.root / "meta"
        self.episodes_stats_path = meta_dir / "episodes_stats.jsonl"
        self.episode_path = meta_dir / "episodes.jsonl"
        self.info_path = meta_dir / "info.json"
        self.task_jsonl_path = meta_dir / "tasks.jsonl"

        # 视频文件（原逻辑保留）
        self.video_paths = {}
        video_dir = self.root / "videos" / "chunk-000"
        if video_dir.exists():
            for video_folder in video_dir.iterdir():
                if video_folder.is_dir():
                    mp4_files = list(video_folder.glob("*.mp4"))
                    if mp4_files:
                        self.video_paths[video_folder.name] = mp4_files
        else:
            self.logger.error(f"[{self.root}], 视频文件夹不存在")

    def get_episodes_file_paths(self, meta: LeRobotDatasetMetadata):
        """
        获取数据集中所有文件的路径（改成生成器，少占内存）
        """
        episodes = range(meta.total_episodes)
        for ep_idx in episodes:
            yield Path(str(meta.get_data_file_path(ep_idx)))

        if len(meta.video_keys) > 0:
            for vid_key in meta.video_keys:
                for ep_idx in episodes:
                    yield Path(str(meta.get_video_file_path(ep_idx, vid_key)))

    def check_parquet_files_exist(self) -> None:
        """
        检查Parquet/视频文件是否存在
        小优化：不再重复构造 Path(self.root)/fpath 两次；并且缺失后只记录一次。
        """
        # 逐文件 exists 仍然是 IO 密集；并行后整体会快很多
        missing_any = False
        for rel_fpath in self.get_episodes_file_paths(self.meta):
            abs_path = self.root / rel_fpath
            if not abs_path.exists():
                missing_any = True
                self.logger.warning(f"[{self.root}], {rel_fpath} 不存在")

        if missing_any and self.root not in self.missing_parquet_episodes:
            self.missing_parquet_episodes.append(self.root)

    def check_info_json_consistency(self) -> None:
        """
        检查 info.json 中的一致性：
        1. total_episodes 和 splits 是否与实际 parquet 文件数量一致
        2. 最后一个 parquet 文件的最后一行 index 是否等于 info.json 的 total_frames - 1
        """
        # 读取 info.json
        try:
            with open(self.info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
        except Exception as e:
            self.logger.warning(f"[{self.root}], 无法读取 info.json: {str(e)}")
            return

        # 获取 parquet 文件目录
        parquet_dir = self.root / "data" / "chunk-000"
        if not parquet_dir.exists():
            self.logger.warning(f"[{self.root}], Parquet 文件目录不存在")
            return

        # 统计实际存在的 parquet 文件数量
        actual_parquet_files = sorted(parquet_dir.glob("episode_*.parquet"))
        actual_count = len(actual_parquet_files)

        # 检查 total_episodes
        total_episodes = info.get("total_episodes")
        if total_episodes is not None:
            if actual_count != total_episodes:
                self.logger.warning(
                    f"[{self.root}], Parquet 文件数量不一致: "
                    f"info.json 中 total_episodes={total_episodes}, "
                    f"实际存在 {actual_count} 个文件"
                )
            else:
                self.logger.info(
                    f"[{self.root}], total_episodes 检查通过: {total_episodes} 个文件"
                )
        else:
            self.logger.warning(f"[{self.root}], info.json 中缺少 total_episodes 字段")

        # 检查 splits
        splits = info.get("splits", {})
        for split_name, split_value in splits.items():
            if isinstance(split_value, str) and ":" in split_value:
                # 解析范围，如 "0:9" -> [0, 9]
                try:
                    start, end = map(int, split_value.split(":"))
                    expected_count = end - start
                    if actual_count != expected_count:
                        self.logger.warning(
                            f"[{self.root}], splits.{split_name} ({split_value}) "
                            f"期望 {expected_count} 个 episode, "
                            f"但实际有 {actual_count} 个 parquet 文件"
                        )
                    else:
                        self.logger.info(
                            f"[{self.root}], splits.{split_name} 检查通过: "
                            f"{split_value} ({expected_count} 个文件)"
                        )
                except ValueError:
                    self.logger.warning(
                        f"[{self.root}], splits.{split_name} 格式错误: {split_value}"
                    )

        # 检查最后一个 parquet 文件的最后一行 index 是否等于 info.json 的 total_frames - 1
        if actual_count == 0:
            self.logger.warning(f"[{self.root}], 没有 parquet 文件，无法检查帧数一致性")
            return

        try:
            import pandas as pd

            last_parquet = actual_parquet_files[-1]
            total_frames = info.get("total_frames", -1)

            # 读取最后一个 parquet 文件的最后一行 index
            df = pd.read_parquet(last_parquet, columns=["index"])
            if df.empty:
                self.logger.warning(f"[{self.root}], 最后一个 parquet 文件为空: {last_parquet.name}")
                return

            last_idx = int(df["index"].iloc[-1])

            # 校验：通常 index 是从 0 开始的，所以最后一位 index 应为 total_frames - 1
            if total_frames > 0:
                if last_idx == total_frames - 1:
                    self.logger.info(
                        f"[{self.root}], 帧数一致性检查通过: Last Index ({last_idx}) == Total Frames ({total_frames}) - 1"
                    )
                else:
                    self.logger.warning(
                        f"[{self.root}], 帧数不一致: Parquet最后索引={last_idx}, "
                        f"Info.json总帧数={total_frames} (期望 last_idx = {total_frames - 1})"
                    )
            else:
                self.logger.warning(f"[{self.root}], info.json 中缺少或无效的 total_frames 字段")

        except Exception as e:
            self.logger.warning(f"[{self.root}], 检查帧数一致性时出错: {str(e)}")

    def _check_json_files_exist(self) -> bool:
        """检查所有必需的文件是否存在"""
        required_paths = [
            (self.episode_path, "episodes.jsonl"),
            (self.info_path, "info.json"),
            (self.task_jsonl_path, "tasks.jsonl"),
        ]

        for path, name in required_paths:
            if not path.exists():
                self.invalid_episodes.append(self.repo_id)
                self.logger.warning(f"[{self.root}], 缺少必要文件: {name}")
                return False
        return True

    def check_video_files_exist(self) -> None:
        """检查所有视频文件是否存在"""
        for folder_name, video_files in self.video_paths.items():
            if not video_files:
                self.logger.warning(
                    f"[{self.root}], 视频文件夹 {folder_name} 中未找到视频文件"
                )

    def check_timestamps(self) -> None:
        """检查时间戳是否与FPS同步（可选，通常最慢）"""
        try:
            dataset = load_dataset(
                "parquet", data_dir=str(self.root / "data"), split="train"
            )

            dataset.set_transform(hf_transform_to_torch)
            timestamps = torch.stack(dataset["timestamp"]).numpy()
            episode_indices = torch.stack(dataset["episode_index"]).numpy()
            self.episode_data_index = get_episode_data_index(self.meta.episodes)

            ep_data_index = {k: t for k, t in self.episode_data_index.items()}

            check_timestamps_sync(
                timestamps,
                episode_indices,
                ep_data_index,
                self.meta.fps,
                0.02,
            )

        except Exception as e:
            self.timestamp_check_failed.append([self.root, str(e)])
            self.logger.warning(f"[{self.root}], 时间戳同步检查失败: {str(e)}")

    def check_names_key(self) -> None:
        for key, value in self.meta.features.items():
            if "names" not in value:
                self.logger.warning(f"[{self.root}], {key} 的 names 键不存在")
            elif (
                len(value["shape"]) == 1
                and isinstance(value["names"], list)
                and len(value["names"]) != value["shape"][0]
            ):
                self.logger.warning(
                    f"[{self.root}], {key} 的 names 键长度可能与对齐信息不匹配"
                )

    def check_video_mp4_valid(self) -> None:
        for folder_name, video_files in self.video_paths.items():
            for video_file in video_files:
                try:
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-v",
                            "error",
                            "-i",
                            str(video_file),
                            "-f",
                            "null",
                            "-",
                        ],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT,
                    )
                except subprocess.CalledProcessError as e:
                    self.logger.warning(
                        f"[{self.root}], 视频文件 {video_file} 无效: {str(e)}"
                    )

    def check(self):
        if self.flag:
            self.check_info_json_consistency()
            self.check_parquet_files_exist()
            self.check_video_files_exist()
            self.check_names_key()
            if self.check_ts:
                self.check_timestamps()
            if self.check_video:
                self.check_video_mp4_valid()  # ffmpeg -v error -i episode_000003.mp4 -f null - 2>&1 | grep -i error


# ---------------------------
# 日志（主进程写文件）
# ---------------------------
def setup_logging(config_name: str = "") -> Tuple[logging.Logger, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if config_name:
        log_filename = f"./outputs/data_validation_report_{config_name}_{timestamp}.log"
    else:
        log_filename = f"./outputs/data_validation_report_{timestamp}.log"
    os.makedirs("./outputs", exist_ok=True)

    logger = logging.getLogger("data_validator")
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_filename, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 避免重复 addHandler（tyro 重入 / 交互式运行时常见）
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger, log_filename


# ---------------------------
# 多进程 worker：处理单个 repo，返回结果 + 子日志
# ---------------------------
def _process_one_repo(
    repo_id: str,
    root_dir: str,
    robot_align_info: RobotAlignInfo,
    check_ts: bool,
    check_video: bool,
) -> Dict[str, Any]:
    cblogger = CollectingLogger()
    repo_root = os.path.join(root_dir, repo_id)
    checker = DatasetChecker(
        repo_root, repo_id, cblogger, robot_align_info, check_ts, check_video
    )
    # 添加进度跟踪
    start_time = datetime.now()
    cblogger.info(f"开始处理数据集: {repo_id}")
    checker.check()
    # 计算处理时间
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    cblogger.info(f"完成处理数据集: {repo_id} (耗时: {processing_time:.2f}秒)")

    return {
        "repo_id": repo_id,
        "invalid_episodes": list(checker.invalid_episodes),
        "missing_parquet_episodes": [str(p) for p in checker.missing_parquet_episodes],
        "timestamp_check_failed": list(checker.timestamp_check_failed),
        "logs": [(r.level, r.msg) for r in cblogger.records],
    }


"""
python tools/public_dataset/data_validator.py \
    --config src/openpi/configs/cfg_pi0.5_pour_water_14_dim.py \
    --num-workers 2
"""


def main(
    config: str = "",
    check_ts: bool = False,  # 可以打开
    check_video: bool = False,  # 一般不打开，检查的很慢
    num_workers: int = 1,
    mp_start_method: str = "spawn",  # Linux也建议spawn，避免datasets/torch在fork下的坑
    chunksize: int = 30,  # executor.map 的任务分发粒度
):
    """
    Args:
        config: openpi 的配置文件路径
        check_ts: 是否进行时间戳检查（最慢）
        num_workers:
            1  = 单进程（方便断点调试）
            >1 = 多进程并行（显著提速）
        mp_start_method: spawn / fork / forkserver
        chunksize: 越大调度开销越小，但负载不均时会影响尾部收敛
    """
    # 从配置文件路径中提取配置名称
    config_name = os.path.splitext(os.path.basename(config))[0] if config else ""
    logger, log_filename = setup_logging(config_name)
    logger.info(f"日志已保存: {log_filename}")

    logger.info("开始数据有效性检查")
    codebase_version = "v2.1"

    cfg = Config.fromfile(config).cfg
    repo_id_list = cfg.data.repo_id
    root_dir = cfg.data.root_dir
    robot_align_info = cfg.data.robot_align_info

    logger.info(f"开始检查 {len(repo_id_list)} 个数据集...")
    logger.info(f"根目录: {root_dir}")
    logger.info(f"代码库版本: {codebase_version}")
    logger.info(
        f"check_ts={check_ts}, num_workers={num_workers}, mp_start_method={mp_start_method}"
    )

    all_invalid_episodes: List[str] = []
    missing_parquet_episodes: List[str] = []
    all_timestamp_check_failed: List[str] = []

    if num_workers <= 1:
        # 单进程：便于 pdb / 断点调试
        for repo_id in tqdm(repo_id_list, desc="检查数据集(单进程)"):
            out = _process_one_repo(
                repo_id, root_dir, robot_align_info, check_ts, check_video
            )
            for level, msg in out["logs"]:
                logger.log(level, msg)

            all_invalid_episodes.extend(out["invalid_episodes"])
            missing_parquet_episodes.extend(out["missing_parquet_episodes"])
            all_timestamp_check_failed.extend(out["timestamp_check_failed"])
    else:
        # 多进程：repo 粒度并行
        try:
            ctx = mp.get_context(mp_start_method)
        except ValueError:
            logger.warning(f"不支持的 mp_start_method={mp_start_method}，回退到 spawn")
            ctx = mp.get_context("spawn")
        # 创建进度条
        pbar = tqdm(total=len(repo_id_list), desc="检查数据集(多进程)")

        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as ex:
            # 使用 submit 和 as_completed 来获取每个任务完成时的进度
            futures = {
                ex.submit(
                    _process_one_repo,
                    repo_id,
                    root_dir,
                    robot_align_info,
                    check_ts,
                    check_video,
                ): repo_id
                for repo_id in repo_id_list
            }

            for future in as_completed(futures):
                repo_id = futures[future]
                try:
                    out = future.result()
                    # 主进程统一打日志，避免多进程写同一文件的锁竞争/乱序
                    for level, msg in out["logs"]:
                        logger.log(level, msg)

                    # 更新进度条，显示当前完成的数据集和处理时间
                    processing_time = out.get("processing_time", 0)
                    pbar.set_postfix_str(
                        f"当前完成: {repo_id} (耗时: {processing_time:.2f}秒)"
                    )
                    pbar.update(1)

                    all_invalid_episodes.extend(out["invalid_episodes"])
                    missing_parquet_episodes.extend(out["missing_parquet_episodes"])
                    all_timestamp_check_failed.extend(out["timestamp_check_failed"])
                except Exception as e:
                    logger.error(f"处理数据集 {repo_id} 时发生错误: {str(e)}")
                    pbar.update(1)

        pbar.close()

    logger.info("数据有效性检查完成")

    # 去重，输出更干净
    all_invalid_episodes = sorted(set(all_invalid_episodes))
    missing_parquet_episodes = sorted(set(missing_parquet_episodes))

    if all_invalid_episodes:
        logger.warning("以下数据集缺少必要文件，可能无法使用:")
        for rid in all_invalid_episodes:
            logger.warning(f"- {rid}")

    if missing_parquet_episodes:
        logger.warning("以下数据集缺少Parquet/视频文件，可能无法使用:")
        for p in missing_parquet_episodes:
            logger.warning(f"- {p}")

    if all_timestamp_check_failed:
        logger.warning("以下数据集的时间戳同步检查失败，可能存在时间戳问题:")
        for rid in all_timestamp_check_failed:
            logger.warning(f"- {rid[0]}")

    logger.info(f"日志已保存: {log_filename}")


if __name__ == "__main__":
    tyro.cli(main)
