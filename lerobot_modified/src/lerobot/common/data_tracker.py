#!/usr/bin/env python
"""
数据批次追踪模块

提供线程安全的JSON读写API，用于记录和管理数据集的录制、上传状态。
支持多进程安全的原子性操作。
"""

import fcntl
import json
import logging
import os
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BatchInfo:
    """批次信息数据类"""

    batch_id: str
    repo_id: str
    local_path: str
    robot_type: str
    robot_id: str
    task_name: str
    tag: str
    recorded_at: str
    num_episodes: int
    avg_duration_s: float
    total_duration_min: float
    total_size_gb: float
    recorded_end_at: Optional[str] = None
    status: str = "pending"  # pending, uploading, completed, failed, paused
    upload_started_at: Optional[str] = None
    upload_completed_at: Optional[str] = None
    upload_duration_min: Optional[float] = None
    upload_size_gb: Optional[float] = None
    upload_progress_percent: int = 0
    upload_error: Optional[str] = None
    upload_retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchInfo":
        """从字典创建实例，兼容旧版 date 字段"""
        data = dict(data)
        if "date" in data and "tag" not in data:
            data["tag"] = data.pop("date")
        elif "date" in data and "tag" in data:
            data.pop("date")
        return cls(**data)


class BatchTracker:
    """批次追踪器 - 管理数据录制和上传状态"""

    def __init__(self, data_root: str | Path, registry_filename: str = "batch_registry.json"):
        """
        初始化批次追踪器

        Args:
            data_root: 数据根目录。支持 ``~`` 与 ``$VAR`` 占位符（与
                ``run_session.sh`` / SessionConfig 的 ``recording.data_root``
                展开规则一致）。如果不在这里展开，YAML 读出的 ``~/...`` 字面量
                会让 Path 把 ``~`` 当成普通目录名，导致守护进程与录制器各自
                操作不同的 ``batch_registry.json``。
            registry_filename: 注册表文件名
        """
        self.data_root = Path(os.path.expandvars(os.path.expanduser(str(data_root))))
        self.registry_path = self.data_root / registry_filename

        # 确保数据根目录存在
        self.data_root.mkdir(parents=True, exist_ok=True)

        # 初始化注册表文件
        if not self.registry_path.exists():
            self._init_registry()

    def _calculate_episode_durations(self, dataset_path: Path) -> tuple[float, float]:
        """
        计算episode的实际时长（基于frame数和fps）

        Args:
            dataset_path: 数据集路径

        Returns:
            tuple[avg_duration_s, total_duration_min]: (平均时长秒, 总时长分钟)
        """
        try:
            # 读取fps
            info_path = dataset_path / "meta" / "info.json"
            if not info_path.exists():
                logger.warning(f"info.json not found in {dataset_path}, returning 0 duration")
                return 0.0, 0.0

            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)

            fps = info.get("fps", 30)  # 默认30fps
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
            logger.error(f"Error calculating episode durations: {e}")
            return 0.0, 0.0

    def _init_registry(self) -> None:
        """初始化空的注册表文件"""
        initial_data = {
            "_metadata": {
                "last_updated": datetime.now().isoformat(),
                "total_batches": 0,
                "total_episodes": 0,
                "total_uploaded_gb": 0.0,
            }
        }
        self._write_json(initial_data)
        logger.info(f"Initialized registry at {self.registry_path}")

    def _read_json(self) -> Dict[str, Any]:
        """线程安全地读取JSON文件"""
        if not self.registry_path.exists():
            self._init_registry()

        with open(self.registry_path, "r", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                data = json.load(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return data

    def _write_json(self, data: Dict[str, Any]) -> None:
        """线程安全地写入JSON文件"""
        # 先写入临时文件，然后原子性替换
        tmp_path = self.registry_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp_path.replace(self.registry_path)

    def _atomic_update(self, update_func: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        """原子性更新JSON文件"""
        with open(self.registry_path, "r+", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                data = json.load(f)
                data = update_func(data)
                # 更新元数据
                data["_metadata"]["last_updated"] = datetime.now().isoformat()
                f.seek(0)
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.truncate()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _get_directory_size(self, path: Path) -> float:
        """计算目录大小（GB）"""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except Exception as e:
            logger.warning(f"Failed to calculate directory size for {path}: {e}")
        return total_size / (1024**3)  # Convert to GB

    def _navigate_to_batch(self, data: Dict[str, Any], key_path: List[str]) -> Optional[Dict[str, Any]]:
        """根据路径导航到batch数据"""
        current = data
        for key in key_path:
            if key not in current:
                return None
            current = current[key]
        return current

    def _set_batch_data(self, data: Dict[str, Any], key_path: List[str], batch_data: Dict[str, Any]) -> None:
        """根据路径设置batch数据"""
        current = data
        for key in key_path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[key_path[-1]] = batch_data

    def record_batch(
        self,
        robot_type: str,
        robot_id: str,
        repo_id: str,
        task_name: str,
        dataset_path: Path,
        num_episodes: int,
        start_time: float,
        end_time: float,
    ) -> BatchInfo:
        """
        记录新的batch信息

        Args:
            robot_type: 机器人类型
            robot_id: 机器人ID
            repo_id: 仓库ID（如 "pack_socks/pack_socks.pink.M.pair_s0s1.200s.20260210.batch.7"）
            task_name: 任务名称
            dataset_path: 数据集本地路径
            num_episodes: episode数量
            start_time: 开始时间（Unix时间戳）
            end_time: 结束时间（Unix时间戳）

        Returns:
            BatchInfo: 记录的batch信息
        """
        dataset_path = Path(dataset_path)
        batch_id = dataset_path.name
        tag = dataset_path.parts[-3] if len(dataset_path.parts) >= 3 else "unknown"
        recorded_at = datetime.fromtimestamp(start_time).isoformat()
        recorded_end_at = datetime.fromtimestamp(end_time).isoformat()

        # 计算基于实际frame的时长统计
        avg_duration_s, total_duration_min = self._calculate_episode_durations(dataset_path)
        
        # 如果计算失败，使用录制会话时间作为fallback
        if total_duration_min == 0.0 and num_episodes > 0:
            logger.warning(f"Failed to calculate accurate duration for {batch_id}, using session time")
            total_duration_s = end_time - start_time
            total_duration_min = total_duration_s / 60
            avg_duration_s = total_duration_s / num_episodes
        
        total_size_gb = self._get_directory_size(dataset_path)

        # 创建BatchInfo
        batch_info = BatchInfo(
            batch_id=batch_id,
            repo_id=repo_id,
            local_path=str(dataset_path.absolute()),
            robot_type=robot_type,
            robot_id=robot_id,
            task_name=task_name,
            tag=tag,
            recorded_at=recorded_at,
            recorded_end_at=recorded_end_at,
            num_episodes=num_episodes,
            avg_duration_s=round(avg_duration_s, 2),
            total_duration_min=round(total_duration_min, 2),
            total_size_gb=round(total_size_gb, 2),
        )

        # 保存到注册表
        key_path = [robot_type, task_name, tag, batch_id]

        def update(data: Dict[str, Any]) -> Dict[str, Any]:
            self._set_batch_data(data, key_path, batch_info.to_dict())
            # 更新元数据
            data["_metadata"]["total_batches"] = data["_metadata"].get("total_batches", 0) + 1
            data["_metadata"]["total_episodes"] = data["_metadata"].get("total_episodes", 0) + num_episodes
            return data

        self._atomic_update(update)
        logger.info(f"Recorded batch: {batch_id} ({num_episodes} episodes, {total_size_gb:.2f} GB)")
        return batch_info

    def update_upload_status(
        self,
        batch_key: str,
        status: str,
        progress: Optional[int] = None,
        error: Optional[str] = None,
        upload_size_gb: Optional[float] = None,
    ) -> None:
        """
        更新batch的上传状态

        Args:
            batch_key: batch的唯一标识符（格式：robot_type/task_name/tag/batch_id）
            status: 状态（uploading, completed, failed, paused）
            progress: 上传进度百分比（0-100）
            error: 错误信息
            upload_size_gb: 上传的数据大小（GB）
        """
        key_path = self._normalize_batch_key_path(batch_key.split("/"))
        now = datetime.now().isoformat()

        def update(data: Dict[str, Any]) -> Dict[str, Any]:
            batch_data = self._navigate_to_batch(data, key_path)
            if batch_data is None:
                logger.warning(f"Batch not found: {batch_key}")
                return data

            batch_data["status"] = status

            if progress is not None:
                batch_data["upload_progress_percent"] = progress

            if error is not None:
                batch_data["upload_error"] = error

            if upload_size_gb is not None:
                batch_data["upload_size_gb"] = round(upload_size_gb, 2)

            # 根据状态更新时间戳
            if status == "uploading" and batch_data.get("upload_started_at") is None:
                batch_data["upload_started_at"] = now
                batch_data["upload_retry_count"] = batch_data.get("upload_retry_count", 0) + 1

            elif status == "completed":
                batch_data["upload_completed_at"] = now
                batch_data["upload_progress_percent"] = 100
                # 计算上传时长
                if batch_data.get("upload_started_at"):
                    start = datetime.fromisoformat(batch_data["upload_started_at"])
                    end = datetime.fromisoformat(now)
                    duration_min = (end - start).total_seconds() / 60
                    batch_data["upload_duration_min"] = round(duration_min, 2)
                # 更新全局已上传数据量
                if upload_size_gb:
                    data["_metadata"]["total_uploaded_gb"] = (
                        data["_metadata"].get("total_uploaded_gb", 0.0) + upload_size_gb
                    )

            elif status == "failed":
                batch_data["upload_error"] = error or "Unknown error"

            return data

        self._atomic_update(update)
        logger.debug(f"Updated upload status for {batch_key}: {status} ({progress}%)")

    def get_pending_batches(self, limit: Optional[int] = None) -> List[BatchInfo]:
        """
        获取待上传的batch列表

        Args:
            limit: 返回数量限制

        Returns:
            List[BatchInfo]: 待上传的batch列表
        """
        data = self._read_json()
        pending_batches = []

        def traverse(node: Dict[str, Any], path: List[str] = []) -> None:
            for key, value in node.items():
                if key == "_metadata":
                    continue
                if isinstance(value, dict):
                    # 检查是否是batch数据（包含status字段）
                    if "status" in value and value["status"] == "pending":
                        try:
                            batch_info = BatchInfo.from_dict(value)
                            pending_batches.append(batch_info)
                        except Exception as e:
                            logger.warning(f"Failed to parse batch at {'/'.join(path + [key])}: {e}")
                    else:
                        # 继续递归
                        traverse(value, path + [key])

        traverse(data)

        # 按录制时间排序（先录制的先上传）
        pending_batches.sort(key=lambda x: x.recorded_at)

        if limit:
            return pending_batches[:limit]
        return pending_batches

    def get_batch_info(self, batch_key: str) -> Optional[BatchInfo]:
        """
        获取指定batch的信息

        Args:
            batch_key: batch的唯一标识符

        Returns:
            Optional[BatchInfo]: batch信息，如果不存在返回None
        """
        data = self._read_json()
        key_path = self._normalize_batch_key_path(batch_key.split("/"))
        batch_data = self._navigate_to_batch(data, key_path)

        if batch_data is None:
            return None

        try:
            return BatchInfo.from_dict(batch_data)
        except Exception as e:
            logger.warning(f"Failed to parse batch {batch_key}: {e}")
            return None

    def _normalize_batch_key_path(self, key_path: List[str]) -> List[str]:
        """
        将 batch_key 路径规范化为 registry 的 4 层结构。

        registry 结构为 robot_type -> task_name -> tag -> batch_id（4 层），
        但设计文档中曾使用 robot_type/robot_id/task_name/tag/batch_id（5 段）。
        当 key_path 为 5 段且前两段相同（robot_type==robot_id）时，去掉重复的 robot_id。
        """
        if len(key_path) == 5 and key_path[0] == key_path[1]:
            # 5 段格式：robot_type/robot_id/task_name/tag/batch_id
            # 去掉第二段 robot_id，得到 4 段
            return [key_path[0], key_path[2], key_path[3], key_path[4]]
        return key_path

    def get_all_batches(
        self, status_filter: Optional[str] = None, tag_filter: Optional[str] = None
    ) -> List[BatchInfo]:
        """
        获取所有batch信息

        Args:
            status_filter: 状态过滤（可选）
            tag_filter: tag过滤（可选）

        Returns:
            List[BatchInfo]: batch列表
        """
        data = self._read_json()
        all_batches = []

        def traverse(node: Dict[str, Any], path: List[str] = []) -> None:
            for key, value in node.items():
                if key == "_metadata":
                    continue
                if isinstance(value, dict):
                    if "status" in value:
                        try:
                            batch_info = BatchInfo.from_dict(value)
                            if status_filter and batch_info.status != status_filter:
                                pass
                            elif tag_filter and batch_info.tag != tag_filter:
                                pass
                            else:
                                all_batches.append(batch_info)
                        except Exception as e:
                            logger.warning(f"Failed to parse batch at {'/'.join(path + [key])}: {e}")
                    else:
                        traverse(value, path + [key])

        traverse(data)
        return all_batches

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息

        Returns:
            Dict[str, Any]: 统计信息字典
        """
        data = self._read_json()
        metadata = data.get("_metadata", {})

        # 统计各状态的batch数量
        all_batches = self.get_all_batches()
        status_counts = {}
        for batch in all_batches:
            status_counts[batch.status] = status_counts.get(batch.status, 0) + 1

        # 统计今日数据
        today = datetime.now().strftime("%Y%m%d")
        today_batches = self.get_all_batches(tag_filter=today)
        today_episodes = sum(b.num_episodes for b in today_batches)
        today_size_gb = sum(b.total_size_gb for b in today_batches)
        
        # 今日有效数据时长：只统计已上传完成的batch
        today_uploaded_batches = [b for b in today_batches if b.status == "completed"]
        today_duration_hours = sum(b.total_duration_min for b in today_uploaded_batches) / 60  # 转换为小时

        return {
            "total_batches": metadata.get("total_batches", 0),
            "total_episodes": metadata.get("total_episodes", 0),
            "total_uploaded_gb": metadata.get("total_uploaded_gb", 0.0),
            "status_counts": status_counts,
            "today_batches": len(today_batches),
            "today_episodes": today_episodes,
            "today_size_gb": round(today_size_gb, 2),
            "today_duration_hours": round(today_duration_hours, 2),
            "last_updated": metadata.get("last_updated"),
        }

    def get_batch_key(self, batch_info: BatchInfo) -> str:
        """
        生成batch的唯一标识符

        Args:
            batch_info: batch信息

        Returns:
            str: batch_key
        """
        return f"{batch_info.robot_type}/{batch_info.task_name}/{batch_info.tag}/{batch_info.batch_id}"

    def record_batch_info(
        self,
        robot_type: str,
        robot_id: str,
        repo_id: str,
        dataset_root: str | Path,
        num_episodes: int,
        num_expected_episodes: int,
        session_start_time: float,
    ) -> BatchInfo:
        """Record batch info after a recording session.

        Convenience wrapper around record_batch() used by both record.py
        and record_unified.py so the logic lives in one place.

        Args:
            robot_type: Robot type string.
            robot_id: Robot ID string.
            repo_id: Dataset repo ID.
            dataset_root: Path to the dataset directory.
            num_episodes: Number of episodes actually recorded.
            num_expected_episodes: Target number of episodes.
            session_start_time: Unix timestamp when the session started.

        Returns:
            BatchInfo recorded into the registry.
        """
        task_name = repo_id.split("/")[0] if "/" in repo_id else "unknown_task"
        session_end_time = time.time()

        batch_info = self.record_batch(
            robot_type=robot_type,
            robot_id=robot_id,
            repo_id=repo_id,
            task_name=task_name,
            dataset_path=Path(dataset_root),
            num_episodes=num_episodes,
            start_time=session_start_time,
            end_time=session_end_time,
        )

        status_msg = (
            "complete"
            if num_episodes >= num_expected_episodes
            else f"partial ({num_episodes}/{num_expected_episodes})"
        )
        logger.info(
            f"Batch recorded to registry ({status_msg}): "
            f"{batch_info.batch_id} ({num_episodes} episodes)"
        )
        return batch_info

    def reset_orphaned_uploads(self) -> int:
        """
        重置孤立的上传状态（用于守护进程启动时）

        将所有状态为"uploading"的batch重置为"pending"

        Returns:
            int: 重置的batch数量
        """
        count = 0

        def update(data: Dict[str, Any]) -> Dict[str, Any]:
            nonlocal count

            def traverse(node: Dict[str, Any]) -> None:
                nonlocal count
                for key, value in node.items():
                    if key == "_metadata":
                        continue
                    if isinstance(value, dict):
                        if "status" in value and value["status"] == "uploading":
                            value["status"] = "pending"
                            value["upload_progress_percent"] = 0
                            count += 1
                        else:
                            traverse(value)

            traverse(data)
            return data

        self._atomic_update(update)

        if count > 0:
            logger.info(f"Reset {count} orphaned uploading batch(es) to pending")

        return count
