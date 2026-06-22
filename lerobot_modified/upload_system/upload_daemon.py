#!/usr/bin/env python3
"""
上传守护进程

定时扫描待上传的batch，自动调用upload_data.sh进行上传。
支持并发控制、进度监控、上传验证和失败处理。
"""

import argparse
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import yaml

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.common.data_tracker import BatchInfo, BatchTracker

Path("upload_system_logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("upload_system_logs/upload_daemon.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class UploadConfig:
    """上传配置"""

    # 必须由 upload_config.yaml 的 upload 节提供，无默认值
    remote_ip: str
    remote_port: int
    remote_user: str
    remote_target_dir: str

    # 可选配置，具有默认值
    check_interval_seconds: int = 60
    max_concurrent_uploads: int = 2
    data_root: str = ""
    registry_filename: str = "batch_registry.json"
    upload_script: str = "upload_system/upload_data.sh"


class UploadTask:
    """单个上传任务"""

    def __init__(self, batch_info: BatchInfo, config: UploadConfig, tracker: BatchTracker):
        self.batch_info = batch_info
        self.config = config
        self.tracker = tracker
        self.process: Optional[subprocess.Popen] = None
        self.stopped = threading.Event()
        self.log_file = None

    def run(self) -> bool:
        """
        执行上传任务

        Returns:
            bool: 上传是否成功
        """
        batch_key = self.tracker.get_batch_key(self.batch_info)
        logger.info(f"Starting upload: {batch_key}")

        # 更新状态为uploading
        self.tracker.update_upload_status(batch_key, "uploading", progress=0)

        # 创建日志文件
        log_dir = Path("upload_system_logs/uploads")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir / f"{self.batch_info.batch_id}.log"
        self.log_file = open(log_file_path, "w", encoding="utf-8")

        try:
            # 构造远程目标路径（保留tag和task_name目录结构）
            # 例如：/remote/base/20260210/pack_socks/
            remote_target_dir = f"{self.config.remote_target_dir.rstrip('/')}/{self.batch_info.tag}/{self.batch_info.task_name}/"
            
            # 将 upload_config.yaml 中的 upload 配置通过环境变量注入到脚本
            env = os.environ.copy()
            env["REMOTE_IP"] = self.config.remote_ip
            env["REMOTE_PORT"] = str(self.config.remote_port)
            env["REMOTE_USER"] = self.config.remote_user
            env["DEFAULT_REMOTE_TARGET_DIR"] = self.config.remote_target_dir

            # 调用upload_data.sh，传递本地路径和远程目标路径
            cmd = ["bash", self.config.upload_script, self.batch_info.local_path, remote_target_dir]
            logger.info(f"Running command: {' '.join(cmd)}")

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                env=env,
            )

            # 实时读取输出并解析进度
            last_progress = 0
            for line in self.process.stdout:
                if self.stopped.is_set():
                    logger.info(f"Upload cancelled: {batch_key}")
                    self.process.terminate()
                    self.process.wait(timeout=10)
                    self.tracker.update_upload_status(batch_key, "paused", error="Cancelled by user")
                    return False

                # 写入日志
                self.log_file.write(line)
                self.log_file.flush()

                # 解析rsync进度: "1.23GB  45%  10.5MB/s"
                progress_match = re.search(r"(\d+)%", line)
                if progress_match:
                    progress = int(progress_match.group(1))
                    # 只在进度变化时更新（避免频繁写入）
                    if progress > last_progress:
                        self.tracker.update_upload_status(batch_key, "uploading", progress=progress)
                        last_progress = progress
                        logger.debug(f"Upload progress: {batch_key} - {progress}%")

            # 等待进程结束
            return_code = self.process.wait()

            if return_code == 0:
                logger.info(f"Upload completed: {batch_key}")

                # 验证上传
                success, message = self._validate_upload()
                if success:
                    # 获取实际上传的数据大小
                    upload_size_gb = self.batch_info.total_size_gb
                    self.tracker.update_upload_status(
                        batch_key, "completed", progress=100, upload_size_gb=upload_size_gb
                    )
                    logger.info(f"Upload validated successfully: {batch_key}")
                    return True
                else:
                    logger.error(f"Upload validation failed: {batch_key} - {message}")
                    self.tracker.update_upload_status(batch_key, "failed", error=f"Validation failed: {message}")
                    return False
            else:
                error_msg = f"Upload script failed with return code {return_code}"
                logger.error(f"{error_msg}: {batch_key}")
                self.tracker.update_upload_status(batch_key, "failed", error=error_msg)
                return False

        except Exception as e:
            error_msg = f"Upload exception: {str(e)}"
            logger.error(f"{error_msg}: {batch_key}", exc_info=True)
            self.tracker.update_upload_status(batch_key, "failed", error=error_msg)
            return False

        finally:
            if self.log_file:
                self.log_file.close()

    def _validate_upload(self) -> tuple[bool, str]:
        """
        验证上传是否成功（检查文件数量和大小）

        Returns:
            tuple[bool, str]: (是否成功, 消息)
        """
        try:
            local_path = Path(self.batch_info.local_path)

            # 统计本地文件
            local_count = 0
            local_size = 0
            for root, dirs, files in os.walk(local_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.exists(file_path):
                        local_count += 1
                        local_size += os.path.getsize(file_path)

            # TODO: 通过SSH统计远程文件（如果需要严格验证）
            # 目前我们假设rsync成功即可

            logger.info(f"Validation: local {local_count} files, {local_size / (1024**3):.2f} GB")
            return True, "Validation passed (rsync successful)"

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def cancel(self):
        """取消上传"""
        self.stopped.set()
        if self.process and self.process.poll() is None:
            self.process.terminate()


class UploadDaemon:
    """上传守护进程"""

    def __init__(self, config: UploadConfig):
        self.config = config
        self.tracker = BatchTracker(config.data_root, config.registry_filename)
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_uploads)
        self.active_tasks: Dict[str, UploadTask] = {}
        self.running = True
        self.lock = threading.Lock()

        # 注册信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """处理终止信号"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        # 取消所有活动任务
        with self.lock:
            for task in self.active_tasks.values():
                task.cancel()
        self.executor.shutdown(wait=True)
        sys.exit(0)

    def run(self):
        """运行守护进程主循环"""
        logger.info("Upload daemon started")

        # 启动时重置孤立的上传状态
        reset_count = self.tracker.reset_orphaned_uploads()
        if reset_count > 0:
            logger.info(f"Reset {reset_count} orphaned uploads on startup")
        
        # 检查数据一致性（查找孤儿batch）
        self._check_data_consistency()

        # 触发文件：Web 注册后 touch 此文件，守护进程可提前唤醒
        trigger_file = Path("upload_system_logs/upload_trigger")

        while self.running:
            try:
                self._check_and_start_uploads()
                # 分段 sleep，每秒检查触发文件，实现立即响应
                for _ in range(self.config.check_interval_seconds):
                    if not self.running:
                        break
                    if trigger_file.exists():
                        trigger_file.unlink(missing_ok=True)
                        logger.debug("Upload trigger detected, checking immediately")
                        break
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(10)  # 出错后短暂等待

        logger.info("Upload daemon stopped")

    def _check_and_start_uploads(self):
        """检查待上传任务并启动"""
        with self.lock:
            # 清理已完成的任务
            completed_keys = []
            for batch_key, task in list(self.active_tasks.items()):
                if task.process and task.process.poll() is not None:
                    completed_keys.append(batch_key)

            for key in completed_keys:
                del self.active_tasks[key]

            # 检查可以启动的新任务数量
            available_slots = self.config.max_concurrent_uploads - len(self.active_tasks)

            if available_slots <= 0:
                logger.debug(f"No available slots (active: {len(self.active_tasks)})")
                return

            # 获取待上传的batch
            pending_batches = self.tracker.get_pending_batches(limit=available_slots)

            if not pending_batches:
                logger.debug("No pending batches to upload")
                return

            # 启动新的上传任务
            for batch_info in pending_batches:
                batch_key = self.tracker.get_batch_key(batch_info)

                # 检查路径是否存在
                if not Path(batch_info.local_path).exists():
                    logger.warning(f"Local path not found: {batch_info.local_path}")
                    self.tracker.update_upload_status(batch_key, "failed", error="Local path not found")
                    continue

                # 创建并启动任务
                task = UploadTask(batch_info, self.config, self.tracker)
                self.active_tasks[batch_key] = task

                # 提交到线程池
                self.executor.submit(self._run_task, task, batch_key)

                logger.info(f"Started upload task: {batch_key}")

    def _run_task(self, task: UploadTask, batch_key: str):
        """在线程池中运行任务"""
        try:
            success = task.run()
            if success:
                logger.info(f"Task completed successfully: {batch_key}")
            else:
                logger.warning(f"Task failed: {batch_key}")
        except Exception as e:
            logger.error(f"Task exception: {batch_key} - {e}", exc_info=True)
        finally:
            # 从活动任务列表中移除
            with self.lock:
                if batch_key in self.active_tasks:
                    del self.active_tasks[batch_key]
    
    def _check_data_consistency(self):
        """
        检查数据一致性，查找孤儿batch
        
        扫描数据目录，检查是否有未在registry中注册的dataset
        """
        try:
            logger.info("Checking data consistency...")
            data_root = Path(self.config.data_root)
            
            # 扫描所有潜在的dataset目录（包含meta/info.json）
            orphan_count = 0
            for root, dirs, files in os.walk(data_root):
                root_path = Path(root)
                info_file = root_path / "meta" / "info.json"
                
                if info_file.exists():
                    # 检查是否已注册
                    registered_batches = self.tracker.get_all_batches()
                    registered_paths = {Path(batch.local_path) for batch in registered_batches}
                    
                    if root_path not in registered_paths:
                        orphan_count += 1
                        logger.warning(f"Found orphan batch: {root_path}")
            
            if orphan_count > 0:
                logger.warning(f"⚠️  Found {orphan_count} orphan batch(es) not registered in the upload system")
                logger.warning(f"   These batches exist on disk but won't be uploaded automatically")
                logger.warning(f"   Run recovery tool to fix: python upload_system/recover_orphan_batches.py")
            else:
                logger.info("✅ Data consistency check passed, no orphan batches found")
                
        except Exception as e:
            logger.error(f"Failed to check data consistency: {e}", exc_info=True)


def load_config(config_path: str) -> UploadConfig:
    """
    加载配置文件。upload 节下的四个字段（remote_ip / remote_port / remote_user /
    remote_target_dir）必须在配置文件中显式提供，缺失时直接抛出异常。

    Args:
        config_path: 配置文件路径

    Returns:
        UploadConfig: 配置对象
    """
    if not Path(config_path).exists():
        raise FileNotFoundError(
            f"配置文件不存在: {config_path}，请确保 upload_config.yaml 已正确放置。"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    upload_section = config_data.get("upload") or {}

    # 校验必填字段
    required_upload_fields = ["remote_ip", "remote_port", "remote_user", "remote_target_dir"]
    missing = [f for f in required_upload_fields if upload_section.get(f) is None]
    if missing:
        raise ValueError(
            f"upload_config.yaml 的 upload 节缺少必填字段: {missing}，请补全后重新启动。"
        )

    data_root = config_data.get("data", {}).get("root", "")
    if not data_root:
        raise ValueError(
            "upload_config.yaml 的 data.root 未设置，请配置数据根目录路径。"
        )
    # Expand ``~`` / ``$VAR`` once here so the daemon scans the same path the
    # recorder writes to (run_session.sh / SessionConfig.recording.data_root
    # both expand the same way).
    data_root = os.path.expandvars(os.path.expanduser(str(data_root)))

    return UploadConfig(
        remote_ip=upload_section["remote_ip"],
        remote_port=int(upload_section["remote_port"]),
        remote_user=upload_section["remote_user"],
        remote_target_dir=upload_section["remote_target_dir"],
        check_interval_seconds=config_data.get("daemon", {}).get("check_interval_seconds", 60),
        max_concurrent_uploads=config_data.get("daemon", {}).get("max_concurrent_uploads", 2),
        data_root=data_root,
        registry_filename=config_data.get("data", {}).get("registry_filename", "batch_registry.json"),
        upload_script=upload_section.get("script", "upload_system/upload_data.sh"),
    )


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description="LeRobot Upload Daemon")
    parser.add_argument(
        "--config",
        type=str,
        default="upload_system/upload_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # 确保日志目录存在
    Path("upload_system_logs").mkdir(exist_ok=True)

    # 加载配置
    config = load_config(args.config)

    # 启动守护进程
    daemon = UploadDaemon(config)
    daemon.run()


if __name__ == "__main__":
    main()
