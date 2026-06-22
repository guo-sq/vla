#!/usr/bin/env python3
# coding=utf-8
#
# Copyright @2025 Wujiedongli Technology Inc. All rights reserved.
# Authors: Yawei Wang <yawei.wang@anyverseintelligence.com>

import logging
import os
from pathlib import Path

import oss2
import yaml

logger = logging.getLogger("OSSUploader")
# 禁止 oss2库 打印 INFO 级别的 Exception 日志 (如 404 Not Found)
logging.getLogger("oss2").setLevel(logging.WARNING)

LOG_DIR = os.getenv("ROBOT_DAQ_LOG_DIR", "/opt/logs/robot_daq")
SERVER_CONFIG_FILE = Path(LOG_DIR, "server_config.yaml")
# 加载服务配置
server_config = {}
if os.path.exists(SERVER_CONFIG_FILE):
    with open(SERVER_CONFIG_FILE, "r") as reader:
        server_config = yaml.safe_load(reader.read())
# OSS Config (for direct sync)
OSS_ENDPOINT = os.getenv(
    "OSS_ENDPOINT",
    server_config.get("oss", {}).get("endpoint", "oss-cn-hangzhou.aliyuncs.com"),
)
OSS_BUCKET = os.getenv(
    "OSS_BUCKET", server_config.get("oss", {}).get("bucket_name", "dataset-robot")
)
OSS_ACCESS_KEY = os.getenv(
    "OSS_ACCESS_KEY", server_config.get("oss", {}).get("access_key_id", "")
)
OSS_SECRET_KEY = os.getenv(
    "OSS_SECRET_KEY", server_config.get("oss", {}).get("access_key_secret", "")
)
REMOTE_UPLOAD_DIR = os.getenv("REMOTE_UPLOAD_DIR", "anyverse")


class OSSUploader:
    def __init__(
        self, bucket_name=None, endpoint=None, access_key=None, secret_key=None
    ):
        """
        初始化 OSS Uploader
        依赖 daq_core/config.py 中的 OSS 配置，也可以通过参数覆盖
        """
        self.access_key = access_key or OSS_ACCESS_KEY
        self.secret_key = secret_key or OSS_SECRET_KEY
        self.endpoint_raw = endpoint or OSS_ENDPOINT
        self.bucket_name = bucket_name or OSS_BUCKET

        if not all(
            [self.access_key, self.secret_key, self.endpoint_raw, self.bucket_name]
        ):
            logger.error(
                "OSS Config missing. Please check server_config.yaml, env vars or pass arguments."
            )
            raise ValueError("Missing OSS Configuration")

        self.auth = oss2.Auth(self.access_key, self.secret_key)
        # Endpoint 处理: oss2 需要 http/https 前缀，如果配置中没有，这里可以补全，或者直接使用
        if not self.endpoint_raw.startswith("http"):
            # 默认为 https，或者根据具体情况
            endpoint = f"https://{self.endpoint_raw}"
        else:
            endpoint = self.endpoint_raw

        self.bucket = oss2.Bucket(self.auth, endpoint, self.bucket_name)
        logger.info(
            f"OSSUploader initialized for bucket: {self.bucket_name} at {endpoint}"
        )

    def _progress_callback(
        self, filename, consumed_bytes, total_bytes, user_callback=None
    ):
        """
        上传进度回调
        """
        if total_bytes:
            rate = int(100 * (float(consumed_bytes) / float(total_bytes)))
            if user_callback:
                user_callback(filename, consumed_bytes, total_bytes, rate)

            # 为了减少日志量，可以只在特定百分比或完成时打印
            rate = int(100 * (float(consumed_bytes) / float(total_bytes)))
            # 为了减少日志量，可以只在特定百分比或完成时打印
            # 这里简单实现为每 10% 打印一次，或者直接打印当前进度（视需求而定）
            # 注意：resumable_upload 会频繁调用此回调
            if rate % 10 == 0 and rate > 0:
                logger.debug(
                    f"Uploading {filename}: {rate}% ({consumed_bytes}/{total_bytes})"
                )

    def _calculate_file_crc64(self, file_path):
        """Calculates the CRC64 of a local file."""
        crc64 = oss2.utils.Crc64()
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                crc64.update(chunk)
        return str(crc64.crc)

    def upload_file(self, local_file_path, remote_object_key, progress_callback=None):
        """
        使用断点续传上传单个文件
        """
        try:
            # 检查远程文件是否存在且一致
            try:
                head = self.bucket.head_object(remote_object_key)
                remote_size = head.content_length
                local_size = os.path.getsize(local_file_path)

                if remote_size == local_size:
                    remote_crc = head.headers.get("x-oss-hash-crc64ecma")
                    if remote_crc:
                        local_crc = self._calculate_file_crc64(local_file_path)
                        if local_crc == remote_crc:
                            logger.info(
                                f"Skipping {local_file_path}: already exists and identical."
                            )
                            return True
            except oss2.exceptions.NotFound:
                pass
            except Exception as e:
                logger.warning(
                    f"Error checking remote object {remote_object_key}: {e}. Proceeding with upload."
                )

            # oss2.resumable_upload 支持断点续传
            # store 指定保存断点信息目录，None 默认在 ~/.ossutil_checkpoint
            # num_threads 并发线程数
            logger.debug(f"Start uploading {local_file_path} to {remote_object_key}")

            result = oss2.resumable_upload(
                self.bucket,
                remote_object_key,
                local_file_path,
                progress_callback=lambda c, t: self._progress_callback(
                    os.path.basename(local_file_path), c, t, progress_callback
                ),
                num_threads=4,  # 使用多线程加速
            )

            if result.status == 200:
                logger.debug(f"Successfully uploaded {local_file_path}")
                return True
            else:
                logger.warning(
                    f"Upload finished with status {result.status} for {local_file_path}"
                )
                return result.status == 200

        except oss2.exceptions.OssError as e:
            logger.error(f"Failed to upload {local_file_path}: {e}")
            return False
        except Exception as e:
            logger.exception(f"Unexpected error uploading {local_file_path}: {e}")
            return False

    def upload_directory(self, local_dir, remote_prefix=None, progress_callback=None):
        """
        上传整个目录
        :param local_dir: 本地目录路径
        :param remote_prefix: OSS 上的前缀 (目录)。如果为 None，则使用 local_dir 的 basename
        """
        local_dir = Path(local_dir)
        if not local_dir.exists() or not local_dir.is_dir():
            logger.error(f"Directory not found: {local_dir}")
            return False

        if remote_prefix is None:
            # 默认使用 REMOTE_UPLOAD_DIR 加上目录名，或者模仿 config 里的逻辑
            # 这里简单起见，如果没传 remote_prefix，就放在 REMOTE_UPLOAD_DIR 下，或者保持相对路径
            remote_prefix = str(Path(REMOTE_UPLOAD_DIR) / local_dir.name)

        logger.debug(f"Starting upload directory: {local_dir} -> {remote_prefix}")

        success_count = 0
        fail_count = 0

        for root, dirs, files in os.walk(local_dir):
            for file in files:
                local_file_path = str(Path(root) / file)

                # 计算相对路径，构建 OSS key
                rel_path = os.path.relpath(local_file_path, str(local_dir))
                remote_object_key = str(Path(remote_prefix) / rel_path)

                # 统一路径分隔符为 /
                remote_object_key = remote_object_key.replace(os.sep, "/")

                # 去除开头的 /，防止在 OSS 生成根目录下的空文件夹或 / 文件夹
                if remote_object_key.startswith("/"):
                    remote_object_key = remote_object_key.lstrip("/")

                if self.upload_file(
                    local_file_path,
                    remote_object_key,
                    progress_callback=progress_callback,
                ):
                    success_count += 1
                else:
                    fail_count += 1

        logger.info(
            f"Directory upload finished. Success: {success_count}, Failed: {fail_count}"
        )
        return fail_count == 0

    def upload_directories(self, directories):
        """
        上传多个目录
        :param directories: 目录列表 [path1, path2, ...]
        """
        results = {}
        for directory in directories:
            results[directory] = self.upload_directory(directory)

        return results

    def _normalize_remote_key(self, remote_key):
        return str(remote_key).replace("\\", "/").lstrip("/")

    def _iter_files_with_prefix(self, prefix):
        for obj in oss2.ObjectIterator(self.bucket, prefix=prefix):
            if obj.key == prefix or obj.key.endswith("/"):
                continue
            yield obj

    def download_file(self, remote_object_key, local_file_path, progress_callback=None):
        """
        使用断点续传下载单个文件
        """
        remote_object_key = self._normalize_remote_key(remote_object_key)
        local_file_path = Path(local_file_path)
        local_file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            head = self.bucket.head_object(remote_object_key)
            remote_size = head.content_length
            remote_crc = head.headers.get("x-oss-hash-crc64ecma")

            if (
                local_file_path.exists()
                and local_file_path.stat().st_size == remote_size
            ):
                if (
                    remote_crc
                    and self._calculate_file_crc64(local_file_path) == remote_crc
                ):
                    logger.info(
                        f"Skipping {local_file_path}: already exists and identical."
                    )
                    if progress_callback:
                        progress_callback(
                            remote_object_key, remote_size, remote_size, 100
                    )
                    return True

            oss2.resumable_download(
                self.bucket,
                remote_object_key,
                str(local_file_path),
                progress_callback=lambda c, t: self._progress_callback(
                    os.path.basename(remote_object_key), c, t, progress_callback
                ),
                num_threads=4,
            )

            local_size = (
                local_file_path.stat().st_size if local_file_path.exists() else -1
            )
            if local_size != remote_size:
                logger.error(
                    f"Downloaded size mismatch for {remote_object_key}: "
                    f"local={local_size}, remote={remote_size}"
                )
                return False

            if remote_crc and self._calculate_file_crc64(local_file_path) != remote_crc:
                logger.error(f"Downloaded CRC mismatch for {remote_object_key}")
                return False

            if progress_callback:
                progress_callback(remote_object_key, remote_size, remote_size, 100)

            if local_file_path.exists():
                logger.debug(f"Successfully downloaded {remote_object_key}")
                return True

            return False
        except oss2.exceptions.OssError as e:
            logger.error(f"Failed to download {remote_object_key}: {e}")
            return False
        except Exception as e:
            logger.exception(f"Unexpected error downloading {remote_object_key}: {e}")
            return False

    def download_path(self, remote_path, local_dir, progress_callback=None):
        """
        下载 OSS 文件或目录到本地目录，目录下载会保留目录内的相对层级。
        :param remote_path: OSS 文件 Key 或目录前缀
        :param local_dir: 本地目标目录
        """
        remote_path = self._normalize_remote_key(remote_path)
        local_dir = Path(local_dir).expanduser()

        try:
            if not remote_path.endswith("/"):
                try:
                    self.bucket.head_object(remote_path)
                    local_file_path = local_dir / Path(remote_path).name
                    return self.download_file(
                        remote_path,
                        local_file_path,
                        progress_callback=progress_callback,
                    )
                except oss2.exceptions.NotFound:
                    pass

            prefix = remote_path if remote_path.endswith("/") else f"{remote_path}/"
            objects = list(self._iter_files_with_prefix(prefix))
            if not objects:
                logger.error(f"Remote path not found or empty: {remote_path}")
                return False

            total_bytes = sum(obj.size for obj in objects)
            downloaded_bytes = 0
            target_root = local_dir

            for obj in objects:
                rel_path = obj.key[len(prefix) :]
                if not rel_path:
                    continue

                local_file_path = target_root / rel_path

                def aggregate_progress(filename, consumed, total, rate):
                    if progress_callback:
                        current_total = total_bytes or total
                        current_consumed = downloaded_bytes + consumed
                        current_rate = (
                            int(100 * current_consumed / current_total)
                            if current_total
                            else 100
                        )
                        progress_callback(
                            obj.key, current_consumed, current_total, current_rate
                        )

                if not self.download_file(
                    obj.key,
                    local_file_path,
                    progress_callback=aggregate_progress,
                ):
                    return False

                downloaded_bytes += obj.size

            if progress_callback:
                progress_callback(remote_path, total_bytes, total_bytes, 100)
            return True
        except oss2.exceptions.OssError as e:
            logger.error(f"Failed to download {remote_path}: {e}")
            return False
        except Exception as e:
            logger.exception(f"Unexpected error downloading {remote_path}: {e}")
            return False
