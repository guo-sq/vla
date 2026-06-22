"""
跨进程共享的 Episode 缓存系统（异步写入版）

针对大规模数据集（2亿帧、1000+ repos）的多 Worker 训练场景，
使用 /dev/shm 共享内存实现跨进程的 LRU 缓存，避免重复加载。

核心设计:
- cache miss 时立即返回数据，save_to_disk 在后台线程异步完成
- 每台机器独立的 /dev/shm，多机训练各自缓存
- 同机多 worker (spawn) 各自维护独立的 metadata 文件，避免锁竞争
- 本地内存缓存 → 共享内存缓存 → parquet 三级查找

使用方式:
    # 在主进程中初始化
    SharedEpisodeCache.initialize(
        cache_dir="/dev/shm/openpi_cache",
        max_size_gb=200,
    )

    # 在 Worker 进程中获取单例
    cache = SharedEpisodeCache.get_instance()
    data = cache.get_or_load(repo_id, ep_idx, loader_fn)
"""

# ruff: noqa: RUF002, RUF003, PLC0415

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Any

_shared_cache_instance: "SharedEpisodeCache | None" = None
_shared_cache_init_kwargs: dict | None = None
_instance_lock = threading.Lock()


def _get_dir_size(path: Path) -> int:
    """递归计算目录的总大小（字节）"""
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
    except OSError:
        pass
    return total


class SharedEpisodeCache:
    """
    跨进程共享的 Episode 缓存（异步写入版）

    特点:
    - cache miss 时立即返回数据，不阻塞训练
    - save_to_disk 在后台线程池中异步执行
    - 使用 /dev/shm 作为共享内存存储（每台机器独立）
    - 每个 worker 维护独立的 metadata 文件，避免锁竞争
    - LRU 淘汰策略
    - _pending_saves 防止同一 episode 被重复写入
    """

    _instance: "SharedEpisodeCache | None" = None

    def __init__(
        self,
        cache_dir: str = "/dev/shm/openpi_cache",
        max_size_gb: float = 1000.0,
        metadata_file: str = "cache_metadata.json",
        async_write_workers: int = 2,
    ):
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = int(max_size_gb * 1024**3)

        # Per-worker metadata: 每个 worker 写自己的文件，避免锁竞争
        self._worker_id = os.getpid()
        self._worker_metadata_file = f".metadata.worker_{self._worker_id}.json"
        self._worker_metadata_path = self.cache_dir / self._worker_metadata_file

        # 全局 metadata 路径（仅用于兼容性清理旧文件）
        self._global_metadata_path = self.cache_dir / metadata_file

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._thread_lock = threading.Lock()
        # 本 worker 的 metadata（只包含本 worker 写入的条目）
        self._metadata = self._load_worker_metadata()

        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._async_saves = 0
        self._async_errors = 0

        self._pending_saves: set[str] = set()
        self._pending_lock = threading.Lock()

        # ========= access-time 写盘节流 =========
        # 每次 cache hit 都写 metadata JSON 会造成大量小文件写入，引入吞吐抖动。
        # 这里把“更新 last_access 并落盘”改成批量刷新：
        # - 始终在内存里更新 last_access
        # - 满足 (距离上次 flush 超过 interval) 且 (累计更新次数超过 min_updates) 才落盘
        #
        # 环境变量：
        # - OPENPI_SHARED_CACHE_ACCESS_FLUSH_INTERVAL_S: 最小 flush 间隔秒（默认 5）
        # - OPENPI_SHARED_CACHE_ACCESS_FLUSH_MIN_UPDATES: 最小累计更新次数（默认 512）
        self._last_access_flush_ts: float = 0.0
        self._access_updates_since_flush: int = 0
        self._metadata_dirty: bool = False

        # ========= 全局 metadata 合并开销控制 =========
        # 全量合并所有 worker 的 metadata 文件在多 worker 下非常昂贵（大量小文件 IO + JSON parse）。
        # 这里做两层优化：
        # 1) 合并结果做 TTL 缓存（默认 30s）
        # 2) eviction 检查做节流（默认每 10s 或每 50 次异步写入最多触发一次）
        #
        # 环境变量：
        # - OPENPI_SHARED_CACHE_GLOBAL_METADATA_TTL_S: 全局 metadata 缓存 TTL 秒数（默认 30）
        # - OPENPI_SHARED_CACHE_EVICT_MIN_INTERVAL_S: eviction 最小间隔秒数（默认 10）
        # - OPENPI_SHARED_CACHE_EVICT_MIN_ASYNC_SAVES: eviction 最小异步写入次数间隔（默认 50）
        # - OPENPI_SHARED_CACHE_ENABLE_EVICT_THROTTLE: 0 关闭节流（默认开启）
        self._global_metadata_cache: dict | None = None
        self._global_metadata_cache_ts: float = 0.0
        self._last_evict_check_ts: float = 0.0
        self._async_saves_since_last_evict_check: int = 0

        self._write_pool = ThreadPoolExecutor(
            max_workers=async_write_workers,
            thread_name_prefix="cache_writer",
        )

        # fork 安全：记录初始化时的 pid
        self._initialized_pid: int | None = None

        logging.info(
            f"[SharedCache] Initialized: dir={cache_dir}, "
            f"max_size={max_size_gb}GB, async_workers={async_write_workers}, "
            f"worker_id={self._worker_id}, "
            f"local_entries={len(self._metadata.get('entries', {}))}"
        )

        # 标记初始化完成
        self._initialized_pid = self._worker_id

    def _ensure_fork_safe(self) -> None:
        """检测 fork 并重建进程相关资源。

        fork 后子进程会继承父进程的线程池和锁，但这些资源在子进程中是损坏的。
        此方法通过检测 pid 变化来识别 fork，并重建必要的资源。
        """
        current_pid = os.getpid()

        # pid 未变化，无需处理
        if self._initialized_pid == current_pid:
            return

        # 检测到 fork（pid 变化），重建资源
        old_pid = self._worker_id
        self._worker_id = current_pid
        self._worker_metadata_file = f".metadata.worker_{self._worker_id}.json"
        self._worker_metadata_path = self.cache_dir / self._worker_metadata_file

        # 重建线程相关资源（fork 后原资源损坏）
        self._thread_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending_saves = set()
        self._write_pool = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="cache_writer",
        )

        # 重新加载当前 worker 的 metadata
        self._metadata = self._load_worker_metadata()

        logging.info(f"[SharedCache] Fork detected: {old_pid} -> {current_pid}")
        self._initialized_pid = current_pid

    def _load_worker_metadata(self) -> dict:
        """加载当前 worker 的 metadata 文件"""
        if self._worker_metadata_path.exists():
            try:
                with open(self._worker_metadata_path) as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logging.warning(f"[SharedCache] Failed to load worker metadata: {e}, starting fresh")

        return {
            "entries": {},
            "total_size": 0,
            "version": 1,
        }

    def _save_worker_metadata(self):
        """保存当前 worker 的 metadata（无需锁，每个 worker 独占文件）"""
        try:
            tmp_path = self._worker_metadata_path.with_suffix(f".tmp.{self._worker_id}")
            with open(tmp_path, "w") as f:
                json.dump(self._metadata, f, indent=2)
            tmp_path.replace(self._worker_metadata_path)
        except Exception as e:
            logging.warning(f"[SharedCache] Failed to save worker metadata: {e}")
            try:  # noqa: SIM105
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _load_all_metadata(self) -> dict:
        """
        加载所有 worker 的 metadata 并合并（用于统计和 LRU 淘汰决策）
        这个操作只在需要全局视图时调用（如 get_stats、evict）
        """
        merged = {
            "entries": {},
            "total_size": 0,
            "version": 1,
        }

        # 收集所有 worker metadata 文件
        try:
            for meta_file in self.cache_dir.glob(".metadata.worker_*.json"):
                try:
                    with open(meta_file) as f:
                        data = json.load(f)
                        for key, entry in data.get("entries", {}).items():
                            if key not in merged["entries"]:
                                merged["entries"][key] = entry
                                merged["total_size"] += entry.get("size", 0)
                except (OSError, json.JSONDecodeError) as e:
                    logging.debug(f"[SharedCache] Failed to load {meta_file}: {e}")
        except OSError as e:
            logging.warning(f"[SharedCache] Failed to list metadata files: {e}")

        return merged

    def _get_global_metadata(self, *, force_refresh: bool = False) -> dict:
        """获取合并后的全局 metadata（带 TTL 缓存）。

        注意：全局 metadata 只用于统计/淘汰决策，允许短时间不一致。
        """
        ttl_s_raw = os.getenv("OPENPI_SHARED_CACHE_GLOBAL_METADATA_TTL_S", "30")
        try:
            ttl_s = max(float(ttl_s_raw), 0.0)
        except ValueError:
            ttl_s = 30.0

        now = time.time()
        with self._thread_lock:
            if (
                not force_refresh
                and self._global_metadata_cache is not None
                and ttl_s > 0
                and (now - self._global_metadata_cache_ts) < ttl_s
            ):
                return self._global_metadata_cache

        merged = self._load_all_metadata()
        with self._thread_lock:
            self._global_metadata_cache = merged
            self._global_metadata_cache_ts = now
        return merged

    def _should_run_evict_check(self) -> bool:
        enable_raw = os.getenv("OPENPI_SHARED_CACHE_ENABLE_EVICT_THROTTLE", "1")
        enable = enable_raw != "0"
        if not enable:
            return True

        interval_s_raw = os.getenv("OPENPI_SHARED_CACHE_EVICT_MIN_INTERVAL_S", "10")
        min_saves_raw = os.getenv("OPENPI_SHARED_CACHE_EVICT_MIN_ASYNC_SAVES", "50")
        try:
            interval_s = max(float(interval_s_raw), 0.0)
        except ValueError:
            interval_s = 10.0
        try:
            min_saves = max(int(min_saves_raw), 0)
        except ValueError:
            min_saves = 50

        now = time.time()
        with self._thread_lock:
            if self._last_evict_check_ts == 0.0:
                return True
            if interval_s > 0 and (now - self._last_evict_check_ts) < interval_s:  # noqa: SIM102
                if self._async_saves_since_last_evict_check < min_saves:
                    return False
            if min_saves > 0 and self._async_saves_since_last_evict_check < min_saves:  # noqa: SIM102
                if interval_s > 0 and (now - self._last_evict_check_ts) < interval_s:
                    return False
        return True

    def _with_file_lock(self, lock_path: Path, fn: Callable, timeout: float = 30.0):
        """使用文件锁执行操作（仅用于 episode 写入去重，不再用于 metadata）"""
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = None
        try:
            lock_fd = open(lock_path, "w")  # noqa: SIM115
            start_time = time.time()
            while True:
                try:
                    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.time() - start_time >= timeout:
                        logging.warning(f"[SharedCache] File lock timeout: {lock_path}")
                        return None  # 超时跳过，不执行
                    time.sleep(0.05)
            try:
                return fn()
            finally:
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
        finally:
            if lock_fd is not None:
                lock_fd.close()

    @classmethod
    def initialize(
        cls,
        cache_dir: str = "/dev/shm/openpi_cache",
        max_size_gb: float = 1000.0,
    ) -> "SharedEpisodeCache":
        """初始化全局单例，应该在主进程中调用一次"""
        global _shared_cache_instance, _shared_cache_init_kwargs  # noqa: PLW0603

        with _instance_lock:
            _shared_cache_init_kwargs = {
                "cache_dir": cache_dir,
                "max_size_gb": max_size_gb,
            }
            if _shared_cache_instance is None:
                _shared_cache_instance = cls(
                    cache_dir=cache_dir,
                    max_size_gb=max_size_gb,
                )
            return _shared_cache_instance

    @classmethod
    def get_instance(cls) -> "SharedEpisodeCache":
        """获取全局单例，如果未初始化则使用保存的参数或默认参数创建"""
        global _shared_cache_instance, _shared_cache_init_kwargs  # noqa: PLW0603, PLW0602

        with _instance_lock:
            if _shared_cache_instance is None:
                kwargs = _shared_cache_init_kwargs or {}
                _shared_cache_instance = cls(**kwargs)
            return _shared_cache_instance

    @classmethod
    def is_initialized(cls) -> bool:
        return _shared_cache_instance is not None

    def _get_cache_key(self, repo_id: str, ep_idx: int) -> str:
        key_str = f"{repo_id}_{ep_idx}"
        hash_suffix = hashlib.md5(key_str.encode()).hexdigest()[:8]
        safe_repo = repo_id.replace("/", "__")
        return f"{safe_repo}/ep_{ep_idx}_{hash_suffix}"

    def _get_cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / cache_key

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """检查缓存目录是否有效（存在且包含完整的 Arrow 数据）"""
        return cache_path.exists() and (cache_path / "dataset_info.json").exists()

    def get_or_load(
        self,
        repo_id: str,
        ep_idx: int,
        loader_fn: Callable[[], Any],
    ) -> Any:
        """
        获取或加载 episode 数据。

        cache miss 时立即返回数据，save_to_disk 异步在后台完成。
        多机训练下每台机器各自独立缓存到本机 /dev/shm。
        同机多 worker 通过文件锁 + _pending_saves 去重。

        Returns:
            ("cache_hit", cache_path) 或 ("cache_miss", data)
        """
        # fork 安全检测：确保 fork 后重建资源
        self._ensure_fork_safe()

        cache_key = self._get_cache_key(repo_id, ep_idx)
        cache_path = self._get_cache_path(cache_key)

        # 快速路径：缓存命中（检查文件是否存在，跨进程可见）
        if self._is_cache_valid(cache_path):
            self._hits += 1
            self._update_access_time(cache_key)
            return ("cache_hit", cache_path)

        self._misses += 1

        # cache miss：直接加载数据
        data = loader_fn()

        # 异步写入缓存（不阻塞训练）
        self._submit_async_save(cache_key, cache_path, repo_id, ep_idx, data)

        return ("cache_miss", data)

    def _submit_async_save(
        self,
        cache_key: str,
        cache_path: Path,
        repo_id: str,
        ep_idx: int,
        data: Any,
    ):
        """提交异步缓存写入任务。

        通过 _pending_saves 集合在进程内去重，
        通过文件锁在跨进程（同机多 worker）间去重。
        """
        if not hasattr(data, "save_to_disk"):
            return

        with self._pending_lock:
            if cache_key in self._pending_saves:
                return
            # 再次检查是否已经被其他进程写好了
            if self._is_cache_valid(cache_path):
                return
            self._pending_saves.add(cache_key)

        def _do_async_save():
            lock_path = self.cache_dir / ".locks" / f"{cache_key}.lock"
            try:
                self._with_file_lock(
                    lock_path,
                    lambda: self._save_episode_to_cache(cache_key, cache_path, repo_id, ep_idx, data),
                    timeout=60.0,
                )
            except Exception as e:
                self._async_errors += 1
                logging.warning(f"[SharedCache] Async save failed for {cache_key}: {e}")
            finally:
                with self._pending_lock:
                    self._pending_saves.discard(cache_key)
                try:  # noqa: SIM105
                    lock_path.unlink(missing_ok=True)
                except OSError:
                    pass

        try:
            self._write_pool.submit(_do_async_save)
        except RuntimeError:
            with self._pending_lock:
                self._pending_saves.discard(cache_key)

    def _save_episode_to_cache(
        self,
        cache_key: str,
        cache_path: Path,
        repo_id: str,
        ep_idx: int,
        data: Any,
    ):
        """将 episode 数据保存到共享缓存（在文件锁内执行）。

        调用方保证已持有 cache_key 对应的文件锁。
        双重检查：锁内再确认缓存尚不存在。
        """
        if self._is_cache_valid(cache_path):
            return

        tmp_path = cache_path.with_name(cache_path.name + f".tmp.{os.getpid()}")
        try:
            if tmp_path.exists():
                shutil.rmtree(tmp_path, ignore_errors=True)
            tmp_path.parent.mkdir(parents=True, exist_ok=True)

            # 静默 save_to_disk 的 tqdm 进度条输出
            from datasets.utils.logging import disable_progress_bar
            from datasets.utils.logging import enable_progress_bar

            disable_progress_bar()
            try:
                # with_format(None) 创建一个去掉 transform 的新视图，
                # 不影响原 data 的 format，避免 set_transform 导致的序列化错误
                save_data = data.with_format(None)
                save_data.save_to_disk(str(tmp_path))
            finally:
                enable_progress_bar()

            if cache_path.exists():
                shutil.rmtree(cache_path, ignore_errors=True)
            tmp_path.rename(cache_path)

            cache_size = _get_dir_size(cache_path)
            with self._thread_lock:
                self._metadata.setdefault("entries", {})[cache_key] = {
                    "size": cache_size,
                    "last_access": time.time(),
                    "path": str(cache_path),
                    "repo_id": repo_id,
                    "ep_idx": ep_idx,
                }
                self._metadata["total_size"] = sum(e["size"] for e in self._metadata["entries"].values())
                self._save_worker_metadata()

            self._async_saves += 1
            with self._thread_lock:
                self._async_saves_since_last_evict_check += 1
            self._evict_if_needed()

            logging.debug(f"[SharedCache] Async cached {cache_key} ({cache_size / 1024**2:.2f}MB)")
        except Exception as e:
            logging.warning(f"[SharedCache] Failed to save cache for {cache_key}: {e}")
            shutil.rmtree(tmp_path, ignore_errors=True)
            shutil.rmtree(cache_path, ignore_errors=True)

    def _update_access_time(self, cache_key: str):
        """更新 cache_key 的访问时间（带节流，避免 hit 时频繁写盘）。"""
        interval_raw = os.getenv("OPENPI_SHARED_CACHE_ACCESS_FLUSH_INTERVAL_S", "5")
        min_updates_raw = os.getenv("OPENPI_SHARED_CACHE_ACCESS_FLUSH_MIN_UPDATES", "512")
        try:
            interval_s = max(float(interval_raw), 0.0)
        except ValueError:
            interval_s = 5.0
        try:
            min_updates = max(int(min_updates_raw), 0)
        except ValueError:
            min_updates = 512

        now = time.time()
        should_flush = False

        with self._thread_lock:
            entries = self._metadata.get("entries", {})
            e = entries.get(cache_key)
            if e is None:
                return

            e["last_access"] = now
            self._metadata_dirty = True
            self._access_updates_since_flush += 1

            # 首次 flush 允许尽快落一次盘，后续按节流规则
            if (
                self._last_access_flush_ts == 0.0
                or interval_s == 0.0
                or (
                    (now - self._last_access_flush_ts) >= interval_s
                    and (min_updates == 0 or self._access_updates_since_flush >= min_updates)
                )
            ):
                should_flush = True

            if not should_flush:
                return

            # 只在需要 flush 时才写盘
            try:
                self._save_worker_metadata()
                self._metadata_dirty = False
                self._last_access_flush_ts = now
                self._access_updates_since_flush = 0
            except Exception:
                # 写盘失败不影响训练，留待下次 flush 重试
                pass

    def _evict_if_needed(self):
        """LRU 淘汰，确保缓存不超过最大大小

        使用全局 metadata（合并所有 worker）来做淘汰决策，
        但只更新本 worker 的 metadata。
        """
        if not self._should_run_evict_check():
            return

        with self._thread_lock:
            self._last_evict_check_ts = time.time()
            self._async_saves_since_last_evict_check = 0

        # 加载全局 metadata 做决策（带 TTL 缓存以减少小文件 IO）
        global_metadata = self._get_global_metadata(force_refresh=False)
        total_size = global_metadata.get("total_size", 0)

        if total_size <= self.max_size_bytes:
            return

        entries = global_metadata.get("entries", {})
        sorted_keys = sorted(entries.keys(), key=lambda k: entries[k].get("last_access", 0))

        keys_to_remove = []
        evicted_size = 0
        for cache_key in sorted_keys:
            if total_size - evicted_size <= self.max_size_bytes * 0.9:
                break

            # 不淘汰正在写入的 key
            with self._pending_lock:
                if cache_key in self._pending_saves:
                    continue

            entry_info = entries[cache_key]
            cache_path = Path(entry_info.get("path", ""))
            if cache_path.exists():
                try:
                    shutil.rmtree(cache_path)
                    self._evictions += 1
                    evicted_size += entry_info.get("size", 0)
                    keys_to_remove.append(cache_key)
                    logging.debug(f"[SharedCache] Evicted {cache_key}")
                except OSError as e:
                    logging.warning(f"[SharedCache] Failed to evict {cache_key}: {e}")

        # 从本 worker 的 metadata 中移除被淘汰的条目
        if keys_to_remove:
            with self._thread_lock:
                for key in keys_to_remove:
                    self._metadata.get("entries", {}).pop(key, None)
                self._metadata["total_size"] = sum(e["size"] for e in self._metadata.get("entries", {}).values())
                self._save_worker_metadata()

    def get_stats(self) -> dict:
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0

        with self._pending_lock:
            pending = len(self._pending_saves)

        # 使用全局 metadata 统计（带 TTL 缓存）
        global_metadata = self._get_global_metadata(force_refresh=False)

        return {
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": hit_rate,
            "total_entries": len(global_metadata.get("entries", {})),
            "total_size_mb": global_metadata.get("total_size", 0) / 1024**2,
            "async_saves": self._async_saves,
            "async_errors": self._async_errors,
            "pending_saves": pending,
            "worker_id": self._worker_id,
        }

    def shutdown(self, wait: bool = True):  # noqa: FBT001, FBT002
        """关闭后台写入线程池。训练结束时调用。"""
        try:
            self._write_pool.shutdown(wait=wait)
            if wait:
                logging.info(
                    f"[SharedCache] Shutdown complete. "
                    f"async_saves={self._async_saves}, async_errors={self._async_errors}"
                )
        except Exception as e:
            logging.warning(f"[SharedCache] Error during shutdown: {e}")

    def clear(self):
        """清除所有缓存（包括所有 worker 的 metadata）"""
        # 清除缓存文件
        try:
            for entry in self.cache_dir.glob("*"):
                if entry.is_dir() and not entry.name.startswith("."):
                    try:  # noqa: SIM105
                        shutil.rmtree(entry)
                    except OSError:
                        pass
        except OSError:
            pass

        # 清除所有 worker metadata 文件
        try:
            for meta_file in self.cache_dir.glob(".metadata.worker_*.json"):
                try:  # noqa: SIM105
                    meta_file.unlink()
                except OSError:
                    pass
        except OSError:
            pass

        # 清除旧的锁文件（兼容性）
        try:
            lock_file = self.cache_dir / ".metadata.lock"
            if lock_file.exists():
                lock_file.unlink()
        except OSError:
            pass

        # 重置本 worker 的 metadata
        with self._thread_lock:
            self._metadata = {
                "entries": {},
                "total_size": 0,
                "version": 1,
            }
            self._save_worker_metadata()

        logging.info("[SharedCache] Cache cleared")

    def __getstate__(self):
        """序列化支持（spawn worker 场景）。
        线程池和锁不可序列化，在 __setstate__ 中重建。
        """
        state = self.__dict__.copy()
        state["_thread_lock"] = None
        state["_pending_lock"] = None
        state["_write_pool"] = None
        state["_pending_saves"] = set()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._thread_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending_saves = set()
        self._write_pool = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="cache_writer",
        )

    def __del__(self):
        try:
            if hasattr(self, "_write_pool") and self._write_pool is not None:
                self._write_pool.shutdown(wait=False)
        except Exception:
            pass


class EpisodeCacheManager:
    """
    Episode 缓存管理器

    两级查找：
    1. 共享内存缓存（/dev/shm 的 Arrow 文件，同机多 worker 共享）
    2. 从原始 parquet 加载（异步写入 /dev/shm 缓存供后续命中）

    使用 keep_in_memory=True 读取共享缓存，避免 mmap 导致的
    Cannot allocate memory 问题。数据读完即释放，不在进程内长期驻留。
    """

    def __init__(self, shared_cache: SharedEpisodeCache | None = None):
        self._shared_cache = shared_cache or SharedEpisodeCache.get_instance()

    def get_episode(
        self,
        repo_id: str,
        ep_idx: int,
        loader_fn: Callable[[], Any],
    ) -> Any:
        """
        获取 episode 数据

        优先级:
        1. 共享内存缓存命中 → load_from_disk(keep_in_memory=True)
        2. 都没命中 → loader_fn() 从 parquet 加载，异步写入共享缓存
        """
        result = self._shared_cache.get_or_load(repo_id, ep_idx, loader_fn)

        if result[0] == "cache_hit":
            cache_path = result[1]
            try:
                import datasets

                return datasets.load_from_disk(str(cache_path), keep_in_memory=True)
            except Exception as e:
                logging.warning(f"[CacheManager] Failed to load from cache: {e}")
                return loader_fn()
        else:
            return result[1]

    def get_stats(self) -> dict:
        return self._shared_cache.get_stats()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_shared_cache"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._shared_cache = SharedEpisodeCache.get_instance()


# 便捷函数
def get_shared_cache() -> SharedEpisodeCache:
    """获取共享缓存单例"""
    return SharedEpisodeCache.get_instance()


def _get_cache_dir_from_env() -> str:
    """从环境变量获取缓存目录"""
    return os.getenv("OPENPI_SHARED_CACHE_DIR", "/dev/shm/openpi_cache")


def _get_cache_size_gb_from_env() -> float:
    """从环境变量获取缓存大小（GB）"""
    raw = os.getenv("OPENPI_SHARED_CACHE_SIZE_GB", "1000.0")
    try:
        return float(raw)
    except ValueError:
        logging.warning(f"Invalid OPENPI_SHARED_CACHE_SIZE_GB={raw}, using default 1000.0")
        return 1000.0


def init_shared_cache(
    cache_dir: str | None = None,
    max_size_gb: float | None = None,
) -> SharedEpisodeCache:
    """初始化共享缓存

    配置优先级: 参数 > 环境变量 > 默认值

    环境变量:
        - OPENPI_SHARED_CACHE_DIR: 缓存目录 (默认: /dev/shm/openpi_cache)
        - OPENPI_SHARED_CACHE_SIZE_GB: 最大缓存大小 GB (默认: 1000.0)
    """
    final_cache_dir = cache_dir if cache_dir is not None else _get_cache_dir_from_env()
    final_max_size_gb = max_size_gb if max_size_gb is not None else _get_cache_size_gb_from_env()
    return SharedEpisodeCache.initialize(cache_dir=final_cache_dir, max_size_gb=final_max_size_gb)
