from collections.abc import Callable
from enum import IntEnum
import hashlib
import logging
import os
from pathlib import Path
import random
import time
from typing import Any

import datasets
from datasets import load_dataset
from lerobot.common.datasets.compute_stats import aggregate_stats
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import get_episode_data_index
from lerobot.common.datasets.utils import hf_transform_to_torch
from lerobot.common.datasets.utils import load_jsonlines
from lerobot.common.datasets.video_utils import get_safe_default_codec
import numpy as np
import packaging.version
import pyarrow.parquet as pq
import torch

import openpi.training.config as _config
from openpi.training.demo_difficulty.sampling import load_difficulty_sample_weights
from openpi.training.frame_attributes_preprocessors import DatasetContext
from openpi.training.frame_attributes_preprocessors import run_frame_attr_preprocessor_pipeline

# 导入共享缓存模块
from openpi.training.shared_cache import EpisodeCacheManager
from openpi.transforms import flatten_dict
from openpi.transforms import unflatten_dict

CODEBASE_VERSION = "v2.1"

logger = logging.getLogger(__name__)


class TaskState(IntEnum):
    INVALID = 0
    DEVIATION = 1
    FAILURE = 2
    SUCCESS = 3


class AnyverseDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path,
        delta_indices: dict | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        tolerance_s: float = 2e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        parquet_dir: str | None = None,
        video_backend: str | None = None,
        robot_align_info: dict | None = None,
        align_dim: int = 28,
        unify_action_space: bool = False,
        subtask_info: dict | None = None,
        use_generalizable_prompt=False,
        frame_skip: int = 1,
        lazy_load: bool = False,
        enforce_segment_continuity: bool = False,
        disable_action_padding: bool = False,
        frame_attributes_preprocessors: list | None = None,
        use_state_as_action: bool = False,
        difficulty_label_file: str | None = None,
        difficulty_label_strict: bool = False,
    ):
        self.robot_align_info = robot_align_info.robot_align_info
        self._lazy_load = lazy_load
        self._episode_frame_map: list[tuple[int, int]] = []
        self.use_state_as_action = use_state_as_action

        self._cache_manager: EpisodeCacheManager | None = None
        self._episode_cache: dict[int, datasets.Dataset] | None = None
        self._cache_hits = 0
        self._cache_misses = 0
        self._init_cache(repo_id)

        # Start of LeRobotDataset init logic (adapted to avoid download for local path)
        # super().__init__(...)
        self.repo_id = repo_id
        self.root = Path(root)
        self.image_transforms = image_transforms
        self.delta_timestamps = None
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision
        self.video_backend = video_backend if video_backend else get_safe_default_codec()
        self.delta_indices = None

        # Unused attributes
        self.image_writer = None
        self.episode_buffer = None
        self.parquet_dir = parquet_dir if parquet_dir else "data"

        # import pdb; pdb.set_trace()
        if not self.root.exists():
            self.root.mkdir(parents=True, exist_ok=True)
        # self.root.mkdir(exist_ok=True, parents=True)

        # Load metadata
        self.meta = LeRobotDatasetMetadata(self.repo_id, self.root, self.revision, force_cache_sync=force_cache_sync)

        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = aggregate_stats(episodes_stats)
        else:
            self.stats = {}  # Default if version < 2.1 or no episodes

        # 根据延迟加载模式决定是否加载全部数据
        if self._lazy_load:
            logging.info(f"[Dataset] {repo_id}: Lazy load mode enabled, skipping full dataset loading")
            self.hf_dataset = None  # 延迟加载模式下不预加载
            self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)

            # 构建索引映射
            self._episode_frame_map = self._build_episode_index_map()

            # episode_index -> position in episode_data_index
            ep_list = self.episodes if self.episodes else sorted(self.meta.episodes.keys())
            self._ep_idx_to_pos = {ep_idx: pos for pos, ep_idx in enumerate(ep_list)}
        else:
            # Load actual data (原有逻辑)
            try:
                if force_cache_sync:
                    raise FileNotFoundError
                self.hf_dataset = self.load_hf_dataset()
            except (
                AssertionError,
                FileNotFoundError,
                NotADirectoryError,
                Exception,
            ) as e:
                logging.warning(f"Initial load failed: {e}. Attempting to proceed without download due to local path.")
                self.hf_dataset = self.load_hf_dataset()

            self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)

            # Check timestamps - skipping or adapting
            try:
                _timestamps = torch.stack(self.hf_dataset["timestamp"]).numpy()
                _episode_indices = torch.stack(self.hf_dataset["episode_index"]).numpy()
                # We skip further checks for now to be safe
            except Exception as e:
                logging.warning(f"Timestamp check skipped/failed: {e}")

        # End of LeRobotDataset init logic

        self.delta_indices = delta_indices
        self.enforce_segment_continuity = enforce_segment_continuity

        self.align_dim = align_dim
        self.unify_action_space = unify_action_space
        self.subtask_info = subtask_info
        self.disable_action_padding = disable_action_padding
        self.update_meta_info()

        # 处理 valid_mask 等 - 需要根据延迟加载模式调整
        if self._lazy_load:
            # 延迟加载模式:从元数据推断 valid_mask
            total_frames = len(self._episode_frame_map)
            valid_mask_tensor = torch.ones(total_frames, dtype=torch.bool)
            valid_raw_indices = torch.where(valid_mask_tensor)[0]
            valid_weights = torch.ones(total_frames, dtype=torch.long)

            self.action_mask_tensor = torch.ones(total_frames, dtype=torch.bool)
            self.segment_id_tensor = torch.zeros(total_frames, dtype=torch.long)
            self.optimality_arr = torch.ones(total_frames, dtype=bool)
            self.pred_value_tensor = None
            self.indicator_tensor = None
            self.is_negative_episode_tensor = None
            self.episode_boundary_tensor = None
            self.episode_prompt_map = None
            if self.subtask_info is not None:
                self.subtask_index_tensor = torch.full((total_frames,), -1, dtype=torch.long)
            logging.info(f"Lazy load: initialized {total_frames} frames with default valid masks")
        else:
            # 原有逻辑
            processors = frame_attributes_preprocessors or []
            ctx = DatasetContext(
                repo_id=self.repo_id,
                hf_dataset=self.hf_dataset,
                episode_data_index=self.episode_data_index,
                meta=self.meta,
                delta_indices=self.delta_indices,
                robot_type=self.meta.robot_type,
                root=str(self.root),  # Add root parameter
            )
            attrs = run_frame_attr_preprocessor_pipeline(processors, ctx)

            valid_mask_tensor = torch.from_numpy(attrs.valid_mask)
            sample_weight_all = torch.from_numpy(attrs.sample_weight).long()
            self.segment_id_tensor = torch.from_numpy(attrs.segment_id).long()
            self.pred_value_tensor = (
                torch.from_numpy(attrs.pred_value).float() if attrs.pred_value is not None else None
            )
            self.indicator_tensor = torch.from_numpy(attrs.indicator).bool() if attrs.indicator is not None else None
            self.optimality_arr = torch.from_numpy(attrs.optimality)
            self.is_negative_episode_tensor = (
                torch.from_numpy(attrs.is_negative_episode).bool() if attrs.is_negative_episode is not None else None
            )
            self.episode_boundary_tensor = (
                torch.from_numpy(attrs.episode_boundary).to(torch.int8) if attrs.episode_boundary is not None else None
            )
            from openpi.training.frame_attributes_preprocessors.base import EXTRA_EPISODE_PROMPT_MAP

            self.episode_prompt_map = ctx.extras.get(EXTRA_EPISODE_PROMPT_MAP)
            # Read prompts for cross-negative flip (written by ValueReturnsPreprocessor)
            self.positive_prompt = ctx.extras.get("positive_prompt")
            self.negative_prompt = ctx.extras.get("negative_prompt")
            self.action_mask_tensor = valid_mask_tensor.clone()

            if self.disable_action_padding and self.delta_indices is not None:
                action_horizon = len(self.delta_indices["action"])
                valid_end_indices = torch.where(valid_mask_tensor[:-1] & (~valid_mask_tensor[1:]))[0] + 1
                padding_end_indices = torch.cat([valid_end_indices, self.episode_data_index["to"]])
                for idx in padding_end_indices:
                    start_idx = max(0, idx - action_horizon)
                    valid_mask_tensor[start_idx:idx] = False

            valid_raw_indices = torch.where(valid_mask_tensor)[0]
            valid_weights = sample_weight_all[valid_raw_indices]
            logging.info(f"Filtering valid frames: {len(valid_mask_tensor)} -> {len(valid_raw_indices)}")

            if self.subtask_info is not None:
                if "subtask_index" in self.hf_dataset.features:
                    self.subtask_index_tensor = torch.tensor(self.hf_dataset["subtask_index"], dtype=torch.long)
                else:
                    self.subtask_index_tensor = torch.ones(len(self.hf_dataset), dtype=torch.long) * -1

        if difficulty_label_file is not None:
            difficulty_weights = load_difficulty_sample_weights(
                root=self.root,
                total_frames=len(valid_mask_tensor),
                episode_data_index=self.episode_data_index,
                meta_episodes=self.meta.episodes,
                episodes=self.episodes,
                label_file=difficulty_label_file,
                strict=difficulty_label_strict,
            )
            if difficulty_weights is not None:
                difficulty_weight_tensor = torch.from_numpy(difficulty_weights).long()
                valid_weights = valid_weights * difficulty_weight_tensor[valid_raw_indices]
                keep_mask = valid_weights > 0
                valid_raw_indices = valid_raw_indices[keep_mask]
                valid_weights = valid_weights[keep_mask]
                logging.info(
                    "Applied difficulty labels for %s: sampler_frames=%d/%d, mean_weight=%.3f",
                    self.repo_id,
                    len(valid_raw_indices),
                    len(valid_mask_tensor),
                    valid_weights.float().mean().item() if len(valid_weights) else 0.0,
                )

        self._sampler_indices = torch.repeat_interleave(valid_raw_indices, valid_weights)

        sampled_count = len(self._sampler_indices)
        logging.info(
            f"Frame statistics: valid_frames={valid_mask_tensor.sum().item()}, "
            f"sampled_frames={sampled_count}, total_frames={len(valid_mask_tensor)}"
        )

        self.use_generalizable_prompt = use_generalizable_prompt
        if self.use_generalizable_prompt:
            self.tasks_multi_prompts = self.get_tasks_multi_prompts()
        self._original_sample_count = len(self._sampler_indices)
        if frame_skip > 1:
            self._sampler_indices = self._sampler_indices[::frame_skip]
        self.frame_skip = frame_skip
        self._known_bad_indices: set[int] = set()

        # 打印加载模式统计日志
        skipped_count = len(self._sampler_indices)
        mode_str = "lazy_load" if self._lazy_load else "eager_load"
        logging.info(
            f"Dataset '{repo_id}' [{mode_str}]: frame_skip={frame_skip}, "
            f"original_samples={self._original_sample_count}, skipped_samples={skipped_count}, "
            f"reduction_ratio={skipped_count / self._original_sample_count:.2%}"
        )

    # ── private helpers ──────────────────────────────────────────────

    def update_meta_info(self):
        """兼容Lerobot v2.0"""
        # Add names for features that don't have it
        for v in self.meta.features.values():
            if "names" not in v:
                v["names"] = "null"

        # Remap dict keys
        if self.meta.robot_type in self.robot_align_info:
            features_dict = flatten_dict(self.meta.features)
            meta_mapping = self.robot_align_info[self.meta.robot_type].get_meta_mapping_dict()
            if len(meta_mapping):
                features_dict = {meta_mapping.get(k, k): v for k, v in features_dict.items()}
            self.meta.info["features"] = unflatten_dict(features_dict)

    def update_hf_dataset_name(self, hf_dataset):
        """Mapping some keys"""
        if self.meta.robot_type in self.robot_align_info:
            hf_dataset_mapping = self.robot_align_info[self.meta.robot_type].get_hf_dataset_mapping_dict()
            if len(hf_dataset_mapping):
                # Update dataset features with camera mapping
                for key, value in hf_dataset_mapping.items():
                    hf_dataset = hf_dataset.rename_column(key, value)

                self.meta.info["features"] = {hf_dataset_mapping.get(k, k): v for k, v in self.meta.features.items()}
        return hf_dataset

    def get_data_file_path(self, ep_index: int) -> Path:
        ep_chunk = self.meta.get_episode_chunk(ep_index)
        data_path = self.meta.info["data_path"]
        refractor_path = data_path.replace("data/", f"{self.parquet_dir}/")
        if os.path.exists(refractor_path):
            data_path = refractor_path
        fpath = data_path.format(episode_chunk=ep_chunk, episode_index=ep_index)
        return Path(fpath)

    def _build_episode_index_map(self) -> list[tuple[int, int]]:
        """构建全局索引 -> (episode_index, frame_index) 的映射。

        只依赖 meta.episodes 中的元数据,不需要加载实际 parquet 数据。

        Returns:
            List of (episode_index, frame_index) tuples, where global index is the list position.
        """
        episode_frame_map = []

        # 获取需要加载的 episodes 列表
        ep_list = self.episodes if self.episodes is not None else sorted(self.meta.episodes.keys())

        for ep_idx in ep_list:
            if ep_idx not in self.meta.episodes:
                logging.warning(f"Episode {ep_idx} not found in meta.episodes, skipping.")
                continue
            ep_length = self.meta.episodes[ep_idx]["length"]
            episode_frame_map.extend((ep_idx, frame_idx) for frame_idx in range(ep_length))

        logging.info(
            f"Built episode index map for {self.repo_id}: "
            f"{len(ep_list)} episodes, {len(episode_frame_map)} total frames"
        )
        return episode_frame_map

    def _init_cache(self, repo_id: str):
        """初始化缓存系统(共享缓存或本地缓存)

        本地缓存默认无上限 (0 = 不限制),因为 pq.read_table 创建的
        Dataset 完全驻留内存,不持有文件描述符。如果内存紧张,可通过
        OPENPI_LOCAL_CACHE_SIZE 环境变量限制每个 repo 的缓存 episode 数。
        """
        raw = os.getenv("OPENPI_LOCAL_CACHE_SIZE", "0")
        max_local_cache = int(raw)  # 0 = unlimited

        if not self._lazy_load:
            # eager 模式不需要缓存
            self._episode_cache = None
            self._local_cache_max = 0
            self._local_cache_order: list[int] = []
            return

        # lazy_load 模式:尝试使用共享缓存
        try:
            self._cache_manager = EpisodeCacheManager()
            self._episode_cache = None
            self._local_cache_max = max_local_cache
            self._local_cache_order: list[int] = []
            logging.info(f"[Dataset] {repo_id}: Using shared cache mode")
        except Exception as e:
            logging.warning(f"[Dataset] {repo_id}: Failed to init shared cache: {e}, falling back to local cache")
            self._cache_manager = None
            self._episode_cache = {}
            self._local_cache_max = max_local_cache
            self._local_cache_order: list[int] = []

    def _load_episode_data(self, ep_idx: int) -> datasets.Dataset:
        """按需加载单个 episode 的 parquet 数据(带缓存)。

        Args:
            ep_idx: Episode index to load.

        Returns:
            datasets.Dataset for the requested episode.
        """
        if not self._lazy_load:
            return self.hf_dataset

        # 尝试使用共享缓存 (如果 _cache_manager 存在)
        if self._cache_manager is not None:
            try:
                ep_dataset = self._cache_manager.get_episode(
                    self.repo_id,
                    ep_idx,
                    lambda: self._load_parquet(ep_idx, apply_transform=False),
                )
                ep_dataset.set_transform(hf_transform_to_torch)

                # 周期性日志
                stats = self._cache_manager.get_stats()
                total = stats["hits"] + stats["misses"]
                if total <= 10 or total % 100 == 0:
                    logging.info(
                        f"[SharedCache] {self.repo_id} ep {ep_idx}: "
                        f"hit_rate={stats['hit_rate']:.2%}, entries={stats['total_entries']}"
                    )
                return ep_dataset
            except Exception as e:
                logging.warning(f"[SharedCache] {self.repo_id} ep {ep_idx}: {e}, fallback to local cache")

        # 使用本地缓存
        return self._load_episode_from_local_cache(ep_idx)

    def _load_episode_from_local_cache(self, ep_idx: int) -> datasets.Dataset:
        """从本地缓存加载 episode,未命中则从磁盘加载。

        当 _local_cache_max > 0 时启用 LRU 淘汰;为 0 时不限制缓存大小。
        """
        if self._episode_cache is not None and ep_idx in self._episode_cache:
            self._cache_hits += 1
            if self._cache_hits % 100 == 0:
                logging.info(
                    f"[LazyLoad] {self.repo_id} ep {ep_idx}: cache hit "
                    f"(hits={self._cache_hits}, misses={self._cache_misses}, size={len(self._episode_cache)})"
                )
            return self._episode_cache[ep_idx]

        self._cache_misses += 1
        ep_dataset = self._load_parquet(ep_idx, apply_transform=True)

        if self._episode_cache is not None:
            if self._local_cache_max > 0:
                while len(self._episode_cache) >= self._local_cache_max and self._local_cache_order:
                    evict_key = self._local_cache_order.pop(0)
                    self._episode_cache.pop(evict_key, None)
                self._local_cache_order.append(ep_idx)

            self._episode_cache[ep_idx] = ep_dataset

        if self._cache_misses <= 10 or self._cache_misses % 50 == 0:
            cache_size = len(self._episode_cache) if self._episode_cache else 0
            logging.info(
                f"[LazyLoad] {self.repo_id} ep {ep_idx}: loaded from disk "
                f"(hits={self._cache_hits}, misses={self._cache_misses}, size={cache_size})"
            )

        return ep_dataset

    def _load_parquet(self, ep_idx: int, apply_transform: bool = True) -> datasets.Dataset:
        """从磁盘加载单个 parquet 文件。

        使用 PyArrow 直接读取,避免 HF datasets 的 Arrow 缓存机制
        导致的临时文件和文件描述符泄漏 (Too many open files)。
        """
        parquet_path = self.root / self.get_data_file_path(ep_idx)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        logging.debug(f"Loading episode {ep_idx} from {parquet_path}")
        table = pq.read_table(str(parquet_path))
        ep_dataset = datasets.Dataset(table)
        del table

        ep_dataset = self.update_hf_dataset_name(ep_dataset)
        if apply_transform:
            ep_dataset.set_transform(hf_transform_to_torch)
        return ep_dataset

    def _get_episode_and_frame_index(self, global_idx: int) -> tuple[int, int]:
        """将全局索引转换为 (episode_index, frame_index)。

        Args:
            global_idx: Global frame index in the dataset.

        Returns:
            Tuple of (episode_index, frame_index within episode).
        """
        if global_idx < 0 or global_idx >= len(self._episode_frame_map):
            raise IndexError(f"Global index {global_idx} out of range [0, {len(self._episode_frame_map)})")
        return self._episode_frame_map[global_idx]

    def _get_hf_load_num_proc(self) -> int | None:
        env_value = os.getenv("OPENPI_HF_LOAD_NUM_PROC")
        if env_value is not None:
            try:
                parsed = int(env_value)
                return parsed if parsed > 1 else None
            except ValueError:
                logging.warning(f"Invalid OPENPI_HF_LOAD_NUM_PROC={env_value}, fallback to default.")

        cpu_count = os.cpu_count() or 1
        default_proc = min(8, cpu_count)
        return default_proc if default_proc > 1 else None

    def _get_arrow_cache_path(self, files: list[str] | None = None) -> Path:
        cache_root = self.root / ".hf_arrow_cache"
        cache_root.mkdir(parents=True, exist_ok=True)

        if files is None:
            # 使用self.parquet_dir
            if self.parquet_dir is not None and (self.root / self.parquet_dir).exists():
                dataset_dir = self.parquet_dir
            else:
                dataset_dir = "data"
            cache_key = f"all_{dataset_dir}"
        else:
            digest = hashlib.sha1("\n".join(files).encode("utf-8")).hexdigest()[:16]
            cache_key = f"episodes_{len(files)}_{digest}"

        return cache_root / cache_key

    def load_hf_dataset(self) -> datasets.Dataset:
        """hf_dataset contains all the observations, states, actions, rewards, etc."""
        num_proc = self._get_hf_load_num_proc()
        use_arrow_cache = os.getenv("OPENPI_DISABLE_HF_ARROW_CACHE", "1") != "1"

        def _load_from_cache_or_parquet(
            cache_path: Path,
            load_kwargs: dict[str, Any],
        ) -> datasets.Dataset:
            if use_arrow_cache and cache_path.exists():
                logging.info(f"Loading HF dataset from arrow cache: {cache_path}")
                try:
                    return datasets.load_from_disk(str(cache_path))
                except Exception as cache_error:
                    logging.warning(
                        f"HF arrow cache invalid at {cache_path}, fallback to parquet reload: {cache_error}"
                    )
                    try:
                        import shutil

                        shutil.rmtree(cache_path, ignore_errors=True)
                    except Exception as remove_error:
                        logging.warning(f"Failed to remove invalid HF arrow cache {cache_path}: {remove_error}")

            hf_dataset = load_dataset("parquet", **load_kwargs)
            if self.use_state_as_action:
                logging.info(f"Using state as action for {self.repo_id}")
                hf_dataset = hf_dataset.map(lambda x: {"action": x["observation.state"]})
                logging.info("State as action done!")
            if use_arrow_cache:
                try:
                    hf_dataset.save_to_disk(str(cache_path))
                except Exception as cache_error:
                    logging.warning(f"Failed to save HF arrow cache to {cache_path}: {cache_error}")
            return hf_dataset

        if self.episodes is None:
            path = str(self.root / self.parquet_dir)
            if not os.path.exists(path):
                path = str(self.root / "data")
            cache_path = self._get_arrow_cache_path()
            load_kwargs = {"data_dir": path, "split": "train"}
            if num_proc is not None:
                load_kwargs["num_proc"] = num_proc
            hf_dataset = _load_from_cache_or_parquet(cache_path, load_kwargs)
        else:
            files = [str(self.root / self.get_data_file_path(ep_idx)) for ep_idx in self.episodes]
            cache_path = self._get_arrow_cache_path(files)
            load_kwargs = {"data_files": files, "split": "train"}
            if num_proc is not None:
                load_kwargs["num_proc"] = num_proc
            hf_dataset = _load_from_cache_or_parquet(cache_path, load_kwargs)

        # TODO(aliberts): hf_dataset.set_format("torch")
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def get_tasks_multi_prompts(
        self,
    ):
        """Get multiple prompts for each task if available, used for prompt generalization.
        The multi-prompts are stored in a jsonl file with the format:
            [
                {"task_index": 0, "prompts": [prompt1, prompt2, ...]},
                {"task_index": 1, "prompts": [prompt1, prompt2, ...]},
                ...
            ]
        """
        tasks_multi_prompts_path = os.path.join(self.root, "annotations/task_multi_prompts.jsonl")
        if os.path.exists(tasks_multi_prompts_path):
            tasks_multi_prompts = load_jsonlines(Path(tasks_multi_prompts_path))
        else:
            tasks_multi_prompts = None
        tasks_multi_prompts_dict = {}
        if tasks_multi_prompts:
            tasks_multi_prompts_dict = {
                item["task_index"]: item["prompts"]
                for item in sorted(tasks_multi_prompts, key=lambda x: x["task_index"])
            }
        return tasks_multi_prompts_dict

    def __len__(self):
        return len(self._sampler_indices)

    def _get_optimality(self, raw_idx: int) -> Any:
        return int(self.optimality_arr[raw_idx])

    def _getitem_impl(self, idx: int) -> dict:
        # Hugging Face Dataset needs index to be int type
        raw_idx = int(self._sampler_indices[idx])

        # 根据加载模式选择不同的获取逻辑
        item = self._getitem_lazy(raw_idx, idx) if self._lazy_load else self._getitem_eager(raw_idx, idx)

        # 公共后处理逻辑
        if self.delta_indices is not None:
            for key in self.delta_indices:
                if f"{key}_is_valid" in item:
                    # TODO(jy) whether only need action_mask
                    item["action_mask"] = item[f"{key}_is_valid"] & (~item[f"{key}_is_pad"])
                    if self.enforce_segment_continuity:
                        is_segment_continuous = self.segment_id_tensor[raw_idx] == item[f"{key}_segment_id"]
                        item["action_mask"] = item["action_mask"] & is_segment_continuous
                    break

        item = self.update_stats_and_actions(item)

        # get subtask
        if self.subtask_info is not None:
            item["subtask"] = self.get_subtask_anno(int(item["episode_index"]), raw_idx)
        task_idx = item["task_index"].item()
        if (
            self.use_generalizable_prompt
            and self.tasks_multi_prompts is not None
            and task_idx in self.tasks_multi_prompts
        ):
            # TODO: can set the probability for different prompts
            item["task"] = random.choice(self.tasks_multi_prompts[task_idx])
        elif self.unify_action_space:
            item["task"] = (
                self.meta.tasks[task_idx]
                + f" Action fps: {self.meta.fps} \n"
                + f"Robot type: {self.meta.robot_type} \n"
            )
        else:
            item["task"] = self.meta.tasks[task_idx]
        item["robot_type"] = self.meta.robot_type
        item["optimality"] = self._get_optimality(raw_idx)
        if self.indicator_tensor is not None:
            item["indicator"] = self.indicator_tensor[raw_idx]
        return item

    def _getitem_lazy(self, raw_idx: int, idx: int) -> dict:
        """延迟加载模式的 __getitem__ 逻辑

        Args:
            raw_idx: 原始数据索引 (从 sampler_indices 映射得到)
            idx: 请求的索引

        Returns:
            单个样本的字典
        """
        ep_idx, frame_idx = self._get_episode_and_frame_index(raw_idx)

        try:
            ep_dataset = self._load_episode_data(ep_idx)
            item = ep_dataset[frame_idx]

            query_indices = None
            if self.delta_indices is not None:
                ep_pos = self._ep_idx_to_pos[ep_idx]
                ep_start = self.episode_data_index["from"][ep_pos].item()
                ep_end = self.episode_data_index["to"][ep_pos].item()

                query_indices = {
                    key: [max(ep_start, min(ep_end - 1, raw_idx + delta)) for delta in delta_idx]
                    for key, delta_idx in self.delta_indices.items()
                }
                padding = {
                    f"{key}_is_pad": torch.BoolTensor(
                        [(raw_idx + delta < ep_start) | (raw_idx + delta >= ep_end) for delta in delta_idx]
                    )
                    for key, delta_idx in self.delta_indices.items()
                }

                query_data = {}
                for key, q_idx in query_indices.items():
                    if key not in self.meta.video_keys and key in self.meta.names:
                        local_indices = [gi - ep_start for gi in q_idx]
                        query_data[key] = torch.stack(ep_dataset.select(local_indices)[key])
                        query_data[f"{key}_is_valid"] = self.action_mask_tensor[q_idx]
                        query_data[f"{key}_segment_id"] = self.segment_id_tensor[q_idx]

                item = {**item, **padding, **query_data}

            if len(self.meta.video_keys) > 0:
                current_ts = item["timestamp"].item()
                query_timestamps = {}
                for vid_key in self.meta.video_keys:
                    if query_indices is not None and vid_key in query_indices:
                        ep_start_local = self.episode_data_index["from"][self._ep_idx_to_pos[ep_idx]].item()
                        local_indices = [gi - ep_start_local for gi in query_indices[vid_key]]
                        timestamps = ep_dataset.select(local_indices)["timestamp"]
                        query_timestamps[vid_key] = torch.stack(timestamps).tolist()
                    else:
                        query_timestamps[vid_key] = [current_ts]
                video_frames = self._query_videos(query_timestamps, ep_idx)
                item = {**video_frames, **item}

            if self.image_transforms is not None:
                for cam in self.meta.camera_keys:
                    if cam in item:
                        item[cam] = self.image_transforms(item[cam])

        except RuntimeError as e:
            if "Could not push packet to decoder" in str(e) or "Invalid data found when processing input" in str(e):
                dataset_len = len(self._sampler_indices)
                new_idx = random.randint(0, dataset_len - 1)
                logging.warning(
                    f"Caught video decoding error at episode {ep_idx}, frame {frame_idx} (mapped from {idx}). "
                    f"Error: {e}. "
                    f"Retrying with random index {new_idx}."
                )
                return self.__getitem__(new_idx)
            raise e

        return item

    def _getitem_eager(self, raw_idx: int, idx: int) -> dict:
        """原有加载模式的 __getitem__ 逻辑

        Args:
            raw_idx: 原始数据索引 (从 sampler_indices 映射得到)
            idx: 请求的索引

        Returns:
            单个样本的字典
        """
        try:
            return super().__getitem__(raw_idx)
        except RuntimeError as e:
            # "Could not push packet to decoder" 是 torchcodec 报错的关键字
            # "Invalid data found when processing input" 是 FFmpeg 常见报错
            if "Could not push packet to decoder" in str(e) or "Invalid data found when processing input" in str(e):
                dataset_len = len(self._sampler_indices)

                # 随机选择一个新的索引
                new_idx = random.randint(0, dataset_len - 1)

                # 使用 logging 模块记录警告,包含原始索引和新索引
                logging.warning(
                    f"Caught video decoding error at index {raw_idx} (mapped from {idx}). "
                    f"Error: {e}. "
                    f"Retrying with random index {new_idx}."
                )
                return self.__getitem__(new_idx)
            # 如果是其他类型的 RuntimeError，直接抛出，不进行处理
            raise e

    def _handle_getitem_error(self, exc: Exception, idx: int, attempt: int, max_attempts: int) -> bool:
        if isinstance(exc, FileNotFoundError):
            is_recoverable = True
        elif isinstance(exc, RuntimeError):
            message = str(exc).lower()
            recoverable_patterns = (
                "could not open input file",
                "no such file or directory",
                "could not push packet to decoder",
                "invalid data found when processing input",
            )
            is_recoverable = any(pattern in message for pattern in recoverable_patterns)
        else:
            is_recoverable = False

        if is_recoverable:
            logging.warning(
                "Skipping bad sample for %s at idx=%d (attempt %d/%d): %s",
                self.repo_id,
                idx,
                attempt,
                max_attempts,
                exc,
            )
        return is_recoverable

    def _get_max_retries(self, dataset_len: int) -> int:
        env_value = os.getenv("OPENPI_BAD_SAMPLE_MAX_RETRIES")
        if env_value is not None:
            try:
                parsed = int(env_value)
                if parsed >= 0:
                    return parsed
                # negative means: scan the whole dataset once
                return max(dataset_len - 1, 0)
            except ValueError:
                logging.warning(f"Invalid OPENPI_BAD_SAMPLE_MAX_RETRIES={env_value}, fallback to default.")

        # Default: adaptive for partially downloaded datasets.
        # Try up to one full pass (minus current sample) and at least 64 retries.
        return max(64, dataset_len - 1)

    def __getitem__(self, idx) -> dict:
        if idx < 0 or idx >= len(self._sampler_indices):
            raise IndexError(idx)
        current_idx = int(idx)
        dataset_len = len(self._sampler_indices)
        max_retries = self._get_max_retries(dataset_len)
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            if current_idx in self._known_bad_indices:
                if len(self._known_bad_indices) >= dataset_len:
                    break
                current_idx = (current_idx + 1) % dataset_len
                continue

            try:
                return self._getitem_impl(current_idx)
            except Exception as exc:
                if not self._handle_getitem_error(exc, current_idx, attempt + 1, max_retries + 1):
                    raise
                last_error = exc
                self._known_bad_indices.add(current_idx)
                if dataset_len <= 1:
                    break
                current_idx = (current_idx + 1) % dataset_len

        raise RuntimeError(
            f"Failed to fetch a valid sample after {max_retries + 1} attempts for repo {self.repo_id}."
        ) from last_error

    def update_stats_and_actions(self, item):
        """
        Update the stats and action according to the robot model.

        Args:
            item (dict): The item to update.
        """
        # task_index = item["task_index"]
        # print(f"task_index: {task_index}")
        # for key, value in item.items():
        #     if type(value) == type("123"):
        #         print(f"{key}, value: {value}")
        #     else:
        #         print(f"{key}, value.shape: {value.shape}")
        robot_type = self.meta.robot_type
        if self.unify_action_space and robot_type in self.robot_align_info:
            robot_align_info = self.robot_align_info[robot_type]

            state_key_prior = next(iter(robot_align_info.state_meta_source_dict.values()))
            # state_frame_num = item[state_key_prior].shape[0]
            if isinstance(state_key_prior, tuple):
                state_key_prior, _ = state_key_prior
            device = item[state_key_prior].device
            # define state& action
            # TODO(heyuan): expand state dimention
            state = torch.zeros(
                self.align_dim,
                dtype=torch.float32,
                device=device,
            )
            action_key_prior = next(iter(robot_align_info.action_meta_source_dict.values()))
            action_frame_num = item[action_key_prior].shape[0]

            actions = torch.zeros(
                [action_frame_num, self.align_dim],
                dtype=torch.float32,
                device=device,
            )

            joint_eef_dof_mask = torch.zeros(
                [action_frame_num, self.align_dim],
                dtype=torch.bool,
                device=device,
            )

            # update joint state
            for (
                robot_part_name,
                target_dof,
            ) in robot_align_info.get_state_name_dict().items():
                for src_dof, tgt_dof in target_dof.items():
                    state_meta_key = robot_align_info.state_meta_source_dict[robot_part_name]
                    frame_offset = None
                    if isinstance(state_meta_key, tuple):
                        state_meta_key, frame_offset = state_meta_key
                    assert src_dof in self.meta.features[state_meta_key]["names"]
                    state_index = self.meta.features[state_meta_key]["names"].index(src_dof)

                    # print(f"state_meta_key: {state_meta_key}, src_dof: {src_dof}, state_index: {state_index}, current_state shape", item[state_meta_key].shape)
                    if frame_offset is not None:
                        state[tgt_dof] = item[state_meta_key][frame_offset][state_index]
                    else:
                        state[tgt_dof] = item[state_meta_key][state_index]
            item["observation.state"] = state
            # update action
            for (
                robot_part_name,
                target_dof,
            ) in robot_align_info.get_action_name_dict().items():
                for src_dof, tgt_dof in target_dof.items():
                    action_meta_key = robot_align_info.action_meta_source_dict[robot_part_name]
                    if src_dof not in self.meta.features[action_meta_key]["names"]:
                        logging.warning(
                            f"Source DOF {src_dof} not found in meta features for key {action_meta_key}. Skipping."
                        )
                    assert src_dof in self.meta.features[action_meta_key]["names"]
                    action_index = self.meta.features[action_meta_key]["names"].index(src_dof)
                    # print(f"action_meta_key: {action_meta_key}, src_dof: {src_dof}, action_index: {action_index}, current_action shape", item[action_meta_key].shape)
                    actions[:, tgt_dof] = item[action_meta_key][:, action_index]
                    joint_eef_dof_mask[:, tgt_dof] = True

            item["action"] = actions
            item["joint_eef_dof_mask"] = joint_eef_dof_mask
        else:
            # NOTE(heyuan): assure "observation.state" and "action" is properly processed
            item["joint_eef_dof_mask"] = torch.ones(
                [item["action"].shape[0], self.align_dim],
                dtype=torch.bool,
                device=item["action"].device,
            )

        return item

    def get_subtask_anno(self, epi_ind: int, idx: int) -> str:
        subtask_anno_path = os.path.join(self.root, self.subtask_info["subtask_anno_path"])
        if not os.path.exists(subtask_anno_path):
            return None
        subtask_anno = load_jsonlines(subtask_anno_path)
        subtask_anno = {
            item["episode_index"]: item["subtasks"] for item in sorted(subtask_anno, key=lambda x: x["episode_index"])
        }
        episode_subtasks = subtask_anno.get(epi_ind)
        if episode_subtasks is None:
            return None
        if str(self.subtask_index_tensor[idx].item()) not in episode_subtasks:
            return None
        return episode_subtasks[str(self.subtask_index_tensor[idx].item())]

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        data = {}
        for key, q_idx in query_indices.items():
            if key not in self.meta.video_keys and key in self.meta.names:
                data[key] = torch.stack(self.hf_dataset.select(q_idx)[key])
                data[f"{key}_is_valid"] = self.action_mask_tensor[q_idx]
                data[f"{key}_segment_id"] = self.segment_id_tensor[q_idx]
        return data


class MultiAnyverseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_config: _config.DataConfig,
        action_horizon: int,
        image_transforms: Callable | None = None,
        download_videos: bool = True,
        video_backend: str | None = None,
    ):
        super().__init__()

        self.root = Path(data_config.root_dir)
        self.repo_ids = data_config.repo_id if isinstance(data_config.repo_id, list) else [data_config.repo_id]

        self.delta_indices = {key: list(range(action_horizon)) for key in data_config.action_sequence_keys}

        self.robot_align_info = data_config.robot_align_info
        self.align_dim = data_config.align_dim
        self.tolerance_s = data_config.tolerance_s
        self.unify_action_space = data_config.unify_action_space
        self.subtask_info = data_config.subtask_info
        self.frame_skip = data_config.frame_skip
        self.lazy_load = data_config.lazy_load

        frame_attributes_preprocessors = getattr(data_config, "frame_attributes_preprocessors", None)
        enforce_segment_continuity = getattr(data_config, "enforce_segment_continuity", False)
        disable_action_padding = getattr(data_config, "disable_action_padding", False)
        use_state_as_action = getattr(data_config, "use_state_as_action", False)
        difficulty_label_file = getattr(data_config, "difficulty_label_file", None)
        difficulty_label_strict = getattr(data_config, "difficulty_label_strict", False)

        episodes = data_config.episode
        if data_config.episode_fail and data_config.dataset_length:
            assert len(data_config.dataset_length) == len(data_config.episode_fail), (
                f"dataset_length size: {len(data_config.dataset_length)} != episode_fail size:{len(data_config.episode_fail)}"
            )
            assert len(data_config.episode_fail) == len(self.repo_ids), (
                f"episode_fail size: {len(data_config.episode_fail)} != repo_ids size:{len(self.repo_ids)}"
            )
            episodes = {}
            for repo_id, episode_f, data_len in zip(
                self.repo_ids, data_config.episode_fail, data_config.dataset_length, strict=True
            ):
                episodes[repo_id] = [i for i in range(data_len) if i not in episode_f]

        # 创建多个数据集,添加进度日志
        total_start = time.time()
        logging.info(
            f"[MultiDataset] Starting to initialize {len(self.repo_ids)} datasets (lazy_load={self.lazy_load})"
        )

        self._datasets = []
        for idx, repo_id in enumerate(self.repo_ids):
            ds_start = time.time()
            logging.info(f"[MultiDataset] Initializing dataset {idx + 1}/{len(self.repo_ids)}: {repo_id}")

            ds = AnyverseDataset(
                repo_id,
                root=self.root / repo_id,
                delta_indices=self.delta_indices,
                episodes=episodes[repo_id] if episodes else None,
                image_transforms=image_transforms,
                download_videos=download_videos,
                parquet_dir=data_config.parquet_dir,
                use_generalizable_prompt=data_config.use_generalizable_prompt,
                video_backend=video_backend,
                robot_align_info=self.robot_align_info,
                align_dim=self.align_dim,
                unify_action_space=self.unify_action_space,
                subtask_info=self.subtask_info,
                tolerance_s=self.tolerance_s,
                frame_skip=self.frame_skip,
                lazy_load=self.lazy_load,
                frame_attributes_preprocessors=frame_attributes_preprocessors,
                enforce_segment_continuity=enforce_segment_continuity,
                disable_action_padding=disable_action_padding,
                use_state_as_action=use_state_as_action,
                difficulty_label_file=difficulty_label_file,
                difficulty_label_strict=difficulty_label_strict,
            )

            ds_elapsed = time.time() - ds_start
            self._datasets.append(ds)
            logging.info(
                f"[MultiDataset] Dataset {idx + 1}/{len(self.repo_ids)} '{repo_id}' initialized in {ds_elapsed:.2f}s"
            )

        total_elapsed = time.time() - total_start
        logging.info(f"[MultiDataset] All {len(self._datasets)} datasets initialized in {total_elapsed:.2f}s")

        self.cum_sizes = []
        total = 0
        for ds in self._datasets:
            total += len(ds)
            self.cum_sizes.append(total)

        # 打印多数据集汇总统计
        total_original = sum(getattr(ds, "_original_sample_count", len(ds)) for ds in self._datasets)
        total_skipped = self.cum_sizes[-1] if self.cum_sizes else 0
        logging.info(
            f"MultiAnyverseDataset: total_datasets={len(self._datasets)}, "
            f"frame_skip={self.frame_skip}, "
            f"total_original_samples={total_original}, total_skipped_samples={total_skipped}, "
            f"reduction_ratio={total_skipped / total_original:.2%}"
        )

    def __len__(self):
        return self.cum_sizes[-1] if self.cum_sizes else 0

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")

        for ds_idx, cumulative_size in enumerate(self.cum_sizes):
            if idx < cumulative_size:
                sub_idx = idx - self.cum_sizes[ds_idx - 1] if ds_idx > 0 else idx
                return self._datasets[ds_idx][sub_idx]
        return None

    @property
    def num_frames(self) -> int:
        """Number of samples/frames."""
        return sum(d.num_frames for d in self._datasets)

    @property
    def num_episodes(self) -> int:
        """Number of episodes."""
        return sum(d.num_episodes for d in self._datasets)


def main():
    repo_id = "pack_socks.black.M.s0s1s2_takeover.1000s.20260129.batch.14"
    root = f"/mnt/workspace/datf/openpi_modified/anyverse_human_data_record/arxx5_bimanual/pack_socks/{repo_id}"

    action_horizon = 30
    sequence_keys = ["action"]
    delta_indices = {key: list(range(action_horizon)) for key in sequence_keys}
    dataset = AnyverseDataset(
        repo_id,
        root=root,
        delta_indices=delta_indices,
    )

    for i in range(1090, 1100):
        print(f"\nidx: {i}")
        for k, v in dataset[i].items():
            if isinstance(v, np.ndarray | torch.Tensor):
                if v.ndim > 1:
                    print(f"{k}: shape: {v.shape}, dtype: {v.dtype}")
                else:
                    print(f"{k}: {v}")
            else:
                print(f"{k}: {v}")

    print()


if __name__ == "__main__":
    main()
