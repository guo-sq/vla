from collections.abc import Iterator, Sequence
from collections import Counter
import logging
import multiprocessing
import os
import typing
import dataclasses
import time
from typing import Literal, Protocol, SupportsIndex, TypeVar
from typing import List, Generic

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch
from torch.utils.data import ConcatDataset

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
from openpi.training.anyverse_dataset import MultiAnyverseDataset
import openpi.transforms as _transforms
from openpi.training.shared_cache import SharedEpisodeCache, init_shared_cache

T_co = TypeVar("T_co", covariant=True)


### ---实现 weighted dataset mixture --- ###
# class WeightedMixedDataset(Dataset):
#     """按权重混合多个数据集的自定义数据集"""
#     def __init__(self, datasets, weights=None):
#         self.datasets = datasets
#         self.concat_dataset = ConcatDataset(datasets)
#         # 计算每个数据集的样本数
#         self.dataset_sizes = [len(ds) for ds in datasets]
#         # 默认为等权重，若指定权重则按权重分配
#         self.weights = weights if weights is not None else [1.0 / len(datasets)] * len(datasets)
#         # 为每个样本分配权重（同一数据集内的样本权重相同）
#         self.sample_weights = []
#         for i, size in enumerate(self.dataset_sizes):
#             self.sample_weights.extend([self.weights[i]] * size)

#     def __len__(self):
#         return len(self.concat_dataset)

#     def __getitem__(self, index):
#         return self.concat_dataset[index]

# # 在 create_mixture_torch_dataset 中使用
# def create_mixture_torch_dataset(
#     data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
# ) -> Dataset:
#     # （前面逻辑同方案1：加载单个数据集到 datasets 列表）
#     # ...（省略校验和单个数据集加载代码）...

#     # 按权重混合（示例：给每个 repo_id 分配权重，如 [0.6, 0.4]）
#     # 可从 data_config 中读取权重，这里简化为手动指定
#     weights = [0.5, 0.5]  # 假设两个数据集各占 50%
#     if len(weights) != len(datasets):
#         raise ValueError("Number of weights must match number of datasets.")

#     combined_dataset = WeightedMixedDataset(datasets, weights)
#     return combined_dataset

# # 使用时，可配合 DataLoader 和 WeightedRandomSampler 实现按权重采样
# # 示例：
# # dataset = create_mixture_torch_dataset(...)
# # sampler = WeightedRandomSampler(dataset.sample_weights, len(dataset))
# # dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

### ---实现 weighted dataset mixture 结束--- ###


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError(
            "Subclasses of IterableDataset should implement __iter__."
        )

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError(
            "Subclasses of DataLoader should implement data_config."
        )

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(
        self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        # print(f"--TransformedDataset __getitem__ index: {index}--- \n dataset item: {self._dataset[index]}")
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


##--mixture concat dataset
class ConcatDataset(Dataset[T_co], Generic[T_co]):
    """拼接多个Dataset（包括TransformedDataset）的实现，遵循Dataset协议"""

    def __init__(self, datasets: List[Dataset[T_co]]):
        self.datasets = datasets  # 存储多个TransformedDataset实例
        # 计算累计长度（用于索引映射）
        self.cumulative_sizes = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cumulative_sizes.append(total)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        index = int(index)  # 转换为整数索引
        # 找到索引属于哪个子数据集
        for i, cumulative_size in enumerate(self.cumulative_sizes):
            if index < cumulative_size:
                # 计算在子数据集中的相对索引
                if i == 0:
                    sub_index = index
                else:
                    sub_index = index - self.cumulative_sizes[i - 1]
                return self.datasets[i][sub_index]  # 调用子数据集的__getitem__
        # 索引越界
        raise IndexError(
            f"Index {index} out of range for ConcatDataset with total length {self.__len__()}"
        )

    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [
                    jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)
                ]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(
                    data_rng, shape=shape, minval=-1.0, maxval=1.0
                )
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)
        observation_dict = {
            key: value
            for key, value in observation.to_dict().items()
            if value is not None
        }

        return {
            **observation_dict,
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    root = os.path.join(data_config.root_dir, repo_id)
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=root)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        root=root,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)]
            for key in data_config.action_sequence_keys
        },
        episodes=data_config.episode,
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(
            dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)]
        )

    if data_config.prompt_from_episode:
        dataset = TransformedDataset(
            dataset, [_transforms.PromptFromEpisodeTask(dataset_meta.episodes)]
        )

    return dataset


def create_anyverse_dataset(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
) -> Dataset:
    """Create a dataset for training."""
    if data_config.repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if data_config.repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    return MultiAnyverseDataset(
        data_config,
        model_config.action_horizon,
    )


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        filter_dict_path=data_config.filter_dict_path,
    )


def transform_dataset(
    dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False
) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = None
    # print(f"---data_config in transform_dataset------:\n {data_config}")
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.public_dataset_map_transform.inputs,
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = None
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


# 训练时的数据入口
def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    data_split: Literal["train", "val", "test"] | None = None,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_factory = config.data
    if data_split is not None and hasattr(data_factory, "split"):
        data_factory = dataclasses.replace(data_factory, split=data_split)

    data_config = data_factory.create(config.assets_dirs, config.model)
    logging.info(f"---data_config: {data_config}")
    data_config = dataclasses.replace(
        data_config,
        episode_fail=config.data.episode_fail,
        dataset_length=config.data.dataset_length,
    )

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    # print(f"--data_loader:{config}")
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


# --- 多节点训练入口 ---
def create_distributed_torch_data_loader(
    config: _config.TrainConfig,
    rank: int,
    world_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = True,
    num_batches: int | None = None,
):
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"---data_config: {data_config}")
    data_config = dataclasses.replace(
        data_config,
        episode_fail=config.data.episode_fail,
        dataset_length=config.data.dataset_length,
    )

    # 1. 初始化共享缓存（只在 rank 0 初始化，其他进程会检测到已存在）
    # lazy_load=True 时自动使用共享缓存，配置通过环境变量控制:
    #   - OPENPI_SHARED_CACHE_DIR: 缓存目录 (默认: /dev/shm/openpi_cache)
    #   - OPENPI_SHARED_CACHE_SIZE_GB: 最大缓存大小 GB (默认: 1000.0)
    if data_config.lazy_load:
        cache_init_start = time.time()
        try:
            init_shared_cache()
            logging.info(
                f"[DataLoader] Rank {rank}: Shared cache initialized in {time.time() - cache_init_start:.2f}s"
            )
        except Exception as e:
            logging.warning(
                f"[DataLoader] Rank {rank}: Failed to init shared cache: {e}"
            )

    # 2. 创建 DataConfig
    logging.info(
        f"[DataLoader] Rank {rank}: Starting data loader creation (lazy_load={data_config.lazy_load})"
    )
    print(f"data_config:{data_config}")

    # 3. 创建数据集
    ds_start = time.time()
    dataset = create_anyverse_dataset(data_config, config.model)
    logging.info(
        f"[DataLoader] Rank {rank}: Dataset created in {time.time() - ds_start:.2f}s"
    )

    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # 5. 分布式采样器 (多机 Loss 正常的关键)
    sampler = torch.utils.data.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=config.seed,
        drop_last=True,
    )

    local_batch_size = config.batch_size // world_size
    # if config.batch_size % world_size != 0:
    #     raise ValueError(f"Batch size {config.batch_size} not divisible by world size {world_size}")
    print(
        f"----->rank {rank}: global batch size {config.batch_size}, local batch size {local_batch_size}",
        flush=True,
    )
    # 7. 构造 PyTorch DataLoader
    mp_context = (
        multiprocessing.get_context("spawn") if config.num_workers > 0 else None
    )
    torch_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=local_batch_size,
        num_workers=config.num_workers,
        sampler=sampler,
        collate_fn=_collate_fn,
        persistent_workers=config.num_workers > 0,
        drop_last=True,
        multiprocessing_context=mp_context,
    )

    # 8. 包装为 JAX 适配器
    data_loader = DistributedTorchDataLoader(
        torch_loader,
        sharding=sharding,
        num_batches=num_batches,
    )
    return DataLoaderImpl(data_config, data_loader)


class DistributedTorchDataLoader:
    """分布式数据加载器，支持 JAX 多机训练"""

    def __init__(self, data_loader, sharding=None, num_batches=None):
        self._data_loader = data_loader
        self._num_batches = num_batches
        self._sharding = sharding or jax.sharding.NamedSharding(
            jax.sharding.Mesh(jax.devices(), ("B",)), jax.sharding.PartitionSpec("B")
        )
        self._epoch = int(0)

        # 多机环境下，确保所有进程使用相同的初始 epoch
        sync_devices(f"Epoch {self._epoch} start")

    def __iter__(self):
        num_items = 0
        max_retries = 3  # 最大重试次数

        while True:
            # 设置 epoch 以确保每个进程使用不同的随机种子
            if hasattr(self._data_loader.sampler, "set_epoch"):
                self._data_loader.sampler.set_epoch(self._epoch)

            # # 多机环境下同步 epoch
            # sync_devices(f"Epoch {self._epoch} start")

            retry_count = 0
            while retry_count < max_retries:
                try:
                    for batch in self._data_loader:
                        if (
                            self._num_batches is not None
                            and num_items >= self._num_batches
                        ):
                            return

                        num_items += 1

                        # 将 PyTorch tensor 转换为 JAX 数组，并进行分片
                        yield jax.tree.map(
                            lambda x: jax.make_array_from_process_local_data(
                                self._sharding, x.numpy() if torch.is_tensor(x) else x
                            ),
                            batch,
                        )
                    break  # 成功完成，跳出重试循环
                except Exception as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        logging.error(
                            f"Data loading failed after {max_retries} retries: {e}"
                        )
                        raise
                    logging.warning(
                        f"Data loading failed (attempt {retry_count}/{max_retries}): {e}"
                    )
                    # 多机环境下同步重试
                    sync_devices(f"Data loading retry {retry_count}")

            self._epoch = int(self._epoch + 1)

            if self._num_batches is not None and num_items >= self._num_batches:
                break


def is_distributed():
    """Check if we're running in distributed mode."""
    return jax.process_count() > 1


def sync_devices(name: str):
    """Synchronize all devices in distributed mode."""
    if is_distributed():
        from jax.experimental import multihost_utils

        multihost_utils.sync_global_devices(name)


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    print("---create_torch_data_loader---")
    print(f"data_config:{data_config}")

    dataset = create_anyverse_dataset(data_config, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        # For distributed JAX, adjust batch size accordingly
        local_batch_size = (
            batch_size // jax.process_count() if is_distributed() else batch_size
        )

    weighted_sampler = _build_repo_weighted_sampler(dataset, data_config, seed=seed)
    if weighted_sampler is not None:
        if sampler is not None:
            logging.warning(
                "repo_sampling_weights is ignored because DistributedSampler is active."
            )
        else:
            sampler = weighted_sampler
            shuffle = False

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(
        data_config, action_horizon, batch_size, shuffle=shuffle
    )
    dataset = transform_iterable_dataset(
        dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True
    )

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        pin_memory: bool = True,
        prefetch_factor: int | None = None,
        persistent_workers: bool = True,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        # Support both single-node and multi-node training
        # if jax.process_count() > 1:
        #     raise NotImplementedError(
        #         "Data loading with multiple processes is not supported."
        #     )

        if len(dataset) < local_batch_size:
            raise ValueError(
                f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)})."
            )

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        loader_kwargs = {}
        if num_workers > 0 and prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0 and persistent_workers,
            pin_memory=pin_memory,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
            **loader_kwargs,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(
                        lambda x: jax.make_array_from_process_local_data(
                            self._sharding, x
                        ),
                        batch,
                    )
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(
        lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items
    )


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        # Support both single-node and multi-node training
        # if jax.process_count() > 1:
        #     raise NotImplementedError(
        #         "Data loading with multiple processes is not supported."
        #     )

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(
                    lambda x: jax.make_array_from_process_local_data(self._sharding, x),
                    batch,
                )


class DataLoaderImpl(DataLoader):
    def __init__(
        self,
        data_config: _config.DataConfig,
        data_loader: TorchDataLoader | RLDSDataLoader,
    ):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            if "actions_mask" in batch:
                actions_mask = batch["actions_mask"]
            else:
                target_shape = batch["actions"].shape[:2]
                actions_mask = jnp.ones(target_shape, dtype=jnp.bool_)

            yield _model.Observation.from_dict(batch), batch["actions"], actions_mask


class RepoWeightedRandomSampler(torch.utils.data.Sampler[int]):
    """Random sampler that first samples repo id by weight, then samples inside that repo."""

    def __init__(
        self,
        *,
        dataset_sizes: list[int],
        repo_weights: list[float],
        num_samples: int,
        seed: int = 0,
    ):
        if len(dataset_sizes) != len(repo_weights):
            raise ValueError(
                f"dataset_sizes length {len(dataset_sizes)} != repo_weights length {len(repo_weights)}"
            )
        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {num_samples}")

        self._dataset_sizes = [int(s) for s in dataset_sizes]
        self._repo_weights = [float(w) for w in repo_weights]
        self._num_samples = int(num_samples)
        self._seed = int(seed)

        if any(size <= 0 for size in self._dataset_sizes):
            raise ValueError("All dataset_sizes must be > 0 for weighted repo sampling")
        if any(weight < 0 for weight in self._repo_weights):
            raise ValueError("repo_weights must be >= 0")
        if sum(self._repo_weights) <= 0:
            raise ValueError("At least one repo weight must be > 0")

        self._repo_probs = torch.tensor(self._repo_weights, dtype=torch.float)
        self._repo_probs = self._repo_probs / self._repo_probs.sum()

        offsets = []
        total = 0
        for size in self._dataset_sizes:
            offsets.append(total)
            total += size
        self._dataset_offsets = offsets

        self._generator = torch.Generator()
        self._generator.manual_seed(self._seed)

    def __iter__(self):
        for _ in range(self._num_samples):
            repo_index = int(
                torch.multinomial(
                    self._repo_probs,
                    1,
                    replacement=True,
                    generator=self._generator,
                ).item()
            )
            local_index = int(
                torch.randint(
                    low=0,
                    high=self._dataset_sizes[repo_index],
                    size=(1,),
                    generator=self._generator,
                ).item()
            )
            yield self._dataset_offsets[repo_index] + local_index

    def __len__(self) -> int:
        return self._num_samples


def _resolve_repo_weight(
    repo_id: str,
    weight_spec: dict[str, float] | list[float] | None,
    repo_index: int,
) -> float:
    if weight_spec is None:
        return 1.0

    if isinstance(weight_spec, list):
        if repo_index >= len(weight_spec):
            return 1.0
        return float(weight_spec[repo_index])

    if repo_id in weight_spec:
        return float(weight_spec[repo_id])

    for pattern, value in weight_spec.items():
        if pattern.endswith("*") and repo_id.startswith(pattern[:-1]):
            return float(value)

    return 1.0


def _build_repo_weighted_sampler(
    dataset,
    data_config: _config.DataConfig,
    *,
    seed: int,
) -> RepoWeightedRandomSampler | None:
    weight_spec = data_config.repo_sampling_weights
    if not weight_spec:
        return None

    repo_ids = getattr(dataset, "repo_ids", None)
    sub_datasets = getattr(dataset, "_datasets", None)
    if not isinstance(repo_ids, list) or not isinstance(sub_datasets, list):
        logging.warning(
            "repo_sampling_weights is set but dataset type does not expose repo_ids/_datasets, skipping weighted sampling."
        )
        return None

    dataset_sizes = [len(ds) for ds in sub_datasets]
    repo_weights = [
        _resolve_repo_weight(repo_id, weight_spec, idx)
        for idx, repo_id in enumerate(repo_ids)
    ]

    if all(weight == 1.0 for weight in repo_weights):
        return None

    if any(weight < 0 for weight in repo_weights):
        raise ValueError("repo_sampling_weights must be >= 0")

    weight_summary = Counter()
    for repo_id, weight in zip(repo_ids, repo_weights):
        weight_summary[weight] += 1
    logging.info(
        "Using repo-weighted sampler with %d repos; weight distribution: %s",
        len(repo_ids),
        dict(weight_summary),
    )

    return RepoWeightedRandomSampler(
        dataset_sizes=dataset_sizes,
        repo_weights=repo_weights,
        num_samples=len(dataset),
        seed=seed,
    )
