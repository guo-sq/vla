from collections import Counter
from collections.abc import Iterator, Sequence
import dataclasses
import logging
import multiprocessing
import os
import typing
from typing import Generic, Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
from openpi.training.rl_dataset import LeRobotRLDataset
from openpi.training.rl_dataset import MultiRLAnyverseDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


### ---实现 weighted dataset mixture --- ###
# class WeightedMixedDataset(Dataset):
#     """按权重混合多个数据集的自定义数据集"""
#     def __init__(self, datasets, weights=None):
#         self.datasets = datasets
#         self.concat_dataset = ConcatDataset(datasets)
#         # 计算每个数据集的样本数
#         self.dataset_sizes = [len(ds) for ds in datasets]
#         # 默认为等权重,若指定权重则按权重分配
#         self.weights = weights if weights is not None else [1.0 / len(datasets)] * len(datasets)
#         # 为每个样本分配权重(同一数据集内的样本权重相同)
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
#     # (前面逻辑同方案1:加载单个数据集到 datasets 列表)
#     # ...(省略校验和单个数据集加载代码)...

#     # 按权重混合(示例:给每个 repo_id 分配权重,如 [0.6, 0.4])
#     # 可从 data_config 中读取权重,这里简化为手动指定
#     weights = [0.5, 0.5]  # 假设两个数据集各占 50%
#     if len(weights) != len(datasets):
#         raise ValueError("Number of weights must match number of datasets.")

#     combined_dataset = WeightedMixedDataset(datasets, weights)
#     return combined_dataset

# # 使用时,可配合 DataLoader 和 WeightedRandomSampler 实现按权重采样
# # 示例:
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
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedRLDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        # print(f"--TransformedDataset __getitem__ index: {index}--- \n dataset item: {self._dataset[index]}")
        # cur, nxt = self._dataset[index]
        # return self._transform(cur), self._transform(nxt)
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


##--mixture concat dataset
class ConcatDataset(Dataset[T_co], Generic[T_co]):
    """拼接多个Dataset(包括TransformedDataset)的实现,遵循Dataset协议"""

    def __init__(self, datasets: list[Dataset[T_co]]):
        self.datasets = datasets  # 存储多个TransformedDataset实例
        # 计算累计长度(用于索引映射)
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
                sub_index = index if i == 0 else index - self.cumulative_sizes[i - 1]
                return self.datasets[i][sub_index]  # 调用子数据集的__getitem__
        # 索引越界
        raise IndexError(f"Index {index} out of range for ConcatDataset with total length {self.__len__()}")

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
                individual_samples = [jax.tree.map(lambda x, _i=i: x[_i], sample) for i in range(batch_size)]

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
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_rl_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    episode_fail: int | None = None,
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    root = os.path.join(data_config.root_dir, repo_id)
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=root)
    dataset = LeRobotRLDataset(
        data_config.repo_id,
        root=root,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        episodes=data_config.episode,
        episode_fail=episode_fail,
    )

    if data_config.prompt_from_task:
        dataset = TransformedRLDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    if data_config.prompt_from_episode:
        dataset = TransformedRLDataset(dataset, [_transforms.PromptFromEpisodeTask(dataset_meta.episodes)])

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

    value_net_cfg = getattr(data_config, "value_net_cfg", None) or {}
    if (
        value_net_cfg.get("returns_norm_strategy") == "per_task"
        and value_net_cfg.get("pinned_task_to_norm_length") is None
    ):
        raise ValueError(
            "per_task strategy requires precomputed pinned stats, but none were "
            "loaded into value_net_cfg. Run: "
            "python scripts/compute_rl_norm_stats.py --config <your-config>  "
            "to generate rl_norm_stats.json for every repo before training."
        )

    return MultiRLAnyverseDataset(data_config, model_config.action_horizon)


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


def transform_rl_dataset(
    dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False
) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    # print(f"---data_config in transform_dataset------:\n {data_config}")
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedRLDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
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
    norm_stats = {}
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
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


# 训练时的数据入口
def create_rl_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
    drop_last: bool = True,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
        drop_last: Whether to drop the last incomplete batch. Set to False for inference
            to ensure all frames are processed.
    """
    data_config = config.data.create(config.assets_dirs, config.model)
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
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
        episode_fail=config.data.episode_fail,
        dataset_length=config.data.dataset_length,
        drop_last=drop_last,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
    episode_fail: list[list[int]] | None = None,
    dataset_length: list[int] | None = None,
    drop_last: bool = True,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

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
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
        drop_last: Whether to drop the last incomplete batch. Set to False for inference
            to ensure all frames are processed.
    """
    print("---create_torch_data_loader---")
    print(f"data_config:{data_config}")
    dataset = create_anyverse_dataset(data_config, model_config)
    dataset = transform_rl_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchRLDataLoader for both frameworks
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
        local_batch_size = batch_size // jax.process_count()

    weighted_sampler = _build_repo_weighted_sampler(dataset, data_config, seed=seed)
    if weighted_sampler is not None:
        if sampler is not None:
            logging.warning("repo_sampling_weights is ignored because DistributedSampler is active.")
        else:
            sampler = weighted_sampler
            shuffle = False

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchRLDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
        drop_last=drop_last,
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
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchRLDataLoader:
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
        seed: int = 0,
        framework: str = "jax",
        drop_last: bool = True,
        target_batch_size: int | None = None,
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
            drop_last: Whether to drop the last incomplete batch. Set to False for inference
                to ensure all frames are processed.
            target_batch_size: If set, every batch is padded up to this fixed shape regardless
                of how many real frames it contains. Required by sparse inference because the
                XLA shape contract must stay constant across batches. Pad rows are tagged with
                ``frame_index = -1`` so ``sparse_inference_loop`` can filter them out.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches
        self._target_batch_size = target_batch_size

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=drop_last,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def _pad_batch(self, batch, num_devices: int):
        """Pad dimension 0 to ``target_batch_size`` (if set) or a multiple of ``num_devices``.

        Returns ``(padded_batch, original_batch_size)``. Pad rows carry ``frame_index = -1``
        so downstream consumers (e.g. ``sparse_inference_loop``) can filter them out instead
        of treating them as real frames and silently double-counting the last real row.
        """
        batch_size = next(iter(jax.tree.leaves(batch))).shape[0]
        if self._target_batch_size is not None:
            target = self._target_batch_size
        else:
            remainder = batch_size % num_devices
            target = batch_size if remainder == 0 else batch_size + (num_devices - remainder)
        if target == batch_size:
            return batch, batch_size
        pad_size = target - batch_size

        def _pad(x):
            pad_width = [(0, pad_size)] + [(0, 0)] * (len(x.shape) - 1)
            return np.pad(x, pad_width, mode="edge")

        padded = jax.tree.map(_pad, batch)
        if isinstance(padded, dict) and "frame_index" in padded:
            padded["frame_index"][batch_size:] = -1
        return padded, batch_size

    def __iter__(self):
        num_items = 0
        num_devices = jax.local_device_count() if self._sharding is not None else 1
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
                    batch, _ = self._pad_batch(batch, num_devices)
                    yield jax.tree.map(
                        lambda x: jax.make_array_from_process_local_data(self._sharding, x),
                        batch,
                    )
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    # print(f"---_collate_fn items: {items} ---")
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


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

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

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
        data_loader: TorchRLDataLoader | RLDSDataLoader,
    ):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        # for batch_cur, batch_nxt in self._data_loader:
        #     yield _model.Observation.from_dict(batch_cur), batch_cur["actions"], \
        #         _model.Observation.from_dict(batch_nxt), batch_nxt["actions"]
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
            raise ValueError(f"dataset_sizes length {len(dataset_sizes)} != repo_weights length {len(repo_weights)}")
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
    repo_weights = [_resolve_repo_weight(repo_id, weight_spec, idx) for idx, repo_id in enumerate(repo_ids)]

    if all(weight == 1.0 for weight in repo_weights):
        return None

    if any(weight < 0 for weight in repo_weights):
        raise ValueError("repo_sampling_weights must be >= 0")

    weight_summary = Counter()
    for _repo_id, weight in zip(repo_ids, repo_weights, strict=False):
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


class SparseFrameSampler(torch.utils.data.Sampler[int]):
    """Sampler that yields only head + tail (+ optional middle) frames per episode.

    Reads ``dataset.episode_mapping`` (and cumulative offsets for multi-dataset wrappers)
    to build a fixed, deterministic list of global position indices. Designed for sparse
    inference where only the first and last frame predictions per episode are needed
    (e.g., head_pred/tail_pred for the 2D episode classifier), turning a 16h dense sweep
    into a ~30min sparse sweep.

    Supports:
    - Single datasets exposing ``episode_mapping`` and (optionally) ``_valid_frame_indices``
      for raw-frame -> position conversion.
    - ``MultiRLAnyverseDataset``-like wrappers exposing ``_datasets`` and ``cum_sizes``.

    Args:
        dataset: A dataset (or multi-dataset wrapper) exposing ``episode_mapping``.

    Example:
        >>> sampler = SparseFrameSampler(dataset)
        >>> loader = DataLoader(dataset, sampler=sampler, batch_size=16)
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.sparse_indices: list[int] = self._build_indices()

    @staticmethod
    def _unwrap(dataset):
        """Drill through ``TransformedDataset``-style wrappers that only hold
        a single ``_dataset`` attribute, until we reach the concrete RL dataset
        that actually exposes ``episode_mapping`` / ``_datasets``.
        """
        while (
            getattr(dataset, "episode_mapping", None) is None
            and getattr(dataset, "_datasets", None) is None
            and getattr(dataset, "_dataset", None) is not None
        ):
            dataset = dataset._dataset  # noqa: SLF001 — unwrap wrapper pattern
        return dataset

    def _build_indices(self) -> list[int]:
        inner = self._unwrap(self.dataset)
        sub_datasets = getattr(inner, "_datasets", None)
        if sub_datasets is None:
            sub_datasets = [inner]
            offsets = [0]
        else:
            cum_sizes = getattr(self.dataset, "cum_sizes", None)
            if cum_sizes is None:
                cum_sizes = []
                total = 0
                for sub_ds in sub_datasets:
                    total += len(sub_ds)
                    cum_sizes.append(total)
            offsets = [0] + [int(cs) for cs in cum_sizes[:-1]]

        indices: list[int] = []
        for offset, sub_ds in zip(offsets, sub_datasets, strict=False):
            indices.extend(self._extract_sparse_from_episode(sub_ds, offset))

        return sorted(set(indices))

    def _extract_sparse_from_episode(self, sub_ds, offset: int) -> list[int]:
        ep_map: dict[int, tuple[int, int]] = getattr(sub_ds, "episode_mapping", {}) or {}
        valid_frame_indices = getattr(sub_ds, "_valid_frame_indices", None)

        out: list[int] = []
        for _ep_from, (raw_start, raw_end) in sorted(ep_map.items()):
            head_pos = self._raw_to_position(raw_start, valid_frame_indices)
            tail_pos = self._raw_to_position(raw_end, valid_frame_indices)

            out.append(offset + head_pos)
            if tail_pos != head_pos:
                out.append(offset + tail_pos)

        return out

    @staticmethod
    def _raw_to_position(raw_index: int, valid_frame_indices) -> int:
        """Convert a raw frame index to its position inside ``_valid_frame_indices``.

        ``episode_mapping`` stores raw frame indices (values of ``sorted_indices[left]``),
        but ``Dataset.__getitem__`` interprets its argument as a position in
        ``_sampler_indices`` / ``_valid_frame_indices``. For mock datasets that omit
        ``_valid_frame_indices``, fall through to identity.
        """
        if valid_frame_indices is None:
            return int(raw_index)
        return int(torch.searchsorted(valid_frame_indices, torch.tensor(raw_index)).item())

    def __iter__(self) -> Iterator[int]:
        return iter(self.sparse_indices)

    def __len__(self) -> int:
        return len(self.sparse_indices)


def _build_rl_sampler(
    dataset,
    data_config: "_config.DataConfig | None",
    *,
    seed: int,
    existing_sampler: torch.utils.data.Sampler | None = None,
    sparse_frame_sampling: bool = False,
) -> tuple[torch.utils.data.Sampler | None, bool]:
    """Select the sampler for an RL torch data loader and report whether shuffle must be forced off.

    Priority order (highest first):

    1. ``sparse_frame_sampling=True`` -> :class:`SparseFrameSampler` (forces shuffle off,
       because sparse indices are pre-sorted and downstream inference expects that order).
    2. ``existing_sampler`` -- e.g. a ``DistributedSampler`` already constructed by the
       caller -- is passed through untouched; it manages its own shuffle.
    3. ``data_config.repo_sampling_weights`` -> :class:`RepoWeightedRandomSampler`, also
       forces shuffle off (weighted sampling supplies its own order).
    4. No sampler -> return ``(None, False)`` and let the caller keep its shuffle flag.

    Args:
        dataset: The dataset the sampler will index into.
        data_config: Data config used only to look up repo sampling weights. ``None`` is
            accepted for unit tests that bypass the weighted-sampler branch entirely.
        seed: Seed passed through to weighted sampling.
        existing_sampler: An already-constructed sampler (e.g., ``DistributedSampler``)
            that should take precedence over weighted sampling but yield to sparse mode.
        sparse_frame_sampling: When ``True``, build a :class:`SparseFrameSampler`.

    Returns:
        ``(sampler, force_shuffle_off)``. The caller must disable shuffle iff
        ``force_shuffle_off`` is ``True``.
    """
    if sparse_frame_sampling:
        sparse_sampler = SparseFrameSampler(dataset)
        logging.info("Sparse frame sampling enabled: %d indices", len(sparse_sampler))
        return sparse_sampler, True

    if existing_sampler is not None:
        return existing_sampler, False

    if data_config is None:
        return None, False

    weighted = _build_repo_weighted_sampler(dataset, data_config, seed=seed)
    if weighted is not None:
        return weighted, True

    return None, False
