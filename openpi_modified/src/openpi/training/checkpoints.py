from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import json
import logging
import shutil
from typing import Protocol

from etils import epath
import jax
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


def save_config_to_checkpoint(
    checkpoint_dir: epath.Path | str,
    config,
    config_name: str | None = None,
    config_file_path: str | None = None,
) -> None:
    """Save the training config to the checkpoint directory.

    If config_file_path is provided, copies the .py config file directly.
    Otherwise, generates a .py file from the TrainConfig object.

    Args:
        checkpoint_dir: The checkpoint directory to save the config to.
        config: The TrainConfig object or path to config file.
        config_name: Name of the config (used to generate filename).
        config_file_path: Path to the original .py config file (if available).
    """
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()

    # Determine config name
    if config_name is None:
        config_name = getattr(config, "name", "config")

    config_path = checkpoint_dir / f"{config_name}.py"

    # Only save on rank 0 in distributed setting
    if jax.process_index() != 0:
        return

    try:
        if config_file_path is not None:
            # Copy the original .py config file
            shutil.copy2(config_file_path, config_path)
            logging.info(f"Copied config file to {config_path}")
        else:
            # Generate .py file from config
            config_dict = dataclasses.asdict(config)

            # Convert pathlib.Path objects to strings for JSON serialization
            def convert_paths(obj):
                if isinstance(obj, epath.Path):
                    return str(obj)
                if isinstance(obj, dict):
                    return {k: convert_paths(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [convert_paths(item) for item in obj]
                return obj

            config_dict = convert_paths(config_dict)

            # Save as JSON with .py extension (wrapped in comment for readability)
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(f"# Config: {config_name}\n")
                f.write("# Auto-generated config file\n\n")
                f.write("config = ")
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            logging.info(f"Saved config to {config_path}")
    except Exception as e:
        logging.warning(f"Failed to save config: {e}")


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str,
    *,
    keep_period: int | None,
    overwrite: bool,
    resume: bool,
    config=None,
    config_file_path: str | None = None,
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False

    # 在多机环境下，只在rank 0上执行目录操作
    if jax.process_index() == 0:
        if checkpoint_dir.exists():
            if overwrite:
                checkpoint_dir.rmtree()
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
            elif resume:
                resuming = True
            else:
                raise FileExistsError(
                    f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                    "to indicate how to handle it."
                )
        else:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save config to checkpoint directory for version control (only for new runs)
        if config is not None and not resuming:
            save_config_to_checkpoint(checkpoint_dir, config, config_file_path=config_file_path)

    # 多机环境下同步所有进程
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils

        multihost_utils.sync_global_devices("Checkpoint directory initialized")

    # 延迟创建CheckpointManager，避免在NCCL未完全初始化时同步
    def create_manager():
        return ocp.CheckpointManager(
            checkpoint_dir,
            item_handlers={
                "assets": CallbackHandler(),
                "train_state": ocp.PyTreeCheckpointHandler(use_ocdbt=True),
                "params": ocp.PyTreeCheckpointHandler(use_ocdbt=True),
            },
            options=ocp.CheckpointManagerOptions(
                max_to_keep=1,
                keep_period=keep_period,
                create=False,
                async_options=ocp.AsyncOptions(timeout_secs=7200),
            ),
        )

    # 先在rank 0上创建manager并检查是否有checkpoint
    if jax.process_index() == 0:
        mngr = create_manager()
        # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
        # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
        # checkpoint, since it will fail.
        if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
            logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
            resuming = False
        # 广播resuming状态到所有进程
        if jax.process_count() > 1:
            from jax.experimental import multihost_utils

            resuming = multihost_utils.broadcast_one_to_all(resuming)
    else:
        # 其他进程创建manager
        mngr = create_manager()
        # 其他进程接收resuming状态
        if jax.process_count() > 1:
            from jax.experimental import multihost_utils

            resuming = multihost_utils.broadcast_one_to_all(resuming)

    # 所有进程在此同步，确保所有节点完成checkpoint manager初始化
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils

        multihost_utils.sync_global_devices("Checkpoint manager initialization complete")

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None:
            if data_config.asset_id is not None:
                asset_dir = data_config.asset_id
            elif isinstance(data_config.repo_id, list):
                asset_dir = "_".join(data_config.repo_id)
            else:
                asset_dir = data_config.repo_id
            logging.info(f"Saving norm stats to {directory / asset_dir}")
            _normalize.save(directory / asset_dir, norm_stats)

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)

        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        default_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

        # Create PyTree restore args mapping so Orbax will restore arrays using our
        # `default_sharding` (mapping saved shards into current device layout).
        train_state_restore = ocp.args.PyTreeRestore(
            item=train_state,
            restore_args=jax.tree_map(
                lambda _: ocp.ArrayRestoreArgs(sharding=default_sharding, restore_type=jax.Array),
                train_state,
            ),
        )

        params_item = {"params": params}
        params_restore = ocp.args.PyTreeRestore(
            item=params_item,
            restore_args=jax.tree_map(
                lambda _: ocp.ArrayRestoreArgs(sharding=default_sharding, restore_type=jax.Array),
                params_item,
            ),
        )

        restored = checkpoint_manager.restore(
            step,
            args=ocp.args.Composite(train_state=train_state_restore, params=params_restore),
        )
    return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    if isinstance(asset_id, list):
        asset_id = "_".join(asset_id)
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(
    state: training_utils.TrainState,
) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
