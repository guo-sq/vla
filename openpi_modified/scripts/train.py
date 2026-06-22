import os

MASTER_ADDR = os.environ.get("MASTER_ADDR", None)
MASTER_PORT = os.environ.get("MASTER_PORT", None)
# Ensure temporary files do not go to /tmp (which may be small on cluster nodes).
_tmp_base = os.environ.get("OPENPI_TMPDIR") or "/root/.cache/tmp"
try:
    os.makedirs(_tmp_base, exist_ok=True)
    os.environ["TMPDIR"] = _tmp_base
    os.environ["TEMP"] = _tmp_base
    os.environ["TMP"] = _tmp_base
    os.environ["CUDA_CACHE_PATH"] = os.path.join(_tmp_base, "cuda_cache")
    os.makedirs(os.environ["CUDA_CACHE_PATH"], exist_ok=True)
except Exception:
    pass
import dataclasses
import functools
import logging
import platform
import queue
import threading
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import tyro
import wandb

from openpi.configs.robot_cfg.base import AlignActionDim
import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

os.environ.setdefault("WANDB_MODE", "online")


def _action_dim_names(action_dim: int) -> list[str]:
    names = AlignActionDim.names()
    out: list[str] = []
    for idx in range(int(action_dim)):
        if idx < len(names):
            out.append(names[idx])
        else:
            out.append(f"ACTION_DIM_{idx}")
    return out


def _expand_action_dim_loss_metrics(
    per_dim_loss: np.ndarray,
    *,
    topk: int,
) -> dict[str, float]:
    values = np.asarray(per_dim_loss)
    if values.ndim != 1:
        values = values.reshape(-1)

    dim_names = _action_dim_names(int(values.shape[0]))
    if topk > 0 and topk < values.shape[0]:
        order = np.argsort(values)[::-1]
        keep = sorted(order[:topk].tolist())
    else:
        keep = list(range(values.shape[0]))

    return {f"loss/action_dim/{dim_names[i]}": float(values[i]) for i in keep}


def _expand_action_dim_count_metrics(
    per_dim_count: np.ndarray,
    *,
    topk: int,
    ref_per_dim_loss: np.ndarray | None = None,
) -> dict[str, float]:
    values = np.asarray(per_dim_count)
    if values.ndim != 1:
        values = values.reshape(-1)

    dim_names = _action_dim_names(int(values.shape[0]))

    if topk > 0 and topk < values.shape[0] and ref_per_dim_loss is not None:
        ref = np.asarray(ref_per_dim_loss).reshape(-1)
        ref = np.where(np.isfinite(ref), ref, -np.inf)
        order = np.argsort(ref)[::-1]
        keep = sorted(order[:topk].tolist())
    else:
        keep = list(range(values.shape[0]))

    return {f"loss/action_dim_count/{dim_names[i]}": float(values[i]) for i in keep}


def _summarize_action_dim_validity(
    per_dim_count: np.ndarray,
) -> dict[str, float]:
    values = np.asarray(per_dim_count)
    if values.ndim != 1:
        values = values.reshape(-1)

    total_num = int(values.shape[0])
    valid_num = int(np.sum(values > 0))
    valid_ratio = float(valid_num / total_num) if total_num > 0 else 0.0

    return {
        "loss/action_dim_total_num": float(total_num),
        "loss/action_dim_valid_num": float(valid_num),
        "loss/action_dim_valid_ratio": valid_ratio,
    }


def _summarize_tensor(name: str, x: np.ndarray) -> str:
    arr = np.asarray(x)
    total = arr.size
    finite_mask = np.isfinite(arr)
    finite_count = int(finite_mask.sum())
    if total == 0:
        return f"{name}: empty"
    if finite_count == 0:
        return f"{name}: shape={arr.shape}, finite=0/{total}"

    finite_vals = arr[finite_mask]
    abs_vals = np.abs(finite_vals)
    q95 = float(np.percentile(abs_vals, 95))
    q99 = float(np.percentile(abs_vals, 99))
    vmax = float(np.max(abs_vals))
    return (
        f"{name}: shape={arr.shape}, finite={finite_count}/{total}, "
        f"|x|_p95={q95:.4g}, |x|_p99={q99:.4g}, |x|_max={vmax:.4g}"
    )


def log_batch_numeric_health(
    batch: tuple[_model.Observation, _model.Actions, _model.ActionsMask],
) -> None:
    observation, actions, actions_mask = batch

    state_np = np.asarray(multihost_utils.process_allgather(observation.state))
    actions_np = np.asarray(multihost_utils.process_allgather(actions))
    mask_np = np.asarray(multihost_utils.process_allgather(actions_mask)).astype(bool)

    mask_ratio = float(mask_np.mean()) if mask_np.size else 0.0
    if mask_np.ndim == actions_np.ndim - 1:
        masked_actions = actions_np[mask_np]
    else:
        masked_actions = actions_np.reshape(-1, actions_np.shape[-1])

    logging.info(_summarize_tensor("health/state", state_np))
    logging.info(_summarize_tensor("health/actions", actions_np))
    logging.info(_summarize_tensor("health/actions_masked", masked_actions))
    logging.info(
        "health/actions_mask: shape=%s, valid_ratio=%.4f",
        mask_np.shape,
        mask_ratio,
    )


def is_distributed():
    """Check if we're running in distributed mode."""
    return jax.process_count() > 1


def sync_devices(name: str):
    """Synchronize all devices in distributed mode."""
    if is_distributed():

        multihost_utils.sync_global_devices(name)


def broadcast_value(value):
    """Broadcast a value from rank 0 to all ranks in distributed mode."""
    if is_distributed():
        return multihost_utils.broadcast_one_to_all(value)
    return value


def setup_nccl_env():
    """Setup NCCL environment variables for distributed training."""
    defaults = {
        # "NCCL_IB_DISABLE": "1",  # 禁用InfiniBand
        # "NCCL_P2P_DISABLE": "1",  # 禁用P2P
        # "NCCL_SOCKET_IFNAME": "eth0",  # 指定网络接口
        "NCCL_DEBUG": "INFO",  # 启用调试信息
        "NCCL_TIMEOUT": "1800",  # 设置超时时间为30分钟
        "NCCL_BLOCKING_WAIT": "1",  # 启用阻塞等待
        # "NCCL_ALGO": "Ring",  # 使用Ring算法
        # "NCCL_PROTO": "Simple",  # 使用Simple协议
        # "NCCL_NET_GDR_LEVEL": "0",  # 禁用GPU Direct RDMA
        # "NCCL_NVLS_ENABLE": "0",  # 禁用NVLS
        # "NCCL_IGNORE_CPU_AFFINITY": "1",  # 忽略CPU亲和性
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


class PrefetchGenerator(threading.Thread):
    """A general prefetch generator.
    Reference: https://stackoverflow.com/questions/7323664/python-generator-pre-fetch
    Args:
        generator: Python generator.
        num_prefetch_queue (int): Number of prefetch queue.
    """

    def __init__(self, generator, num_prefetch_queue):
        super().__init__(daemon=True)
        self.queue = queue.Queue(maxsize=num_prefetch_queue)
        self.generator = generator
        self.exception = None
        self.start()

    def run(self):
        try:
            for item in self.generator:
                self.queue.put(item)
            self.queue.put(None)
        except Exception as e:
            self.exception = e
            self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if self.exception:
            raise self.exception
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {
        "DEBUG": "D",
        "INFO": "I",
        "WARNING": "W",
        "ERROR": "E",
        "CRITICAL": "C",
    }

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(
    config: _config.TrainConfig,
    *,
    resuming: bool,
    log_code: bool = False,
    enabled: bool = True,
):
    if not enabled:
        wandb.init(mode="disabled")
        return

    # Only initialize wandb on rank 0 in distributed training
    if is_distributed() and jax.process_index() != 0:
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    try:
        at.check_pytree_equality(
            expected=params_shape,
            got=loaded_params,
            check_shapes=True,
            check_dtypes=True,
        )
    except ValueError as e:
        logging.warning(
            "Attention: Parameter shape/dtype check failed but will be skipped due to OPENPI_SKIP_PARAM_CHECK: %s",
            e,
        )

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    *,
    resume: bool,
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(
            params,
            config.freeze_filter,
            lambda p: p.replace(p.value.astype(jnp.bfloat16)),
        )

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions, _model.ActionsMask],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        actions_mask: _model.ActionsMask,
    ):
        result = model.compute_loss(rng, observation, actions, max_delay=config.rtc_max_delay, train=True)
        (chunked_loss, postfix_mask, per_dim_chunked_loss, subtask_loss, fast_action_loss) = result

        final_mask = jnp.logical_and(actions_mask, postfix_mask)
        action_loss = jnp.sum(chunked_loss * final_mask) / (jnp.sum(final_mask) + 1e-8)
        per_dim_loss = jnp.full((actions.shape[-1],), jnp.nan, dtype=chunked_loss.dtype)
        per_dim_count = jnp.zeros((actions.shape[-1],), dtype=chunked_loss.dtype)
        if per_dim_chunked_loss is not None:
            masked_per_dim = per_dim_chunked_loss * final_mask[..., None]
            if (
                observation.joint_eef_dof_mask is not None
                and observation.joint_eef_dof_mask.shape == per_dim_chunked_loss.shape
            ):
                dof_mask = observation.joint_eef_dof_mask.astype(per_dim_chunked_loss.dtype)
                masked_per_dim = masked_per_dim * dof_mask
                per_dim_count = jnp.sum(final_mask[..., None] * dof_mask, axis=(0, 1))
            else:
                per_dim_count = jnp.sum(final_mask, axis=(0, 1))
            denom = per_dim_count + 1e-8
            numer = jnp.sum(masked_per_dim, axis=(0, 1))
            per_dim_loss = numer / denom
        subtask_loss = jnp.mean(subtask_loss)
        fast_action_loss = jnp.mean(fast_action_loss)
        loss = action_loss
        if config.model.pi05_subtask_fast:
            if config.model.pi05_with_subtask:
                loss += subtask_loss
            if config.model.pi05_with_fast_action:
                loss += fast_action_loss

        return loss, (action_loss, per_dim_loss, per_dim_count, subtask_loss, fast_action_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions, actions_mask = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)

    # Use value_and_grad with has_aux=True to return additional loss values
    (loss, (action_loss, per_dim_loss, per_dim_count, subtask_loss, fast_action_loss)), grads = nnx.value_and_grad(
        loss_fn, argnums=diff_state, has_aux=True
    )(model, train_rng, observation, actions, actions_mask)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params,
                new_params,
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    current_lr = config.lr_schedule.create()(state.step)
    info = {
        "total_loss": loss,
        "action_loss": action_loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
        "learning_rate": current_lr,
    }
    if config.model.pi05_subtask_fast:
        if config.model.pi05_with_subtask:
            info["subtask_loss"] = subtask_loss
        if config.model.pi05_with_fast_action:
            info["fast_action_loss"] = fast_action_loss
    info["loss/action_dim_vector"] = per_dim_loss
    info["loss/action_dim_count_vector"] = per_dim_count
    return new_state, info


def main(config: _config.TrainConfig, config_file_path: str | None = None):
    # Setup distributed training environment if needed
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        setup_nccl_env()

        # Set XLA environment variables
        os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")

        jax.distributed.initialize(
            f"{MASTER_ADDR}:{MASTER_PORT}",
            int(os.environ["WORLD_SIZE"]),
            int(os.environ["RANK"]),
        )
        # Set master addr and port after jax distributed initialization
        if MASTER_ADDR:
            os.environ["MASTER_ADDR"] = MASTER_ADDR
        if MASTER_PORT:
            os.environ["MASTER_PORT"] = MASTER_PORT

        # Ensure JAX compilation cache directory exists
        _jax_cache_dir = epath.Path("~/.cache/jax").expanduser()
        _jax_cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Total processes = {jax.process_count()}")
    else:
        # Single node training
        print("Single node training detected")

    init_logging()
    logging.info(f"Running on: {platform.node()}")
    logging.info(
        "effective action_dim_loss_topk=%d (0 means all dims)",
        int(config.action_dim_loss_topk),
    )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Synchronize before initializing checkpoint manager
    sync_devices("Before checkpoint manager initialization")

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
        config=config,
        config_file_path=config_file_path,
    )

    # Synchronize after checkpoint manager initialization
    sync_devices("After checkpoint manager initialization")

    # Initialize wandb only on rank 0
    if jax.process_index() == 0:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # Choose data loader based on whether we're in distributed mode
    if is_distributed():
        data_loader = _data_loader.create_distributed_torch_data_loader(
            config=config,
            rank=jax.process_index(),
            world_size=jax.process_count(),
            sharding=data_sharding,
            shuffle=True,
            skip_norm_stats=False,  # <--- 必须是 False
            num_batches=config.num_train_steps + 100,
        )
        logging.info(f"Rank {jax.process_index()}: Total processes = {jax.process_count()}")
    else:
        data_loader = _data_loader.create_data_loader(
            config,
            sharding=data_sharding,
            shuffle=True,
        )

    # Synchronize after data loader initialization
    sync_devices("Dataset initialization complete")

    data_iter = iter(data_loader)

    if is_distributed():
        logging.info(f"Waiting for data loader to be initialized on rank {jax.process_index()}")
        data_iter = PrefetchGenerator(data_iter, 16)

    logging.info(f"Initializing data loader on rank {jax.process_index()}")
    batch = next(data_iter)

    # Synchronize after first batch is loaded
    sync_devices("First batch loaded")

    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")
    log_batch_numeric_health(batch)

    # 记录第一个 batch 的图像到 wandb。
    # 多机训练时 batch[0].images 中的 tensor 是分布式 sharded array, 直接在 rank 0 调用
    # np.array() 会触发隐式 all-gather, 但 rank 1 此时不参与, 导致死锁。
    # 正确做法是先让所有 rank 共同执行 process_allgather 完成通信, 再由 rank 0 负责 log。
    if is_distributed():
        images_to_log_np = {k: np.asarray(multihost_utils.process_allgather(v)) for k, v in batch[0].images.items()}
    else:
        images_to_log_np = {k: np.asarray(v) for k, v in batch[0].images.items()}
    if jax.process_index() == 0:
        _num_samples = min(5, next(iter(images_to_log_np.values())).shape[0])
        images_to_log = [
            wandb.Image(np.concatenate([img[i] for img in images_to_log_np.values()], axis=1))
            for i in range(_num_samples)
        ]
        wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=bool(resuming))
    jax.block_until_ready(train_state)

    # Synchronize after train state initialization
    sync_devices("Train state initialization complete")
    logging.info(f"Rank {jax.process_index()}: Train state initialized and synchronized")

    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))

            per_dim_loss = reduced_info.pop("loss/action_dim_vector", None)
            per_dim_count = reduced_info.pop("loss/action_dim_count_vector", None)
            if per_dim_loss is not None:
                reduced_info.update(
                    _expand_action_dim_loss_metrics(
                        per_dim_loss,
                        topk=config.action_dim_loss_topk,
                    )
                )
            if per_dim_count is not None:
                reduced_info.update(
                    _expand_action_dim_count_metrics(
                        per_dim_count,
                        topk=config.action_dim_loss_topk,
                        ref_per_dim_loss=per_dim_loss,
                    )
                )
                reduced_info.update(_summarize_action_dim_validity(per_dim_count))

            subtask_loss = reduced_info.pop("subtask_loss", None)
            if subtask_loss is not None:
                reduced_info["subtask_loss"] = subtask_loss
            fast_action_loss = reduced_info.pop("fast_action_loss", None)
            if fast_action_loss is not None:
                reduced_info["fast_action_loss"] = fast_action_loss

            info_str = ", ".join(
                f"{k}={v:.8f}" if k == "learning_rate" else f"{k}={v:.4f}" for k, v in reduced_info.items()
            )
            pbar.write(f"Step {step}: {info_str}")

            # Log to wandb only on rank 0
            if jax.process_index() == 0:
                wandb.log(reduced_info, step=step)

            infos = []
        batch = next(data_iter)
        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    config, config_path = tyro.cli(_config.cli)
    main(config, config_file_path=config_path)
