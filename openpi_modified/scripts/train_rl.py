import dataclasses
import functools
import logging
import platform
import time
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import tyro
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader_rl as _data_loader_rl
from openpi.training.frame_attributes_preprocessors import ValueReturnsPreprocessor
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


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
            "==Qi== Parameter shape/dtype check failed but will be skipped due to OPENPI_SKIP_PARAM_CHECK: %s",
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
    ):
        value_loss = model.compute_rl_loss(rng, observation, actions, train=True)
        return jnp.mean(value_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions, actions_mask = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

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
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


@at.typecheck
def validation_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions, _model.ActionsMask],
) -> dict[str, at.Array]:
    """Run a single validation step (no gradient updates).

    Note: no `.eval()` call — it is a no-op inside a jit trace, and
    `model.compute_rl_loss(train=False)` already carries the train/eval flag.
    """
    model = nnx.merge(state.model_def, state.params)

    val_rng = jax.random.fold_in(rng, state.step)
    observation, actions, actions_mask = batch
    loss = jnp.mean(model.compute_rl_loss(val_rng, observation, actions, train=False))

    return {"val/loss": loss}


def _build_val_data_factory(config: _config.TrainConfig) -> Any:
    """Build a val-aware copy of config.data.

    Deep-replaces base_config to:
    1. Force ``ValueReturnsPreprocessor.exclude_failures=False`` — the train
       pipeline drops failure episodes via valid_mask to keep the value model
       from collapsing to -1, but evaluating on that same filtered subset
       hides whether the model still collapses on failure cases. Keep failure
       episodes in val so val loss can observe the collapse signal.
    2. Force ``value_net_cfg["cross_negative_rate"]=0.0`` — cross_negative is
       a train-time augmentation (``rl_dataset.py:402-412`` flips GT + prompt
       with this probability). Applying it at val turns val/loss into a
       mixed "normal + flipped" loss whose absolute value is no longer
       comparable to baselines without the augmentation. Disable at val only.
    """
    val_base_config = config.data.base_config
    if val_base_config is not None and val_base_config.frame_attributes_preprocessors:
        val_preprocessors = [
            (
                dataclasses.replace(p, exclude_failures=False)
                if isinstance(p, ValueReturnsPreprocessor) and p.exclude_failures
                else p
            )
            for p in val_base_config.frame_attributes_preprocessors
        ]
        val_base_config = dataclasses.replace(val_base_config, frame_attributes_preprocessors=val_preprocessors)
    if val_base_config is not None:
        vnc = getattr(val_base_config, "value_net_cfg", None)
        if vnc and vnc.get("cross_negative_rate", 0.0) > 0.0:
            val_base_config = dataclasses.replace(
                val_base_config,
                value_net_cfg={**vnc, "cross_negative_rate": 0.0},
            )
    return dataclasses.replace(
        config.data,
        repo_id=config.validation_repo_id,
        root_dir=config.validation_root_dir or config.data.root_dir,
        base_config=val_base_config,
    )


def run_validation(
    train_rng: at.KeyArrayLike,
    train_state: training_utils.TrainState,
    val_data_loader,
    pval_step,
    mesh: jax.sharding.Mesh,
    val_num_batches: int,
    val_repo_id: Any,
) -> dict[str, float]:
    """Run validation on multiple batches and return averaged metrics.

    StopIteration handling:
    - i == 0: loader is empty → hard error with repo_id context.
    - i > 0:  loader exhausted mid-run → warn once and restart; if the second
      next() is also empty, break out with whatever batches we have.
    """
    logging.info(f"Running validation on {val_num_batches} batches...")

    val_metrics_list = []
    val_data_iter = iter(val_data_loader)
    warned_exhausted = False
    for i in range(val_num_batches):
        try:
            val_batch = next(val_data_iter)
        except StopIteration:
            if i == 0:
                raise RuntimeError(
                    f"Validation data loader for repo_id={val_repo_id!r} is empty. "
                    f"Check validation_repo_id / validation_root_dir / dataset contents."
                ) from None
            if not warned_exhausted:
                logging.warning(
                    f"Validation loader exhausted at batch {i}/{val_num_batches} — "
                    f"restarting iterator (subsequent batches will repeat samples)."
                )
                warned_exhausted = True
            val_data_iter = iter(val_data_loader)
            try:
                val_batch = next(val_data_iter)
            except StopIteration:
                logging.warning(
                    f"Validation loader still empty after restart at batch {i}; "
                    f"breaking early with {len(val_metrics_list)} batches collected."
                )
                break

        # Fold in batch index to avoid identical RNG across batches
        batch_rng = jax.random.fold_in(train_rng, i)
        with sharding.set_mesh(mesh):
            val_metrics = pval_step(batch_rng, train_state, val_batch)

        val_metrics_list.append(jax.device_get(val_metrics))

    avg_val_metrics: dict[str, float] = {}
    if val_metrics_list:
        for key in val_metrics_list[0]:
            avg_val_metrics[key] = float(np.mean([m[key] for m in val_metrics_list]))

    logging.info(f"Validation metrics: {avg_val_metrics}")
    return avg_val_metrics


def main(config: _config.TrainConfig, config_file_path: str | None = None):
    t0 = time.perf_counter()
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
        config=config,
        config_file_path=config_file_path,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
    t1 = time.perf_counter()
    logging.info(f"Timing: ckpt manager took {t1 - t0:.3f}s")  # 5s

    t0 = time.perf_counter()
    data_loader = _data_loader_rl.create_rl_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    t1 = time.perf_counter()
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")
    logging.info(f"Timing: data loader creation + first-batch fetch took {t1 - t0:.3f}s")

    # Create validation data loader if configured
    val_data_loader = None
    if config.validation_repo_id is not None:
        logging.info("Creating validation data loader...")
        val_data_factory = _build_val_data_factory(config)
        val_config = dataclasses.replace(config, data=val_data_factory, validation_repo_id=None)
        val_data_loader = _data_loader_rl.create_rl_data_loader(
            val_config,
            sharding=data_sharding,
            shuffle=False,
        )
        logging.info(
            f"Validation data loader created (interval={config.validation_interval}, batches={config.validation_num_batches})"
        )

    # Log images from first batch to sanity check.
    t0 = time.perf_counter()
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)
    t1 = time.perf_counter()
    logging.info(f"Timing: logging first-batch images took {t1 - t0:.3f}s")  # 135s

    t0 = time.perf_counter()
    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    t1 = time.perf_counter()
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")
    logging.info(f"Timing: train state initialization took {t1 - t0:.3f}s")  # 9s

    t0 = time.perf_counter()

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    # jit the validation step once up front to avoid re-tracing on every call
    # (otherwise ~40 recompiles across a 20k-step / 500-interval run).
    pval_step = None
    if val_data_loader is not None:
        pval_step = jax.jit(
            functools.partial(validation_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=replicated_sharding,
        )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    t1 = time.perf_counter()
    logging.info(f"Timing: train_step jit took {t1 - t0:.3f}s")  #

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

        # Run validation at specified intervals (also force on the last step so
        # run end metrics aren't skipped by modulo).
        is_last_step = step == config.num_train_steps - 1
        should_val = (
            val_data_loader is not None and step > 0 and (step % config.validation_interval == 0 or is_last_step)
        )
        if should_val:
            val_metrics = run_validation(
                train_rng,
                train_state,
                val_data_loader,
                pval_step,
                mesh,
                config.validation_num_batches,
                config.validation_repo_id,
            )
            wandb.log(val_metrics, step=step)
            pbar.write(f"Step {step} validation: {val_metrics}")

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    config, config_path = tyro.cli(_config.cli)
    main(config, config_file_path=config_path)
