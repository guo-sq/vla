import dataclasses
import os
import pathlib

import orbax.checkpoint as ocp
import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

import openpi.models.pi0_config as pi0_config
from openpi.training import base_cfg as _base_cfg
import openpi.training.checkpoints as _checkpoints

from . import train_rl


def _make_debug_train_config() -> _base_cfg.TrainConfig:
    return _base_cfg.TrainConfig(
        name="debug",
        data=_base_cfg.FakeDataConfig(),
        batch_size=2,
        num_workers=0,
        model=pi0_config.Pi0Config(
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            enable_rl_value_head=True,
        ),
        save_interval=100,
        overwrite=False,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    )


@pytest.fixture
def lightweight_checkpointing(monkeypatch: pytest.MonkeyPatch):
    """Use compact synchronous checkpointing for this smoke test."""

    def initialize_checkpoint_dir_for_test(
        checkpoint_dir,
        *,
        keep_period,
        overwrite,
        resume,
        config=None,
        config_file_path=None,
    ):
        checkpoint_dir = _checkpoints.epath.Path(checkpoint_dir).resolve()
        resuming = False

        if checkpoint_dir.exists():
            if overwrite:
                checkpoint_dir.rmtree()
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
            elif resume:
                resuming = True
            else:
                raise FileExistsError(
                    f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                    "to indicate how to handle it."
                )
        else:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if config is not None and not resuming:
            _checkpoints.save_config_to_checkpoint(checkpoint_dir, config, config_file_path=config_file_path)

        manager = ocp.CheckpointManager(
            checkpoint_dir,
            item_handlers={
                "assets": _checkpoints.CallbackHandler(),
                "train_state": ocp.PyTreeCheckpointHandler(use_ocdbt=False),
                "params": ocp.PyTreeCheckpointHandler(use_ocdbt=False),
            },
            options=ocp.CheckpointManagerOptions(
                max_to_keep=1,
                keep_period=keep_period,
                create=False,
            ),
        )

        if resuming and tuple(manager.all_steps()) in [(), (0,)]:
            resuming = False

        return manager, resuming

    monkeypatch.setattr(_checkpoints, "initialize_checkpoint_dir", initialize_checkpoint_dir_for_test)


@pytest.mark.smoke
@pytest.mark.rl
def test_train_rl(tmp_path: pathlib.Path, lightweight_checkpointing):
    del lightweight_checkpointing
    config = dataclasses.replace(
        _make_debug_train_config(),
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test",
        resume=False,
        num_train_steps=2,
        log_interval=1,
    )
    train_rl.main(config)
