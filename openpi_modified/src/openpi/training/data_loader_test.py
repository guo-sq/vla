import jax
import pytest

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import base_cfg as _base_cfg
from openpi.training import data_loader as _data_loader


def _make_train_config(batch_size: int = 4) -> _base_cfg.TrainConfig:
    return _base_cfg.TrainConfig(
        name="test_fake_loader",
        exp_name="test",
        model=pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48),
        data=_base_cfg.FakeDataConfig(),
        batch_size=batch_size,
        num_workers=0,
    )


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(
        dataset, local_batch_size=4, num_batches=2, num_workers=2
    )
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _make_train_config()

    loader = _data_loader.create_data_loader(
        config, skip_norm_stats=True, num_batches=2
    )
    batches = list(loader._data_loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for batch in batches:
        actions = batch["actions"]
        assert actions.shape == (
            config.batch_size,
            config.model.action_horizon,
            config.model.action_dim,
        )


def test_with_real_dataset():
    config = _make_train_config(batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader._data_loader)

    assert len(batches) == 2

    for batch in batches:
        actions = batch["actions"]
        assert actions.shape == (
            config.batch_size,
            config.model.action_horizon,
            config.model.action_dim,
        )


def test_data_loader_respects_num_batches():
    num_batches = 2
    config = _make_train_config(batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        skip_norm_stats=True,
        num_batches=num_batches,
        shuffle=False,
    )
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader._data_loader)

    assert len(batches) == num_batches
    for _, batch in enumerate(batches):
        actions = batch["actions"]
        assert actions.shape == (
            config.batch_size,
            config.model.action_horizon,
            config.model.action_dim,
        )