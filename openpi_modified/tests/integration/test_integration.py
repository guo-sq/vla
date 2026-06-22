import dataclasses
import os
from pathlib import Path

import pytest

from openpi.training import config as _config
from openpi.training.base_cfg import TrainConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
HOTSPOT_CONFIGS = [
    REPO_ROOT / "src/openpi/configs/cfg_opensource/cfg_pi0.5_28_dim.anyverse.py",
    REPO_ROOT / "src/openpi/configs/cfg_opensource/cfg_pi0.5_28_dim.all_public_datasets.py",
    REPO_ROOT / "src/openpi/configs/cfg_opensource/cfg_single_robot.anyverse.py",
]


@pytest.fixture
def fake_dataset_listing(monkeypatch: pytest.MonkeyPatch):
    real_listdir = os.listdir

    def safe_listdir(path: str):
        try:
            return real_listdir(path)
        except OSError:
            return ["placeholder"]

    monkeypatch.setattr(os, "listdir", safe_listdir)


@pytest.mark.integration
@pytest.mark.pretrain
@pytest.mark.smoke
@pytest.mark.parametrize("config_path", HOTSPOT_CONFIGS)
def test_dev_anyverse_hotspot_configs_build_data_contracts(config_path: Path, fake_dataset_listing) -> None:
    del fake_dataset_listing
    cfg = _config.get_config(str(config_path))
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)

    assert isinstance(cfg, TrainConfig)
    assert data_cfg.repo_id is not None
    assert isinstance(data_cfg.repo_id, list)
    assert len(data_cfg.repo_id) > 0
    assert data_cfg.unify_action_space is True
    assert len(data_cfg.action_sequence_keys) > 0
    assert len(data_cfg.model_transforms.inputs) > 0


@pytest.mark.integration
@pytest.mark.pretrain
@pytest.mark.smoke
def test_hotspot_model_config_can_be_retargeted_to_fake_repo(fake_dataset_listing) -> None:
    del fake_dataset_listing
    cfg = _config.get_config(str(HOTSPOT_CONFIGS[0]))
    fake_data_factory = dataclasses.replace(cfg.data, repo_id="fake")
    fake_data_cfg = fake_data_factory.create(cfg.assets_dirs, cfg.model)

    assert fake_data_cfg.repo_id == "fake"
    assert len(fake_data_cfg.repack_transforms.inputs) > 0
    assert len(fake_data_cfg.data_transforms.inputs) > 0
    assert len(fake_data_cfg.model_transforms.inputs) > 0
