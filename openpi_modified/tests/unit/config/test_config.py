import os
from pathlib import Path

import pytest

from openpi.training import config as _config
from openpi.training.base_cfg import TrainConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
ANYVERSE_CONFIG = REPO_ROOT / "src/openpi/configs/cfg_opensource/cfg_pi0.5_28_dim.anyverse.py"
PUBLIC_CONFIG = REPO_ROOT / "src/openpi/configs/cfg_opensource/cfg_pi0.5_28_dim.all_public_datasets.py"


@pytest.fixture
def fake_dataset_listing(monkeypatch: pytest.MonkeyPatch):
    real_listdir = os.listdir

    def safe_listdir(path: str):
        try:
            return real_listdir(path)
        except OSError:
            return ["placeholder"]

    monkeypatch.setattr(os, "listdir", safe_listdir)


@pytest.mark.config
@pytest.mark.smoke
@pytest.mark.parametrize("config_path", [ANYVERSE_CONFIG, PUBLIC_CONFIG])
def test_hotspot_configs_load_from_file(config_path: Path, fake_dataset_listing) -> None:
    del fake_dataset_listing
    cfg = _config.get_config(str(config_path))

    assert isinstance(cfg, TrainConfig)
    assert cfg.name
    assert cfg.model.action_horizon > 0
    assert cfg.model.max_token_len >= 48


@pytest.mark.config
def test_get_config_builds_data_contract_for_anyverse(fake_dataset_listing) -> None:
    del fake_dataset_listing
    cfg = _config.get_config(str(ANYVERSE_CONFIG))
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)

    assert data_cfg.repo_id is not None
    assert isinstance(data_cfg.repo_id, list)
    assert len(data_cfg.repo_id) > 0
    assert data_cfg.unify_action_space is True
    assert len(data_cfg.action_sequence_keys) > 0
    assert len(data_cfg.model_transforms.inputs) > 0


@pytest.mark.config
def test_get_config_unknown_name_suggests_closest_match() -> None:
    with pytest.raises(ValueError, match=r"Did you mean '.+'\?"):
        _config.get_config("debgu")
