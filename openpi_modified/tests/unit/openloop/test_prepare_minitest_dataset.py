from pathlib import Path

import pytest

from scripts.prepare_minitest_dataset import MINITEST_DATASET_SPECS
from scripts.prepare_minitest_dataset import prepare_dataset


@pytest.mark.posttrain
def test_prepare_dataset_copies_dataset_root_without_extra_nesting(tmp_path: Path) -> None:
    source_root = tmp_path / "source_root"
    (source_root / "meta").mkdir(parents=True)
    (source_root / "meta" / "info.json").write_text("{}", encoding="utf-8")

    spec = {
        "robot_type": "galaxea",
        "oss_path": str(source_root),
        "cpfs_path": "galaxea/demo_dataset",
        "max_episodes": 3,
    }

    result = prepare_dataset(spec, tmp_path / "cpfs")

    assert result["status"] == "success"
    assert (tmp_path / "cpfs" / "galaxea" / "demo_dataset" / "meta" / "info.json").exists()
    assert not (tmp_path / "cpfs" / "galaxea" / "demo_dataset" / "source_root").exists()


@pytest.mark.posttrain
def test_prepare_dataset_preserves_episode_children_for_episode_directories(tmp_path: Path) -> None:
    source_root = tmp_path / "episodes_root"
    episode_dir = source_root / "episode_0001"
    (episode_dir / "meta").mkdir(parents=True)
    (episode_dir / "meta" / "info.json").write_text("{}", encoding="utf-8")

    spec = {
        "robot_type": "agilex",
        "oss_path": str(source_root),
        "cpfs_path": "robomind_2/agilex/fold_clothes/success_episodes",
        "max_episodes": 3,
    }

    result = prepare_dataset(spec, tmp_path / "cpfs")

    assert result["status"] == "success"
    assert (tmp_path / "cpfs" / "robomind_2" / "agilex" / "fold_clothes" / "success_episodes" / "episode_0001").exists()


@pytest.mark.posttrain
def test_minitest_dataset_specs_match_configured_repo_ids() -> None:
    spec_paths = {spec["cpfs_path"] for spec in MINITEST_DATASET_SPECS}

    assert "robomind_2/tianyi/close_drawer_under_combined_cabinet/success_episodes" in spec_paths
    assert "galaxea/Adjust_The_Air_Conditioner_Temperature_20250711_006" in spec_paths
    assert "intern_a1/pickup_a_bag_of_bread_into_the_basket/set_0" in spec_paths
