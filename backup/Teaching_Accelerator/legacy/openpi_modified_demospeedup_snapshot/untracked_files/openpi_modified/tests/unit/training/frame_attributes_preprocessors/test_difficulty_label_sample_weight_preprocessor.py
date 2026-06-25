import json
from types import SimpleNamespace

import numpy as np

from openpi.training.demo_difficulty.sampling import load_difficulty_sample_weights
from openpi.training.frame_attributes_preprocessors.base import FrameAttributes
from openpi.training.frame_attributes_preprocessors.sample_weight_preprocessor import (
    DifficultyLabelSampleWeightPreprocessor,
)


class _LenDataset:
    def __init__(self, n: int):
        self._n = n

    def __len__(self):
        return self._n


def _write_labels(repo_root, records):
    label_path = repo_root / "meta" / "difficulty_labels.jsonl"
    label_path.parent.mkdir(parents=True)
    with label_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return label_path


def test_load_difficulty_sample_weights_maps_real_episode_indices(tmp_path):
    repo_root = tmp_path / "repo"
    _write_labels(
        repo_root,
        [
            {"episode_index": 20, "sample_weight": [0, 1]},
            {"episode_index": 10, "sample_weight": [1, 0, 1]},
        ],
    )
    episode_data_index = {
        "from": np.array([0, 3], dtype=np.int64),
        "to": np.array([3, 5], dtype=np.int64),
    }

    weights = load_difficulty_sample_weights(
        root=repo_root,
        total_frames=5,
        episode_data_index=episode_data_index,
        meta_episodes={10: {}, 20: {}},
    )

    np.testing.assert_array_equal(weights, [1, 0, 1, 0, 1])


def test_difficulty_preprocessor_multiplies_existing_sample_weight(tmp_path):
    repo_root = tmp_path / "repo"
    _write_labels(
        repo_root,
        [
            {"episode_index": 0, "sample_weight": [1, 0, 1]},
            {"episode_index": 1, "sample_weight": [0, 1]},
        ],
    )
    ctx = SimpleNamespace(
        repo_id="repo",
        root=str(repo_root),
        hf_dataset=_LenDataset(5),
        episode_data_index={
            "from": np.array([0, 3], dtype=np.int64),
            "to": np.array([3, 5], dtype=np.int64),
        },
        meta=SimpleNamespace(episodes={0: {}, 1: {}}),
    )
    attrs = FrameAttributes(sample_weight=np.array([2, 2, 2, 2, 0], dtype=np.int32))

    DifficultyLabelSampleWeightPreprocessor()(ctx, attrs)

    np.testing.assert_array_equal(attrs.sample_weight, [2, 0, 2, 0, 0])
