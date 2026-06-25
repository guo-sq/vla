from __future__ import annotations

import numpy as np
import pytest
import torch

from action_chunk_bc.data import build_action_chunk_targets
from action_chunk_bc.inference import disagreement_from_predictions
from action_chunk_bc.model import ActionChunkMLP
from action_chunk_bc.sidecar import validate_record


def test_action_chunk_tail_padding_and_mask() -> None:
    actions = np.arange(5 * 2, dtype=np.float32).reshape(5, 2)
    chunks, mask = build_action_chunk_targets(actions, horizon=4)

    assert chunks.shape == (5, 4, 2)
    assert mask.shape == (5, 4)
    np.testing.assert_array_equal(chunks[0], actions[[0, 1, 2, 3]])
    np.testing.assert_array_equal(chunks[3], actions[[3, 4, 4, 4]])
    np.testing.assert_array_equal(mask[3], np.array([True, True, False, False]))
    np.testing.assert_array_equal(mask[4], np.array([True, False, False, False]))


def test_mlp_output_shape() -> None:
    model = ActionChunkMLP(input_dim=43, horizon=16, action_dim=14, hidden_dim=32, layers=2)
    out = model(torch.zeros(7, 43))

    assert tuple(out.shape) == (7, 16, 14)


def test_disagreement_from_predictions() -> None:
    same = [
        np.zeros((3, 2, 2), dtype=np.float32),
        np.zeros((3, 2, 2), dtype=np.float32),
    ]
    different = [
        np.zeros((3, 2, 2), dtype=np.float32),
        np.ones((3, 2, 2), dtype=np.float32),
    ]

    assert np.allclose(disagreement_from_predictions(same), 0.0)
    assert np.all(disagreement_from_predictions(different) > 0.0)


def test_sidecar_validate_record_lengths_and_spans() -> None:
    record = {
        "repo_id": "repo",
        "episode_index": 0,
        "task": ["task"],
        "length": 3,
        "fps": 30,
        "bc_precision_score": [0.1, 0.9, 0.2],
        "ensemble_disagreement_score": [0.9, 0.1, 0.8],
        "label": ["casual", "precision", "neutral"],
        "acceleration_stride": [4, 2, 2],
        "hard_spans": [{"start_frame": 1, "end_frame": 3, "start_s": 0.0333, "end_s": 0.1, "duration_s": 0.0667}],
    }
    validate_record(record)

    broken = dict(record)
    broken["bc_precision_score"] = [0.1]
    with pytest.raises(ValueError, match="length mismatch"):
        validate_record(broken)
