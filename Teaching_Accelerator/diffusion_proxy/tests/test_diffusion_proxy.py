from __future__ import annotations

import numpy as np
import torch

from diffusion_proxy.data import build_action_chunk_targets
from diffusion_proxy.diffusion import DiffusionSchedule
from diffusion_proxy.fusion import fuse_records
from diffusion_proxy.model import ActionChunkDenoiser
from diffusion_proxy.utils import moving_average
from diffusion_proxy.vision import GRID_EMBEDDING_DIM
from diffusion_proxy.vision import frame_embedding


def test_frame_embedding_shape() -> None:
    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    emb = frame_embedding(frame)
    assert emb.shape == (GRID_EMBEDDING_DIM,)
    assert emb.dtype == np.float32


def test_action_chunk_padding() -> None:
    actions = np.arange(5 * 2, dtype=np.float32).reshape(5, 2)
    chunks, mask = build_action_chunk_targets(actions, 4)
    assert chunks.shape == (5, 4, 2)
    np.testing.assert_array_equal(chunks[3], actions[[3, 4, 4, 4]])
    np.testing.assert_array_equal(mask[3], np.array([True, True, False, False]))


def test_denoiser_output_shape_and_schedule() -> None:
    model = ActionChunkDenoiser(cond_dim=12, horizon=4, action_dim=3, hidden_dim=32, blocks=2)
    x0 = torch.zeros(5, 4, 3)
    cond = torch.zeros(5, 12)
    t = torch.arange(5) % 10
    schedule = DiffusionSchedule(10)
    noisy = schedule.q_sample(x0, t, torch.ones_like(x0))
    out = model(noisy, cond, t)
    assert tuple(out.shape) == (5, 4, 3)


def test_moving_average_preserves_length_and_edges() -> None:
    values = np.array([1.0, 1.0, 10.0, 1.0, 1.0], dtype=np.float32)
    smoothed = moving_average(values, half_window=1)
    assert smoothed.shape == values.shape
    assert smoothed[0] > 0.9
    assert smoothed[2] < 10.0


def test_fuse_records_minimal() -> None:
    rule = {
        "repo_id": "repo",
        "episode_index": 0,
        "task": ["task"],
        "length": 4,
        "fps": 30,
        "hard_score": [0.9, 0.1, 0.2, 0.8],
        "casualness_score": [0.1, 0.8, 0.7, 0.2],
        "gripper_event_score": [0.9, 0.0, 0.0, 0.8],
        "turn_score": [0.0, 0.0, 0.0, 0.0],
        "jerk_score": [0.2, 0.0, 0.0, 0.2],
        "label": ["precision", "casual", "neutral", "precision"],
        "hard_spans": [],
    }
    diffusion = {
        "repo_id": "repo",
        "episode_index": 0,
        "task": ["task"],
        "length": 4,
        "fps": 30,
        "diffusion_precision_score": [0.8, 0.2, 0.3, 0.7],
        "diffusion_entropy_score": [0.2, 0.9, 0.8, 0.3],
        "label": ["precision", "casual", "casual", "precision"],
        "hard_spans": [],
    }
    records, summary = fuse_records([rule], [diffusion], precision_quantile=0.5, casual_quantile=0.5)
    assert len(records) == 1
    assert len(records[0]["label"]) == 4
    assert summary["num_episodes"] == 1
