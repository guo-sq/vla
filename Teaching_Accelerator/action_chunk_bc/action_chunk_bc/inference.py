"""BC ensemble inference and label conversion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from action_chunk_bc.data import DatasetArrays
from action_chunk_bc.data import stats_from_checkpoint
from action_chunk_bc.data import normalize_features
from action_chunk_bc.model import ActionChunkMLP
from action_chunk_bc.utils import merge_boolean_spans
from action_chunk_bc.utils import robust_unit_scale


def load_model_checkpoint(path: Path, device: torch.device) -> tuple[ActionChunkMLP, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ActionChunkMLP(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def infer_checkpoint(
    path: Path,
    arrays: DatasetArrays,
    *,
    batch_size: int = 4096,
    device: str | None = None,
) -> tuple[np.ndarray, dict]:
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, ckpt = load_model_checkpoint(path, device_obj)
    stats = stats_from_checkpoint(ckpt["data_stats"])
    features = normalize_features(arrays.features, stats)
    loader = DataLoader(TensorDataset(torch.as_tensor(features, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    preds: list[np.ndarray] = []
    with torch.inference_mode():
        for (x,) in loader:
            x = x.to(device_obj, non_blocking=True)
            preds.append(model(x).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(preds, axis=0), ckpt


def disagreement_from_predictions(predictions: list[np.ndarray]) -> np.ndarray:
    if len(predictions) < 2:
        raise ValueError("Need at least two prediction tensors")
    mean = None
    m2 = None
    count = 0
    for pred in predictions:
        count += 1
        if mean is None:
            mean = pred.astype(np.float32)
            m2 = np.zeros_like(mean, dtype=np.float32)
        else:
            delta = pred - mean
            mean += delta / float(count)
            m2 += delta * (pred - mean)
    assert m2 is not None
    var = m2 / float(max(count - 1, 1))
    return var.mean(axis=(1, 2)).astype(np.float32)


def ensemble_disagreement(
    checkpoint_paths: list[Path],
    arrays: DatasetArrays,
    *,
    batch_size: int = 4096,
    device: str | None = None,
) -> tuple[np.ndarray, dict]:
    if len(checkpoint_paths) < 2:
        raise ValueError("Need at least two checkpoints for ensemble disagreement")
    preds = []
    first_ckpt = None
    for path in checkpoint_paths:
        pred, ckpt = infer_checkpoint(path, arrays, batch_size=batch_size, device=device)
        if first_ckpt is None:
            first_ckpt = ckpt
        preds.append(pred)
    assert first_ckpt is not None
    raw = disagreement_from_predictions(preds)
    return raw, first_ckpt


def action_speed_scores(arrays: DatasetArrays) -> np.ndarray:
    values: list[np.ndarray] = []
    for episode in arrays.episodes:
        velocity = np.diff(episode.actions, axis=0, prepend=episode.actions[:1])
        values.append(np.linalg.norm(velocity, axis=1).astype(np.float32))
    return robust_unit_scale(np.concatenate(values, axis=0))


def labels_from_disagreement(
    arrays: DatasetArrays,
    disagreement_raw: np.ndarray,
    *,
    precision_quantile: float = 0.75,
    casual_quantile: float = 0.65,
    static_speed_quantile: float = 0.10,
    precision_stride: int = 2,
    neutral_stride: int = 2,
    casual_stride: int = 4,
) -> dict:
    disagreement = robust_unit_scale(disagreement_raw)
    precision_score = (1.0 - disagreement).astype(np.float32)
    speed = action_speed_scores(arrays)
    precision_threshold = float(np.quantile(precision_score, precision_quantile))
    casual_threshold = float(np.quantile(disagreement, casual_quantile))
    static_threshold = float(np.quantile(speed, static_speed_quantile))

    records_by_key = {}
    label_counts = {"precision": 0, "neutral": 0, "casual": 0}
    total_spans = 0
    for episode in arrays.episodes:
        key = (episode.repo_id, episode.episode_index)
        sl = arrays.episode_slices[key]
        ep_precision = precision_score[sl]
        ep_disagreement = disagreement[sl]
        ep_speed = speed[sl]
        labels = np.full(episode.length, "neutral", dtype=object)
        labels[ep_disagreement >= casual_threshold] = "casual"
        precision_mask = (ep_precision >= precision_threshold) & (ep_speed > static_threshold)
        labels[precision_mask] = "precision"
        strides = np.full(episode.length, neutral_stride, dtype=np.int32)
        strides[labels == "precision"] = precision_stride
        strides[labels == "casual"] = casual_stride
        spans = merge_boolean_spans(
            labels == "precision",
            fps=episode.fps,
            min_span_frames=15,
            merge_gap_frames=10,
            padding_frames=8,
            score=ep_precision,
        )
        for label in label_counts:
            label_counts[label] += int(np.count_nonzero(labels == label))
        total_spans += len(spans)
        records_by_key[key] = {
            "bc_precision_score": ep_precision,
            "ensemble_disagreement_score": ep_disagreement,
            "label": [str(x) for x in labels.tolist()],
            "acceleration_stride": strides,
            "hard_spans": spans,
        }
    return {
        "records_by_key": records_by_key,
        "summary": {
            "thresholds": {
                "bc_precision_min": precision_threshold,
                "ensemble_disagreement_casual_min": casual_threshold,
                "static_speed_max": static_threshold,
                "precision_quantile": precision_quantile,
                "casual_quantile": casual_quantile,
                "static_speed_quantile": static_speed_quantile,
            },
            "label_counts": label_counts,
            "num_hard_spans": total_spans,
            "num_episodes": len(arrays.episodes),
            "num_frames": int(len(arrays.features)),
            "action_dim": 14,
        },
    }
