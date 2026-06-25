"""Training helpers for the BC ensemble."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm

from action_chunk_bc.data import DatasetArrays
from action_chunk_bc.data import NormalizationStats
from action_chunk_bc.data import fit_normalization
from action_chunk_bc.data import normalize_features
from action_chunk_bc.data import normalize_targets
from action_chunk_bc.data import stats_to_tensors
from action_chunk_bc.model import ActionChunkMLP
from action_chunk_bc.utils import seed_everything


@dataclass(frozen=True)
class TrainConfig:
    horizon: int = 16
    action_dim: int = 14
    input_dim: int = 43
    hidden_dim: int = 256
    layers: int = 3
    batch_size: int = 1024
    epochs: int = 80
    lr: float = 1e-3
    weight_decay: float = 1e-4
    train_fraction: float = 0.90


def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none")
    mask_f = mask.to(dtype=loss.dtype).unsqueeze(-1)
    denom = torch.clamp(mask_f.sum() * pred.shape[-1], min=1.0)
    return (loss * mask_f).sum() / denom


def _split_indices(num_samples: int, seed: int, train_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    train_count = int(round(num_samples * train_fraction))
    train_count = min(max(train_count, 1), num_samples - 1)
    return indices[:train_count], indices[train_count:]


def train_one_seed(
    arrays: DatasetArrays,
    *,
    seed: int,
    checkpoint_dir: Path,
    config: TrainConfig,
    stats: NormalizationStats | None = None,
    device: str | None = None,
) -> Path:
    seed_everything(seed)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    stats = stats or fit_normalization(arrays)
    features = normalize_features(arrays.features, stats)
    targets = normalize_targets(arrays.targets, stats)
    masks = arrays.target_mask.astype(bool)

    train_idx, val_idx = _split_indices(len(features), seed, config.train_fraction)
    tensors = {
        "features": torch.as_tensor(features, dtype=torch.float32),
        "targets": torch.as_tensor(targets, dtype=torch.float32),
        "masks": torch.as_tensor(masks, dtype=torch.bool),
    }

    train_ds = TensorDataset(tensors["features"][train_idx], tensors["targets"][train_idx], tensors["masks"][train_idx])
    val_ds = TensorDataset(tensors["features"][val_idx], tensors["targets"][val_idx], tensors["masks"][val_idx])
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, drop_last=False)

    model = ActionChunkMLP(
        input_dim=config.input_dim,
        horizon=config.horizon,
        action_dim=config.action_dim,
        hidden_dim=config.hidden_dim,
        layers=config.layers,
    ).to(device_obj)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history: list[dict[str, float]] = []

    for epoch in tqdm(range(config.epochs), desc=f"seed {seed}", leave=False):
        model.train()
        total = 0.0
        count = 0
        for x, y, mask in train_loader:
            x = x.to(device_obj, non_blocking=True)
            y = y.to(device_obj, non_blocking=True)
            mask = mask.to(device_obj, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = masked_smooth_l1(model(x), y, mask)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * len(x)
            count += len(x)

        model.eval()
        val_total = 0.0
        val_count = 0
        with torch.inference_mode():
            for x, y, mask in val_loader:
                x = x.to(device_obj, non_blocking=True)
                y = y.to(device_obj, non_blocking=True)
                mask = mask.to(device_obj, non_blocking=True)
                loss = masked_smooth_l1(model(x), y, mask)
                val_total += float(loss.detach().cpu()) * len(x)
                val_count += len(x)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": total / max(count, 1),
                "val_loss": val_total / max(val_count, 1),
            }
        )

    path = checkpoint_dir / f"seed_{seed}.pt"
    torch.save(
        {
            "seed": seed,
            "model_state": model.state_dict(),
            "model_config": {
                "input_dim": config.input_dim,
                "horizon": config.horizon,
                "action_dim": config.action_dim,
                "hidden_dim": config.hidden_dim,
                "layers": config.layers,
            },
            "train_config": config.__dict__,
            "data_stats": stats_to_tensors(stats),
            "history": history,
            "num_train_samples": int(len(train_idx)),
            "num_val_samples": int(len(val_idx)),
        },
        path,
    )
    return path

