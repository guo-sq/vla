"""Training helpers for out-of-fold action diffusion proxy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm

from diffusion_proxy.data import DatasetArrays
from diffusion_proxy.data import NormalizationStats
from diffusion_proxy.data import fit_normalization
from diffusion_proxy.data import normalize_condition
from diffusion_proxy.data import normalize_targets
from diffusion_proxy.data import stats_to_tensors
from diffusion_proxy.data import subset_arrays
from diffusion_proxy.diffusion import DiffusionSchedule
from diffusion_proxy.model import ActionChunkDenoiser
from diffusion_proxy.utils import seed_everything


@dataclass(frozen=True)
class DiffusionTrainConfig:
    horizon: int = 16
    action_dim: int = 14
    hidden_dim: int = 512
    blocks: int = 4
    time_dim: int = 128
    diffusion_steps: int = 100
    batch_size: int = 1024
    epochs: int = 80
    lr: float = 3e-4
    weight_decay: float = 1e-4
    ema_decay: float = 0.995


def masked_noise_mse(pred: torch.Tensor, noise: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = torch.nn.functional.mse_loss(pred, noise, reduction="none")
    mask_f = mask.to(dtype=loss.dtype).unsqueeze(-1)
    denom = torch.clamp(mask_f.sum() * pred.shape[-1], min=1.0)
    return (loss * mask_f).sum() / denom


def _make_loader(arrays: DatasetArrays, stats: NormalizationStats, batch_size: int, *, shuffle: bool) -> DataLoader:
    cond = normalize_condition(arrays.condition, stats)
    targets = normalize_targets(arrays.targets, stats)
    masks = arrays.target_mask.astype(bool)
    ds = TensorDataset(
        torch.as_tensor(cond, dtype=torch.float32),
        torch.as_tensor(targets, dtype=torch.float32),
        torch.as_tensor(masks, dtype=torch.bool),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _ema_update(model: torch.nn.Module, ema_state: dict[str, torch.Tensor], decay: float) -> None:
    with torch.no_grad():
        for name, param in model.state_dict().items():
            if torch.is_floating_point(param):
                ema_state[name].mul_(decay).add_(param.detach(), alpha=1.0 - decay)
            else:
                ema_state[name].copy_(param)


def _initial_ema_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _run_epoch(
    model: ActionChunkDenoiser,
    loader: DataLoader,
    schedule: DiffusionSchedule,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    ema_state: dict[str, torch.Tensor] | None = None,
    ema_decay: float = 0.995,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    grad_context = torch.enable_grad() if training else torch.inference_mode()
    with grad_context:
        for cond, target, mask in loader:
            cond = cond.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            t = torch.randint(0, schedule.num_steps, (len(cond),), device=device, dtype=torch.long)
            noise = torch.randn_like(target)
            noisy = schedule.q_sample(target, t, noise)
            pred = model(noisy, cond, t)
            loss = masked_noise_mse(pred, noise, mask)
            if training:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if ema_state is not None:
                    _ema_update(model, ema_state, ema_decay)
            total += float(loss.detach().cpu()) * len(cond)
            count += len(cond)
    return total / max(count, 1)


def train_fold(
    arrays: DatasetArrays,
    *,
    train_keys: set[tuple[str, int]],
    heldout_keys: set[tuple[str, int]],
    fold_index: int,
    checkpoint_dir: Path,
    config: DiffusionTrainConfig,
    seed: int,
    device: str | None = None,
) -> Path:
    seed_everything(seed)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_arrays = subset_arrays(arrays, train_keys)
    heldout_arrays = subset_arrays(arrays, heldout_keys)
    stats = fit_normalization(train_arrays)
    train_loader = _make_loader(train_arrays, stats, config.batch_size, shuffle=True)
    val_loader = _make_loader(heldout_arrays, stats, config.batch_size, shuffle=False)
    model = ActionChunkDenoiser(
        cond_dim=int(arrays.condition.shape[1]),
        horizon=config.horizon,
        action_dim=config.action_dim,
        hidden_dim=config.hidden_dim,
        blocks=config.blocks,
        time_dim=config.time_dim,
    ).to(device_obj)
    schedule = DiffusionSchedule(config.diffusion_steps, device=device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    ema_state = _initial_ema_state(model)
    history: list[dict[str, float]] = []

    for epoch in tqdm(range(config.epochs), desc=f"fold {fold_index}", leave=False):
        train_loss = _run_epoch(
            model,
            train_loader,
            schedule,
            optimizer=optimizer,
            device=device_obj,
            ema_state=ema_state,
            ema_decay=config.ema_decay,
        )
        val_loss = _run_epoch(model, val_loader, schedule, optimizer=None, device=device_obj)
        history.append({"epoch": float(epoch + 1), "train_loss": train_loss, "val_loss": val_loss})

    path = checkpoint_dir / f"fold_{fold_index:02d}.pt"
    torch.save(
        {
            "fold_index": int(fold_index),
            "seed": int(seed),
            "model_state": model.state_dict(),
            "ema_model_state": ema_state,
            "model_config": {
                "cond_dim": int(arrays.condition.shape[1]),
                "horizon": config.horizon,
                "action_dim": config.action_dim,
                "hidden_dim": config.hidden_dim,
                "blocks": config.blocks,
                "time_dim": config.time_dim,
            },
            "train_config": config.__dict__,
            "data_stats": stats_to_tensors(stats),
            "history": history,
            "train_keys": [[repo, int(ep)] for repo, ep in sorted(train_keys)],
            "heldout_keys": [[repo, int(ep)] for repo, ep in sorted(heldout_keys)],
            "num_train_samples": int(len(train_arrays.condition)),
            "num_heldout_samples": int(len(heldout_arrays.condition)),
        },
        path,
    )
    return path


def checkpoint_last_metrics(path: Path) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    history = ckpt.get("history", [])
    return dict(history[-1]) if history else {}
