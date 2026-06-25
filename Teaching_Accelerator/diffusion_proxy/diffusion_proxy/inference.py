"""Inference and label conversion for diffusion proxy checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from diffusion_proxy.data import DatasetArrays
from diffusion_proxy.data import normalize_condition
from diffusion_proxy.data import normalize_targets
from diffusion_proxy.data import stats_from_checkpoint
from diffusion_proxy.diffusion import DiffusionSchedule
from diffusion_proxy.model import ActionChunkDenoiser
from diffusion_proxy.utils import LABEL_CASUAL
from diffusion_proxy.utils import LABEL_NEUTRAL
from diffusion_proxy.utils import LABEL_PRECISION
from diffusion_proxy.utils import merge_boolean_spans
from diffusion_proxy.utils import moving_average
from diffusion_proxy.utils import robust_unit_scale


def load_checkpoint(path: Path, device: torch.device, *, use_ema: bool = True) -> tuple[ActionChunkDenoiser, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ActionChunkDenoiser(**ckpt["model_config"]).to(device)
    state_key = "ema_model_state" if use_ema and "ema_model_state" in ckpt else "model_state"
    model.load_state_dict(ckpt[state_key])
    model.eval()
    return model, ckpt


def _ddim_sample(
    model: ActionChunkDenoiser,
    cond: torch.Tensor,
    *,
    schedule: DiffusionSchedule,
    sampling_steps: int,
    samples_per_frame: int,
    horizon: int,
    action_dim: int,
) -> torch.Tensor:
    bsz = cond.shape[0]
    repeated_cond = cond.repeat_interleave(samples_per_frame, dim=0)
    x = torch.randn((bsz * samples_per_frame, horizon, action_dim), dtype=torch.float32, device=cond.device)
    timesteps = schedule.ddim_timesteps(sampling_steps)
    for idx, t_int in enumerate(timesteps):
        prev_t = timesteps[idx + 1] if idx + 1 < len(timesteps) else -1
        t = torch.full((len(x),), t_int, dtype=torch.long, device=cond.device)
        eps = model(x, repeated_cond, t)
        alpha_t = schedule.alpha_bars[t_int]
        alpha_prev = torch.tensor(1.0, dtype=torch.float32, device=cond.device) if prev_t < 0 else schedule.alpha_bars[prev_t]
        x0 = (x - torch.sqrt(1.0 - alpha_t) * eps) / torch.sqrt(alpha_t)
        x = torch.sqrt(alpha_prev) * x0 + torch.sqrt(1.0 - alpha_prev) * eps
    return x.view(bsz, samples_per_frame, horizon, action_dim)


def infer_checkpoint_heldout(
    path: Path,
    arrays: DatasetArrays,
    *,
    batch_size: int = 256,
    sampling_steps: int = 20,
    samples_per_frame: int = 16,
    device: str | None = None,
    use_ema: bool = True,
) -> tuple[dict[tuple[str, int], dict[str, np.ndarray]], dict[str, Any]]:
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, ckpt = load_checkpoint(path, device_obj, use_ema=use_ema)
    stats = stats_from_checkpoint(ckpt["data_stats"])
    horizon = int(ckpt["model_config"]["horizon"])
    action_dim = int(ckpt["model_config"]["action_dim"])
    schedule = DiffusionSchedule(int(ckpt["train_config"]["diffusion_steps"]), device=device_obj)
    heldout_keys = {(str(repo), int(ep)) for repo, ep in ckpt["heldout_keys"]}

    outputs: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for key in sorted(heldout_keys):
        sl = arrays.episode_slices[key]
        condition = normalize_condition(arrays.condition[sl], stats)
        targets = normalize_targets(arrays.targets[sl], stats)
        masks = arrays.target_mask[sl]
        ds = TensorDataset(
            torch.as_tensor(condition, dtype=torch.float32),
            torch.as_tensor(targets, dtype=torch.float32),
            torch.as_tensor(masks, dtype=torch.bool),
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        entropy_parts: list[np.ndarray] = []
        recon_parts: list[np.ndarray] = []
        with torch.inference_mode():
            for cond, target, mask in loader:
                cond = cond.to(device_obj, non_blocking=True)
                target = target.to(device_obj, non_blocking=True)
                mask = mask.to(device_obj, non_blocking=True)
                samples = _ddim_sample(
                    model,
                    cond,
                    schedule=schedule,
                    sampling_steps=sampling_steps,
                    samples_per_frame=samples_per_frame,
                    horizon=horizon,
                    action_dim=action_dim,
                )
                entropy = samples.var(dim=1, unbiased=False).mean(dim=(1, 2))
                sample_mean = samples.mean(dim=1)
                err = torch.nn.functional.mse_loss(sample_mean, target, reduction="none")
                mask_f = mask.to(dtype=err.dtype).unsqueeze(-1)
                denom = torch.clamp(mask.to(dtype=err.dtype).sum(dim=1) * err.shape[-1], min=1.0)
                recon = (err * mask_f).sum(dim=(1, 2)) / denom
                entropy_parts.append(entropy.detach().cpu().numpy().astype(np.float32))
                recon_parts.append(recon.detach().cpu().numpy().astype(np.float32))
        outputs[key] = {
            "diffusion_entropy_raw": np.concatenate(entropy_parts, axis=0).astype(np.float32),
            "diffusion_reconstruction_error_raw": np.concatenate(recon_parts, axis=0).astype(np.float32),
        }
    return outputs, ckpt


def action_speed_scores(arrays: DatasetArrays) -> np.ndarray:
    values: list[np.ndarray] = []
    for episode in arrays.episodes:
        velocity = np.diff(episode.actions, axis=0, prepend=episode.actions[:1])
        values.append(np.linalg.norm(velocity, axis=1).astype(np.float32))
    return robust_unit_scale(np.concatenate(values, axis=0))


def labels_from_entropy(
    arrays: DatasetArrays,
    entropy_by_key: dict[tuple[str, int], np.ndarray],
    reconstruction_by_key: dict[tuple[str, int], np.ndarray],
    *,
    precision_quantile: float = 0.75,
    casual_quantile: float = 0.65,
    static_speed_quantile: float = 0.10,
    smoothing_half_window: int = 8,
    min_span_frames: int = 15,
    merge_gap_frames: int = 10,
    span_padding_frames: int = 8,
    precision_stride: int = 2,
    neutral_stride: int = 2,
    casual_stride: int = 4,
) -> dict[str, Any]:
    entropy_flat = []
    recon_flat = []
    for episode in arrays.episodes:
        entropy_flat.append(moving_average(entropy_by_key[episode.key], smoothing_half_window))
        recon_flat.append(moving_average(reconstruction_by_key[episode.key], smoothing_half_window))
    entropy_score_flat = robust_unit_scale(np.concatenate(entropy_flat, axis=0))
    reconstruction_score_flat = robust_unit_scale(np.concatenate(recon_flat, axis=0))
    precision_score_flat = (1.0 - entropy_score_flat).astype(np.float32)
    speed = action_speed_scores(arrays)
    precision_threshold = float(np.quantile(precision_score_flat, precision_quantile))
    casual_threshold = float(np.quantile(entropy_score_flat, casual_quantile))
    static_threshold = float(np.quantile(speed, static_speed_quantile))

    records_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    label_counts = {LABEL_PRECISION: 0, LABEL_NEUTRAL: 0, LABEL_CASUAL: 0}
    total_spans = 0
    offset = 0
    for episode in arrays.episodes:
        n = episode.length
        ep_entropy = entropy_score_flat[offset : offset + n]
        ep_recon = reconstruction_score_flat[offset : offset + n]
        ep_precision = precision_score_flat[offset : offset + n]
        ep_speed = speed[offset : offset + n]
        labels = np.full(n, LABEL_NEUTRAL, dtype=object)
        labels[ep_entropy >= casual_threshold] = LABEL_CASUAL
        precision_mask = (ep_precision >= precision_threshold) & (ep_speed > static_threshold)
        labels[precision_mask] = LABEL_PRECISION
        strides = np.full(n, neutral_stride, dtype=np.int32)
        strides[labels == LABEL_PRECISION] = precision_stride
        strides[labels == LABEL_CASUAL] = casual_stride
        spans = merge_boolean_spans(
            labels == LABEL_PRECISION,
            fps=episode.fps,
            min_span_frames=min_span_frames,
            merge_gap_frames=merge_gap_frames,
            padding_frames=span_padding_frames,
            score=ep_precision,
            score_name="mean_diffusion_precision_score",
        )
        for label in label_counts:
            label_counts[label] += int(np.count_nonzero(labels == label))
        total_spans += len(spans)
        records_by_key[episode.key] = {
            "diffusion_precision_score": ep_precision,
            "diffusion_entropy_score": ep_entropy,
            "diffusion_reconstruction_error": ep_recon,
            "label": [str(x) for x in labels.tolist()],
            "acceleration_stride": strides,
            "hard_spans": spans,
        }
        offset += n
    return {
        "records_by_key": records_by_key,
        "summary": {
            "thresholds": {
                "diffusion_precision_min": precision_threshold,
                "diffusion_entropy_casual_min": casual_threshold,
                "static_speed_max": static_threshold,
                "precision_quantile": precision_quantile,
                "casual_quantile": casual_quantile,
                "static_speed_quantile": static_speed_quantile,
                "smoothing_half_window": int(smoothing_half_window),
                "min_span_frames": int(min_span_frames),
                "merge_gap_frames": int(merge_gap_frames),
                "span_padding_frames": int(span_padding_frames),
            },
            "label_counts": label_counts,
            "num_hard_spans": total_spans,
            "num_episodes": len(arrays.episodes),
            "num_frames": int(len(arrays.condition)),
            "action_dim": 14,
        },
    }
