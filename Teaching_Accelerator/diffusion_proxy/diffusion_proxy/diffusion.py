"""DDPM utilities for action-chunk diffusion."""

from __future__ import annotations

import math

import torch


class DiffusionSchedule:
    def __init__(self, num_steps: int = 100, beta_start: float = 1e-4, beta_end: float = 2e-2, device=None) -> None:
        if num_steps < 2:
            raise ValueError(f"num_steps must be >= 2, got {num_steps}")
        self.num_steps = int(num_steps)
        self.device = torch.device(device or "cpu")
        self.betas = torch.linspace(beta_start, beta_end, self.num_steps, dtype=torch.float32, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def to(self, device) -> "DiffusionSchedule":
        return DiffusionSchedule(self.num_steps, float(self.betas[0].cpu()), float(self.betas[-1].cpu()), device=device)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1)
        return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * noise

    def ddim_timesteps(self, sampling_steps: int) -> list[int]:
        sampling_steps = min(max(int(sampling_steps), 1), self.num_steps)
        values = torch.linspace(self.num_steps - 1, 0, sampling_steps, dtype=torch.long)
        return [int(x) for x in values.tolist()]


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    if dim % 2 != 0:
        raise ValueError("sinusoidal embedding dim must be even")
    timesteps = timesteps.float()
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, dtype=torch.float32, device=timesteps.device) * (-math.log(10000.0) / max(half - 1, 1))
    )
    args = timesteps[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

