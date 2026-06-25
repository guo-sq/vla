"""Small conditional MLP denoiser for action chunks."""

from __future__ import annotations

import torch
from torch import nn

from diffusion_proxy.diffusion import sinusoidal_embedding


class ResidualConditionBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x) + cond)


class ActionChunkDenoiser(nn.Module):
    def __init__(
        self,
        *,
        cond_dim: int,
        horizon: int = 16,
        action_dim: int = 14,
        hidden_dim: int = 512,
        blocks: int = 4,
        time_dim: int = 128,
    ) -> None:
        super().__init__()
        if blocks < 1:
            raise ValueError(f"blocks must be >= 1, got {blocks}")
        self.cond_dim = int(cond_dim)
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.time_dim = int(time_dim)
        flat_dim = self.horizon * self.action_dim
        self.x_proj = nn.Linear(flat_dim, hidden_dim)
        self.cond_proj = nn.Sequential(nn.Linear(cond_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.time_proj = nn.Sequential(nn.Linear(time_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.blocks = nn.ModuleList([ResidualConditionBlock(hidden_dim) for _ in range(blocks)])
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, flat_dim))

    def forward(self, noisy_action: torch.Tensor, cond: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        x = noisy_action.reshape(noisy_action.shape[0], -1)
        h = self.x_proj(x)
        t_emb = sinusoidal_embedding(timesteps, self.time_dim)
        c = self.cond_proj(cond) + self.time_proj(t_emb)
        for block in self.blocks:
            h = block(h, c)
        out = self.out(h)
        return out.view(noisy_action.shape[0], self.horizon, self.action_dim)
