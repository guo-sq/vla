"""Small MLP action-chunk behavior cloning model."""

from __future__ import annotations

import torch
from torch import nn


class ActionChunkMLP(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 43,
        horizon: int = 16,
        action_dim: int = 14,
        hidden_dim: int = 256,
        layers: int = 3,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError(f"layers must be >= 1, got {layers}")
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        blocks: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(layers):
            blocks.extend([nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()])
            in_dim = hidden_dim
        blocks.append(nn.Linear(hidden_dim, horizon * action_dim))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.view(x.shape[0], self.horizon, self.action_dim)

