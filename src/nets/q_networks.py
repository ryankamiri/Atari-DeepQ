"""
Q-networks for RAM input. MLP for vanilla DQN (T0); dueling MLP added in later tickets.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPQNetwork(nn.Module):
    """MLP Q-network for 128-dim RAM input. Outputs Q(s, a) for each action."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dims: list[int] = (256, 256)):
        super().__init__()
        dims = [obs_dim] + list(hidden_dims) + [n_actions]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
