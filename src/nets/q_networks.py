"""
Q-networks for RAM input. MLP and dueling MLP for DQN family.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def combine_dueling_streams(value: torch.Tensor, advantage: torch.Tensor) -> torch.Tensor:
    """
    Aggregate dueling streams: Q(s, a) = V(s) + (A(s, a) - mean_a' A(s, a')).

    Parameters
    ----------
    value
        (batch, 1)
    advantage
        (batch, n_actions)
    """
    return value + (advantage - advantage.mean(dim=1, keepdim=True))


class MLPQNetwork(nn.Module):
    """MLP Q-network for RAM input. Outputs Q(s, a) for each action."""

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


class DuelingMLPQNetwork(nn.Module):
    """
    Dueling architecture: shared trunk (from hidden_dims), then value (1) and advantage (n_actions) linear heads.
    If hidden_dims is empty, trunk is Identity and heads read directly from obs_dim.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dims: list[int] = (256, 256)):
        super().__init__()
        hidden_dims = list(hidden_dims)
        if hidden_dims:
            trunk_layers = []
            d_in = obs_dim
            for h in hidden_dims:
                trunk_layers.append(nn.Linear(d_in, h))
                trunk_layers.append(nn.ReLU())
                d_in = h
            self.trunk = nn.Sequential(*trunk_layers)
            trunk_out = hidden_dims[-1]
        else:
            self.trunk = nn.Identity()
            trunk_out = obs_dim
        self.value_head = nn.Linear(trunk_out, 1)
        self.advantage_head = nn.Linear(trunk_out, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.trunk(x)
        v = self.value_head(z)
        a = self.advantage_head(z)
        return combine_dueling_streams(v, a)
