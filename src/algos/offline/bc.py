"""
Behavior Cloning (BC) for offline RL on Atari RAM.

Trains a policy network pi(a | s) by supervised learning:
minimizes cross-entropy loss between predicted action logits
and the actions recorded in the dataset.

BC has no notion of rewards or value — it purely imitates
whatever behavior generated the dataset. As a result:
  - On D_random:     will learn a near-random policy (useless)
  - On D_mixed:      will partially imitate the trained agent
  - On D_expert_ish: will closely imitate the near-greedy agent

Act interface matches DQNAgent so the same evaluate_policy()
utility works unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from src.nets import MLPQNetwork


@dataclass
class BCUpdateResult:
    loss: float
    accuracy: float


class BCAgent:
    """
    Behavior Cloning agent.

    Uses an MLP with the same architecture as the online Q-network
    but treats output logits as action probabilities (softmax at
    inference, cross-entropy at training).

    Parameters
    ----------
    obs_dim : int
        Observation dimensionality (128 for Atari RAM).
    n_actions : int
        Number of discrete actions.
    hidden_dims : list[int]
        Hidden layer sizes (mirrors online agent default).
    device : torch.device
    lr : float
        Adam learning rate.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dims: list[int],
        device: torch.device,
        lr: float = 1e-3,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = device

        # Reuse MLPQNetwork — output logits over actions
        self.policy_net = MLPQNetwork(obs_dim, n_actions, hidden_dims).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def act(self, obs: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Greedy action from policy logits.
        epsilon is accepted for API compatibility with evaluate_policy()
        but epsilon-greedy is not used — BC always acts greedily.
        """
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.policy_net(x)
        return int(logits.argmax(dim=1).item())

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
    ) -> BCUpdateResult:
        """
        One supervised gradient step on a minibatch.

        Parameters
        ----------
        obs : (batch, obs_dim) float32
        actions : (batch,) int64  — target actions from dataset

        Returns
        -------
        BCUpdateResult with cross-entropy loss and classification accuracy.
        """
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(actions, dtype=torch.int64, device=self.device)

        logits = self.policy_net(o)
        loss = self.loss_fn(logits, a)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            accuracy = float((preds == a).float().mean().item())

        return BCUpdateResult(loss=float(loss.item()), accuracy=accuracy)

    def state_dict_for_checkpoint(self) -> dict:
        return {
            "policy_net": self.policy_net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
        }

    def load_state_dict_from_checkpoint(self, ckpt: dict) -> None:
        self.policy_net.load_state_dict(ckpt["policy_net"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])