"""
Vanilla DQN trainer: MLP Q-network, uniform replay, target network, epsilon-greedy.
T1.1: Huber loss, gradient clipping, structured update metrics.
Extensible for Double DQN, dueling, PER in later tickets.
"""
from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn

from src.nets import MLPQNetwork


def linear_schedule(step: int, start: float, end: float, decay_steps: int) -> float:
    """Linear epsilon decay from start to end over decay_steps."""
    if decay_steps <= 0:
        return end
    t = min(1.0, step / decay_steps)
    return start + t * (end - start)


class DQNAgent:
    """
    Vanilla DQN agent. Policy interface: act(obs, epsilon) -> int.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dims: list[int],
        device: torch.device,
        gamma: float = 0.99,
        lr: float = 2.5e-4,
        target_update_interval: int = 1000,
        grad_clip_norm: float = 10.0,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.target_update_interval = target_update_interval
        self.grad_clip_norm = grad_clip_norm
        self.device = device

        self.q_net = MLPQNetwork(obs_dim, n_actions, hidden_dims).to(device)
        self.target_net = copy.deepcopy(self.q_net)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

    def act(self, obs: np.ndarray, epsilon: float = 0.0) -> int:
        """Epsilon-greedy action. obs: (128,) float32."""
        if np.random.random() < epsilon:
            return int(np.random.randint(0, self.n_actions))
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.q_net(x)
        return int(q.argmax(dim=1).item())

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
    ) -> dict[str, float]:
        """
        One batch gradient step. Huber (smooth L1) loss, grad clip.

        Returns
        -------
        dict with keys: loss, q_mean, target_mean, td_abs_mean (all finite floats).
        """
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        r = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        o_next = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        d = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_q = self.target_net(o_next).max(dim=1)[0]
            target = r + self.gamma * (1 - d) * next_q

        q = self.q_net(o).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        with torch.no_grad():
            td_abs = (q.detach() - target).abs()
            metrics = {
                "loss": float(loss.item()),
                "q_mean": float(q.detach().mean().item()),
                "target_mean": float(target.mean().item()),
                "td_abs_mean": float(td_abs.mean().item()),
            }
        return metrics

    def sync_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def state_dict_for_checkpoint(self) -> dict:
        """State for checkpoint (model, target, optimizer)."""
        return {
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict_from_checkpoint(self, ckpt: dict):
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
