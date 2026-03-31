"""
Conservative Q-Learning (CQL) for offline RL on Atari RAM.

CQL addresses OOD action overestimation by adding a regularizer
that explicitly pushes DOWN Q-values for actions not supported
by the dataset, while pushing UP Q-values for dataset actions.

In discrete action spaces the CQL penalty is:

    L_CQL = alpha * (logsumexp_a Q(s, a) - Q(s, a_dataset))

Intuition:
  - logsumexp_a Q(s, a) ~ log(sum of exp Q-values over all actions)
    This is a soft maximum — it is large when any action has a high Q-value
  - Q(s, a_dataset) is the Q-value of the action actually in the dataset
  - Their difference is large when OOD actions have high Q-values
  - Penalizing this difference pushes Q-values down for OOD actions

Combined loss:
    L = L_DQN + alpha * L_CQL

where alpha controls how conservative the agent is.
Higher alpha -> more conservative -> safer but potentially underestimates.

Expected behavior vs naive offline DQN:
  - Much more stable Q-values (no divergence)
  - Better performance across all dataset qualities
  - On D_random: still limited by data quality, but won't diverge
  - On D_expert_ish: best offline performance overall

Reference: Kumar et al., "Conservative Q-Learning for Offline RL", NeurIPS 2020.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from src.algos.dqn import compute_bootstrap_target
from src.nets import DuelingMLPQNetwork, MLPQNetwork


@dataclass
class CQLUpdateResult:
    loss_total: float
    loss_dqn:   float
    loss_cql:   float
    q_mean:     float
    td_abs_mean: float
    cql_gap:    float  # mean(logsumexp Q - Q(s, a_dataset)), measures conservatism

    @property
    def metrics(self) -> dict[str, float]:
        return {
            "loss":      self.loss_total,
            "loss_dqn":  self.loss_dqn,
            "loss_cql":  self.loss_cql,
            "q_mean":    self.q_mean,
            "td_abs_mean": self.td_abs_mean,
            "cql_gap":   self.cql_gap,
        }


class CQLAgent:
    """
    Conservative Q-Learning agent for discrete action spaces.

    Parameters
    ----------
    obs_dim : int
    n_actions : int
    hidden_dims : list[int]
    device : torch.device
    gamma : float
        Discount factor.
    lr : float
        Adam learning rate.
    cql_alpha : float
        CQL regularization weight. Controls conservatism.
        Typical range: 0.5 - 5.0. Start with 1.0.
    target_update_interval : int
        Update steps between target network syncs.
    grad_clip_norm : float
    double_dqn : bool
    dueling : bool
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dims: list[int],
        device: torch.device,
        gamma: float = 0.99,
        lr: float = 1e-3,
        cql_alpha: float = 1.0,
        target_update_interval: int = 1000,
        grad_clip_norm: float = 10.0,
        double_dqn: bool = True,
        dueling: bool = True,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.cql_alpha = cql_alpha
        self.target_update_interval = target_update_interval
        self.grad_clip_norm = grad_clip_norm
        self.double_dqn = double_dqn
        self.device = device
        self._update_count = 0

        net_cls = DuelingMLPQNetwork if dueling else MLPQNetwork
        self.q_net = net_cls(obs_dim, n_actions, hidden_dims).to(device)
        self.target_net = copy.deepcopy(self.q_net)
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

    def act(self, obs: np.ndarray, epsilon: float = 0.0) -> int:
        """Epsilon-greedy action. Matches DQNAgent interface."""
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
    ) -> CQLUpdateResult:
        """
        One combined DQN + CQL gradient step.

        DQN loss: Huber(Q(s,a), TD target)
        CQL loss: mean(logsumexp_a Q(s,a) - Q(s, a_dataset))
        Total:    L_DQN + cql_alpha * L_CQL
        """
        o      = torch.as_tensor(obs,      dtype=torch.float32, device=self.device)
        a      = torch.as_tensor(actions,  dtype=torch.int64,   device=self.device)
        r      = torch.as_tensor(rewards,  dtype=torch.float32, device=self.device)
        o_next = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        d      = torch.as_tensor(dones,    dtype=torch.float32, device=self.device)

        # --- DQN loss ---
        with torch.no_grad():
            q_target_next = self.target_net(o_next)
            q_online_next = self.q_net(o_next) if self.double_dqn else None
            target = compute_bootstrap_target(
                q_target_next, r, d, self.gamma, self.double_dqn, q_online_next
            )

        q_all = self.q_net(o)                               # (batch, n_actions)
        q_sa  = q_all.gather(1, a.unsqueeze(1)).squeeze(1)  # (batch,)
        loss_dqn = nn.functional.smooth_l1_loss(q_sa, target)

        # --- CQL penalty ---
        # logsumexp over actions: soft-max of Q-values, large if any OOD action is high
        logsumexp_q = torch.logsumexp(q_all, dim=1)         # (batch,)
        # CQL gap: how much higher OOD actions are vs dataset action
        cql_gap = (logsumexp_q - q_sa).mean()
        loss_cql = cql_gap                                  # already a scalar mean

        # --- Combined loss ---
        loss = loss_dqn + self.cql_alpha * loss_cql

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        # --- Auto sync target ---
        self._update_count += 1
        if self._update_count % self.target_update_interval == 0:
            self.sync_target()

        with torch.no_grad():
            td_abs = (q_sa.detach() - target).abs().mean().item()

        return CQLUpdateResult(
            loss_total  = float(loss.item()),
            loss_dqn    = float(loss_dqn.item()),
            loss_cql    = float(loss_cql.item()),
            q_mean      = float(q_sa.detach().mean().item()),
            td_abs_mean = float(td_abs),
            cql_gap     = float(cql_gap.item()),
        )

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.q_net.state_dict())

    def state_dict_for_checkpoint(self) -> dict:
        return {
            "q_net":      self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
        }

    def load_state_dict_from_checkpoint(self, ckpt: dict) -> None:
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])