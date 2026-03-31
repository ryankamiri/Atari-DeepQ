"""
Behavior-Regularized Offline DQN (DQN+BC) for offline RL on Atari RAM.

Extends naive offline DQN by adding a supervised behavior cloning
loss term that penalizes the policy for deviating from dataset actions.
This directly addresses the OOD action problem by keeping the implied
greedy policy close to the behavior policy that generated the data.

Combined loss:
    L = L_DQN + lambda * L_BC

where:
    L_DQN = Huber(Q(s,a), r + gamma * max_a' Q_target(s', a'))
    L_BC  = CrossEntropy(Q(s, :), a_dataset)

The BC term acts as a regularizer — higher lambda means stronger
pull toward dataset actions, lower lambda allows more deviation
in pursuit of higher Q-values.

Lambda grid from proposal: {0.1, 1.0, 3.0}

Expected behavior vs naive offline DQN:
  - More stable training (BC term prevents Q divergence)
  - Better performance on D_mixed and D_expert_ish
  - On D_random: BC term hurts since dataset actions are useless
  - Higher lambda -> safer but more constrained policy
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from src.algos.dqn import compute_bootstrap_target, linear_schedule
from src.nets import DuelingMLPQNetwork, MLPQNetwork


@dataclass
class DQNBCUpdateResult:
    loss_total: float
    loss_dqn:   float
    loss_bc:    float
    q_mean:     float
    td_abs_mean: float
    accuracy:   float  # BC classification accuracy on this batch

    @property
    def metrics(self) -> dict[str, float]:
        return {
            "loss":         self.loss_total,
            "loss_dqn":     self.loss_dqn,
            "loss_bc":      self.loss_bc,
            "q_mean":       self.q_mean,
            "td_abs_mean":  self.td_abs_mean,
            "bc_accuracy":  self.accuracy,
        }


class DQNBCAgent:
    """
    Behavior-regularized offline DQN agent.

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
    lam : float
        BC regularization weight (lambda). Tune over {0.1, 1.0, 3.0}.
    target_update_interval : int
        How many update() calls between target network syncs.
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
        lam: float = 1.0,
        target_update_interval: int = 1000,
        grad_clip_norm: float = 10.0,
        double_dqn: bool = True,
        dueling: bool = True,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.lam = lam
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
        self.ce_loss = nn.CrossEntropyLoss()

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
    ) -> DQNBCUpdateResult:
        """
        One combined DQN + BC gradient step.

        DQN loss: Huber between Q(s,a) and TD target.
        BC loss:  CrossEntropy between Q(s,:) logits and dataset actions.
        Total:    L_DQN + lambda * L_BC
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

        q_all = self.q_net(o)                                      # (batch, n_actions)
        q_sa  = q_all.gather(1, a.unsqueeze(1)).squeeze(1)         # (batch,)
        loss_dqn = nn.functional.smooth_l1_loss(q_sa, target)

        # --- BC loss (cross-entropy on Q logits vs dataset actions) ---
        loss_bc = self.ce_loss(q_all, a)

        # --- Combined loss ---
        loss = loss_dqn + self.lam * loss_bc

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
            td_abs  = (q_sa.detach() - target).abs().mean().item()
            preds   = q_all.detach().argmax(dim=1)
            acc     = float((preds == a).float().mean().item())

        return DQNBCUpdateResult(
            loss_total  = float(loss.item()),
            loss_dqn    = float(loss_dqn.item()),
            loss_bc     = float(loss_bc.item()),
            q_mean      = float(q_sa.detach().mean().item()),
            td_abs_mean = float(td_abs),
            accuracy    = acc,
        )

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.q_net.state_dict())

    def state_dict_for_checkpoint(self) -> dict:
        return {
            "q_net":     self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict_from_checkpoint(self, ckpt: dict) -> None:
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])