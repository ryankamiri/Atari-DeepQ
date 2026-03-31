"""
Naive Offline DQN for offline RL on Atari RAM.

Runs standard DQN Bellman regression on a fixed dataset with
no additional constraints. This is intentionally the simplest
possible offline baseline — it will suffer from value
overestimation on out-of-distribution (OOD) actions because
the Bellman bootstrap:

    y = r + gamma * max_a' Q(s', a')

freely selects actions at s' that may never appear in the dataset,
causing Q-values to diverge on unseen state-action pairs.

Expected behavior:
  - D_random:     likely diverges or learns near-nothing
  - D_mixed:      unstable, may partially learn
  - D_expert_ish: best chance of working, still fragile

This is a thin wrapper around the existing DQNAgent — all the
heavy lifting (Double DQN, dueling, Huber loss, grad clipping)
is already implemented there. We just drive it with dataset
batches instead of a live replay buffer.

Act interface matches DQNAgent so evaluate_policy() works unchanged.
"""
from __future__ import annotations

from src.algos.dqn import DQNAgent, UpdateResult


class OfflineDQNAgent(DQNAgent):
    """
    Naive offline DQN agent.

    Inherits everything from DQNAgent unchanged:
      - act(obs, epsilon) -> int
      - update(obs, actions, rewards, next_obs, dones, weights) -> UpdateResult
      - sync_target()
      - state_dict_for_checkpoint() / load_state_dict_from_checkpoint()

    The only difference from online DQN is that update() is called
    with batches from a fixed OfflineDataset instead of a live
    ReplayBuffer. No other changes needed.

    Parameters
    ----------
    obs_dim, n_actions, hidden_dims, device, gamma, lr,
    target_update_interval, grad_clip_norm, double_dqn, dueling
        All passed directly to DQNAgent.
    """
    pass