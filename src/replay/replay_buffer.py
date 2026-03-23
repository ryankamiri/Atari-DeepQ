"""
Uniform replay buffer for DQN. Stores (s, a, r, s', done) with numpy.
T1.4: shared ReplayBatch API with PER buffer; unit IS weights; no-op priority updates.
"""
from __future__ import annotations

import numpy as np

from .batch import ReplayBatch


class ReplayBuffer:
    """Uniform replay buffer. No prioritization."""

    def __init__(self, capacity: int, obs_shape: tuple, obs_dtype: np.dtype = np.float32):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.obs = np.zeros((capacity,) + obs_shape, dtype=obs_dtype)
        self.next_obs = np.zeros((capacity,) + obs_shape, dtype=obs_dtype)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self._pos = 0
        self._size = 0

    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool):
        self.obs[self._pos] = obs
        self.next_obs[self._pos] = next_obs
        self.actions[self._pos] = action
        self.rewards[self._pos] = reward
        self.dones[self._pos] = float(done)
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float | None = None) -> ReplayBatch:
        """Uniform indices; unit IS weights (already in (0, 1]). ``beta`` ignored."""
        indices = np.random.randint(0, self._size, size=batch_size, dtype=np.int64)
        weights = np.ones(batch_size, dtype=np.float32)
        return ReplayBatch(
            obs=self.obs[indices].copy(),
            actions=self.actions[indices].copy(),
            rewards=self.rewards[indices].copy(),
            next_obs=self.next_obs[indices].copy(),
            dones=self.dones[indices].copy(),
            indices=indices,
            weights=weights,
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """No-op for uniform replay."""
        pass

    def __len__(self) -> int:
        return self._size
