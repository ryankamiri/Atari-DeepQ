"""
Uniform replay buffer for DQN. Stores (s, a, r, s', done) with numpy.
"""
from __future__ import annotations

import numpy as np


class ReplayBuffer:
    """Uniform replay buffer. No prioritization (T0)."""

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

    def sample(self, batch_size: int):
        """Returns (obs, actions, rewards, next_obs, dones). No IS weights for uniform buffer."""
        indices = np.random.randint(0, self._size, size=batch_size)
        return (
            self.obs[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_obs[indices],
            self.dones[indices],
        )

    def __len__(self) -> int:
        return self._size
