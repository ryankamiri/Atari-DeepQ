"""
Proportional prioritized experience replay (PER) with sum tree + min tree.
"""
from __future__ import annotations

import numpy as np

from .batch import ReplayBatch
from .sum_min_tree import MinTree, SumTree


class PrioritizedReplayBuffer:
    """
    Ring buffer with proportional sampling. Tree leaves store (raw_priority ** alpha).

    New transitions get the current maximum raw priority. ``update_priorities`` accepts
    raw positive priorities and applies ``alpha`` when writing to the trees.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple,
        alpha: float = 0.6,
        obs_dtype: np.dtype = np.float32,
    ):
        self.capacity = int(capacity)
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.alpha = float(alpha)

        self.obs = np.zeros((capacity,) + obs_shape, dtype=obs_dtype)
        self.next_obs = np.zeros((capacity,) + obs_shape, dtype=obs_dtype)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self._pos = 0
        self._size = 0
        self.max_raw_priority = 1.0

        self._sum_tree = SumTree(capacity)
        self._min_tree = MinTree(capacity)

    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool):
        idx = self._pos
        self.obs[idx] = obs
        self.next_obs[idx] = next_obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = float(done)

        p_tree = self.max_raw_priority**self.alpha
        self._sum_tree.update(idx, p_tree)
        self._min_tree.update(idx, p_tree)

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 0.4) -> ReplayBatch:
        if self._size == 0:
            raise ValueError("cannot sample empty buffer")
        beta = float(beta)
        n = self._size
        total = self._sum_tree.total()
        if total <= 0:
            raise RuntimeError("PER sum tree total is non-positive")

        indices = np.zeros(batch_size, dtype=np.int64)
        for i in range(batch_size):
            mass = np.random.uniform(0, total)
            indices[i] = self._sum_tree.retrieve(mass)

        # P(i) = priority_i / total (priorities are already scaled by alpha in tree)
        probs = np.array([self._leaf_priority(int(j)) for j in indices], dtype=np.float64)
        p = probs / total
        w = (n * p) ** (-beta)

        # Normalize by the buffer-wide maximum IS weight, which comes from the
        # smallest non-zero sampling probability currently in replay.
        min_leaf = float(self._min_tree.min())
        if not np.isfinite(min_leaf) or min_leaf <= 0:
            max_weight = 1.0
        else:
            min_p = min_leaf / total
            max_weight = (n * min_p) ** (-beta)
        if not np.isfinite(max_weight) or max_weight <= 0:
            max_weight = 1.0

        weights = (w / max_weight).astype(np.float32)

        return ReplayBatch(
            obs=self.obs[indices].copy(),
            actions=self.actions[indices].copy(),
            rewards=self.rewards[indices].copy(),
            next_obs=self.next_obs[indices].copy(),
            dones=self.dones[indices].copy(),
            indices=indices,
            weights=weights,
        )

    def _leaf_priority(self, data_idx: int) -> float:
        ti = data_idx + self.capacity - 1
        return float(self._sum_tree.tree[ti])

    def update_priorities(self, indices: np.ndarray, raw_priorities: np.ndarray):
        """Raw positive priorities; alpha applied here. No-op entries ignored."""
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        raw = np.asarray(raw_priorities, dtype=np.float64).reshape(-1)
        assert len(indices) == len(raw)
        for idx, r in zip(indices, raw):
            r = max(float(r), 1e-8)
            self.max_raw_priority = max(self.max_raw_priority, r)
            p_tree = r**self.alpha
            self._sum_tree.update(int(idx), p_tree)
            self._min_tree.update(int(idx), p_tree)

    def __len__(self) -> int:
        return self._size
