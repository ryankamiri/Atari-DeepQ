"""
Binary sum tree and min tree over fixed leaves (capacity).
Used for proportional PER: O(log n) sample and priority update.
"""
from __future__ import annotations

import numpy as np


class SumTree:
    """Sum segment tree; leaf i (data index) at tree node i + capacity - 1."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float64)

    def total(self) -> float:
        return float(self.tree[0])

    def update(self, data_idx: int, priority: float):
        """Set leaf data_idx priority (non-negative)."""
        p = max(0.0, float(priority))
        tree_idx = data_idx + self.capacity - 1
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        self._propagate(tree_idx, change)

    def _propagate(self, idx: int, change: float):
        while idx != 0:
            parent = (idx - 1) // 2
            self.tree[parent] += change
            idx = parent

    def retrieve(self, s: float) -> int:
        """Return data index for cumulative mass s in (0, total]."""
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                return idx - self.capacity + 1
            if s <= self.tree[left] + 1e-12:
                idx = left
            else:
                s -= self.tree[left]
                idx = right


class MinTree:
    """Min segment tree over the same leaf indexing as SumTree."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.tree = np.full(2 * self.capacity - 1, np.inf, dtype=np.float64)

    def min(self) -> float:
        return float(self.tree[0])

    def update(self, data_idx: int, value: float):
        tree_idx = data_idx + self.capacity - 1
        self.tree[tree_idx] = float(value)
        self._recalculate(tree_idx)

    def _recalculate(self, tree_idx: int):
        idx = tree_idx
        while idx != 0:
            parent = (idx - 1) // 2
            left = 2 * parent + 1
            right = left + 1
            self.tree[parent] = min(self.tree[left], self.tree[right])
            idx = parent
