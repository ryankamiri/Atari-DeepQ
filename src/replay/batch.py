"""Shared replay batch container for uniform and prioritized sampling."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReplayBatch:
    """One training minibatch from replay (uniform or PER)."""

    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    dones: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
