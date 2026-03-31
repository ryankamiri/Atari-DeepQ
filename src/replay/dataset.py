"""
Offline dataset loader for Atari RAM experiments.
Loads .npz files produced by scripts/offline_generate_data.py and
provides a sample() method returning a ReplayBatch (same interface
as the online replay buffers, so offline algos plug in identically).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .batch import ReplayBatch


class OfflineDataset:
    """
    Fixed dataset loaded from a .npz file.

    Parameters
    ----------
    path : str or Path
        Path to the .npz file (e.g. artifacts/offline_data/pong/random.npz).
    device : str, optional
        Ignored here — kept for API symmetry. Tensors are created by the
        caller (the training loop) so the dataset stays numpy-only.

    Attributes
    ----------
    obs, actions, rewards, next_obs, dones : np.ndarray
        Raw arrays of shape (N, 128), (N,), (N,), (N, 128), (N,).
    size : int
        Total number of transitions in the dataset.
    """

    def __init__(self, path: str | Path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        data = np.load(path)
        self.obs      = data["obs"].astype(np.float32)       # (N, 128)
        self.actions  = data["actions"].astype(np.int64)     # (N,)
        self.rewards  = data["rewards"].astype(np.float32)   # (N,)
        self.next_obs = data["next_obs"].astype(np.float32)  # (N, 128)
        self.dones    = data["dones"].astype(np.float32)     # (N,)
        self.size     = len(self.obs)

        self._validate()
        print(f"Loaded dataset: {path.name}  "
              f"transitions={self.size:,}  "
              f"obs_shape={self.obs.shape[1:]}")

    def _validate(self) -> None:
        assert self.obs.ndim == 2 and self.obs.shape[1] == 128, \
            f"Expected obs shape (N, 128), got {self.obs.shape}"
        assert self.actions.ndim == 1, \
            f"Expected actions shape (N,), got {self.actions.shape}"
        assert self.rewards.ndim == 1, \
            f"Expected rewards shape (N,), got {self.rewards.shape}"
        assert self.next_obs.shape == self.obs.shape, \
            f"obs/next_obs shape mismatch: {self.obs.shape} vs {self.next_obs.shape}"
        assert self.dones.ndim == 1, \
            f"Expected dones shape (N,), got {self.dones.shape}"
        n = self.size
        for name, arr in [("actions", self.actions), ("rewards", self.rewards),
                           ("next_obs", self.next_obs), ("dones", self.dones)]:
            assert len(arr) == n, \
                f"Array '{name}' has {len(arr)} entries, expected {n}"

    def sample(self, batch_size: int) -> ReplayBatch:
        """
        Sample a random minibatch uniformly from the fixed dataset.
        Returns a ReplayBatch with unit importance weights (same as
        uniform ReplayBuffer, so offline algos need no changes).
        """
        indices = np.random.randint(0, self.size, size=batch_size, dtype=np.int64)
        return ReplayBatch(
            obs=self.obs[indices].copy(),
            actions=self.actions[indices].copy(),
            rewards=self.rewards[indices].copy(),
            next_obs=self.next_obs[indices].copy(),
            dones=self.dones[indices].copy(),
            indices=indices,
            weights=np.ones(batch_size, dtype=np.float32),
        )

    def stats(self) -> dict:
        """Return a summary dict useful for logging and sanity checks."""
        return {
            "size":           self.size,
            "obs_shape":      tuple(self.obs.shape),
            "reward_mean":    float(self.rewards.mean()),
            "reward_std":     float(self.rewards.std()),
            "reward_min":     float(self.rewards.min()),
            "reward_max":     float(self.rewards.max()),
            "reward_nonzero": int(np.count_nonzero(self.rewards)),
            "done_count":     int(self.dones.sum()),
            "action_counts":  {
                int(a): int(c)
                for a, c in zip(*np.unique(self.actions, return_counts=True))
            },
        }

    def __len__(self) -> int:
        return self.size