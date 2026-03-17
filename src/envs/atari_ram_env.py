"""
Atari RAM environment factory: make_env() with RAM obs, normalization, reward clipping, seeding.
Shared by online and offline pipelines.
"""
from __future__ import annotations

import ale_py
import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Register ALE namespace so gym.make("ALE/...") works (required with gymnasium + ale-py).
gym.register_envs(ale_py)


def make_env(env_id: str, seed: int, eval_mode: bool = False) -> gym.Env:
    """
    Create an Atari RAM environment with consistent preprocessing.

    - Uses obs_type="ram" (128 bytes).
    - Converts observations to float32 and normalizes to [0, 1].
    - Clips rewards to [-1, 1].
    - Seeds env reset and action space deterministically.
    - eval_mode=True: no training-only stochastic wrappers; no frame stacking, reward shaping, or life-loss termination.

    Parameters
    ----------
    env_id : str
        Gymnasium env id (e.g. "ALE/Pong-ram-v5").
    seed : int
        Random seed for env and action space.
    eval_mode : bool
        If True, disable training-only behavior (e.g. no sticky actions if we add them later).

    Returns
    -------
    gym.Env
        Wrapped environment with normalized RAM obs and clipped rewards.
    """
    # Support "ALE/Pong-ram-v5" style ids: ale-py registers "ALE/Pong-v5"; we pass obs_type="ram".
    if "-ram-v" in env_id:
        env_id = env_id.replace("-ram-v", "-v")
    env = gym.make(env_id, obs_type="ram")
    env = _NormalizeRAM(env)
    env = _ClipReward(env)
    env = _SeedWrapper(env, seed)
    if eval_mode:
        env = _EvalModeWrapper(env)
    return env


class _NormalizeRAM(gym.ObservationWrapper):
    """Convert RAM obs to float32 in [0, 1]."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=self.env.observation_space.shape,
            dtype=np.float32,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        out = np.asarray(obs, dtype=np.float32)
        if out.max() > 1.0 or out.min() < 0.0:
            out = np.clip(out / 255.0, 0.0, 1.0)
        return out


class _ClipReward(gym.RewardWrapper):
    """Clip rewards to [-1, 1]."""

    def reward(self, reward: float) -> float:
        return float(np.clip(reward, -1.0, 1.0))


class _SeedWrapper(gym.Wrapper):
    """Seed the env once, without rewinding RNG on every episode reset."""

    def __init__(self, env: gym.Env, seed: int):
        super().__init__(env)
        self._seed = seed
        self._has_seeded = False
        self.action_space.seed(seed)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._has_seeded = True
            self._seed = seed
        elif not self._has_seeded:
            seed = self._seed
            self._has_seeded = True
        return self.env.reset(seed=seed, options=options)


class _EvalModeWrapper(gym.Wrapper):
    """Placeholder for eval-only behavior (e.g. disable sticky actions). No-op for T0."""

    pass
