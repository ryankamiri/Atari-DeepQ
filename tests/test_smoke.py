"""
Smoke tests for Ticket 0: make_env, evaluate_policy, and run artifact layout.
Run from project root: pytest tests/test_smoke.py -v
"""
import json
import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np

from src.envs import make_env
from src.envs.atari_ram_env import _SeedWrapper
from src.utils.eval import evaluate_policy


class RandomPolicy:
    """Stub policy with act(obs, epsilon) -> int."""

    def __init__(self, n_actions: int):
        self.n_actions = n_actions

    def act(self, obs, epsilon=0.0):
        return int(np.random.randint(0, self.n_actions))


def test_make_env_obs_shape_and_dtype():
    """Instantiate make_env; reset returns float32 obs of shape (128,)."""
    env = make_env("ALE/Pong-ram-v5", seed=42, eval_mode=False)
    obs, _ = env.reset()
    assert obs.dtype == np.float32, obs.dtype
    assert obs.shape == (128,), obs.shape
    env.close()


def test_make_env_obs_in_range():
    """Verify observation values are in [0, 1]."""
    env = make_env("ALE/Pong-ram-v5", seed=42, eval_mode=False)
    obs, _ = env.reset()
    assert np.all(obs >= 0) and np.all(obs <= 1), f"obs range [{obs.min()}, {obs.max()}]"
    assert env.observation_space.dtype == np.float32
    env.close()


def test_make_env_rewards_clipped():
    """Roll one episode and confirm reward outputs are in [-1, 1]."""
    env = make_env("ALE/Pong-ram-v5", seed=42, eval_mode=False)
    obs, _ = env.reset()
    for _ in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        assert -1 <= reward <= 1, f"reward {reward} not in [-1, 1]"
        if terminated or truncated:
            break
    env.close()


class DummySeedEnv(gym.Env):
    metadata = {}

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.reset_calls = []

    def reset(self, *, seed=None, options=None):
        self.reset_calls.append(seed)
        return np.zeros((1,), dtype=np.float32), {}

    def step(self, action):
        return np.zeros((1,), dtype=np.float32), 0.0, True, False, {}


def test_seed_wrapper_only_uses_default_seed_once():
    """Implicit resets should seed the env once, then leave later resets unseeded."""
    env = _SeedWrapper(DummySeedEnv(), seed=42)
    env.reset()
    env.reset()
    assert env.unwrapped.reset_calls == [42, None]
    env.close()


def test_make_env_explicit_reset_seed_is_deterministic():
    """Passing an explicit reset seed should still reproduce the same initial observation."""
    env = make_env("ALE/Pong-ram-v5", seed=42, eval_mode=True)
    obs_a, _ = env.reset(seed=7)
    obs_b, _ = env.reset(seed=7)
    assert np.array_equal(obs_a, obs_b)
    env.close()


def test_evaluate_policy_returns_dict():
    """Run evaluate_policy with stub policy for 2 episodes; check dict keys and JSON."""
    env = make_env("ALE/Pong-ram-v5", seed=0, eval_mode=True)
    n_actions = env.action_space.n
    env.close()

    policy = RandomPolicy(n_actions)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "eval.json"
        summary = evaluate_policy(
            policy,
            env_id="ALE/Pong-ram-v5",
            n_episodes=2,
            epsilon_eval=0.05,
            seed=123,
            output_path=str(out),
        )
        assert "env_id" in summary
        assert "n_episodes" in summary
        assert summary["n_episodes"] == 2
        assert "epsilon_eval" in summary
        assert "episode_returns" in summary
        assert len(summary["episode_returns"]) == 2
        assert "mean_return" in summary
        assert "std_return" in summary
        assert "episode_lengths" in summary
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded["mean_return"] == summary["mean_return"]
        assert loaded["episode_returns"] == summary["episode_returns"]


def test_evaluate_policy_json_roundtrip():
    """Ensure saved JSON is valid and contains required fields."""
    env = make_env("ALE/Pong-ram-v5", seed=0, eval_mode=True)
    n_actions = env.action_space.n
    env.close()

    policy = RandomPolicy(n_actions)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "eval.json"
        evaluate_policy(
            policy,
            env_id="ALE/Pong-ram-v5",
            n_episodes=1,
            epsilon_eval=0.0,
            seed=1,
            output_path=str(out),
        )
        with open(out) as f:
            data = json.load(f)
    for key in ["env_id", "n_episodes", "epsilon_eval", "episode_returns", "mean_return", "std_return", "episode_lengths"]:
        assert key in data, f"missing key {key}"
