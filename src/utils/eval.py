"""
Unified evaluation harness: evaluate_policy() for any policy with act(obs, epsilon).
Shared by online and offline pipelines.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.envs import make_env


def evaluate_policy(
    policy,
    env_id: str,
    n_episodes: int,
    epsilon_eval: float,
    seed: int,
    output_path: str | None = None,
) -> dict:
    """
    Run evaluation episodes and return a JSON-serializable summary.

    Parameters
    ----------
    policy
        Object with act(obs, epsilon=...) -> int.
    env_id : str
        Gymnasium env id.
    n_episodes : int
        Number of evaluation episodes.
    epsilon_eval : float
        Epsilon for policy.act (e.g. 0.05 for near-greedy).
    seed : int
        Env seed for evaluation.
    output_path : str | None
        If set, save summary dict as JSON here.

    Returns
    -------
    dict
        env_id, n_episodes, epsilon_eval, episode_returns, mean_return, std_return, episode_lengths.
    """
    env = make_env(env_id, seed=seed, eval_mode=True)
    episode_returns: list[float] = []
    episode_lengths: list[int] = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done, truncated = False, False
        total_reward = 0.0
        length = 0
        while not (done or truncated):
            action = policy.act(obs, epsilon=epsilon_eval)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            length += 1
        episode_returns.append(total_reward)
        episode_lengths.append(length)
    env.close()

    arr = np.array(episode_returns)
    mean_return = float(np.mean(arr))
    std_return = float(np.std(arr)) if len(arr) > 1 else 0.0

    summary = {
        "env_id": env_id,
        "n_episodes": n_episodes,
        "epsilon_eval": epsilon_eval,
        "episode_returns": episode_returns,
        "mean_return": mean_return,
        "std_return": std_return,
        "episode_lengths": episode_lengths,
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

    return summary
