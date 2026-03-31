"""
Offline dataset generator for Atari RAM environments.

Loads a trained online agent checkpoint and rolls out episodes at three
epsilon levels to produce datasets of increasing quality:
  - D_random     : epsilon = 1.0  (pure random)
  - D_mixed      : epsilon = 0.30 (partially trained)
  - D_expert_ish : epsilon = 0.05 (near-greedy)

Saves each dataset as a compressed .npz file under artifacts/offline_data/:
  artifacts/offline_data/<env_short>/<split>.npz

Each .npz contains arrays: obs, actions, rewards, next_obs, dones
with shapes (N, 128), (N,), (N,), (N, 128), (N,) respectively.

Usage:
  # Generate all three datasets for all three environments:
  python scripts/generate_offline_data.py \
      --checkpoint artifacts/runs/<run_dir>/checkpoints/latest.pt \
      --config     configs/online/pong_ddqn_dueling_per.yaml

  # Single env, single split, custom size:
  python scripts/generate_offline_data.py \
      --checkpoint artifacts/runs/<run_dir>/checkpoints/latest.pt \
      --config     configs/online/pong_ddqn_dueling_per.yaml \
      --env        ALE/Pong-ram-v5 \
      --splits     random \
      --n-transitions 50000

  # No checkpoint needed for random split:
  python scripts/generate_offline_data.py \
      --env    ALE/Pong-ram-v5 \
      --splits random
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.algos import DQNAgent
from src.envs import make_env
from src.utils import load_checkpoint, set_global_seeds

DEFAULT_SIZES: dict[str, int] = {
    "ALE/Pong-ram-v5":    200_000,
    "ALE/Breakout-ram-v5": 200_000,
    "ALE/Boxing-ram-v5":   300_000,
}

SPLITS: dict[str, float] = {
    "random":     1.00,
    "mixed":      0.30,
    "expert_ish": 0.05,
}

ALL_ENVS = list(DEFAULT_SIZES.keys())


def collect_transitions(
    env_id: str,
    epsilon: float,
    n_transitions: int,
    agent: DQNAgent | None,
    seed: int,
) -> dict[str, np.ndarray]:
    """
    Roll out episodes until n_transitions are collected.
    Returns dict with keys: obs, actions, rewards, next_obs, dones.
    """
    obs_buf      = np.zeros((n_transitions, 128), dtype=np.float32)
    next_obs_buf = np.zeros((n_transitions, 128), dtype=np.float32)
    act_buf      = np.zeros(n_transitions, dtype=np.int64)
    rew_buf      = np.zeros(n_transitions, dtype=np.float32)
    done_buf     = np.zeros(n_transitions, dtype=np.float32)

    env = make_env(env_id, seed=seed, eval_mode=False)
    obs, _ = env.reset()
    collected = 0
    episode = 0

    print(f"  Collecting {n_transitions:,} transitions  epsilon={epsilon:.2f}  env={env_id}")

    while collected < n_transitions:
        # Act
        if agent is not None:
            action = agent.act(obs, epsilon=epsilon)
        else:
            action = env.action_space.sample()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        obs_buf[collected]      = obs
        act_buf[collected]      = action
        rew_buf[collected]      = reward
        next_obs_buf[collected] = next_obs
        done_buf[collected]     = float(done)

        obs = next_obs
        collected += 1

        if done:
            episode += 1
            obs, _ = env.reset()
            if episode % 20 == 0:
                pct = 100.0 * collected / n_transitions
                print(f"    episode {episode:4d}  collected {collected:7,} / {n_transitions:,}  ({pct:.1f}%)")

    env.close()
    print(f"  Done. {collected:,} transitions collected over {episode} episodes.")

    return {
        "obs":      obs_buf,
        "actions":  act_buf,
        "rewards":  rew_buf,
        "next_obs": next_obs_buf,
        "dones":    done_buf,
    }


def env_short(env_id: str) -> str:
    """'ALE/Pong-ram-v5' -> 'pong'"""
    return env_id.split("/")[-1].replace("-ram-v5", "").lower()


def save_dataset(data: dict[str, np.ndarray], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), **data)
    size_mb = out_path.stat().st_size / 1e6
    print(f"  Saved -> {out_path}  ({size_mb:.1f} MB)")


def dataset_path(output_root: Path, env_id: str, split: str) -> Path:
    return output_root / env_short(env_id) / f"{split}.npz"


def load_agent(checkpoint_path: Path, config: dict, env_id: str) -> DQNAgent:
    """Reconstruct agent from checkpoint + config."""
    import torch

    model_cfg  = config.get("model", {})
    train_cfg  = config.get("train", {})
    alg_cfg    = config.get("algorithm", {})

    env = make_env(env_id, seed=0)
    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n
    env.close()

    device = torch.device("cpu")
    agent = DQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden_dims=model_cfg.get("hidden_dims", [256, 256]),
        device=device,
        gamma=float(train_cfg.get("gamma", 0.99)),
        lr=float(train_cfg.get("lr", 2.5e-4)),
        double_dqn=bool(alg_cfg.get("double_dqn", False)),
        dueling=bool(model_cfg.get("dueling", False)),
    )
    ckpt = load_checkpoint(checkpoint_path)
    agent.load_state_dict_from_checkpoint(ckpt["agent"])
    agent.q_net.eval()
    print(f"  Loaded checkpoint: {checkpoint_path}")
    return agent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate offline datasets from a trained online agent."
    )
    p.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to agent checkpoint (.pt). Required for mixed/expert_ish splits.",
    )
    p.add_argument(
        "--config", type=str, default=None,
        help="YAML config used to train the online agent (needed to rebuild agent arch).",
    )
    p.add_argument(
        "--env", type=str, default=None,
        help="Single env id (e.g. ALE/Pong-ram-v5). Defaults to all three envs.",
    )
    p.add_argument(
        "--splits", type=str, nargs="+",
        default=["random", "mixed", "expert_ish"],
        choices=list(SPLITS.keys()),
        help="Which splits to generate.",
    )
    p.add_argument(
        "--n-transitions", type=int, default=None,
        help="Override transition count (default: per-env size from proposal).",
    )
    p.add_argument(
        "--output-root", type=str,
        default=str(ROOT / "artifacts" / "offline_data"),
        help="Root directory for saved datasets.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed.",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Re-generate even if the .npz already exists.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)

    envs = [args.env] if args.env else ALL_ENVS

    # Load config if provided
    config: dict = {}
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}

    # Validate: non-random splits need a checkpoint
    needs_agent = [s for s in args.splits if s != "random"]
    if needs_agent and not args.checkpoint:
        raise ValueError(
            f"Splits {needs_agent} require --checkpoint. "
            "Pass --splits random to generate only the random dataset without a checkpoint."
        )

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None

    for env_id in envs:
        print(f"\n{'='*60}")
        print(f"Environment: {env_id}")
        print(f"{'='*60}")

        n = args.n_transitions or DEFAULT_SIZES.get(env_id, 200_000)

        # Load agent once per env (reuse across splits)
        agent: DQNAgent | None = None
        if checkpoint_path is not None and needs_agent:
            print(f"\nLoading agent for {env_id}...")
            agent = load_agent(checkpoint_path, config, env_id)

        for split in args.splits:
            epsilon = SPLITS[split]
            out_path = dataset_path(output_root, env_id, split)

            if out_path.exists() and not args.overwrite:
                print(f"\n  [skip] {out_path} already exists. Use --overwrite to regenerate.")
                continue

            print(f"\n--- Split: {split}  (epsilon={epsilon}) ---")
            seed = args.seed + hash(env_id + split) % 10_000
            set_global_seeds(seed)

            # Random split never uses the agent
            active_agent = None if split == "random" else agent
            data = collect_transitions(env_id, epsilon, n, active_agent, seed)
            save_dataset(data, out_path)

            # Print quick stats
            r = data["rewards"]
            print(f"  Reward stats: mean={r.mean():.4f}  std={r.std():.4f}  "
                  f"min={r.min():.1f}  max={r.max():.1f}  "
                  f"nonzero={np.count_nonzero(r):,} ({100*np.mean(r!=0):.1f}%)")

    print(f"\nAll done. Datasets saved under: {output_root}")


if __name__ == "__main__":
    main()