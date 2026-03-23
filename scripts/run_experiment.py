#!/usr/bin/env python3
"""
Single entrypoint for bootstrap training runs.
Usage: python scripts/run_experiment.py --config <path> [--resume <ckpt>]
"""
from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

# Project root on path so "src" package is importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs import make_env
from src.algos import DQNAgent
from src.algos.dqn import linear_schedule
from src.replay import ReplayBuffer
from src.utils import (
    ensure_run_metadata,
    evaluate_policy,
    load_checkpoint,
    save_checkpoint,
    RunLogger,
    set_global_seeds,
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_run_dir(config: dict | None, resume_path: Path | None = None) -> Path:
    """Run directory: from resume path or new timestamped name."""
    if resume_path is not None:
        # run_dir/checkpoints/latest.pt -> run_dir
        return resume_path.resolve().parent.parent
    out = Path(config["experiment"]["output_root"])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = config["experiment"]["name"]
    seed = config["experiment"]["seed"]
    return out / "runs" / f"{ts}_{name}_seed{seed}"


def _base_train_row(
    global_step: int,
    episode_count: int,
    episode_return: float,
    episode_length: int,
    epsilon: float,
    last_train: dict[str, float],
    buffer_size: int,
    return_ema: float | None,
) -> dict:
    row = {
        "global_step": global_step,
        "episode": episode_count,
        "train_return": episode_return,
        "train_episode_length": episode_length,
        "epsilon": epsilon,
        "loss": last_train["loss"],
        "q_mean": last_train["q_mean"],
        "target_mean": last_train["target_mean"],
        "td_abs_mean": last_train["td_abs_mean"],
        "buffer_size": buffer_size,
        "episode_return_moving_avg": return_ema if return_ema is not None else "",
    }
    return row


def run_training(config: dict, run_dir: Path, resume_path: Path | None = None):
    exp = config["experiment"]
    env_cfg = config["env"]
    model_cfg = config["model"]
    train_cfg = config["train"]
    explore_cfg = config["exploration"]
    eval_cfg = config["eval"]
    log_cfg = config.get("log", {})

    device = _device(exp.get("device"))
    seed = exp["seed"]
    env_id = env_cfg["id"]

    set_global_seeds(seed)
    ensure_run_metadata(run_dir, config)

    env = make_env(env_id, seed=seed, eval_mode=False)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    env.close()

    hidden_dims = model_cfg.get("hidden_dims", [256, 256])
    replay_capacity = int(train_cfg["replay_capacity"])
    batch_size = int(train_cfg["batch_size"])
    learning_starts = int(train_cfg["learning_starts"])
    train_freq = int(train_cfg["train_freq"])
    total_steps = int(train_cfg["total_steps"])
    gamma = float(train_cfg["gamma"])
    lr = float(train_cfg["lr"])
    target_update_interval = int(train_cfg["target_update_interval"])
    checkpoint_interval = int(train_cfg["checkpoint_interval"])
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 10.0))
    eval_interval = int(eval_cfg["eval_interval"])
    n_episodes_eval = int(eval_cfg["n_episodes"])
    epsilon_eval = float(eval_cfg["epsilon_eval"])
    epsilon_start = float(explore_cfg["epsilon_start"])
    epsilon_end = float(explore_cfg["epsilon_end"])
    decay_steps = int(explore_cfg["decay_steps"])
    csv_flush = log_cfg.get("csv_flush_interval", 1)
    ema_alpha = float(log_cfg.get("episode_return_ema_alpha", 0.1))

    buffer = ReplayBuffer(replay_capacity, obs_shape=(obs_dim,))
    agent = DQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden_dims=hidden_dims,
        device=device,
        gamma=gamma,
        lr=lr,
        target_update_interval=target_update_interval,
        grad_clip_norm=grad_clip_norm,
    )

    global_step = 0
    episode_count = 0
    resume = resume_path is not None
    if resume:
        ckpt = load_checkpoint(resume_path)
        agent.load_state_dict_from_checkpoint(ckpt["agent"])
        global_step = ckpt["global_step"]
        episode_count = ckpt["episode_count"]

    logger = RunLogger(run_dir, csv_flush_interval=csv_flush, resume=resume)
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(env_id, seed=seed, eval_mode=False)
    obs, _ = env.reset()
    episode_return = 0.0
    episode_length = 0
    last_train = {"loss": 0.0, "q_mean": 0.0, "target_mean": 0.0, "td_abs_mean": 0.0}
    return_ema: float | None = None

    try:
        while global_step < total_steps:
            epsilon = linear_schedule(global_step, epsilon_start, epsilon_end, decay_steps)
            action = agent.act(obs, epsilon=epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.add(obs, action, reward, next_obs, done)
            episode_return += reward
            episode_length += 1
            obs = next_obs

            buf_len = len(buffer)

            if done:
                episode_count += 1
                if return_ema is None:
                    return_ema = episode_return
                else:
                    return_ema = ema_alpha * episode_return + (1.0 - ema_alpha) * return_ema
                row = _base_train_row(
                    global_step,
                    episode_count,
                    episode_return,
                    episode_length,
                    epsilon,
                    last_train,
                    buf_len,
                    return_ema,
                )
                logger.log_metrics(row, global_step)
                obs, _ = env.reset()
                episode_return = 0.0
                episode_length = 0

            global_step += 1

            if (
                global_step >= learning_starts
                and global_step % train_freq == 0
                and len(buffer) >= batch_size
            ):
                batch = buffer.sample(batch_size)
                last_train = agent.update(*batch)
                for k, v in last_train.items():
                    if not math.isfinite(v):
                        raise RuntimeError(f"Non-finite training metric {k}={v} at step {global_step}")
                row = _base_train_row(
                    global_step,
                    episode_count,
                    episode_return,
                    episode_length,
                    epsilon,
                    last_train,
                    len(buffer),
                    return_ema,
                )
                logger.log_metrics(row, global_step)

            if global_step % target_update_interval == 0 and global_step >= learning_starts:
                agent.sync_target()

            if eval_interval and global_step % eval_interval == 0 and global_step >= learning_starts:
                summary = evaluate_policy(
                    agent,
                    env_id=env_id,
                    n_episodes=n_episodes_eval,
                    epsilon_eval=epsilon_eval,
                    seed=seed + 1,
                    output_path=str(eval_dir / f"eval_step_{global_step}.json"),
                )
                row = _base_train_row(
                    global_step,
                    episode_count,
                    episode_return,
                    episode_length,
                    epsilon,
                    last_train,
                    len(buffer),
                    return_ema,
                )
                row["eval_mean_return"] = summary["mean_return"]
                row["eval_std_return"] = summary["std_return"]
                logger.log_metrics(row, global_step)

            if checkpoint_interval and global_step % checkpoint_interval == 0:
                save_checkpoint(
                    run_dir,
                    agent,
                    global_step,
                    episode_count,
                    config,
                    filename=f"step_{global_step}.pt",
                )

            if global_step >= total_steps:
                break
    finally:
        env.close()

    save_checkpoint(run_dir, agent, global_step, episode_count, config, filename="latest.pt")
    logger.close()
    print(f"Run complete. global_step={global_step} run_dir={run_dir}")


def _device(device_str: str | None) -> torch.device:
    if device_str is None or device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def main():
    parser = argparse.ArgumentParser(description="Run DQN training from YAML config.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint (e.g. .../checkpoints/latest.pt).")
    args = parser.parse_args()

    resume_path = Path(args.resume) if args.resume else None
    if resume_path is not None:
        run_dir = get_run_dir(None, resume_path)
        config_path = run_dir / "config.yaml"
        if config_path.exists():
            config = load_config(str(config_path))
        else:
            config = load_config(args.config)
    else:
        config = load_config(args.config)
        run_dir = get_run_dir(config, None)
    run_training(config, run_dir, resume_path)


if __name__ == "__main__":
    main()
