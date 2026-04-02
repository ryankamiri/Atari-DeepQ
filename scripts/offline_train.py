#!/usr/bin/env python3
"""
Offline RL training entrypoint for Atari RAM environments.

Trains one of four offline agents (BC, naive DQN, DQN+BC, CQL)
on a fixed dataset produced by scripts/offline_generate_data.py.

Usage:
    python scripts/train_offline.py --config configs/offline/pong_bc_random.yaml
    python scripts/train_offline.py --config configs/offline/pong_cql_random.yaml --seed 1
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.algos.offline.bc import BCAgent
from src.algos.offline.cql import CQLAgent
from src.algos.offline.dqn_bc import DQNBCAgent
from src.algos.offline.dqn_offline import OfflineDQNAgent
from src.envs import make_env
from src.replay.dataset import OfflineDataset
from src.utils import evaluate_policy, set_global_seeds


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------
def build_agent(config: dict, obs_dim: int, n_actions: int, device: torch.device):
    method     = config["offline"]["method"]
    model_cfg  = config.get("model", {})
    train_cfg  = config["train"]
    alg_cfg    = config.get("algorithm", {})
    hidden     = model_cfg.get("hidden_dims", [256, 256])
    lr         = float(train_cfg.get("lr", 1e-3))
    dueling    = bool(model_cfg.get("dueling", True))
    double_dqn = bool(alg_cfg.get("double_dqn", True))
    grad_clip  = float(train_cfg.get("grad_clip_norm", 10.0))
    gamma      = float(train_cfg.get("gamma", 0.99))
    target_upd = int(train_cfg.get("target_update_interval", 1000))

    if method == "bc":
        return BCAgent(
            obs_dim=obs_dim, n_actions=n_actions,
            hidden_dims=hidden, device=device, lr=lr,
        )
    elif method == "dqn_offline":
        return OfflineDQNAgent(
            obs_dim=obs_dim, n_actions=n_actions,
            hidden_dims=hidden, device=device,
            gamma=gamma, lr=lr,
            target_update_interval=target_upd,
            grad_clip_norm=grad_clip,
            double_dqn=double_dqn, dueling=dueling,
        )
    elif method == "dqn_bc":
        lam = float(config["offline"].get("lam", 1.0))
        return DQNBCAgent(
            obs_dim=obs_dim, n_actions=n_actions,
            hidden_dims=hidden, device=device,
            gamma=gamma, lr=lr, lam=lam,
            target_update_interval=target_upd,
            grad_clip_norm=grad_clip,
            double_dqn=double_dqn, dueling=dueling,
        )
    elif method == "cql":
        cql_alpha = float(config["offline"].get("cql_alpha", 1.0))
        return CQLAgent(
            obs_dim=obs_dim, n_actions=n_actions,
            hidden_dims=hidden, device=device,
            gamma=gamma, lr=lr, cql_alpha=cql_alpha,
            target_update_interval=target_upd,
            grad_clip_norm=grad_clip,
            double_dqn=double_dqn, dueling=dueling,
        )
    else:
        raise ValueError(f"Unknown offline method: {method!r}. "
                         f"Choose from: bc, dqn_offline, dqn_bc, cql")


# ---------------------------------------------------------------------------
# One update step (dispatches by method)
# ---------------------------------------------------------------------------
def update_agent(agent, batch, method: str) -> dict[str, float]:
    if method == "bc":
        result = agent.update(batch.obs, batch.actions)
        return result.__dict__
    else:
        result = agent.update(
            batch.obs, batch.actions, batch.rewards,
            batch.next_obs, batch.dones,
        )
        return result.metrics


# ---------------------------------------------------------------------------
# Run directory
# ---------------------------------------------------------------------------
def get_run_dir(config: dict) -> Path:
    out   = Path(config["experiment"]["output_root"])
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    name  = config["experiment"]["name"]
    seed  = config["experiment"]["seed"]
    return out / "offline_runs" / f"{ts}_{name}_seed{seed}"


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(run_dir: Path, agent, grad_step: int, config: dict,
                    filename: str = "latest.pt") -> None:
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "agent":     agent.state_dict_for_checkpoint(),
        "grad_step": grad_step,
        "config":    config,
    }, ckpt_dir / filename)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def run_training(config: dict, run_dir: Path) -> None:
    exp_cfg     = config["experiment"]
    train_cfg   = config["train"]
    eval_cfg    = config["eval"]
    offline_cfg = config["offline"]

    seed        = int(exp_cfg["seed"])
    env_id      = config["env"]["id"]
    method      = offline_cfg["method"]
    dataset_path = Path(offline_cfg["dataset_path"])
    total_steps  = int(train_cfg["total_steps"])
    batch_size   = int(train_cfg["batch_size"])
    eval_interval    = int(eval_cfg["eval_interval"])
    n_episodes_eval  = int(eval_cfg["n_episodes"])
    epsilon_eval     = float(eval_cfg["epsilon_eval"])
    ckpt_interval    = int(train_cfg.get("checkpoint_interval", total_steps))
    log_interval     = int(train_cfg.get("log_interval", 500))

    set_global_seeds(seed)

    # Device
    device_str = exp_cfg.get("device", "auto")
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    # Dataset
    dataset = OfflineDataset(dataset_path)

    # Agent
    env_tmp   = make_env(env_id, seed=seed)
    obs_dim   = env_tmp.observation_space.shape[0]
    n_actions = env_tmp.action_space.n
    env_tmp.close()

    agent = build_agent(config, obs_dim, n_actions, device)

    # Logging
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.dump(config, sort_keys=False))
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "metrics.csv"
    csv_file  = open(metrics_path, "w", newline="")
    csv_writer = None  # initialised on first row

    print(f"\nOffline training: method={method}  env={env_id}  "
          f"dataset={dataset_path.name}  steps={total_steps:,}  seed={seed}")
    print(f"Run dir: {run_dir}\n")

    # Training loop
    for grad_step in range(1, total_steps + 1):
        batch  = dataset.sample(batch_size)
        metrics = update_agent(agent, batch, method)

        # Sanity check for non-finite values
        for k, v in metrics.items():
            if isinstance(v, float) and not math.isfinite(v):
                raise RuntimeError(f"Non-finite metric {k}={v} at step {grad_step}")

        # Periodic console log
        if grad_step % log_interval == 0 or grad_step == 1:
            parts = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"  step {grad_step:6d}/{total_steps}  {parts}")

        # Evaluation
        if grad_step % eval_interval == 0:
            summary = evaluate_policy(
                agent,
                env_id=env_id,
                n_episodes=n_episodes_eval,
                epsilon_eval=epsilon_eval,
                seed=seed + 1,
                output_path=str(eval_dir / f"eval_step_{grad_step}.json"),
            )
            metrics["eval_mean_return"] = summary["mean_return"]
            metrics["eval_std_return"]  = summary["std_return"]
            print(f"  [eval] step={grad_step}  "
                  f"mean_return={summary['mean_return']:.2f}  "
                  f"std={summary['std_return']:.2f}")

        # CSV logging
        metrics["grad_step"] = grad_step
        if csv_writer is None:
            fieldnames = ["grad_step"] + [k for k in metrics if k != "grad_step"]
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames,
                                        extrasaction="ignore")
            csv_writer.writeheader()
        csv_writer.writerow(metrics)
        csv_file.flush()

        # Checkpoint
        if grad_step % ckpt_interval == 0 or grad_step == total_steps:
            save_checkpoint(run_dir, agent, grad_step, config,
                            filename=f"step_{grad_step}.pt")

    save_checkpoint(run_dir, agent, total_steps, config, filename="latest.pt")
    csv_file.close()
    print(f"\nTraining complete. grad_step={total_steps}  run_dir={run_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Offline RL training.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to offline YAML config.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override experiment.seed.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = copy.deepcopy(yaml.safe_load(f))

    if args.seed is not None:
        config.setdefault("experiment", {})
        config["experiment"]["seed"] = int(args.seed)

    run_dir = get_run_dir(config)
    run_training(config, run_dir)


if __name__ == "__main__":
    main()