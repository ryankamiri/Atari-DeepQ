"""
Checkpoint save/load. T0: model, target, optimizer, global_step, episode_count, config ref.
No replay buffer or RNG state.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import torch
import yaml


def save_checkpoint(
    run_dir: Path,
    agent,
    global_step: int,
    episode_count: int,
    config: dict,
    filename: str = "latest.pt",
):
    """
    Save checkpoint to run_dir/checkpoints/<filename>.
    Also write config snapshot and git hash to run_dir if not already present.
    """
    ckpt_dir = Path(run_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / filename

    ckpt = {
        "agent": agent.state_dict_for_checkpoint(),
        "global_step": global_step,
        "episode_count": episode_count,
        "config": config,
    }
    torch.save(ckpt, path)


def load_checkpoint(path: str | Path):
    """
    Load checkpoint dict. Keys: agent (state_dict for agent), global_step, episode_count, config.
    """
    return torch.load(path, map_location="cpu", weights_only=False)


def get_git_hash() -> str:
    """Return current git commit hash or 'unknown'."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=Path(__file__).resolve().parent.parent.parent,
        )
        if out.returncode == 0 and out.stdout:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def ensure_run_metadata(run_dir: Path, config: dict):
    """Write config.yaml and git_commit.txt to run_dir if not present."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    git_path = run_dir / "git_commit.txt"
    if not git_path.exists():
        git_path.write_text(get_git_hash() + "\n")
