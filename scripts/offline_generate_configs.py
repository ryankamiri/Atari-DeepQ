#!/usr/bin/env python3
"""
Auto-generates all offline training YAML configs.

Produces one config per combination of:
  - 3 environments  (pong, breakout, boxing)
  - 3 dataset splits (random, mixed, expert_ish)
  - 4 methods        (bc, dqn_offline, dqn_bc, cql)
  - lambda sweep for dqn_bc: {0.1, 1.0, 3.0}
  - alpha  sweep for cql:    {0.5, 1.0, 3.0}

Total: 3 * 3 * 2  (bc + dqn_offline, no sweep)
     + 3 * 3 * 3  (dqn_bc lambda sweep)
     + 3 * 3 * 3  (cql alpha sweep)
     = 18 + 27 + 27 = 72 configs

Usage:
    python scripts/generate_offline_configs.py
    python scripts/generate_offline_configs.py --out-dir configs/offline
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
ENVS = {
    "pong":     "ALE/Pong-ram-v5",
    "breakout": "ALE/Breakout-ram-v5",
    "boxing":   "ALE/Boxing-ram-v5",
}

SPLITS = ["random", "mixed", "expert_ish"]

DQN_BC_LAMBDAS  = [0.1, 1.0, 3.0]
CQL_ALPHAS      = [0.5, 1.0, 3.0]

# ---------------------------------------------------------------------------
# Shared train / eval blocks
# ---------------------------------------------------------------------------
def base_train(env_short: str) -> dict:
    return {
        "total_steps":          50_000,
        "batch_size":           256,
        "lr":                   1e-3,
        "gamma":                0.99,
        "grad_clip_norm":       10.0,
        "target_update_interval": 1000,
        "checkpoint_interval":  50_000,
        "log_interval":         2_000,
    }

EVAL_BLOCK = {
    "eval_interval": 10_000,
    "n_episodes":    20,
    "epsilon_eval":  0.05,
}

# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------
def _base(env_short: str, env_id: str, split: str,
          method: str, name: str, dueling: bool, double_dqn: bool) -> dict:
    return {
        "experiment": {
            "name":        name,
            "seed":        0,
            "device":      "auto",
            "output_root": "artifacts",
        },
        "env":   {"id": env_id},
        "model": {"hidden_dims": [256, 256], "dueling": dueling},
        "algorithm": {"double_dqn": double_dqn},
        "offline": {
            "method":       method,
            "dataset_path": f"artifacts/offline_data/{env_short}/{split}.npz",
        },
        "train": base_train(env_short),
        "eval":  EVAL_BLOCK,
    }


def make_bc(env_short: str, env_id: str, split: str) -> tuple[str, dict]:
    name = f"{env_short}_bc_{split}"
    cfg  = _base(env_short, env_id, split, "bc", name,
                 dueling=False, double_dqn=False)
    return name, cfg


def make_dqn_offline(env_short: str, env_id: str, split: str) -> tuple[str, dict]:
    name = f"{env_short}_dqn_offline_{split}"
    cfg  = _base(env_short, env_id, split, "dqn_offline", name,
                 dueling=True, double_dqn=True)
    return name, cfg


def make_dqn_bc(env_short: str, env_id: str, split: str,
                lam: float) -> tuple[str, dict]:
    lam_str = str(lam).replace(".", "p")          # 1.0 -> "1p0"
    name    = f"{env_short}_dqn_bc_{split}_lam{lam_str}"
    cfg     = _base(env_short, env_id, split, "dqn_bc", name,
                    dueling=True, double_dqn=True)
    cfg["offline"]["lam"] = lam
    return name, cfg


def make_cql(env_short: str, env_id: str, split: str,
             alpha: float) -> tuple[str, dict]:
    alpha_str = str(alpha).replace(".", "p")
    name      = f"{env_short}_cql_{split}_alpha{alpha_str}"
    cfg       = _base(env_short, env_id, split, "cql", name,
                      dueling=True, double_dqn=True)
    cfg["offline"]["cql_alpha"] = alpha
    return name, cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="configs/offline")
    args   = parser.parse_args()
    out    = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    configs: list[tuple[str, dict]] = []

    for env_short, env_id in ENVS.items():
        for split in SPLITS:
            configs.append(make_bc(env_short, env_id, split))
            configs.append(make_dqn_offline(env_short, env_id, split))
            for lam in DQN_BC_LAMBDAS:
                configs.append(make_dqn_bc(env_short, env_id, split, lam))
            for alpha in CQL_ALPHAS:
                configs.append(make_cql(env_short, env_id, split, alpha))

    for name, cfg in configs:
        path = out / f"{name}.yaml"
        with open(path, "w") as f:
            yaml.dump(cfg, f, sort_keys=False, default_flow_style=False)

    print(f"Generated {len(configs)} configs in {out}/")
    # Print summary table
    from collections import Counter
    methods = Counter(cfg["offline"]["method"] for _, cfg in configs)
    print("\nBreakdown by method:")
    for method, count in sorted(methods.items()):
        print(f"  {method:15s}: {count} configs")


if __name__ == "__main__":
    main()