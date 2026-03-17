# Atari RAM DQN

Reproducible online/offline RL on Atari RAM environments (Gymnasium ALE). This repo implements vanilla DQN and will extend to Rainbow-lite (Double DQN + Dueling + PER) and offline methods (BC, offline DQN, CQL).

## Setup

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Install Atari ROMs (required for ALE):

   ```bash
   pip install autorom
   autorom --accept-license
   ```

   Or with `ale-py` only: download ROMs and set `ALE_ROM_PATH` if needed.

## Testing

From the project root (with dependencies installed):

```bash
PYTHONPATH=. python -m pytest tests/test_smoke.py -v
```

Requires ALE ROMs (see setup). Tests cover `make_env` (obs shape/dtype/range, reward clipping) and `evaluate_policy` (return dict and JSON output).

## Smoke run (Ticket 0)

After setup, run a short Pong RAM training job (~5 minutes on CPU):

```bash
python scripts/run_experiment.py --config configs/smoke/pong_tiny.yaml
```

This creates a run directory under `artifacts/runs/` with TensorBoard logs, metrics CSV, config snapshot, git hash, eval JSON summaries, and checkpoints.

## Run artifact layout

Each run directory is named:

`artifacts/runs/<timestamp>_<experiment_name>_seed<seed>/`

Contents:

- `config.yaml` — snapshot of the config used for the run
- `git_commit.txt` — git commit hash at launch
- `metrics.csv` — training metrics (global_step, episode, train_return, loss, epsilon, eval_mean_return, etc.)
- `eval/` — JSON summaries per evaluation (e.g. `eval_step_50000.json`)
- `checkpoints/` — `latest.pt` and optional step checkpoints (e.g. `step_50000.pt`)
- `tensorboard/` — TensorBoard event files

View logs:

```bash
tensorboard --logdir artifacts/runs/<run_dir>/tensorboard
```

## Resume a run

To continue training from the latest checkpoint:

```bash
python scripts/run_experiment.py --config configs/smoke/pong_tiny.yaml --resume artifacts/runs/<run_dir>/checkpoints/latest.pt
```

Resume restores model, target network, optimizer, training step, episode count, and epsilon scheduler state. The replay buffer is not restored (practical resume only).

## First real baseline

For a full Pong DQN baseline run (2M steps, 3 seeds in separate runs):

```bash
python scripts/run_experiment.py --config configs/online/pong_dqn.yaml
```

Use a different `experiment.seed` or run name per seed so each run has its own directory.

## Environments

- `ALE/Pong-ram-v5`
- `ALE/Breakout-ram-v5`
- `ALE/Boxing-ram-v5`

Observations are 128-dim RAM, normalized to float32 in [0, 1]. Rewards are clipped to [-1, 1]. No frame stacking or reward shaping.
