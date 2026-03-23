# T1.1 Pong Vanilla DQN Results

This note stores the actual T1.1 experiment evidence currently available in `Atari-DeepQ`. It is meant to be paper-facing: concise, factual, and tied to concrete run directories.

## Experiment setup

- Environment: `ALE/Pong-ram-v5`
- Observation pipeline: 128-byte RAM normalized to `float32` in `[0, 1]`
- Reward processing: clipped to `[-1, 1]`
- Evaluation policy: epsilon-greedy with `epsilon_eval = 0.05`
- Tuning eval budget: `5` episodes per checkpoint
- Shared T1.1 trainer changes:
  - Huber loss
  - gradient clipping (`10.0`)
  - structured DQN diagnostics (`loss`, `q_mean`, `target_mean`, `td_abs_mean`)
  - global seeding for Python, NumPy, and Torch

## Executed runs

| Run directory | Steps | Key hyperparameters | First eval | Best eval | Final eval | Takeaway |
|---|---:|---|---:|---:|---:|---|
| `artifacts/runs/20260322_194459_pong_dqn_tune_seed42` | 200k | `lr=2.5e-4`, `learning_starts=20000`, `decay_steps=500000` | `-21.0` @ 20k | `-20.8` @ 60k | `-21.0` @ 200k | Stable run, but no lasting improvement |
| `artifacts/runs/20260322_194753_pong_dqn_tune_ls10000_lr1e4_decay250k_quick_seed42` | 100k | `lr=1e-4`, `learning_starts=10000`, `decay_steps=250000` | `-21.0` @ 10k | `-20.4` @ 70k | `-21.0` @ 100k | Best interim result seen so far, but not stable by the end |
| `artifacts/runs/20260322_192922_pong_dqn_tune_quick_seed42` | 80k | `lr=2.5e-4`, `learning_starts=20000`, `decay_steps=500000` | `-20.8` @ 20k | `-20.8` @ 20k | `-21.0` @ 80k | Quick precursor run; also regressed by the end |

## What we can honestly claim right now

- The T1.1 implementation is **code-complete and stable**:
  - `python3 -m pytest -q` passes
  - the training runner completes end to end
  - checkpoints, CSV logs, TensorBoard logs, and eval JSON files are produced consistently
- Vanilla DQN on Pong RAM is **not yet meeting the intended T1.1 success criterion** under the runs executed so far.
- The best interim eval observed in the current artifact set is `-20.4` at `70k` steps.
- None of the executed runs finished above `-21.0`, so we do **not** have evidence yet for a “frozen winner” config.

## Paper-ready interpretation

For the writeup, the safest interpretation is:

> In our bounded early Pong RAM experiments, vanilla DQN was stable to train but showed only weak and inconsistent improvement. The best interim mean evaluation return we observed was `-20.4` at `70k` steps, but the runs we executed regressed to `-21.0` by their final checkpoints. This makes vanilla DQN a reasonable baseline implementation, but not yet a strong-performing online agent under our current tuning budget.

This supports the narrative that:

- plain DQN is a necessary baseline implementation
- Pong RAM is nontrivial even before moving to Breakout and Boxing
- the stronger online stack in later tickets (Double DQN + dueling + PER) is justified

## Recommended next online experiment

If Ryan wants one next run before starting T1.2, use:

- config: `configs/online/pong_dqn.yaml`
- current provisional candidate:
  - `lr=1e-4`
  - `learning_starts=10000`
  - `decay_steps=250000`

Reason: among the runs currently on disk, this combination produced the strongest interim eval signal, even though it has not yet shown stable end-of-run improvement.
