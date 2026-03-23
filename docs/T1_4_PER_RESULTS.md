# T1.4 — Prioritized Experience Replay Results

This note records the actual T1.4 experiment evidence currently available in `Atari-DeepQ`. It is intended for the paper workflow and only makes claims supported by completed artifacts on disk.

Post-review note: the runs summarized here were produced after correcting the PER importance-sampling normalization to use the buffer-wide minimum sampling probability rather than the largest weight inside each sampled batch.

## What changed in T1.4

- Added proportional prioritized replay with:
  - priorities
  - importance-sampling weights
  - priority updates from absolute TD error
- Kept the single trainer path intact:
  - `algorithm.double_dqn`
  - `model.dueling`
  - `replay.prioritized`
- Added replay diagnostics to `metrics.csv` and TensorBoard:
  - `beta`
  - `is_weight_mean`

## Validation summary

- Test suite:
  - command: `python3 -m pytest -q`
  - result: `34 passed`
- Coverage now includes:
  - uniform replay unit-weight behavior
  - PER sampling shapes and finite normalized weights
  - PER sampling bias after priority updates
  - global IS-weight normalization behavior
  - weighted `DQNAgent.update(...)`
  - subprocess integration run with DDQN + dueling + PER

## Executed runs

### 1. PER smoke run

- Config: `configs/smoke/pong_tiny_ddqn_dueling_per.yaml`
- Run directory: `artifacts/runs/20260322_205806_pong_tiny_ddqn_dueling_per_seed42`
- Outcome: completed cleanly, wrote eval JSONs, `metrics.csv`, TensorBoard logs, and checkpoints

Smoke eval summary:

| Run | First eval | Best eval | Final eval |
|---|---:|---:|---:|
| `pong_tiny_ddqn_dueling_per` | `-20.5` @ 2500 | `-20.0` @ 5000 | `-20.0` @ 5000 |

### 2. Bounded 200k DDQN + dueling vs DDQN + dueling + PER comparison

These two runs use the same current bounded tune budget:

- `200000` total steps
- `lr=1e-4`
- `learning_starts=10000`
- `decay_steps=250000`
- `5` eval episodes every `10000` steps

| Variant | Run directory | First eval | Best eval | Final eval | Stability |
|---|---|---:|---:|---:|---|
| DDQN + dueling | `artifacts/runs/20260322_203635_pong_ddqn_dueling_tune_seed42` | `-21.0` @ 10000 | `-18.8` @ 160000 | `-20.6` @ 200000 | Clean run |
| DDQN + dueling + PER | `artifacts/runs/20260322_205806_pong_ddqn_dueling_per_tune_seed42` | `-20.2` @ 10000 | `-12.4` @ 150000 | `-16.6` @ 200000 | Clean run |

For the completed PER run:

- Final eval file: `artifacts/runs/20260322_205806_pong_ddqn_dueling_per_tune_seed42/eval/eval_step_200000.json`
- Checkpoints present:
  - `step_50000.pt`
  - `step_100000.pt`
  - `step_150000.pt`
  - `step_200000.pt`
  - `latest.pt`
- No NaNs or crashes were observed in the logged loss/Q/TD/PER metrics

## What we can honestly claim

- Prioritized replay is integrated correctly into the existing DDQN + dueling path and trains stably on Pong RAM.
- The PER run completed at the full `200k` step budget with valid eval JSONs and checkpoints.
- In this one-seed bounded comparison, PER improved both the best checkpoint and the final checkpoint relative to the matched non-PER run:
  - best checkpoint: `-12.4` vs `-18.8`
  - final checkpoint: `-16.6` vs `-20.6`

That makes T1.4 a stronger result than T1.2 or T1.3 individually:

- the main outcome is still stable integration
- but unlike the earlier bounded comparisons, this run also shows a clear positive performance signal under the current budget

## Paper-ready interpretation

Use language like:

> Prioritized experience replay integrated cleanly into our Atari RAM DQN pipeline and remained stable throughout a bounded 200k-step Pong RAM run. In a one-seed comparison against the matched Double DQN + dueling configuration, adding PER improved both the strongest intermediate checkpoint (`-12.4` vs `-18.8`) and the final evaluation return (`-16.6` vs `-20.6`). While still limited to a bounded one-seed validation run, this is our clearest positive online improvement so far and supports using DDQN + dueling + PER as the main online agent for the full experiment matrix.
