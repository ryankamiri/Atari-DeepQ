# T1.3 — Dueling Architecture Results

This note records the actual T1.3 experiment evidence currently available in `Atari-DeepQ`. It is intended for the paper workflow and only makes claims supported by completed artifacts on disk.

## What changed in T1.3

- Added a model-level toggle: `model.dueling: true/false`
- Added a dueling MLP with:
  - shared trunk
  - value head `V(s)`
  - advantage head `A(s, a)`
  - aggregation `Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a))`
- Kept the trainer, replay buffer, checkpointing, CSV/TensorBoard logging, and Double DQN logic unchanged

## Validation summary

- Test suite:
  - command: `python3 -m pytest -q`
  - result: `24 passed`
- Coverage now includes:
  - `combine_dueling_streams`
  - dueling network forward shape
  - empty-`hidden_dims` dueling construction
  - agent class selection for dueling on/off
  - `update()` across all four `(double_dqn x dueling)` combinations
  - subprocess integration run with `double_dqn=true` and `dueling=true`

## Executed runs

### 1. Dueling smoke run

- Config: `configs/smoke/pong_tiny_dueling.yaml`
- Run directory: `artifacts/runs/20260322_203141_pong_tiny_dueling_seed42`
- Outcome: completed cleanly, wrote eval JSONs, `metrics.csv`, and checkpoints

Smoke eval summary:

| Run | First eval | Best eval | Final eval |
|---|---:|---:|---:|
| `pong_tiny_dueling` | `-21.0` @ 2500 | `-20.0` @ 5000 | `-20.0` @ 5000 |

### 2. Bounded 200k DDQN vs DDQN+dueling comparison

These two runs use the same current tune budget:

- `200000` total steps
- `lr=1e-4`
- `learning_starts=10000`
- `decay_steps=250000`
- `5` eval episodes every `10000` steps

| Variant | Run directory | First eval | Best eval | Final eval | Stability |
|---|---|---:|---:|---:|---|
| DDQN | `artifacts/runs/20260322_201922_pong_ddqn_tune_seed42` | `-21.0` @ 10000 | `-20.2` @ 140000 | `-20.6` @ 200000 | Clean run |
| DDQN + dueling | `artifacts/runs/20260322_203635_pong_ddqn_dueling_tune_seed42` | `-21.0` @ 10000 | `-18.8` @ 160000 | `-20.6` @ 200000 | Clean run |

For the completed DDQN+dueling run:

- Final eval file: `artifacts/runs/20260322_203635_pong_ddqn_dueling_tune_seed42/eval/eval_step_200000.json`
- Checkpoints present:
  - `step_50000.pt`
  - `step_100000.pt`
  - `step_150000.pt`
  - `step_200000.pt`
  - `latest.pt`
- No NaNs or crashes were observed in the logged loss/Q/TD metrics

## What we can honestly claim

- The dueling architecture is implemented correctly and toggles cleanly through `model.dueling`.
- Dueling composes with `algorithm.double_dqn` without requiring a separate trainer path.
- In the completed one-seed bounded comparison, DDQN+dueling was **stable** and reached a better best checkpoint than plain DDQN:
  - DDQN best checkpoint: `-20.2`
  - DDQN + dueling best checkpoint: `-18.8`
- The final checkpoint was the same for both variants:
  - DDQN final: `-20.6`
  - DDQN + dueling final: `-20.6`

That means T1.3 supports a modest, accurate takeaway:

- dueling did not break training
- dueling produced a stronger mid-training checkpoint in this one-seed run
- dueling did not yet produce a better final bounded-run result under the current budget

## Paper-ready interpretation

Use language like:

> The dueling architecture integrated cleanly into our existing DQN/DDQN training pipeline and remained stable throughout a bounded 200k-step Pong RAM run. In a one-seed comparison against matched Double DQN, the dueling variant reached a stronger best checkpoint (`-18.8` vs `-20.2`) but ended with the same final evaluation return (`-20.6`). At this stage, the main result of T1.3 is that dueling is a clean and stable architectural extension, with suggestive but not conclusive performance benefit under our limited evaluation budget.

This is a good setup for T1.4:

- PER can now be added on top of the existing DDQN + dueling path
- the real online headline comparisons should wait until the stacked agent is complete
