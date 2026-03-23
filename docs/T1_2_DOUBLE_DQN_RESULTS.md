# T1.2 — Double DQN (toggle) — experiment note

## Config contract

- `algorithm.double_dqn: true` enables Double DQN bootstrap:
  - action selection: `a* = argmax Q_online(s')`
  - target evaluation: `Q_target(s', a*)`
- Missing `algorithm` or `algorithm.double_dqn: false` preserves vanilla DQN:
  - `max_a Q_target(s', a)`

## Tracked configs

| Purpose | Path |
|--------|------|
| Smoke (DDQN) | `configs/smoke/pong_tiny_ddqn.yaml` |
| Bounded Pong validation (DDQN) | `configs/online/pong_ddqn_tune.yaml` |
| Matched vanilla comparison | `configs/online/pong_dqn_tune.yaml` |

## Executed validation

### 1. Test suite

- Command: `python3 -m pytest -q`
- Result: `17 passed`
- Coverage includes:
  - deterministic target-math test for vanilla vs Double DQN
  - parametrized update test for both modes
  - tiny subprocess integration run with `algorithm.double_dqn: true`

### 2. Smoke run

- Config: `configs/smoke/pong_tiny_ddqn.yaml`
- Run directory: `artifacts/runs/20260322_200318_pong_tiny_ddqn_seed42`
- Outcome: completed cleanly and produced `metrics.csv`, eval JSONs, and checkpoints.

### 3. Bounded Pong comparison

| Variant | Run directory | First eval | Best eval | Final eval | Outcome |
|---|---|---:|---:|---:|---|
| Vanilla DQN | `artifacts/runs/20260322_202158_pong_dqn_tune_seed42` | `-21.0` @ 10k | `-20.4` @ 70k | `-20.4` @ 200k | Clean run |
| Double DQN | `artifacts/runs/20260322_201922_pong_ddqn_tune_seed42` | `-21.0` @ 10k | `-20.2` @ 140k | `-20.6` @ 200k | Clean run |

Both runs used the same current tune budget:

- `200000` total steps
- `lr=1e-4`
- `learning_starts=10000`
- `decay_steps=250000`
- `5` eval episodes every `10000` steps

## What we can honestly claim

- The Double DQN implementation is **correctly integrated and stable**:
  - same runner
  - same artifact layout
  - same checkpointing behavior
  - no NaNs or crashes in the bounded run
- In this one-seed, 200k-step comparison, Double DQN was **comparable** to vanilla DQN, not clearly better:
  - DDQN achieved the better **best checkpoint** (`-20.2` vs `-20.4`)
  - vanilla DQN achieved the better **final checkpoint** (`-20.4` vs `-20.6`)
- With only one seed and `5` eval episodes per checkpoint, the difference is too small to treat as a meaningful performance regression or win.

## Paper-ready interpretation

Use language like:

> Double DQN integrated cleanly into the existing training pipeline and completed the same bounded Pong RAM run without instability. In a one-seed 200k-step comparison against the matched vanilla DQN config, Double DQN produced a slightly better best checkpoint (`-20.2` vs `-20.4`) but a slightly worse final checkpoint (`-20.6` vs `-20.4`). At this stage, the main result of T1.2 is implementation correctness and stability rather than a decisive performance gain.

This is a good bridge into later tickets:

- T1.3 adds dueling architecture
- T1.4 adds prioritized replay
- the full online headline comparison should wait until the stacked agent is complete
