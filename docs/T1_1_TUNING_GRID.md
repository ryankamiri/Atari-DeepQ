# T1.1 bounded Pong vanilla DQN tuning grid

This file describes the **planned** bounded tuning grid for T1.1. The actual executed runs and outcomes live in `docs/T1_1_PONG_BASELINE_RESULTS.md`.

Run **one seed (`42`)** with `configs/online/pong_dqn_tune.yaml` as the template. Fix all hyperparameters except these three:

| Run | `train.lr` | `train.learning_starts` | `exploration.decay_steps` |
|-----|------------|-------------------------|---------------------------|
| 1 | `1e-4` | `10000` | `250000` |
| 2 | `1e-4` | `10000` | `500000` |
| 3 | `1e-4` | `10000` | `1000000` |
| 4 | `1e-4` | `20000` | `250000` |
| 5 | `1e-4` | `20000` | `500000` |
| 6 | `1e-4` | `20000` | `1000000` |
| 7 | `2.5e-4` | `10000` | `250000` |
| 8 | `2.5e-4` | `10000` | `500000` |
| 9 | `2.5e-4` | `10000` | `1000000` |
| 10 | `2.5e-4` | `20000` | `250000` |
| 11 | `2.5e-4` | `20000` | `500000` |
| 12 | `2.5e-4` | `20000` | `1000000` |

Selection rule when the full grid is eventually completed:

- Highest `mean_return` in the final eval JSON for that run.
- Tie-break 1: smoother upward eval trend across checkpoints.
- Tie-break 2: saner `loss`, `q_mean`, and `td_abs_mean` behavior in `metrics.csv`.

Current status:

- The full 12-run grid has **not** been completed yet.
- The tracked 200k default tune config did **not** satisfy the T1.1 success criterion.
- The most promising quick candidate observed so far is `lr=1e-4`, `learning_starts=10000`, `decay_steps=250000`, but it still regressed to `-21.0` by its final `100k` eval.
