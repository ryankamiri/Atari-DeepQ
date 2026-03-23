# T1.5 — Online Matrix Results

This note records the completed T1.5 headline online matrix for `Atari-DeepQ`.

## Locked protocol

- Envs: `ALE/Pong-ram-v5`, `ALE/Breakout-ram-v5`, `ALE/Boxing-ram-v5`
- Agents:
  - `DQN`
  - `DDQN + dueling + PER`
- Seeds: `42, 43, 44`
- Training steps: `2,000,000`
- Evaluation cadence: every `50,000` steps
- Evaluation episodes: `20`

## Completed run set

All `18` headline runs completed and were included in aggregation.

- Pong / DQN:
  - `artifacts/runs/20260323_094619_pong_dqn_seed42`
  - `artifacts/runs/20260323_094619_pong_dqn_seed43`
  - `artifacts/runs/20260323_094619_pong_dqn_seed44`
- Pong / DDQN + dueling + PER:
  - `artifacts/runs/20260323_094619_pong_ddqn_dueling_per_seed42`
  - `artifacts/runs/20260323_094619_pong_ddqn_dueling_per_seed43`
  - `artifacts/runs/20260323_094619_pong_ddqn_dueling_per_seed44`
- Breakout / DQN:
  - `artifacts/runs/20260323_095228_breakout_dqn_seed42`
  - `artifacts/runs/20260323_095228_breakout_dqn_seed43`
  - `artifacts/runs/20260323_095228_breakout_dqn_seed44`
- Breakout / DDQN + dueling + PER:
  - `artifacts/runs/20260323_095228_breakout_ddqn_dueling_per_seed42`
  - `artifacts/runs/20260323_095228_breakout_ddqn_dueling_per_seed43`
  - `artifacts/runs/20260323_095228_breakout_ddqn_dueling_per_seed44`
- Boxing / DQN:
  - `artifacts/runs/20260323_095431_boxing_dqn_seed42`
  - `artifacts/runs/20260323_101642_boxing_dqn_seed43`
  - `artifacts/runs/20260323_101642_boxing_dqn_seed44`
- Boxing / DDQN + dueling + PER:
  - `artifacts/runs/20260323_095431_boxing_ddqn_dueling_per_seed42`
  - `artifacts/runs/20260323_101642_boxing_ddqn_dueling_per_seed43`
  - `artifacts/runs/20260323_101642_boxing_ddqn_dueling_per_seed44`

## Output artifacts

- Summary JSON:
  - `artifacts/figures/online/online_matrix_summary.json`
- Final tables:
  - `artifacts/figures/online/online_final_eval_table.csv`
  - `artifacts/figures/online/online_final_eval_table.md`
- Learning curves:
  - `artifacts/figures/online/learning_curve_pong.png`
  - `artifacts/figures/online/learning_curve_breakout.png`
  - `artifacts/figures/online/learning_curve_boxing.png`

## Final eval table

The headline metric is the `20`-episode evaluation mean return at the final `2,000,000`-step checkpoint, aggregated across `3` seeds.

| env_id | variant | n_seeds | final_eval_mean +/- std |
|---|---:|---:|---:|
| ALE/Pong-ram-v5 | DQN | 3 | -13.20 +/- 2.22 |
| ALE/Pong-ram-v5 | DDQN+Dueling+PER | 3 | -9.50 +/- 1.76 |
| ALE/Breakout-ram-v5 | DQN | 3 | 11.62 +/- 1.14 |
| ALE/Breakout-ram-v5 | DDQN+Dueling+PER | 3 | 20.15 +/- 6.14 |
| ALE/Boxing-ram-v5 | DQN | 3 | 46.85 +/- 3.59 |
| ALE/Boxing-ram-v5 | DDQN+Dueling+PER | 3 | 57.13 +/- 12.81 |

## Headline interpretation

- The main online agent, `DDQN + dueling + PER`, outperformed vanilla `DQN` on all three environments in final `2M`-step evaluation mean return.
- Pong improved from `-13.20 +/- 2.22` to `-9.50 +/- 1.76`.
- Breakout improved from `11.62 +/- 1.14` to `20.15 +/- 6.14`.
- Boxing improved from `46.85 +/- 3.59` to `57.13 +/- 12.81`.

For the paper, the strongest clean statement supported by this matrix is that the Rainbow-lite online agent was consistently stronger than the vanilla DQN baseline across Pong, Breakout, and Boxing under the locked `3`-seed, `2M`-step evaluation protocol.
