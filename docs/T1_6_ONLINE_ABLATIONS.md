# T1.6 — Online Ablations (Breakout Only)

This note records the completed T1.6 online ablation section for `Atari-DeepQ`.

## Locked protocol

- Env: `ALE/Breakout-ram-v5`
- Variants:
  - `DDQN + dueling + PER`
  - `DDQN + dueling`
  - `DDQN + PER`
- Seeds: `42, 43, 44`
- Training steps: `2,000,000`
- Evaluation cadence: every `50,000` steps
- Evaluation episodes: `20`

## Run set

Reused completed main-agent runs from T1.5:

- `artifacts/runs/20260323_095228_breakout_ddqn_dueling_per_seed42`
- `artifacts/runs/20260323_095228_breakout_ddqn_dueling_per_seed43`
- `artifacts/runs/20260323_095228_breakout_ddqn_dueling_per_seed44`

New T1.6 ablation runs:

- `DDQN + dueling`:
  - `artifacts/runs/20260323_174512_breakout_ddqn_dueling_seed42`
  - `artifacts/runs/20260323_174512_breakout_ddqn_dueling_seed43`
  - `artifacts/runs/20260323_174512_breakout_ddqn_dueling_seed44`
- `DDQN + PER`:
  - `artifacts/runs/20260323_174512_breakout_ddqn_per_seed42`
  - `artifacts/runs/20260323_174512_breakout_ddqn_per_seed43`
  - `artifacts/runs/20260323_174512_breakout_ddqn_per_seed44`

All `9` runs in the ablation report completed the full protocol and were included in aggregation.

## Output artifacts

- Figure:
  - `artifacts/figures/online/ablations/breakout_ablation_curves.png`
- Final tables:
  - `artifacts/figures/online/ablations/breakout_ablation_final_table.csv`
  - `artifacts/figures/online/ablations/breakout_ablation_final_table.md`
- Summary JSON:
  - `artifacts/figures/online/ablations/breakout_ablation_summary.json`

## Final ablation table

The headline metric is the final `2,000,000`-step eval mean return aggregated across `3` seeds.

| variant | n_seeds | mean +/- std | delta_vs_main |
|---|---:|---:|---:|
| `DDQN + dueling + PER` | 3 | `20.15 +/- 6.14` | `0.00` |
| `DDQN + dueling` | 3 | `14.37 +/- 4.67` | `-5.78` |
| `DDQN + PER` | 3 | `18.15 +/- 3.56` | `-2.00` |

## Interpretation

- Removing **PER** caused the larger drop on Breakout:
  - `DDQN + dueling + PER` to `DDQN + dueling`: `-5.78`
- Removing **dueling** also hurt performance, but less:
  - `DDQN + dueling + PER` to `DDQN + PER`: `-2.00`

Under this locked Breakout-only ablation, **PER appears to be the more important of the two components** for the final online result, while dueling still provides a smaller positive gain.

## Stability notes

- All `6` new T1.6 ablation runs completed without crash.
- The aggregated ablation report status is `complete`.
- Like the T1.5 headline runs, some individual seeds peaked earlier than the final `2M` checkpoint, so the paper should present these as final-protocol comparisons rather than fully converged best-checkpoint results.
