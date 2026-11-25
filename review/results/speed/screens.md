# Quick screens on the FINAL set-up (validation only)

FINAL set-up (review/warmstart/run_final.sh: P3 pre-commitment, adi_trading market, re-tuned rule (b), b = 0.6 x rule VaR,
ATM fix), seed 1, 6,000 episodes, `--skip_test` (fresh test set untouched). One change per row vs the control.
Best validation VaR95 (8,192 validation paths), lower is better. Rule (b) re-tuned: 24.69 / 27.64 / 32.66.

| Screen | 0.5% | 1% | 2% |
|---|---|---|---|
| control (FINAL set-up, 6k episodes) | 23.81 | 26.20 | 30.53 |
| half updates (`--samples_per_insert 16`) | 23.82 | 26.19 | 30.99 |
| half learning rates (`--lr 3e-4 --actor_lr 1.5e-4`) | 23.70 | 26.05 | 30.66 |
| less exploration noise (`--sigma 0.5 --sigma_floor 0.1`) | 23.88 | 26.19 | 30.96 |
| original DRL learning rate (`--lr 1e-4 --actor_lr 1e-4`) | 24.35 | 26.49 | 32.48 |
| batch 512 (stopped on request; 2% not run) | 23.95 | 25.80 | - |

No change beats the control clearly at all three costs (seed-to-seed spread is about 0.3-0.5), so FINAL was kept as it is.
The original learning rate 1e-4 is clearly worse (too slow to leave the rule in 6k episodes).
Lead from earlier batches: more gradient updates per sample hurt (A2 vs A1, E1b vs E1a).
Script: logs_torch/screen/run_screens*.sh in the main tree (run_final.sh screen mode).
