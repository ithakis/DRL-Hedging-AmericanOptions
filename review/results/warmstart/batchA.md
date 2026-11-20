# Batch A (training speed / stability recipe), seed 1, Heston P r = 2%, VaR95

A1 = 16 parallel paths, Polyak tau 0.005, reward scale 0.1, eval-only actor EMA 0.999, lr critic 6e-4 / actor 3e-4.
A2 = A1 + 4x more gradient steps per sample (batch 128, one update per 2 env steps).
PR3 = retrain recipe (PR #3), same seed 1, its own validation log. Validation VaR95 (seed 2025), test VaR95 (seed 654).
"Reach" = first episode within 1.0 of the run's best validation VaR95 up to 20k episodes. "Late sd" = std of validation VaR95 after episode 10k.

| Cost | Recipe | Best val, first 20k (ep) | Best val, 40k | Reach | Late sd | s per 1,000 eps | Test VaR95 (selected) | Rule (b) test |
|---|---|---|---|---|---|---|---|---|
| 0.5% | PR3 | 27.25 (20000) | 25.93 | 17000 | 1.29 | 157 | 25.84 | 24.87 |
| 0.5% | A1 | 27.88 (7504) | - | 2000 | 0.45 | 154 | 29.04 | 24.87 |
| 0.5% | A2 | 29.51 (12512) | - | 2000 | 0.48 | 337 | 29.84 | 24.87 |
| 1% | PR3 | 31.86 (20000) | 29.93 | 11000 | 0.32 | 261 | 29.83 | 27.43 |
| 1% | A1 | 31.63 (13504) | - | 3008 | 0.37 | 158 | 32.30 | 27.43 |
| 1% | A2 | 33.29 (16000) | - | 9008 | 0.45 | 323 | 33.34 | 27.43 |
| 2% | PR3 | 39.79 (20000) | 37.52 | 16000 | 1.42 | 262 | 37.26 | 32.56 |
| 2% | A1 | 37.92 (4512) | - | 2512 | 0.36 | 156 | 39.65 | 32.56 |
| 2% | A2 | 40.66 (13504) | - | 8512 | 0.54 | 333 | 41.88 | 32.56 |

PR3 test = its seed-1 run selected over 40k episodes. Timing is with 6 runs in parallel on a loaded Mac (not comparable across days).
