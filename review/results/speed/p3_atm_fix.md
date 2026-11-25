# P3 (pre-commitment, b = 0.6 x rule VaR): ATM fix on / off, validation only

Speed branch (ad60894 + d17b0dd, mixture loss), 12,000 episodes, 6 runs in parallel.
Numbers: best validation VaR95 (terminal) of the selected checkpoint, 8,192 validation paths (seed 2025).
Lower is better. Fix - no fix < 0 means the ATM fix helps. No test-set numbers (exploratory test set).

| Cost | Seed | Rule (b) val | No fix: val (sel. ep) | ATM fix: val (sel. ep) | Fix - no fix |
|---|---|---|---|---|---|
| 0.5% | 1 | 24.59 | 24.38 (7504) | 24.23 (9008) | -0.15 |
| 0.5% | 2 | 24.59 | 24.40 (7008) | 24.59 (1008) | +0.18 |
| 0.5% | 3 | 24.59 | 24.66 (1008) | 24.66 (1008) | +0.00 |
| 1.0% | 1 | 27.87 | 26.88 (11504) | 26.76 (11504) | -0.12 |
| 1.0% | 2 | 27.87 | 26.80 (11504) | 27.39 (11504) | +0.59 |
| 1.0% | 3 | 27.87 | 27.31 (12000) | 27.38 (9008) | +0.08 |
| 2.0% | 1 | 32.80 | 30.61 (9008) | 30.58 (9008) | -0.03 |
| 2.0% | 2 | 32.80 | 31.42 (7504) | 31.00 (12000) | -0.41 |
| 2.0% | 3 | 32.80 | 31.41 (11504) | 31.07 (11504) | -0.33 |

| Cost | Mean no fix | Mean fix | Mean fix - no fix | SD over seeds | n |
|---|---|---|---|---|---|
| 0.5% | 24.48 | 24.49 | +0.01 | 0.16 | 3 |
| 1.0% | 26.99 | 27.18 | +0.18 | 0.37 | 3 |
| 2.0% | 31.15 | 30.89 | -0.26 | 0.20 | 3 |

Seconds per 1,000 episodes (6 parallel): no fix 132, fix 131.
