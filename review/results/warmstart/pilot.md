# Warm-start pilot (Heston P, r = 2%, VaR95), test paths seed 654, 8,192 paths

Selection: best validation VaR95 checkpoint (episode 0 = the rule itself is not selectable).
Gain vs rule (b) = rule (b) - RL, paired bootstrap SE; positive = RL better.

| Variant | Cost | Seed | Sel. ep | Val VaR95 (ep 0 = rule) | Val VaR95 (sel.) | Test VaR95 | Rule (b) test | Gain vs rule (b) | RL of PR #3 (test) |
|---|---|---|---|---|---|---|---|---|---|
| gate0 | 0.5% | 1 | 16000 | 24.60 | 24.77 | 24.82 (0.50) | 24.87 | 0.05 (0.25) | 26.25 |
| gate0 | 1.0% | 1 | 8000 | 27.87 | 28.33 | 28.14 (0.43) | 27.43 | -0.71 (0.25) | 29.95 |
| gate0 | 2.0% | 1 | 7000 | 32.80 | 33.50 | 33.57 (0.45) | 32.56 | -1.01 (0.34) | 35.13 |
| gate05 | 0.5% | 1 | 20000 | 24.60 | 24.70 | 25.08 (0.44) | 24.87 | -0.20 (0.18) | 26.25 |
| gate05 | 1.0% | 1 | 19000 | 27.87 | 28.01 | 27.64 (0.51) | 27.43 | -0.21 (0.18) | 29.95 |
| gate05 | 2.0% | 1 | 6000 | 32.80 | 32.90 | 32.94 (0.49) | 32.56 | -0.38 (0.20) | 35.13 |
| res02lr | 0.5% | 1 | 9000 | 24.59 | 24.98 | 24.63 (0.65) | 24.87 | 0.24 (0.32) | 26.25 |
| res02lr | 1.0% | 1 | 7000 | 27.93 | 28.70 | 28.65 (0.58) | 27.43 | -1.22 (0.36) | 29.95 |
| res02lr | 2.0% | 1 | 19000 | 32.83 | 34.47 | 34.50 (0.50) | 32.56 | -1.94 (0.34) | 35.13 |
| res05l2 | 0.5% | 1 | 8000 | 24.62 | 25.17 | 25.04 (0.44) | 24.87 | -0.16 (0.28) | 26.25 |
| res05l2 | 1.0% | 1 | 19000 | 27.92 | 28.50 | 28.45 (0.49) | 27.43 | -1.02 (0.25) | 29.95 |
| res05l2 | 2.0% | 1 | 17000 | 32.83 | 34.26 | 34.24 (0.50) | 32.56 | -1.68 (0.29) | 35.13 |

## Gate pilot: validation checkpoints after the warm-up

| Run | Rule (b) val VaR95 | Best val VaR95 | Checkpoints better than rule | Agreement with rule at best | Trade share at best |
|---|---|---|---|---|---|
| HestonP_var_spread0.005_seed1_gate0 | 24.60 | 24.77 (16000) | 0 of 18 | 0.67 | 0.70 |
| HestonP_var_spread0.005_seed1_gate05 | 24.60 | 24.70 (20000) | 0 of 18 | 0.93 | 0.65 |
| HestonP_var_spread0.01_seed1_gate0 | 27.87 | 28.33 (8000) | 0 of 18 | 0.66 | 0.52 |
| HestonP_var_spread0.01_seed1_gate05 | 27.87 | 28.01 (19000) | 0 of 18 | 0.95 | 0.43 |
| HestonP_var_spread0.02_seed1_gate0 | 32.80 | 33.50 (7000) | 0 of 18 | 0.69 | 0.44 |
| HestonP_var_spread0.02_seed1_gate05 | 32.80 | 32.90 (6000) | 0 of 18 | 0.96 | 0.24 |
