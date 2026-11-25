# Improvement screens (validation seed 2025 only)

best = best val VaR95 after the freeze; last4 = mean of the last 4 validation checks. Gains = ctrl - arm, paired by seed (positive = arm better); se over seeds.

| arm   |   cost |   n |   done |   best |   last4 |   val_cost |   cvar |   paired |   gain_best |   gain_last4 |   se_last4 |
|:------|-------:|----:|-------:|-------:|--------:|-----------:|-------:|---------:|------------:|-------------:|-----------:|
| b30   |   0.01 |   1 |      0 | 27.008 |  27.249 |      5.168 | 40.496 |        1 |      -0.998 |       -1.034 |        nan |
| b45   |   0.01 |   1 |      0 | 26.056 |  26.294 |      5.373 | 38.817 |        1 |      -0.046 |       -0.079 |        nan |
| ctrl  |   0.01 |   1 |      0 | 26.01  |  26.215 |      5.249 | 38.343 |      nan |     nan     |      nan     |        nan |
| b30   |   0.02 |   1 |      0 | 32.498 |  32.841 |      6.04  | 48.936 |        1 |      -1.891 |       -1.997 |        nan |
| b45   |   0.02 |   1 |      0 | 32.398 |  32.628 |      6.143 | 48.208 |        1 |      -1.791 |       -1.785 |        nan |
| ctrl  |   0.02 |   1 |      0 | 30.607 |  30.843 |      7.012 | 44.279 |      nan |     nan     |      nan     |        nan |
