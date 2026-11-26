# Improvement screens (validation seed 2025 only)

best = best val VaR95 after the freeze; last4 = mean of the last 4 validation checks. Gains = ctrl - arm, paired by seed (positive = arm better); se over seeds.

| arm     |   cost |   n |   done |   best |   last4 |   val_cost |   cvar |   paired |   gain_best |   gain_last4 |   se_last4 |
|:--------|-------:|----:|-------:|-------:|--------:|-----------:|-------:|---------:|------------:|-------------:|-----------:|
| cand    |  0.005 |   3 |      3 | 23.209 |  23.405 |      3.002 | 33.364 |        3 |       0.26  |        0.354 |      0.132 |
| candrv  |  0.005 |   2 |      0 | 24.014 |  24.512 |      3.181 | 34.814 |        2 |      -0.497 |       -0.638 |      0.117 |
| ctrl314 |  0.005 |   3 |      3 | 23.469 |  23.76  |      3.056 | 33.623 |      nan |     nan     |      nan     |    nan     |
| cand    |  0.01  |   3 |      3 | 25.67  |  25.96  |      5.255 | 36.443 |        3 |       0.031 |        0.11  |      0.244 |
| candrv  |  0.01  |   2 |      0 | 27.042 |  27.41  |      5.355 | 38.873 |        2 |      -1.529 |       -1.521 |      0.152 |
| ctrl314 |  0.01  |   3 |      3 | 25.701 |  26.07  |      5.211 | 36.835 |      nan |     nan     |      nan     |    nan     |
| cand    |  0.02  |   3 |      3 | 30.424 |  30.833 |      7.183 | 42.6   |        3 |      -0.084 |       -0.163 |      0.143 |
| candrv  |  0.02  |   2 |      0 | 31.495 |  32.381 |      7.159 | 45.883 |        2 |      -1.284 |       -1.876 |      0.092 |
| ctrl314 |  0.02  |   3 |      3 | 30.34  |  30.67  |      7.191 | 42.894 |      nan |     nan     |      nan     |    nan     |
