# Warm-start pilot (Heston P, r = 2%, VaR95), test paths seed 654, 8,192 paths

Selection: best validation VaR95 checkpoint (episode 0 = the rule itself is not selectable).
Gain vs rule (b) = rule (b) - RL, paired bootstrap SE; positive = RL better.

| Variant | Cost | Seed | Sel. ep | Val VaR95 (ep 0 = rule) | Val VaR95 (sel.) | Test VaR95 | Rule (b) test | Gain vs rule (b) | RL of PR #3 (test) |
|---|---|---|---|---|---|---|---|---|---|
| res02lr | 0.5% | 1 | 9000 | 24.59 | 24.98 | 24.63 (0.65) | 24.87 | 0.24 (0.32) | 26.25 |
| res02lr | 1.0% | 1 | 7000 | 27.93 | 28.70 | 28.65 (0.58) | 27.43 | -1.22 (0.36) | 29.95 |
| res02lr | 2.0% | 1 | 19000 | 32.83 | 34.47 | 34.50 (0.50) | 32.56 | -1.94 (0.34) | 35.13 |
| res05l2 | 0.5% | 1 | 8000 | 24.62 | 25.17 | 25.04 (0.44) | 24.87 | -0.16 (0.28) | 26.25 |
| res05l2 | 1.0% | 1 | 19000 | 27.92 | 28.50 | 28.45 (0.49) | 27.43 | -1.02 (0.25) | 29.95 |
| res05l2 | 2.0% | 1 | 17000 | 32.83 | 34.26 | 34.24 (0.50) | 32.56 | -1.68 (0.29) | 35.13 |
