# Warm-start pilot 1 (stopped at ~29,000 of 40,000 episodes): validation VaR95

Residual on top of rule (b), critic warm-up 2,000 episodes (actor frozen), seed 1. Validation paths seed 2025, 8,192 paths.
Episode 0 = untrained agent = rule (b). No checkpoint after the warm-up is better than the rule on validation, so the runs were stopped.

| Run | Rule (b) val VaR95 | Best val VaR95 after warm-up (episode) | Last val VaR95 | Checkpoints better than rule |
|---|---|---|---|---|
| HestonP_var_spread0.005_seed1_res02 | 24.59 | 24.69 (27000) | 24.95 (29000) | 0 of 27 |
| HestonP_var_spread0.005_seed1_res05 | 24.62 | 25.25 (19000) | 26.57 (30000) | 0 of 28 |
| HestonP_var_spread0.01_seed1_res02 | 27.93 | 28.04 (25000) | 28.67 (30000) | 0 of 28 |
| HestonP_var_spread0.01_seed1_res05 | 27.92 | 28.92 (17000) | 29.44 (29000) | 0 of 27 |
| HestonP_var_spread0.02_seed1_res02 | 32.83 | 34.08 (28000) | 34.51 (29000) | 0 of 27 |
| HestonP_var_spread0.02_seed1_res05 | 32.83 | 34.84 (25000) | 35.81 (30000) | 0 of 28 |
