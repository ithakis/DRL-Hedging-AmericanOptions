# Main table, block A (BS, VaR95 agents), test paths seed 654, 8,192 paths

VaR95 of the episode loss, standard errors in brackets. Gain = baseline - RL (positive = RL better), paired bootstrap SE.

| Cost | Agent | Seed | RL VaR95 | CVaR95 | Mean P&L | E[cost] | Gamma ratio | Vega ratio | Delta | DG | DV | Rule (a) | Rule (b) | Gain vs DG | Gain vs rule (b) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5% | selected | 4 | 5.06 (0.04) | 6.03 (0.07) | -2.96 (0.01) | 2.96 | 0.72 | nan | 25.48 (0.58) | 5.31 (0.04) | - | 5.09 (0.03) | 5.03 (0.04) | 0.24 (0.03) | -0.03 (0.03) |
| 0.5% | median | 3 | 5.06 (0.03) | 5.95 (0.07) | -2.96 (0.01) | 2.96 | 0.74 | nan | 25.48 (0.58) | 5.31 (0.04) | - | 5.09 (0.03) | 5.03 (0.04) | 0.25 (0.03) | -0.03 (0.03) |
| 1.0% | selected | 2 | 8.26 (0.07) | 9.43 (0.08) | -4.44 (0.02) | 4.41 | 0.43 | nan | 25.48 (0.58) | 9.65 (0.06) | - | 8.47 (0.06) | 8.13 (0.07) | 1.39 (0.07) | -0.13 (0.06) |
| 1.0% | median | 4 | 8.27 (0.07) | 9.65 (0.10) | -4.33 (0.03) | 4.31 | 0.45 | nan | 25.48 (0.58) | 9.65 (0.06) | - | 8.47 (0.06) | 8.13 (0.07) | 1.37 (0.08) | -0.15 (0.06) |
| 2.0% | selected | 3 | 12.82 (0.11) | 15.03 (0.14) | -5.49 (0.05) | 5.48 | 0.19 | nan | 25.48 (0.58) | 18.69 (0.08) | - | 13.36 (0.14) | 12.45 (0.11) | 5.87 (0.12) | -0.36 (0.09) |
| 2.0% | median | 5 | 13.01 (0.12) | 15.37 (0.16) | -5.76 (0.05) | 5.75 | 0.25 | nan | 25.48 (0.58) | 18.69 (0.08) | - | 13.36 (0.14) | 12.45 (0.11) | 5.68 (0.13) | -0.55 (0.10) |

## Gate: selected BS agent vs tuned gamma-fraction rule (a) on the same test paths

| Cost | RL VaR95 | Rule (a) VaR95 (ours) | RL - rule (paired SE) | Pass | Sven's rule VaR (his paths) |
|---|---|---|---|---|---|
| 0.5% | 5.06 | 5.09 (a=0.9) | -0.03 (0.02) | yes | 5.16 |
| 1.0% | 8.26 | 8.47 (a=0.6) | -0.22 (0.06) | yes | 8.55 |
| 2.0% | 12.82 | 13.36 (a=0.3) | -0.54 (0.13) | yes | 13.57 |

Pass = RL VaR95 is not worse than the rule by more than 2 paired standard errors.
