# VaR95, CVaR95 and mean P&L on the same test paths (seed 4242, 8,192 paths, adi_trading)

Levels with bootstrap SE. Differences paired on the same paths (2,000 resamples): VaR/CVaR = baseline - DRL, mean P&L = DRL - baseline; positive = DRL better.

| Cost | Policy | VaR95 | CVaR95 | Mean P&L |
|---|---|---|---|---|
| 0.5% | DG | 31.38 (0.61) | 42.53 (0.78) | -3.21 (0.19) |
| 0.5% | Rule (b) | 24.44 (0.34) | 34.04 (0.70) | -2.96 (0.14) |
| 0.5% | Richer rule | 24.16 (0.39) | 33.69 (0.68) | -3.22 (0.14) |
| 0.5% | DRL selected (seed 4) | 23.16 (0.38) | 33.15 (0.73) | -2.81 (0.15) |
| 0.5% | DRL median (seed 5) | 23.08 (0.33) | 33.19 (0.74) | -2.75 (0.15) |
| 1.0% | DG | 34.85 (0.64) | 46.30 (0.80) | -6.50 (0.19) |
| 1.0% | Rule (b) | 27.54 (0.50) | 36.96 (0.68) | -5.11 (0.15) |
| 1.0% | Richer rule | 27.56 (0.41) | 36.44 (0.64) | -4.79 (0.15) |
| 1.0% | DRL selected (seed 1) | 25.35 (0.45) | 36.36 (0.81) | -4.97 (0.15) |
| 1.0% | DRL median (seed 9) | 25.88 (0.39) | 36.11 (0.75) | -5.18 (0.15) |
| 2.0% | DG | 41.87 (0.66) | 53.96 (0.83) | -13.07 (0.19) |
| 2.0% | Rule (b) | 32.94 (0.45) | 42.59 (0.69) | -7.10 (0.17) |
| 2.0% | Richer rule | 32.60 (0.64) | 42.06 (0.67) | -7.18 (0.16) |
| 2.0% | DRL selected (seed 8) | 31.15 (0.50) | 42.85 (0.86) | -6.81 (0.17) |
| 2.0% | DRL median (seed 9) | 30.93 (0.52) | 42.77 (0.86) | -6.94 (0.17) |

| Cost | Comparison | dVaR95 | dCVaR95 | dMean P&L |
|---|---|---|---|---|
| 0.5% | DRL selected (seed 4) vs DG | 8.22 (0.56) | 9.38 (0.68) | 0.40 (0.13) |
| 0.5% | DRL selected (seed 4) vs Rule (b) | 1.29 (0.28) | 0.89 (0.47) | 0.15 (0.07) |
| 0.5% | DRL selected (seed 4) vs Richer rule | 1.01 (0.28) | 0.54 (0.45) | 0.42 (0.06) |
| 0.5% | DRL median (seed 5) vs DG | 8.30 (0.56) | 9.34 (0.72) | 0.47 (0.13) |
| 0.5% | DRL median (seed 5) vs Rule (b) | 1.36 (0.25) | 0.85 (0.45) | 0.21 (0.06) |
| 0.5% | DRL median (seed 5) vs Richer rule | 1.08 (0.27) | 0.50 (0.43) | 0.48 (0.06) |
| 1.0% | DRL selected (seed 1) vs DG | 9.50 (0.57) | 9.94 (0.65) | 1.53 (0.13) |
| 1.0% | DRL selected (seed 1) vs Rule (b) | 2.19 (0.34) | 0.59 (0.45) | 0.14 (0.06) |
| 1.0% | DRL selected (seed 1) vs Richer rule | 2.21 (0.30) | 0.07 (0.43) | -0.18 (0.07) |
| 1.0% | DRL median (seed 9) vs DG | 8.97 (0.58) | 10.19 (0.67) | 1.32 (0.13) |
| 1.0% | DRL median (seed 9) vs Rule (b) | 1.66 (0.35) | 0.84 (0.47) | -0.07 (0.06) |
| 1.0% | DRL median (seed 9) vs Richer rule | 1.68 (0.30) | 0.33 (0.42) | -0.39 (0.06) |
| 2.0% | DRL selected (seed 8) vs DG | 10.72 (0.59) | 11.10 (0.64) | 6.27 (0.13) |
| 2.0% | DRL selected (seed 8) vs Rule (b) | 1.79 (0.41) | -0.27 (0.65) | 0.29 (0.09) |
| 2.0% | DRL selected (seed 8) vs Richer rule | 1.44 (0.48) | -0.80 (0.64) | 0.37 (0.08) |
| 2.0% | DRL median (seed 9) vs DG | 10.94 (0.62) | 11.18 (0.71) | 6.13 (0.14) |
| 2.0% | DRL median (seed 9) vs Rule (b) | 2.01 (0.41) | -0.19 (0.63) | 0.15 (0.08) |
| 2.0% | DRL median (seed 9) vs Richer rule | 1.67 (0.49) | -0.72 (0.62) | 0.23 (0.07) |
