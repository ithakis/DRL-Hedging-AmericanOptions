# Hedging the American book with European Greeks (test seed 4242, 8,192 paths)

All P&L with American prices; European Greeks only for the hedge decisions and the delta hedge. Difference = European - American (positive = European Greeks worse), paired bootstrap SE.

| Cost | Policy | VaR95 American Greeks | VaR95 European Greeks | Difference | CVaR95 American | CVaR95 European | Difference |
|---|---|---|---|---|---|---|---|
| 0.5% | DG | 31.38 | 32.36 | +0.98 (0.18) | 42.53 | 43.94 | +1.41 (0.08) |
| 0.5% | Rule (b) | 24.44 | 24.94 | +0.50 (0.10) | 34.04 | 34.56 | +0.52 (0.05) |
| 1.0% | DG | 34.85 | 35.72 | +0.87 (0.16) | 46.30 | 47.68 | +1.37 (0.08) |
| 1.0% | Rule (b) | 27.54 | 28.13 | +0.59 (0.16) | 36.96 | 37.54 | +0.58 (0.05) |
| 2.0% | DG | 41.87 | 42.78 | +0.91 (0.17) | 53.96 | 55.26 | +1.30 (0.08) |
| 2.0% | Rule (b) | 32.94 | 33.50 | +0.56 (0.12) | 42.59 | 43.23 | +0.65 (0.05) |

Rule (b) with European Greeks re-tuned on validation: 0.5%: a=0.4, b=0.4, w=0.75; 1.0%: a=0.3, b=0.3, w=0.5; 2.0%: a=0.2, b=0.3, w=1.0
