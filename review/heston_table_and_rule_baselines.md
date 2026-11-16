# Heston pricing table and cost-aware rule baselines

Referee comment addressed: *the baseline (flatten gamma / vega every step) is too weak; add cost-aware baselines.*
Branch: `claude/project-thread-zgv2tr`. The RL agent is not touched.

## Summary

* A precomputed QuantLib table replaces the per-option FD solve. Building the market for 32 paths takes
  **0.15 s instead of 155 s** (about 1000x). 8,192 paths now take about 13 s per path set.
  The table itself is built once per parameter set (14 min on 4 cores, 364,500 QuantLib solves) and cached.
* Table error vs direct QuantLib solves is small: 99% of points have price error < 5e-4, delta < 2e-4,
  gamma < 3e-3 (for gamma values up to 4). Episode P&L of the Delta and Delta-Gamma baselines changes by
  at most 0.004 and 0.13 per path (P&L std 35 and 15).
* **Under Heston, a tuned rule beats full Delta-Gamma by about 20% in VaR95 and CVaR95 at every option
  cost (0.5%, 1%, 2%), with standard errors of about 0.6 (gains of 6 to 13).** It beats Delta-Vega by
  40 to 48%. Adding a no-trade band helps only a little (0 to 1.5 extra points, clearest at 2% cost).
* The gain is almost the same at 0.5% cost as at 2% cost. So most of the gain does not come from saving
  option costs. It comes from hedging a mix of about 0.2 to 0.4 of gamma and 0.2 to 0.3 of vega instead of
  all gamma (DG) or all vega (DV).
* A proportional cost on delta trades (0 to 10 bp) lowers the gain over DG a little (e.g. CVaR95 at 1%
  option cost: 21% to 15%) and raises the gain over DV. The conclusion does not change.

For the paper this means the RL agent under Heston must be compared with these tuned rules, not only with
DG and DV. Under the old baselines the RL gains will look larger than they are.

## 1. Pricing table (`environment/heston_table.py`)

* American put is homogeneous of degree 1 in (S, K): `P = K * p(S/K, tau, v)`. The table is built at K = 1
  and rescaled (price and vega times K, gamma divided by K, delta unchanged).
* Axes: tau = 1..60 **integer trading days** (the environment steps one day, so no interpolation in time);
  81 log-moneyness points in [0.5, 2.0], dense near ATM with a width that scales with sqrt(tau);
  25 volatility points in [0.02, 1.0] (simulated paths stay in [0.09, 0.51]).
* Each node uses the same engine as the environment: `Utils._American_Option_Heston`
  (FdHestonVanillaEngine, Modified Craig-Sneyd, 50 grid points = `numerical_accuracy='low'`), QuantLib 1.37
  as in `environment.yaml`. Price, delta, gamma are the engine outputs.
* Interpolation: bicubic spline in (log m, vol) per tau, separately for price, delta, gamma, vega.
  Outside the moneyness grid: intrinsic value (deep ITM, delta -1) or zero (deep OTM); vol is clamped.
  No path point fell outside the grid.
* **Vega change (please check).** Vega keeps its meaning (parallel shift of sqrt(v0) and sqrt(theta),
  per 1 vol point), but the table uses a central bump of 0.5 vol points. The original code uses a forward
  bump of about 1e-4 vol points. With such a tiny bump the FD mesh changes between the two solves and the
  vega has spikes (we found nodes with vega -0.16 where the true value is about 0). Option `vega_mode=
  'env_forward'` reproduces the old definition.
* Parameters are in `HestonTableConfig` (kappa, theta, volvol, rho, r, q, days per year, grid, engine size).
  `HestonTableConfig.from_utils(utils)` takes them from the environment, and the cache file name holds a
  hash of the config. So a robustness bump (e.g. volvol 0.4) automatically gets its own table.

### Accuracy (table vs direct QuantLib solve, 4,000 random points, random strikes K in [5, 20])

Points are not grid nodes. Regions: bulk (m in [0.75, 1.3], tau 1-60), near maturity (tau 1-3 days,
m in [0.9, 1.1]), deep ITM (m in [0.55, 0.75]), deep OTM (m in [1.3, 1.9]); vol in [0.08, 0.6].
Errors for one option with the drawn strike: price and vega in currency, delta per share, gamma per
currency unit. Max abs error / mean abs error:

| Region | n | Price max / mean | Delta max / mean | Gamma max / mean | Vega max / mean |
|---|---|---|---|---|---|
| all | 4000 | 3.2e-03 / 1.8e-05 | 1.7e-03 / 1.1e-05 | 1.4e-01 / 2.7e-04 | 4.8e-03 / 2.6e-05 |
| bulk | 2000 | 1.8e-03 / 6.5e-06 | 8.9e-04 / 7.6e-06 | 7.5e-03 / 1.6e-04 | 3.3e-03 / 9.2e-06 |
| deep ITM (m<0.75) | 666 | 3.2e-03 / 7.7e-05 | 3.6e-04 / 1.4e-05 | 1.8e-03 / 7.5e-05 | 4.8e-03 / 1.1e-04 |
| deep OTM (m>1.3) | 668 | 1.4e-05 / 4.7e-07 | 1.3e-05 / 1.4e-06 | 2.3e-04 / 1.3e-05 | 3.7e-05 / 3.1e-07 |
| near maturity (1-3d) | 666 | 1.1e-03 / 1.2e-05 | 1.7e-03 / 2.9e-05 | 1.4e-01 / 1.0e-03 | 2.3e-03 / 1.5e-05 |

| | Price | Delta | Gamma | Vega |
|---|---|---|---|---|
| largest value in sample | 8.73 | 1 | 4.01 | 0.0378 |
| 99th pct abs error | 4.9e-04 | 1.8e-04 | 3.1e-03 | 6.8e-04 |

Notes:
* The largest gamma errors (0.14) are at tau = 1 day, near ATM, with vol below 0.1: the gamma spike is
  very narrow there. The paths almost never have vol below 0.1 (0.1% quantile is 0.17).
* The largest vega errors (about 5e-3) are deep ITM, where the true vega is about 0. Here the direct
  QuantLib solve itself is noisy (it gives 5e-3 at one point and 2e-7 at the next), so this is engine
  noise, not interpolation error. Deep ITM options carry almost no vega in the portfolio.

### End-to-end check: same paths, environment built with QuantLib vs with the table

32 paths (seed 2025), episode P&L of the existing baselines:

| Strategy | Option cost | Mean P&L QuantLib | Mean P&L table | Std P&L QuantLib | Max abs path diff | Mean abs path diff |
|---|---|---|---|---|---|---|
| Delta | 0% | -0.208 | -0.208 | 34.71 | 0.004 | 0.001 |
| Delta | 1% | -0.208 | -0.208 | 34.71 | 0.004 | 0.001 |
| Delta-Gamma | 0% | 1.464 | 1.463 | 15.47 | 0.118 | 0.012 |
| Delta-Gamma | 1% | -5.276 | -5.277 | 15.92 | 0.128 | 0.012 |
| Delta-Vega | 0% | 6.284 | 6.365 | 30.78 | 1.907 | 0.149 |
| Delta-Vega | 1% | -6.012 | -5.900 | 30.26 | 2.289 | 0.189 |

Delta and Delta-Gamma agree to about 1e-3 of the P&L std. Delta-Vega differs more (max 1.9, std 31)
because it trades on vega, and the original vega is noisy (see above).

## 2. Environment flag

* `Utils(..., pricing_backend='quantlib' | 'table', table_dir=None)`. Default is `'quantlib'`, so all old
  code and saved portfolios behave as before. `pricing_backend` is stored in `get_params()`, so a saved
  QuantLib portfolio will not load silently into a table run.
* `run.py` and `greek_run.py` get a flag `--pricing_backend` (default `quantlib`). Only the Utils call is
  changed; the agent code is not touched.
* Small fix: the joblib `temp_folder` was hard-coded to the old server path
  (`/home/atsoskouno/...`); it is now the joblib default.

## 3. Rule baselines (`experiments/heston_rule_baselines.py`)

Setup = paper Heston setup (`(EXP 2.1)` scripts): S0 = K = 10, init_vol 0.3, kappa 1, theta 0.09,
volvol 0.3, rho -0.7, r = q = 0, liability: short one 30-day ATM American put at t = 0 plus Poisson(1)
arrivals of 60-day ATM puts (random buy/sell), hedging instrument: a new 30-day ATM American put each day,
30 daily steps, contract size 100, 252 days per year. Delta is hedged with the stock every step, as in the
environment. Option cost = spread x |premium traded| (as in `Portfolio.add`).

Each day the agent (and the rules) can only buy or sell the new ATM option. Let `n_G` be the number of
contracts that makes portfolio gamma zero and `n_V` the number that makes vega zero.

* **DG** (full Delta-Gamma): trade `n_G` (existing `GammaHedgeAgent`).
* **DV** (full Delta-Vega): trade `n_V` (existing `VegaHedgeAgent`). With one new option per day you cannot
  make gamma and vega zero together, so I used this as the "Delta-Gamma-Vega" baseline of the request.
* **(a) fraction**: trade `a n_G + b n_V`, clipped to `[min(0, n_G, n_V), max(0, n_G, n_V)]`, which is the
  RL agent's action range.
* **(b) fraction + band**: same, but no option trade if `|a n_G + b n_V| < w` contracts.

Grid: a, b in {0, 0.1, ..., 1}, w in {0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5} (1,089 rules).
**Tuning** on 8,192 validation paths (seed 2025), separately for each cost and each metric.
**Report** on 8,192 fresh test paths (seed 654, the same evaluation seed as the paper scripts, so a
retrained RL agent can be evaluated on exactly these paths). Risk is measured on the episode loss (-P&L):
VaR95 = 95% quantile, CVaR95 = mean loss beyond VaR95, in currency per episode. Gain = benchmark minus
rule (positive = rule is better). Standard errors in brackets: paired bootstrap over test paths (1,000
resamples). They do not include the uncertainty from tuning.

The vectorised P&L used for tuning was checked against `MainPortfolio.step` of the environment
(16 episodes, DG, DV and a rule): max difference 1e-11.

### Results on the test paths (no delta cost)

| Option cost | Metric | Rule | a | b | w | Rule | Delta-Gamma | Delta-Vega | Gain vs DG | Gain vs DV |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5% | VaR95 | (a) fraction | 0.3 | 0.3 | 0 | 25.05 (0.49) | 31.05 (0.62) | 41.22 (0.82) | 6.00 (0.55) = 19.3% | 16.17 (0.77) = 39.2% |
| 0.5% | VaR95 | (b) fraction + band | 0.3 | 0.3 | 0.3 | 25.03 (0.49) | 31.05 (0.62) | 41.22 (0.82) | 6.02 (0.56) = 19.4% | 16.19 (0.77) = 39.3% |
| 0.5% | CVaR95 | (a) fraction | 0.3 | 0.3 | 0 | 34.74 (0.69) | 44.01 (0.89) | 60.27 (1.37) | 9.27 (0.74) = 21.1% | 25.53 (1.23) = 42.4% |
| 0.5% | CVaR95 | (b) fraction + band | 0.3 | 0.3 | 0.5 | 34.34 (0.68) | 44.01 (0.89) | 60.27 (1.37) | 9.67 (0.73) = 22.0% | 25.92 (1.24) = 43.0% |
| 1% | VaR95 | (a) fraction | 0.2 | 0.2 | 0 | 28.39 (0.48) | 34.72 (0.66) | 48.06 (0.91) | 6.33 (0.57) = 18.2% | 19.67 (0.89) = 40.9% |
| 1% | VaR95 | (b) fraction + band | 0.3 | 0.3 | 0.5 | 27.67 (0.45) | 34.72 (0.66) | 48.06 (0.91) | 7.06 (0.57) = 20.3% | 20.40 (0.86) = 42.4% |
| 1% | CVaR95 | (a) fraction | 0.2 | 0.2 | 0 | 37.95 (0.68) | 47.74 (0.90) | 67.29 (1.40) | 9.79 (0.69) = 20.5% | 29.34 (1.32) = 43.6% |
| 1% | CVaR95 | (b) fraction + band | 0.3 | 0.3 | 0.5 | 37.49 (0.70) | 47.74 (0.90) | 67.29 (1.40) | 10.25 (0.74) = 21.5% | 29.80 (1.26) = 44.3% |
| 2% | VaR95 | (a) fraction | 0.2 | 0.1 | 0 | 34.20 (0.52) | 42.12 (0.64) | 61.85 (0.94) | 7.93 (0.46) = 18.8% | 27.66 (1.01) = 44.7% |
| 2% | VaR95 | (b) fraction + band | 0.2 | 0.3 | 1 | 32.69 (0.55) | 42.12 (0.64) | 61.85 (0.94) | 9.44 (0.63) = 22.4% | 29.17 (0.91) = 47.2% |
| 2% | CVaR95 | (a) fraction | 0.2 | 0.2 | 0 | 44.10 (0.70) | 55.32 (0.92) | 81.57 (1.43) | 11.22 (0.70) = 20.3% | 37.47 (1.34) = 45.9% |
| 2% | CVaR95 | (b) fraction + band | 0.2 | 0.2 | 0.5 | 42.66 (0.70) | 55.32 (0.92) | 81.57 (1.43) | 12.67 (0.70) = 22.9% | 38.91 (1.37) = 47.7% |

| Option cost | Delta-only VaR95 | Delta-only CVaR95 | Mean P&L rule (b, CVaR-tuned) | Mean P&L DG | Mean P&L DV |
|---|---|---|---|---|---|
| 0.5% | 55.94 (1.06) | 80.67 (1.83) | -2.59 | -3.42 | -6.09 |
| 1% | 55.94 (1.06) | 80.67 (1.83) | -5.17 | -6.76 | -12.28 |
| 2% | 55.94 (1.06) | 80.67 (1.83) | -7.17 | -13.46 | -24.64 |

### Delta-cost sensitivity (proportional cost on stock trades, rule (b) re-tuned on validation each time)

Delta cost = c x |change in stock position| x S_t, paid by all strategies (also DG and DV).

**VaR95**, rule (b) re-tuned on validation for every cell. Gain vs DG / gain vs DV, with s.e.

| Delta cost | option 0.5%: vs DG | option 0.5%: vs DV | option 1%: vs DG | option 1%: vs DV | option 2%: vs DG | option 2%: vs DV |
|---|---|---|---|---|---|---|
| 0 bp | 6.02 (0.56) = 19% | 16.19 (0.77) = 39% | 7.06 (0.57) = 20% | 20.40 (0.86) = 42% | 9.44 (0.63) = 22% | 29.17 (0.91) = 47% |
| 1 bp | 5.72 (0.57) = 18% | 16.48 (0.77) = 39% | 6.29 (0.53) = 18% | 20.00 (0.84) = 41% | 9.08 (0.63) = 21% | 29.42 (0.90) = 47% |
| 2 bp | 5.93 (0.61) = 19% | 17.24 (0.75) = 40% | 6.23 (0.54) = 18% | 20.47 (0.82) = 41% | 8.65 (0.57) = 20% | 29.49 (0.91) = 46% |
| 5 bp | 5.59 (0.54) = 17% | 18.56 (0.74) = 40% | 5.83 (0.59) = 16% | 21.59 (0.93) = 41% | 7.83 (0.50) = 18% | 30.21 (0.95) = 45% |
| 10 bp | 6.06 (0.58) = 17% | 21.81 (0.86) = 43% | 5.24 (0.58) = 14% | 24.18 (0.94) = 42% | 6.40 (0.46) = 14% | 31.70 (0.98) = 44% |

**CVaR95**, rule (b) re-tuned on validation for every cell. Gain vs DG / gain vs DV, with s.e.

| Delta cost | option 0.5%: vs DG | option 0.5%: vs DV | option 1%: vs DG | option 1%: vs DV | option 2%: vs DG | option 2%: vs DV |
|---|---|---|---|---|---|---|
| 0 bp | 9.67 (0.73) = 22% | 25.92 (1.24) = 43% | 10.25 (0.74) = 21% | 29.80 (1.26) = 44% | 12.67 (0.70) = 23% | 38.91 (1.37) = 48% |
| 1 bp | 9.43 (0.73) = 21% | 26.40 (1.25) = 43% | 10.01 (0.74) = 21% | 30.27 (1.27) = 44% | 12.28 (0.70) = 22% | 39.25 (1.38) = 47% |
| 2 bp | 9.19 (0.73) = 20% | 26.88 (1.26) = 43% | 9.76 (0.74) = 20% | 30.75 (1.28) = 44% | 11.77 (0.61) = 21% | 39.47 (1.44) = 47% |
| 5 bp | 8.58 (0.71) = 19% | 28.46 (1.30) = 43% | 9.02 (0.74) = 18% | 32.22 (1.31) = 44% | 10.84 (0.62) = 19% | 40.75 (1.48) = 47% |
| 10 bp | 8.68 (0.71) = 18% | 32.32 (1.35) = 45% | 8.02 (0.60) = 15% | 34.98 (1.45) = 44% | 9.26 (0.62) = 16% | 42.93 (1.54) = 46% |

| Delta cost | 0.5%: DG / DV / rule (b) CVaR95 | 1%: DG / DV / rule (b) CVaR95 | 2%: DG / DV / rule (b) CVaR95 |
|---|---|---|---|
| 0 bp | 44.0 / 60.3 / 34.3 | 47.7 / 67.3 / 37.5 | 55.3 / 81.6 / 42.7 |
| 1 bp | 44.4 / 61.4 / 35.0 | 48.2 / 68.4 / 38.1 | 55.7 / 82.7 / 43.5 |
| 2 bp | 44.8 / 62.5 / 35.6 | 48.6 / 69.6 / 38.8 | 56.2 / 83.9 / 44.4 |
| 5 bp | 46.1 / 65.9 / 37.5 | 49.8 / 73.0 / 40.8 | 57.4 / 87.3 / 46.6 |
| 10 bp | 48.1 / 71.8 / 39.5 | 51.9 / 78.9 / 43.9 | 59.6 / 93.2 / 50.3 |

## What was run

```bash
python experiments/validate_heston_table.py --n_points 4000 --n_jobs 4   # builds the table on first use
python experiments/compare_backends.py --n_paths 32                       # QuantLib vs table, timing
python experiments/heston_rule_baselines.py --check_env                   # vectorised P&L == environment
python experiments/heston_rule_baselines.py                               # tuning + test + delta costs
```
Run with `PYTHONPATH=.` from the repo root. Outputs are in `review/results/`. Total time on 4 CPU cores:
table 14 min, validation 1 min, rule study 9 min. The table is cached in `pricing_tables/`.

## Open points

1. Is DV (`VegaHedgeAgent`) the "Delta-Gamma-Vega" baseline you mean? If the paper uses another
   definition, the rule code needs a small change.
2. The vega definition change in the table backend (central 0.5 vol-point bump) should be stated in the
   paper if the new RL runs use the table.
3. Not done (would expand scope): RL agents retrained on the table backend, robustness-bump tables,
   rules with a zero-cost check of why the mix beats DG.
