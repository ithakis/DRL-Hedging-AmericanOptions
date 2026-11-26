# Convergence of the adi_trading pricing table (price, Delta, Gamma, Vega)

400 random off-grid points (100 per region; S/K, trading days to maturity 1-60, vol 0.12-0.6), maturity tau/252. Reference: ADI table on a 2x finer grid (800 x 400, 16 steps/day). Errors per unit strike, |error| max / 99th percentile / mean. Largest |value| in the sample: Price 0.45, Delta 1, Gamma 35.1, Vega 0.00181. Build time (one core): 100 x 50, 2 steps/day 0s, 200 x 100, 4 steps/day 2s, 400 x 200, 8 steps/day 12s, 800 x 400, 16 steps/day 93s.

## all (400 points)

| Grid | Price | Delta | Gamma | Vega |
|---|---|---|---|---|
| 100 x 50, 2 steps/day | 1.1e-04 / 7.7e-05 / 1.2e-05 | 1.3e-02 / 7.3e-03 / 6.0e-04 | 7.5e-01 / 5.2e-01 / 3.1e-02 | 9.3e-06 / 7.0e-06 / 9.8e-07 |
| 200 x 100, 4 steps/day | 4.8e-05 / 2.6e-05 / 3.8e-06 | 2.0e-03 / 1.1e-03 / 1.0e-04 | 5.0e-01 / 2.5e-01 / 1.5e-02 | 5.6e-06 / 2.4e-06 / 2.8e-07 |
| 400 x 200, 8 steps/day (used) | 8.2e-06 / 4.0e-06 / 6.0e-07 | 5.7e-04 / 3.9e-04 / 2.6e-05 | 2.0e-01 / 1.4e-01 / 5.2e-03 | 1.4e-06 / 6.5e-07 / 6.2e-08 |

## bulk (100 points)

| Grid | Price | Delta | Gamma | Vega |
|---|---|---|---|---|
| 100 x 50, 2 steps/day | 4.8e-05 / 4.3e-05 / 1.5e-05 | 1.5e-03 / 1.2e-03 / 2.4e-04 | 6.9e-02 / 3.1e-02 / 4.0e-03 | 9.0e-06 / 3.1e-06 / 8.8e-07 |
| 200 x 100, 4 steps/day | 1.8e-05 / 1.3e-05 / 5.1e-06 | 2.7e-04 / 2.5e-04 / 4.5e-05 | 9.8e-02 / 1.9e-02 / 2.2e-03 | 2.6e-06 / 1.6e-06 / 2.5e-07 |
| 400 x 200, 8 steps/day (used) | 2.5e-06 / 2.3e-06 / 8.2e-07 | 1.4e-04 / 1.0e-04 / 1.2e-05 | 8.6e-02 / 2.3e-02 / 1.9e-03 | 1.4e-06 / 6.8e-07 / 6.3e-08 |

## near maturity (100 points)

| Grid | Price | Delta | Gamma | Vega |
|---|---|---|---|---|
| 100 x 50, 2 steps/day | 1.1e-04 / 9.2e-05 / 2.4e-05 | 1.3e-02 / 1.2e-02 / 1.8e-03 | 7.5e-01 / 6.9e-01 / 1.1e-01 | 7.0e-06 / 7.0e-06 / 1.6e-06 |
| 200 x 100, 4 steps/day | 4.8e-05 / 4.5e-05 / 8.4e-06 | 2.0e-03 / 1.7e-03 / 2.8e-04 | 5.0e-01 / 4.4e-01 / 4.8e-02 | 4.0e-06 / 3.4e-06 / 4.8e-07 |
| 400 x 200, 8 steps/day (used) | 8.2e-06 / 6.7e-06 / 1.3e-06 | 5.7e-04 / 5.2e-04 / 7.4e-05 | 2.0e-01 / 1.6e-01 / 1.5e-02 | 1.0e-06 / 6.6e-07 / 1.1e-07 |

## deep ITM (100 points)

| Grid | Price | Delta | Gamma | Vega |
|---|---|---|---|---|
| 100 x 50, 2 steps/day | 3.7e-05 / 3.1e-05 / 4.3e-06 | 3.1e-03 / 2.5e-03 / 2.8e-04 | 2.3e-01 / 1.7e-01 / 1.4e-02 | 9.3e-06 / 8.7e-06 / 1.0e-06 |
| 200 x 100, 4 steps/day | 9.2e-06 / 6.6e-06 / 9.1e-07 | 1.1e-03 / 9.6e-04 / 7.6e-05 | 1.4e-01 / 1.3e-01 / 8.6e-03 | 5.6e-06 / 2.5e-06 / 2.8e-07 |
| 400 x 200, 8 steps/day (used) | 1.7e-06 / 1.2e-06 / 1.5e-07 | 3.2e-04 / 2.9e-04 / 1.6e-05 | 1.1e-01 / 5.3e-02 / 3.8e-03 | 1.1e-06 / 5.2e-07 / 5.5e-08 |

## deep OTM (100 points)

| Grid | Price | Delta | Gamma | Vega |
|---|---|---|---|---|
| 100 x 50, 2 steps/day | 2.5e-05 / 2.4e-05 / 3.4e-06 | 2.3e-04 / 2.3e-04 / 7.2e-05 | 9.9e-04 / 9.4e-04 / 2.0e-04 | 1.5e-06 / 1.5e-06 / 3.7e-07 |
| 200 x 100, 4 steps/day | 6.8e-06 / 6.5e-06 / 8.7e-07 | 5.1e-05 / 5.0e-05 / 1.6e-05 | 2.5e-04 / 2.4e-04 / 5.0e-05 | 3.7e-07 / 3.7e-07 / 9.3e-08 |
| 400 x 200, 8 steps/day (used) | 1.3e-06 / 1.2e-06 / 1.7e-07 | 1.1e-05 / 1.0e-05 / 3.3e-06 | 4.9e-05 / 4.6e-05 / 9.7e-06 | 7.3e-08 / 7.2e-08 / 1.8e-08 |

