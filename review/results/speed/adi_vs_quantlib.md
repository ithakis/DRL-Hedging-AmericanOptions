# Single-sweep ADI table vs QuantLib table

Parameters: {"kappa": 1.0, "theta": 0.09, "volvol": 0.3, "rho": -0.7, "r": 0.02, "q": 0.0, "TradingDaysPerYear": 252, "max_ttm": 60, "m_min": 0.5, "m_max": 2.0, "n_m": 81, "vol_min": 0.02, "vol_max": 1.0, "n_vol": 25, "N_gridpoints": 50, "vega_mode": "central", "vega_h": 0.005, "eps_iv": 0.02}

## Build time

| Table | Seconds |
|---|---|
| QuantLib (364,500 single-point solves, 4 cores) | about 840 (14 min, logged when built) |
| ADI coarse (100 x 50, 2 steps/day, 3 sweeps, 1 core) | 0.3 |
| ADI medium (200 x 100, 4 steps/day, 3 sweeps, 1 core) | 1.8 |
| ADI fine (400 x 200, 8 steps/day, 3 sweeps, 1 core) | 12.6 |

## Error vs the fine ADI grid (max / mean abs, per unit strike, at all table nodes)

### QuantLib table

| Region | price | delta | gamma | vega |
|---|---|---|---|---|
| all | 4.2e-03 / 1.1e-04 | 1.3e-01 / 7.6e-04 | 1.2e+02 / 6.2e-02 | 1.4e-04 / 3.7e-06 |
| bulk (m 0.75-1.3, vol 0.09-0.6) | 2.3e-03 / 9.1e-05 | 5.4e-02 / 9.6e-04 | 1.4e+01 / 5.9e-02 | 7.3e-05 / 4.6e-06 |
| near maturity (1-3 d, m 0.9-1.1) | 2.3e-03 / 3.6e-04 | 5.4e-02 / 9.3e-03 | 1.4e+01 / 7.9e-01 | 5.3e-05 / 1.7e-05 |
| deep ITM (m < 0.75) | 1.2e-04 / 1.6e-06 | 2.4e-03 / 6.4e-05 | 3.5e-01 / 6.4e-03 | 4.1e-05 / 3.8e-07 |

### ADI coarse

| Region | price | delta | gamma | vega |
|---|---|---|---|---|
| all | 7.9e-04 / 2.2e-05 | 2.6e-01 / 6.3e-04 | 1.7e+02 / 9.0e-02 | 1.6e-04 / 1.8e-06 |
| bulk (m 0.75-1.3, vol 0.09-0.6) | 1.7e-04 / 1.8e-05 | 7.1e-02 / 5.7e-04 | 1.1e+01 / 4.6e-02 | 8.0e-05 / 1.7e-06 |
| near maturity (1-3 d, m 0.9-1.1) | 1.3e-04 / 2.7e-05 | 7.1e-02 / 3.0e-03 | 1.1e+01 / 2.3e-01 | 4.3e-05 / 2.5e-06 |
| deep ITM (m < 0.75) | 3.8e-05 / 9.6e-07 | 4.0e-03 / 8.8e-05 | 3.4e-01 / 4.7e-03 | 1.7e-05 / 3.7e-07 |

### ADI medium

| Region | price | delta | gamma | vega |
|---|---|---|---|---|
| all | 2.2e-04 / 5.9e-06 | 1.1e-01 / 1.4e-04 | 6.1e+01 / 3.8e-02 | 6.9e-05 / 4.7e-07 |
| bulk (m 0.75-1.3, vol 0.09-0.6) | 8.1e-05 / 5.8e-06 | 8.6e-03 / 8.6e-05 | 3.4e+00 / 1.5e-02 | 2.9e-05 / 4.1e-07 |
| near maturity (1-3 d, m 0.9-1.1) | 8.1e-05 / 1.2e-05 | 8.6e-03 / 3.2e-04 | 3.4e+00 / 8.7e-02 | 1.1e-05 / 5.7e-07 |
| deep ITM (m < 0.75) | 7.9e-06 / 1.7e-07 | 1.4e-03 / 1.8e-05 | 2.1e-01 / 1.8e-03 | 8.8e-06 / 9.0e-08 |

## Error vs direct QuantLib solves with a 200-point engine (150 random nodes, 254s)

| Table | price max / mean | delta max / mean | gamma max / mean | vega max / mean |
|---|---|---|---|---|
| QuantLib table (50-point engine) | 9.0e-05 / 2.0e-05 | 2.3e-03 / 2.1e-04 | 3.0e-01 / 1.1e-02 | 7.7e-06 / 8.3e-07 |
| ADI medium | 2.9e-03 / 1.5e-04 | 3.2e-02 / 8.7e-04 | 2.3e+00 / 4.4e-02 | 4.5e-05 / 4.7e-06 |
| ADI fine | 2.9e-03 / 1.5e-04 | 3.2e-02 / 8.5e-04 | 2.2e+00 / 3.7e-02 | 4.6e-05 / 4.7e-06 |
