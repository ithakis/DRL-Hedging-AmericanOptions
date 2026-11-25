# Pricing-table convergence check (QuantLib FdHestonVanillaEngine, Modified Craig-Sneyd)

200 random off-grid points (S/K, days to maturity, vol), 50 per region, vol in [0.12, 0.6]. Reference: direct QuantLib solve with a 400-point engine (400 x 400 x 200, same vega bump). Errors per unit strike, max / mean abs. Largest |value|: price 0.441, delta 1, gamma 21.5, vega 0.00174. Solves took 1099s.

## all (200 points)

| Against the 400-point solve | price | delta | gamma | vega |
|---|---|---|---|---|
| Direct QuantLib, 50-point engine (the table's engine) | 1.2e-04 / 1.5e-05 | 1.9e-03 / 2.4e-04 | 4.4e-01 / 1.4e-02 | 1.2e-05 / 8.2e-07 |
| Direct QuantLib, 100-point engine ('high' accuracy) | 2.3e-05 / 2.5e-06 | 3.1e-04 / 2.8e-05 | 3.9e-01 / 5.9e-03 | 3.2e-06 / 1.6e-07 |
| Direct QuantLib, 200-point engine | 6.1e-06 / 6.2e-07 | 2.3e-04 / 8.2e-06 | 2.8e-01 / 2.3e-03 | 1.3e-06 / 4.2e-08 |
| Table (50-point engine, bicubic interpolation): total error | 1.2e-04 / 1.5e-05 | 2.0e-03 / 2.5e-04 | 4.3e-01 / 1.6e-02 | 9.6e-06 / 7.3e-07 |
| Single-sweep ADI table, 400 x 200, 8 steps/day (independent check) | 8.6e-06 / 6.6e-07 | 1.5e-03 / 5.1e-05 | 3.5e-01 / 7.9e-03 | 7.0e-06 / 1.3e-07 |
| Interpolation only (table vs direct 50-point) | 5.4e-06 / 2.6e-07 | 7.5e-04 / 2.8e-05 | 4.0e-01 / 7.4e-03 | 1.3e-05 / 2.5e-07 |

## bulk (50 points)

| Against the 400-point solve | price | delta | gamma | vega |
|---|---|---|---|---|
| Direct QuantLib, 50-point engine (the table's engine) | 8.6e-05 / 1.8e-05 | 1.5e-03 / 2.1e-04 | 1.3e-01 / 6.5e-03 | 3.3e-06 / 8.0e-07 |
| Direct QuantLib, 100-point engine ('high' accuracy) | 2.3e-05 / 5.8e-06 | 1.1e-04 / 4.8e-05 | 1.0e-01 / 2.8e-03 | 1.0e-06 / 2.4e-07 |
| Direct QuantLib, 200-point engine | 6.1e-06 / 1.5e-06 | 7.6e-05 / 1.3e-05 | 7.1e-02 / 1.7e-03 | 2.4e-07 / 6.2e-08 |
| Table (50-point engine, bicubic interpolation): total error | 8.6e-05 / 1.8e-05 | 1.5e-03 / 2.0e-04 | 8.6e-02 / 6.9e-03 | 2.3e-06 / 6.8e-07 |
| Single-sweep ADI table, 400 x 200, 8 steps/day (independent check) | 5.2e-06 / 5.8e-07 | 1.8e-04 / 2.6e-05 | 7.2e-02 / 3.1e-03 | 1.1e-06 / 7.7e-08 |
| Interpolation only (table vs direct 50-point) | 2.4e-06 / 1.8e-07 | 6.0e-04 / 2.7e-05 | 1.2e-01 / 5.9e-03 | 4.1e-06 / 1.9e-07 |

## near maturity (50 points)

| Against the 400-point solve | price | delta | gamma | vega |
|---|---|---|---|---|
| Direct QuantLib, 50-point engine (the table's engine) | 1.2e-04 / 3.0e-05 | 1.9e-03 / 6.2e-04 | 3.1e-01 / 3.1e-02 | 1.2e-05 / 1.2e-06 |
| Direct QuantLib, 100-point engine ('high' accuracy) | 3.8e-06 / 9.9e-07 | 1.8e-04 / 3.4e-05 | 1.7e-01 / 6.6e-03 | 3.2e-06 / 1.4e-07 |
| Direct QuantLib, 200-point engine | 6.2e-07 / 2.1e-07 | 3.2e-05 / 7.3e-06 | 5.2e-03 / 7.3e-04 | 7.1e-07 / 3.3e-08 |
| Table (50-point engine, bicubic interpolation): total error | 1.2e-04 / 3.0e-05 | 2.0e-03 / 6.5e-04 | 4.0e-01 / 3.8e-02 | 6.8e-06 / 9.3e-07 |
| Single-sweep ADI table, 400 x 200, 8 steps/day (independent check) | 4.6e-06 / 1.4e-06 | 1.5e-03 / 1.2e-04 | 2.9e-01 / 1.6e-02 | 9.3e-07 / 1.2e-07 |
| Interpolation only (table vs direct 50-point) | 3.8e-06 / 3.2e-07 | 7.5e-04 / 3.7e-05 | 4.0e-01 / 1.8e-02 | 1.3e-05 / 3.7e-07 |

## deep ITM (50 points)

| Against the 400-point solve | price | delta | gamma | vega |
|---|---|---|---|---|
| Direct QuantLib, 50-point engine (the table's engine) | 6.0e-05 / 4.7e-06 | 1.7e-03 / 1.2e-04 | 4.4e-01 / 1.8e-02 | 1.0e-05 / 7.4e-07 |
| Direct QuantLib, 100-point engine ('high' accuracy) | 1.4e-05 / 1.4e-06 | 3.1e-04 / 2.0e-05 | 3.9e-01 / 1.4e-02 | 2.3e-06 / 1.4e-07 |
| Direct QuantLib, 200-point engine | 4.1e-06 / 3.6e-07 | 2.3e-04 / 1.1e-05 | 2.8e-01 / 6.7e-03 | 1.3e-06 / 4.3e-08 |
| Table (50-point engine, bicubic interpolation): total error | 5.9e-05 / 4.8e-06 | 1.6e-03 / 1.4e-04 | 4.3e-01 / 1.7e-02 | 9.6e-06 / 7.9e-07 |
| Single-sweep ADI table, 400 x 200, 8 steps/day (independent check) | 8.6e-06 / 5.8e-07 | 8.4e-04 / 5.0e-05 | 3.5e-01 / 1.2e-02 | 7.0e-06 / 3.2e-07 |
| Interpolation only (table vs direct 50-point) | 5.4e-06 / 5.1e-07 | 6.5e-04 / 4.6e-05 | 6.0e-02 / 5.2e-03 | 6.1e-06 / 4.3e-07 |

## deep OTM (50 points)

| Against the 400-point solve | price | delta | gamma | vega |
|---|---|---|---|---|
| Direct QuantLib, 50-point engine (the table's engine) | 3.6e-05 / 8.9e-06 | 1.8e-04 / 3.1e-05 | 2.4e-03 / 3.6e-04 | 1.8e-06 / 5.5e-07 |
| Direct QuantLib, 100-point engine ('high' accuracy) | 7.8e-06 / 1.9e-06 | 5.5e-05 / 8.3e-06 | 4.7e-04 / 7.3e-05 | 4.2e-07 / 1.3e-07 |
| Direct QuantLib, 200-point engine | 1.4e-06 / 3.6e-07 | 1.3e-05 / 1.8e-06 | 1.3e-04 / 1.9e-05 | 1.1e-07 / 3.0e-08 |
| Table (50-point engine, bicubic interpolation): total error | 3.6e-05 / 8.9e-06 | 1.8e-04 / 3.1e-05 | 1.4e-03 / 2.8e-04 | 1.8e-06 / 5.3e-07 |
| Single-sweep ADI table, 400 x 200, 8 steps/day (independent check) | 9.5e-07 / 1.1e-07 | 2.0e-05 / 4.5e-06 | 3.4e-05 / 6.5e-06 | 6.2e-08 / 1.4e-08 |
| Interpolation only (table vs direct 50-point) | 2.8e-07 / 3.8e-08 | 8.6e-06 / 1.5e-06 | 2.4e-03 / 2.3e-04 | 5.7e-07 / 2.9e-08 |

