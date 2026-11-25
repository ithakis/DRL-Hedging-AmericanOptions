# Day-count check (validation only): P3 + ATM fix, QuantLib table vs ADI table with maturity tau/252

Current table: QuantLib table (maturity round(tau*365/252) calendar days / 365), runs P3_atmfix.
adi_trading: ADI table 400x200, maturity tau/252 (the paths' clock), new markets built with it, runs P3_daycount.
Both: P3 config + --kernel_atm_fix, same absolute b (14.75 / 16.72 / 19.68), 12,000 episodes.
Validation VaR95 on 8,192 validation paths (seed 2025; the adi_trading market has its own option prices).
Gain = own market's rule (b) val VaR95 - best val VaR95; positive = RL better. No test-set numbers.

| Cost | Seed | Rule (b) val, table | Rule (b) val, adi_trading | P3_atmfix best val (ep) | P3_daycount best val (ep) | Gain, table | Gain, adi_trading |
|---|---|---|---|---|---|---|---|
| 0.5% | 1 | 24.59 | 24.86 | 24.23 (9008) | 23.46 (11008) | +0.36 | +1.41 |
| 0.5% | 2 | 24.59 | 24.90 | 24.59 (1008) | 23.45 (9008) | +0.00 | +1.45 |
| 0.5% | 3 | 24.59 | 24.85 | 24.66 (1008) | 23.35 (9008) | -0.07 | +1.49 |
| 1.0% | 1 | 27.87 | 27.63 | 26.76 (11504) | 25.80 (6000) | +1.11 | +1.83 |
| 1.0% | 2 | 27.87 | 27.63 | 27.39 (11504) | 25.89 (8000) | +0.48 | +1.74 |
| 1.0% | 3 | 27.87 | 27.60 | 27.38 (9008) | 25.68 (11504) | +0.49 | +1.92 |
| 2.0% | 1 | 32.80 | 32.69 | 30.58 (9008) | 30.83 (7008) | +2.22 | +1.86 |
| 2.0% | 2 | 32.80 | 32.68 | 31.00 (12000) | 30.17 (10000) | +1.80 | +2.51 |
| 2.0% | 3 | 32.80 | 32.65 | 31.07 (11504) | 30.62 (7008) | +1.73 | +2.03 |

| Cost | Mean gain, table | Mean gain, adi_trading | n |
|---|---|---|---|
| 0.5% | +0.10 (0.23) | +1.45 (0.04) | 3 |
| 1.0% | +0.69 (0.36) | +1.83 (0.09) | 3 |
| 2.0% | +1.91 (0.27) | +2.13 (0.34) | 3 |

(SD over seeds in brackets.)

Caveats
- Rule (b) on the adi_trading market uses the (a, b, w) tuned on the current table's market; it was not re-tuned
  for the new market, so its validation VaR95 may be a little worse than a re-tuned rule and the gain a little larger.
- P3's b was kept at the same absolute values; relative to the adi_trading rule VaR it is 0.59-0.61 x.
- s / 1000 episodes 74-123 (up to 8 runs in parallel on 8 cores).

## Rule (b) re-tuned on the adi_trading validation market (`daycount_rule_b.json`, `retune_rule_daycount.py`)

Same grid and selection as the paper baseline (a, b in 0..1 step 0.1, w in 9 values; min validation VaR95).

| Cost | Old (a, b, w) | Val VaR95, old params | Re-tuned (a, b, w) | Val VaR95, re-tuned | P3_daycount best val, seeds 1/2/3 | Gain vs re-tuned rule, seeds 1/2/3 | Mean (SD) |
|---|---|---|---|---|---|---|---|
| 0.5% | 0.3, 0.3, 0.3 | 24.84 | 0.4, 0.4, 0.75 | 24.69 | 23.46 / 23.45 / 23.35 | +1.23 / +1.24 / +1.33 | +1.27 (0.06) |
| 1.0% | 0.3, 0.3, 0.5 | 27.64 | 0.3, 0.3, 0.5 | 27.64 | 25.80 / 25.89 / 25.68 | +1.84 / +1.75 / +1.96 | +1.85 (0.11) |
| 2.0% | 0.2, 0.3, 1.0 | 32.66 | 0.2, 0.3, 1.0 | 32.66 | 30.83 / 30.17 / 30.62 | +1.82 / +2.49 / +2.04 | +2.12 (0.34) |

Old-params values differ from the episode-0 lines by up to 0.06: the episode-0 agent is the warm-started actor
(residual output not exactly 0), not the pure rule.
