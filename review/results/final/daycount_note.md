# Sven item 2: rule re-tuning, identical paths, and the day-count bug

## Checks
- **Rule (b) was re-tuned after the fix.** On the adi_trading validation paths (seed 2025), same grid as before
  (`review/warmstart/richer_rule.py --pricing_backend adi_trading --retune_rule_b`, files
  `review/results/warmstart/richer_rule_params_adi_trading_<cost>.json`). Result (a, b, w) = 0.4, 0.4, 0.75 /
  0.3, 0.3, 0.5 / 0.2, 0.3, 1.0 (validation VaR95 24.69 / 27.64 / 32.66). Only the 0.5% setting changed
  (old 0.3, 0.3, 0.3). The richer rule was tuned on the same validation paths.
- **All methods used identical test paths.** Every final number uses one market: seed 4242, 8,192 paths, cache
  `Heston_8192_seed4242_4dd79463ae85` (adi_trading). The baselines are rolled out on it in `final_table.py`; the
  agents' stored test P&L comes from `train_d4pg.py` with the same flags. Re-running the four reported agents
  (selected and median seeds) on the baselines' market object reproduces their stored test P&L exactly (max
  difference 0.0 on all 8,192 paths). The QuantLib-clock and adi_trading markets of seed 4242 have bit-identical
  S and V paths; only option prices and Greeks differ.

## Paragraph for the response letter (draft)

While re-implementing the pricing, we found an inconsistency in the time convention of the option pricer. The
simulated paths use a trading-day clock (one step = 1/252 year). The QuantLib pricer, however, converted a remaining
maturity of τ trading days into an exercise date of round(τ · 365/252) calendar days and priced it with an
Actual/365 day count. The priced maturity is therefore round(τ · 365/252)/365 years instead of τ/252 years. The two
agree up to day rounding for long maturities, but not for the last days before expiry: one trading day is priced as
1/365 instead of 1/252 year (31% less time), three trading days as 4/365 instead of 3/252 (8% less), and two or four
trading days as about 3.5% more. In our set-up, the initial option and the first hedge options expire at the end of the
episode, so the error matters in the last days of each episode, where Gamma is largest. This convention was also in
the code of the original submission, so the original numbers are affected. All strategies in the original paper were
evaluated on the same simulated market, so the comparisons there were internally consistent, but the levels were
computed with this inconsistency. The revised paper uses only new numbers: all prices and Greeks now come from an ADI
table with maturity τ/252, consistent with the paths, and all rules were re-tuned and all agents retrained on this
market. On the same test paths, the fix changes the VaR95 of the baselines by at most 0.3 (DG +0.23 / +0.17 / +0.01,
rule (b) −0.27 / −0.07 / −0.08 at 0.5 / 1 / 2% cost, `review/results/final/daycount_effect.csv`), which is within one
standard error. The effect on learning is larger: on validation paths, the gain of the pre-commitment agent over
rule (b) was +0.10 / +0.69 / +1.91 with the old convention and +1.45 / +1.83 / +2.13 with the consistent clock
(3 seeds, `review/results/speed/daycount.md`). We do not report these old-clock agent numbers in the paper.
