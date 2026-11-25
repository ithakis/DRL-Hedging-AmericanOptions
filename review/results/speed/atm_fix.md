# E3a seed 1: speed branch, with and without --kernel_atm_fix

Test paths seed 654 (8,192). Gain vs rule (b) = rule (b) - RL, paired bootstrap SE; positive = RL better.

| Run | Cost | Sel. ep | Val VaR95 | Test VaR95 | Rule (b) test | Gain vs rule (b) | s / 1000 ep | timing (s) |
|---|---|---|---|---|---|---|---|---|
| E3a (tuning thread, old code) | 0.5% | 11504 | 24.59 | 24.50 (0.50) | 24.87 | 0.37 (0.33) | 334 |  |
| E3a (tuning thread, old code) | 1.0% | 1008 | 27.82 | 27.30 (0.46) | 27.43 | 0.13 (0.11) | 337 |  |
| E3a (tuning thread, old code) | 2.0% | 1008 | 32.85 | 32.47 (0.49) | 32.56 | 0.09 (0.12) | 332 |  |
| E3a speed branch | 0.5% | 11504 | 24.51 | 24.46 (0.50) | 24.87 | 0.41 (0.34) | 136 | env 7, kernel 148, learn 1465, validation 29 |
| E3a speed branch | 1.0% | 1008 | 27.82 | 27.30 (0.46) | 27.43 | 0.13 (0.11) | 136 | env 7, kernel 150, learn 1466, validation 29 |
| E3a speed branch | 2.0% | 1008 | 32.85 | 32.47 (0.49) | 32.56 | 0.09 (0.12) | 132 | env 7, kernel 142, learn 1432, validation 29 |
| E3a speed + atm_fix | 0.5% | 11504 | 24.52 | 24.71 (0.48) | 24.87 | 0.17 (0.33) | 135 | env 7, kernel 148, learn 1461, validation 31 |
| E3a speed + atm_fix | 1.0% | 10512 | 27.73 | 27.87 (0.40) | 27.43 | -0.44 (0.25) | 135 | env 7, kernel 148, learn 1461, validation 30 |
| E3a speed + atm_fix | 2.0% | 1008 | 32.85 | 32.48 (0.49) | 32.56 | 0.08 (0.12) | 134 | env 7, kernel 143, learn 1454, validation 29 |

Notes
- Speed-branch runs: commit f25968a (sorted loss on), 12,000 episodes, 6 runs in parallel on the 8-core Mac.
  s / 1000 ep (132-136) is with 6 parallel runs; the old-code E3a runs (332-337) were also 6 in parallel,
  so the like-for-like speed-up is about 2.5x. A single run alone is 79 s / 1000 ep (timing.md).
- Speed branch vs old code: same selected episodes; test VaR95 identical at 1% and 2%, 24.46 vs 24.50 at 0.5%
  (float rounding in the loss carried through 11.5k episodes of training).
- ATM fix: no gain. 0.5%: gain 0.17 vs 0.41; 1%: -0.44 (0.25) vs 0.13 (0.11); 2%: same (selected ep 1008 in both).
  Seed 1 only, so this is weak evidence, but nothing here supports turning it on.
- At 1% and 2% the selected agent is episode 1008, i.e. just after the actor is unfrozen (close to the rule).
