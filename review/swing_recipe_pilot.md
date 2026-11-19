# Pilot: swing-options D4PG recipe on the hedging task (2026-11-19)

Question: does the "v64" training recipe of DRL-Swing-Options close the gap between the RL agent and
the tuned banded rule (Heston under P, r = 2%, mu = 3%, lambda_v = 0, VaR95 agents, block B of the retrain)?

## What was run
* The swing agent has a scalar (mean) critic, so it cannot optimise VaR. We keep the hedging quantile
  critic and VaR95 actor objective and take everything else from the swing v64 defaults
  (see `torch_d4pg/agent_swing.py`): 3x64 SiLU/LayerNorm nets, beta-sigmoid output, AdamW 3e-4/6e-4,
  Polyak tau 0.0032, 1-step TD, batch 128, 2 gradient steps every 2 env steps, 512-episode critic
  warm-up, linear pre-squash noise 1.3 -> 0.26, EMA actor for evaluation.
* Same markets and split as the retrain: train seed 1234 (40,000 paths), validation seed 2025 (8,192),
  test seed 654 (8,192). Validation every 1,000 episodes, best checkpoint kept, only that one tested.
* 2 agent seeds x 3 costs, 40,000 episodes, 6 runs in parallel on the M1 Air (1 thread each).

## Test VaR95 (bootstrap SE), all new-pipeline numbers

| Cost | Swing recipe seed 1 | Swing recipe seed 2 | Retrain RL (selected of 10) | Delta-Gamma | Tuned rule (b) |
|---|---|---|---|---|---|
| 0.5% | 27.33 (0.46) | 27.69 (0.58) | 26.25 (0.39) | 30.97 (0.59) | 24.87 (0.53) |
| 1.0% | 32.21 (0.61) | 31.04 (0.40) | 29.95 (0.51) | 34.53 (0.60) | 27.43 (0.44) |
| 2.0% | 38.75 (0.63) | 37.49 (0.57) | 35.13 (0.55) | 41.37 (0.64) | 32.56 (0.50) |

Selected episodes: 31k / 16k (0.5%), 4k / 5k (1%), 7k / 4k (2%): at the higher costs validation VaR
gets worse after the first few thousand episodes. Mean training time 10,850 s per run (about 3 h with
6 in parallel), vs about 6,400 s for the retrain recipe.

## Conclusion
The swing recipe does not beat the tuned rule at any cost, and it is also worse than the retrain
recipe (by 1-3.6 VaR points). It still beats Delta-Gamma. Hypothesis not supported; no full 10-seed run.
