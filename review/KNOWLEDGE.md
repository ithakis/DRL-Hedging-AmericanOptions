# DRL hedging of American puts under Heston: what we tried and what we learned

Journal of the 2026 revision work (submission 4842): retraining, baselines, the final agent, and the search for a larger
gain. In the style of `DRL-Swing-Options/HPT.md`. All numbers are VaR95 of the episode loss (lower is better) on 8,192
paths, at option costs 0.5% / 1% / 2%, unless stated. "Gain" = rule (b) VaR95 − agent VaR95 (positive = agent better).

**Never mix numbers across market families.** Three families exist: (1) the old pricing table with test seed 654 (all
exploration up to 2026-11-25), (2) the `adi_trading` market (consistent τ/252 clock) with fresh test seed 4242 (the
FINAL paper numbers), (3) validation-only screens (seed 2025, and seed 2026 as a second validation set). A second fresh
test seed is still unused for the v2 numbers.

---

## 1. Quick reference

### 1.1 Problem
- Book of American puts (ATM arrivals each day, net count Skellam(0.5, 0.5)), Heston under P (r = 2%, μ = 3%, κ = 1,
  θ = 0.09, ξ = 0.3, ρ = −0.7, initial vol 0.3), 30 trading days per episode.
- One action per day: the fraction of the gamma/vega-neutralising trade in the new ATM hedge option; delta hedged
  automatically. The agent can not unwind older hedges.
- Proportional option cost (spread × |contracts| × price). Ex-post stock (delta) costs 0–5 bp only in robustness.

### 1.2 FINAL method (PR #8, paper numbers)
| Piece | Setting |
|---|---|
| Objective | Pre-commitment (Rockafellar–Uryasev / Bäuerle–Ott): state has accumulated P&L W_t; reward g(W_{t+1}) − g(W_t), g(W) = min(W + b, 0); critic objective = mean; b = 0.6 × rule (b) validation VaR95 |
| Warm start | Residual on tuned rule (b): f = clip(f_rule + 0.5 u, 0, 1); rule's no-trade band kept in the env (`--band w`); 1,000-episode critic-only warm-up |
| Critic target | Expected (kernel) target: 9-node quadrature of the simulator's own QE step (`--kernel_mx 3`), liability/hedge Greeks re-priced by the table, ATM fix for t+1 arrivals (`--kernel_atm_fix`) |
| Learner | D4PG-QR, 99 quantiles, batch 256, samples_per_insert 32, 1-step, 16 parallel paths |
| Nets | 3 × 64 swish-β3 + LayerNorm (actor and critic); LR critic 6e-4 / actor 3e-4 |
| Stability | Polyak τ 0.003, eval-only EMA 0.999, pre-squash noise 1.0 → 0.2 |
| State | S, Γ, γʰ, V, νʰ, time left, vol, W_t |
| Budget | 20k episodes, validation every 500, 10 seeds; checkpoint and seed chosen on validation |
| Pricing | ADI Heston American-put table, maturity τ/252 (`--pricing_backend adi_trading`) |

Reproduce: `review/warmstart/run_final.sh`, `final_table.py`, `robustness_final.py` (see PR #8).

### 1.3 FINAL result (fresh test seed 4242)
| Cost | DG | Rule (b) | Richer rule | DRL selected | DRL median | Gain vs rule (b) |
|---|---|---|---|---|---|---|
| 0.5% | 31.38 | 24.44 | 24.16 | 23.16 | 23.08 | +1.29 (0.30) |
| 1% | 34.85 | 27.54 | 27.56 | 25.35 | 25.88 | +2.19 (0.35) |
| 2% | 41.87 | 32.94 | 32.60 | 31.15 | 30.93 | +1.79 (0.40) |

All 10 seeds beat rule (b) at every cost. Robustness: wins 25/30 bump rows; loses at initial vol 0.15 and 0.45
(0.45 outside training) and vol-of-vol 0.45 at 2%. Delta costs 0–5 bp: gain stays +0.88 to +2.19. CVaR95 about equal
to the rule's.

---

## 2. Key concepts (why the final design looks like this)

1. **Time-consistent per-day VaR over-trades.** PR #3 agents maximised the 5% quantile of the per-step critic
   distribution (nested / step-wise). That is time-consistent but much more risk-averse than the episode VaR, and the
   actor gradient sits on one of 99 quantiles. Agents traded 20–25% more than the rule at 2% and lost to it.
2. **Pre-commitment fixes the objective.** Putting W_t in the state and using the hinge on the episode loss makes the
   objective a plain expectation, optimised by a mean critic. It is the step that flipped the sign of the result
   (2026-11-24). It is time-inconsistent by design; both metrics are reported (terminal VaR95 and step-wise SW-VaR95).
3. **Warm start from the rule.** The optimal policy has a no-trade band, which is hard to learn from scratch. A
   residual on rule (b) plus the band in the env starts the agent at the rule and makes improvements local.
4. **Expected target.** Averaging the TD target over the one-day Heston transition (quadrature of the QE step) removes
   most target noise, as in the swing work (H1). Needed the ATM fix (t+1 arrivals struck at the node, not the
   realised S_{t+1}) to remove a leak.
5. **Same clock for paths and prices.** The old table priced with calendar-day maturities while paths used trading days;
   `adi_trading` fixed this. Rule (b) was re-tuned on that market before the FINAL run.
6. **Selection is noisy.** One validation VaR95 on 8,192 paths has SE ≈ 0.4, larger than most checkpoint-to-checkpoint
   differences late in training. The selected seed is optimistically biased at 2% (val 30.49 vs test 31.14).

---

## 3. History

### 3.1 Retrain and baselines (PR #1–#3, 2026-11-16 to -19)
- PyTorch D4PG-QR port of the original Acme agent (PR #2); Heston pricing table; cost-aware rule baselines (PR #1).
- Rule (b) = banded fractional hedge x = clip(a n_G + b n_V), no trade if |x| < w; tuned on validation. Under Heston it
  beat the retrained step-wise agents at all costs (agent got 65–77% of the rule's gain over DG).

### 3.2 Swing recipe port (PR #4, 2026-11-19): negative
- Ported DRL-Swing-Options v64 (3 × 64 SiLU/LN, β-sigmoid, AdamW, Polyak, 1-step TD, critic warm-up, EMA actor) but
  kept the quantile critic + per-day VaR. Test 27.3/31.0–32.2/37.5–38.8: worse than PR #3 and the rule.
- Why it failed: per-day VaR objective (over-trading), no warm start, sampled targets. Most of the recipe was later
  adopted once the objective and warm start were fixed (nets, EMA, noise schedule, expected target).

### 3.3 Warm start, gate, batches A–F (PR #8, 2026-11-19 to -24; old table, test 654)
| Step | Result |
|---|---|
| Residual on rule (b), seed 1 | 24.63/28.45/34.24 vs rule 24.87/27.43/32.56: better than PR #3, not the rule |
| Gate (trade rule or not) | −0.20/−0.21/−0.38; 0/108 checkpoints beat the rule on validation |
| A1 swing-like stability (16 envs, τ .005, EMA, reward scale) | steadier, worse (29.0/32.3/39.7) |
| A2 +4× updates | worse, 2× slower |
| B1 extra state (time left, vol) | 25.11/28.75/34.01, kept |
| K9 density target (`--kernel_mx 3`) | same level, best ckpt at 5–10k eps, ~5× slower per episode |
| 80k episodes | small gain |
| SBres / SBband / KBresband | residual + band: first validation checkpoints beating the rule |
| CVaR / tail-quantile objectives | hurt |
| E1a small swish-β3 nets | same accuracy, ~30% faster: kept |
| E1b +4× updates | worse (second time) |
| E2 EMA .999, τ .003, pre-squash 1.0→0.2 | small gain, kept |
| E3b band 1.5w | worse |
| F1–F6 (actor at current state, rule inputs, VaR smoothing, noise, dQ/da clip 10, residual 0.2) | none fixed over-trading at 2% |
| **P2/P3 pre-commitment (b = 0.75 / 0.6 × rule VaR)** | **first robust wins: +0.30/+0.56/+0.93 (P2 seed 1), beats rule on both terminal and step-wise metric** |

### 3.4 Speed (PR #9, #11)
- Vectorised bicubic table lookup; one table pass per book in the kernel; numeric-change flag `--kernel_atm_fix`.
- Python 3.14 + uv (PR #11): evaluation bit-identical, training ~30% faster, not bit-identical (different torch).
  Controls must be rerun in the same env (env-to-env difference seen: +0.23 / −0.24 on 3 seeds at 8k).
- Numba ADI pricer: separate thread.

### 3.5 FINAL (2026-11-25): see 1.2–1.3. Hyperparameter screens on FINAL (half updates, half LR, less noise, batch 512,
LR 1e-4): no winner (`review/results/speed/screens.md`).

### 3.6 Improvement search (2026-11-25/26, this thread; validation only)
Protocol: FINAL flags + one change, 8k episodes, seeds 1–3, costs 1% and 2%, paired by seed against a control in the
same env. Score = mean of the last 4 validation checks (less selection noise than the best). Paired SE over seeds
≈ 0.05–0.4. Files: `review/improve/` (arms.py, run_screen.sh, report.py), results `review/results/improve/`.

| Arm | Change | Gain 1% | Gain 2% | Verdict |
|---|---|---|---|---|
| Seed ensemble | average residual of the 10 FINAL actors, eval only | +0.28 (val 2026) | +0.35 | kept (also +0.05 at 0.5%); unbiased check on 2nd validation set |
| b45 / b30 | b = 0.45 / 0.3 × rule VaR | −0.15 / −1.27 | −0.57 / −2.11 | rejected: b = 0.6 near optimal |
| rin | rule inputs n_G, n_V and rule fraction in state | +0.04 | −0.01 | neutral |
| band05 | band 0.5w | −0.36 | −0.17 | rejected |
| upd2 | 2× updates | +0.06 | −0.34 | rejected (third time) |
| mc | scalar mean critic + expected target (swing) | −0.36 | −0.32 | rejected: the quantile critic is better here |
| kap1 | softplus hinge κ = 1 | −0.28 | +0.32 | mixed |
| lam05 / lam15 | hinge + λW | −0.45 / −1.10 | −0.43 / −1.34 | rejected: cuts cost a lot, raises VaR |
| aoc | actor at current state (trains t = 0, no terminal states) | 0.00 | +0.16 | small + |
| dqn | dQ/da / EMA batch mean, clip 3 | +0.04 | +0.14 | small + |
| sq05 | output tanh(z/2) | +0.05 | +0.09 | small + |
| stk | aoc + dqn + sq05 | +0.04 | +0.14 | small + |
| **arr2** | **kernel averages over net arrivals k ∈ {−2..2} (Skellam)** | **+0.15** | **+0.12** | **kept; CVaR95 −1.0 at both costs; ~5–7× slower with the quantile critic** |
| allb | one policy for all b (y = W + b in state, b random) | −0.36 | +0.11 | rejected |
| ramp | hinge floored at 1.25 × rule VaR | +0.29 | −0.28 | rejected: CVaR95 +4.6 |
| **rv0** | **training paths start at log-uniform initial vol 0.12–0.50** | −0.25 | −0.03 | **kept for robustness** |

Robustness screen (validation seed 2025, bumps not in the test grid; gain vs rule (b), 3 seeds):

| Bump | Control | rv0 | arr2 |
|---|---|---|---|
| base | +1.91 / +1.95 | +1.67 / +1.82 | +1.98 / +2.09 |
| init vol 0.18 | −0.33 / +0.02 | +2.00 / +1.19 | +1.25 / +0.38 |
| init vol 0.42 | +0.95 / +1.36 | +2.01 / +2.62 | +1.58 / +2.59 |
| vol-of-vol 0.38 | +1.26 / +0.85 | +1.30 / +0.60 | +1.84 / +1.10 |

Candidate for v2: arr2 + stk (+ rv0), 20k-episode check against a 20k control at 3 costs (running 2026-11-26).

---

## 4. Lessons learned
| Attempt | Result | Lesson |
|---|---|---|
| Per-day (nested) quantile VaR objective | ❌ | Over-trades under costs; use pre-commitment for an episode risk target |
| Pre-commitment + W_t in state | ✅ | Turns VaR into a mean objective; the main lever |
| Warm start residual on the tuned rule + band | ✅ | Starts at a strong policy; learns local improvements |
| Expected (quadrature) target | ✅ | Removes target noise; same as swing H1 |
| Average the target over liability arrivals | ✅ (small) | Remaining target noise was the arrival count; helps the tail (CVaR) |
| Random initial vol in training | ✅ robustness | Domain randomisation needs no new table; small base cost |
| More gradient updates | ❌ ×3 | Unlike swing, more updates hurt here at every stage |
| Scalar mean critic | ❌ | The quantile critic's mean is a better critic here, even for a mean objective |
| Lower b, ramp, λW blends, all-b policy | ❌ | The hinge at 0.6 × rule VaR is a good VaR surrogate; reshaping it trades VaR for cost or CVaR |
| Narrower / wider band | ❌ | The rule's tuned band is right |
| Rule inputs in state | neutral | The residual already knows the rule |
| Seed ensemble | ✅ (small) | +0.05..+0.35 on unbiased validation; removes single-model selection |
| Selecting on one 8k-path validation VaR | ⚠️ | SE 0.4: use paired screens on 3 seeds, a 2nd validation set, the last-4 mean |

## 5. Practical notes
- The Mac sleeps or idles at night: queue the next batch before stopping; check `results.json` and relaunch dead runs.
- Run experiments in a separate worktree when other threads share the main checkout.
- Every new state column must be added in three places: `HedgingEnv._obs`, `HestonKernel.transition` (next state),
  and the loaders (`robustness_final.py`, `stepwise.py`, `ensemble.py`). W must stay the last column.
- Kernel fidelity checks: `review/warmstart/kernel_fidelity.py`, `review/improve/check_rule_frac_kernel.py`,
  `review/improve/check_arrivals.py` (realised node reproduces the env).
