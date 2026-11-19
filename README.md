# DRL Hedging for American Options (GBM & Heston)

This repository accompanies research on risk-sensitive deep reinforcement learning (DRL) for hedging option liabilities under Geometric Brownian Motion (GBM) and Heston stochastic volatility. It implements distributional DRL agents (D4PG with C51/Quantile/IQN critics) and strong Greek-based baselines (Delta, Delta–Gamma, Delta–Vega) in a realistic trading environment with transaction costs, American exercise, and Poisson option arrivals.

If you’re reading the paper, this README maps concepts to code and gives step-by-step instructions to reproduce training and evaluation.


## Reproduce the revised paper (one command)

Environment: `pip install -r requirements-torch.txt` (Python 3.10, PyTorch pipeline in `train_d4pg.py`).

| Command | What it does | Time (M1 Air) |
|---|---|---|
| `make paper` | LaTeX tables + figures from the stored result files and weights, then the PDF | ~2 min |
| `make all` | re-tunes the rule baselines, evaluates all stored agents on the test paths, writes every CSV/markdown result, then `make paper` | ~30-60 min |
| `make retrain` | trains all 45 agents from scratch (`logs_torch/retrain`), exports the weights, then `make all` | several days |

Stored inputs: agents in `review/results/retrain/weights/` (which file is used where: `mapping.csv`), validation
curves in `review/results/retrain/runs/`, Heston price tables in `pricing_tables/`. Simulated markets are cached in
`markets/` (not in git; built on first use). Generated paper files: `review/results/paper/{tables,figures}`, which
`main.tex` includes with `\input` / `\includegraphics`. On GitHub, Actions -> "Build paper" runs `make paper`
(cheap path only; no training or evaluation runs in the cloud).

## What’s implemented

- Market models and pricing
  - GBM: closed-form simulation; European pricing via BSM; American pricing via a fast binomial tree (Numba).
  - Heston: Quadratic–Exponential (QE) simulation with Sobol randoms; American pricing via QuantLib FDM Heston (Modified Craig–Sneyd scheme); European via QuantLib Analytic Heston.
  - P vs Q measure switching with variance risk premium λ_v (see `Utils.get_sim_path_Heston`).
- Hedging environment
  - Gym/dm_env-compatible environment that manages a liability portfolio and ATM hedging options.
  - Actions are continuous in [0, 1] and map to hedge contract shares within dynamic Gamma/Vega-neutral bounds; the stock leg is kept delta-neutral and frictionless.
  - Reward is PnL net of transaction costs (spread).
- DRL agents
  - Distributional D4PG implemented on DeepMind Acme + TensorFlow Sonnet.
  - Critics: C51, Quantile Regression (QR) with Huber/Gaussian/Laplace variants, and Implicit Quantile Networks (IQN).
  - Risk-sensitive objectives: Mean-Std, VaR, CVaR (computed from learned return distributions).
- Baselines
  - Exact Delta, Delta–Gamma, and Delta–Vega hedgers that compute analytic hedge shares each step.
- Logging & artifacts
  - CSV logs and `flags.json` under `./logs/<work_folder>`.
  - Reusable market/portfolio states saved under `./portfolios/.Portfolio_*` for Train/Eval.

## Repository layout (key files)

- `run.py` — Train and/or evaluate the DRL policy; saves/loads policies to/from `./logs/<work_folder>/policy`.
- `greek_run.py` — Evaluate Greek baselines (delta/gamma/vega) over many episodes.
- `environment/`
  - `utils.py` — Simulation (GBM, Heston QE), pricing (GBM binomial; Heston FDM/Analytic via QuantLib), liability aggregation (Poisson), ATM hedging universe.
  - `Trading.py` — Asset and portfolio models; transaction costs; delta-neutral stock leg; persistence.
  - `Environment.py` — Gym wrapper, state, reward, action mapping with Gamma/Vega bounds.
- `agent/`
  - `agent.py` — D4PG assembly (Acme); Greek baseline actors.
  - `distributional.py` — Distributional heads (C51/QR/IQN), risk metrics, quantile losses.
  - `learning.py` — Learner tying objectives (mean-std/VaR/CVaR) to critic types.
- Experiment scripts: `(EXP *) *.sh`, `run.sh`, `Robustness Testing.sh` — orchestration examples for batch runs.
- `requirements.txt`, `environment.yaml` — dependencies (pip and Conda). QuantLib is provided via Conda.

## Environment setup

QuantLib is required for Heston American pricing; we recommend Conda on macOS.

- Conda (recommended)
  - Create the environment from `environment.yaml` to get QuantLib, TensorFlow 2.8, Acme, Sonnet, and the scientific stack with pinned versions.
  - Activate the environment before running experiments.

- Pip (alternative)
  - Install packages from `requirements.txt`.
  - Note: QuantLib wheels aren’t included here; Heston American pricing will require a local QuantLib install if you need that path. Prefer Conda if you plan to run Heston+American.

Performance tips
- Use `--n_jobs -1` to leverage all CPU cores for simulation and pricing.
- `--numerical_accuracy` can be `low` or `high` (higher accuracy is slower).
- For reproducibility: set `--train_seed` and `--eval_seed`.

## Quickstart: Greek baselines (evaluation)

Evaluate Delta, Gamma, or Vega hedging strategies without training.

- GBM, European hedging, 5k episodes:
  - `python greek_run.py --stochastic_process GBM --hed_type European --hed_ttm 20 --spread 0.005 --eval_sim 5120 --strategy delta`
  - Replace `--strategy` with `gamma` or `vega` to test other baselines.

- Heston, American hedging (requires Conda env with QuantLib):
  - `python greek_run.py --stochastic_process Heston --hed_type American --hed_ttm 20 --spread 0.01 --eval_sim 5120 --strategy gamma`

Notes
- On first run, a timestamped folder is created under `./portfolios/.Portfolio_*` with an `Eval` subfolder; subsequent runs can reuse it via `--portfolio_folder`.
- Logs are written to `./logs/greekhedge_stochastic_process=..._spread=..._hedttm=.../` with CSV files for `eval_delta_loop`, `eval_gamma_loop`, or `eval_vega_loop` and a `flags.json` snapshot.

## Train and evaluate a DRL agent

Training
- Typical VaR objective with a quantile-regression critic:
  - `python run.py --obj_func var --threshold 0.95 --critic qr-huber --stochastic_process Heston --hed_type American --hed_ttm 20 --spread 0.005 --train_sim 40000 --eval_sim 5120 --vega_obs True`
- Alternatives
  - Objective: `--obj_func meanstd --std_coef 1.645`, or `--obj_func cvar --threshold 0.95`.
  - Critic: `--critic c51`, `qr-huber`, `qr-gl`, `qr-gl_tl`, `qr-lapl`, `qr-lapl_tl`, or `iqn` (IQN supports `--obj_func cvar`).
  - Heston model: pass `--kappa --theta --volvol --rho` if you want to override defaults.
  - Transaction costs: set `--spread` (e.g., 0.005 for 50 bps round-turn).

Evaluation of a trained policy
- During training, the policy is exported to `./logs/<work_folder>/policy`.
- To evaluate only, load a saved policy via:
  - `python run.py --eval_only True --agent_path ./logs/<work_folder> --stochastic_process ... --hed_type ... --hed_ttm ... --eval_sim 5120`

Artifacts and logs
- `./logs/<work_folder>/` contains CSV logs for `train_loop`, `eval_loop`, and `learner`, a `flags.json`, and an `ok` marker file when runs complete.
- `--logger_prefix` allows grouping multiple runs under a common subfolder.
- Portfolios and environment snapshots live under `./portfolios/.Portfolio_*/{Train,Eval}` and are reused when you pass `--portfolio_folder`.

## PyTorch D4PG-QR (revision pipeline, runs on macOS ARM)

The original agent (`agent/`, `run.py`: dm-acme 0.4 + Reverb + TF) does not install on Apple Silicon.
`train_d4pg.py` trains the same agent in PyTorch (structure from the swing D4PG of
[DRL-Swing-Options](https://github.com/ithakis/DRL-Swing-Options), quantile critic and learner settings
from `agent/`). All revision results come from this pipeline only.

Same as before: state `[S, portfolio gamma, hedge gamma (, portfolio vega, hedge vega)]`, action in [0, 1]
between no hedge and the gamma-/vega-neutral amount, reward = P&L minus option cost, objectives
`meanstd` / `var` / `cvar` on a 99-quantile critic with quantile-Huber loss (`qr-huber`),
n-step 5, discount 1, Adam 1e-4, noise 0.3, hard target update every 100 steps, gradient clip 40,
actor 256-256-256, critic 512-512-256 (all flags, see `--help`).

Install (conda env or pip): `pip install -r requirements-torch.txt` (torch, numba, QuantLib, numpy, scipy, pandas, ...).

**Paths.** Three disjoint path sets: training paths (`--market_seed 1234`, shared by all agent seeds),
validation (`--val_seed 2025`, 8,192 paths) and test (`--test_seed 654`, 8,192 paths, the same as the rule
baselines). Every `--val_every 1000` episodes the greedy policy is run on the validation paths and the
checkpoint with the best validation objective (VaR95 / CVaR95 of the loss, or -(mean - 1.645 std)) is kept.
Only this checkpoint is evaluated on the test paths; results have bootstrap standard errors.
Markets are cached in `markets/` as `.npy` files (memory-mapped, so parallel runs share them).
Heston prices come from the precomputed table (`--pricing_backend table`, built once per parameter set).

**Train** (paper setup is the default: S0 = K = 10, 30-day hedges, Poisson 60-day liabilities):

```bash
# Heston, CVaR95, 1% option cost (vega in the state by default for Heston)
python train_d4pg.py --process Heston --obj_func cvar --spread 0.01 --seed 11 --train_episodes 40000
# GBM (sigma = 0.3), VaR95, 2% cost (no vega in the state by default for GBM)
python train_d4pg.py --process GBM --obj_func var --spread 0.02 --seed 11 --train_episodes 40000
# Heston under P (paper: r = 2%, mu = 3%, lambda_v = 1.5; kappa/theta are the Q values used for pricing)
python train_d4pg.py --process Heston --r 0.02 --mu 0.03 --lambda_v 1.5 --kappa 2.5 --theta 0.036 ...
```

Output in `logs_torch/<run_name>/`: `validation.csv` (score per checkpoint), `results.json` (test metrics
with SEs, selected episode, timings), `best_actor.pt`, `final_actor.pt`, and `policy_best.npz`.
Use `--torch_threads 1` (default) and run several agents in parallel rather than one agent on many cores.

**Evaluate / export.**

```bash
# a trained policy on another market (e.g. robustness: vol-of-vol 0.4), optionally save that market
python evaluate_d4pg.py --policy logs_torch/<run>/policy_best.npz --volvol 0.4 --export_market market.npz
# without this repository: only numpy is needed
python tools/evaluate_policy_numpy.py --policy policy_best.npz --market market_test.npz
```

`policy_best.npz` holds the actor weights (`W0, b0, ln_gamma, ln_beta, W1.., W_out, b_out`) and a
`meta_json` string (cost, state layout, forward pass). `--export_test_market` writes the test market as
`market_test.npz`. `tools/evaluate_policy_numpy.py` is one file (policy forward pass, environment,
metrics) that a co-author can run with numpy alone.

**Checks.** `pytest tests/test_torch_d4pg.py`: the vectorised environment gives the same states and
rewards as `MainPortfolio` (GBM, Heston, Heston under P), the numpy evaluator matches the PyTorch one,
and the fused quantile loss matches the direct port of `agent/distributional.py`.

## Reproducing paper experiments

- Use the provided shell scripts as references:
  - `(EXP 1) Cao GBM Experiments.sh`
  - `(EXP 2.1) Cao Heston Experiments.sh`
  - `(EXP 2.2) Heston Experiments Single Option - European Hedging.sh`
  - `(EXP 3) Stress Test.sh`, `(EXP 4) Heston Test.sh`, `(EXP 5) Heston Test under P.sh`
  - `Robustness Testing.sh`, `run.sh`
- These scripts vary objective functions (Mean-Std/VaR/CVaR), model (GBM/Heston), spreads, hedging maturities, and critic types, and show how to organize results with `--logger_prefix` and `--portfolio_folder`.

## Mapping paper sections to code

- Simulation & measures
  - GBM: `environment/utils.py:get_sim_path_GBM`
  - Heston QE (Sobol) and P/Q w/ λ_v: `environment/utils.py:get_sim_path_Heston`
- Pricing engines
  - GBM American (binomial): `environment/utils.py:American_put_option`
  - Heston American (FDM, Modified Craig–Sneyd): `environment/utils.py:_American_Option_Heston`
  - European under Heston (Analytic): `environment/utils.py:European_put_option`
- Hedging environment and bounds
  - `environment/Environment.py` and `environment/Trading.py`
  - Action alpha in [0,1] mapped to contract shares within Gamma/Vega-neutral bounds; optional state augmentation with Vega via `--vega_obs`.
- DRL agents and objectives
  - Agent assembly: `agent/agent.py` (D4PG), `agent/learning.py` (objectives: mean-std/VaR/CVaR), `agent/distributional.py` (C51/QR/IQN heads & losses).
- Greek baselines
  - `DeltaHedgeAgent`, `GammaHedgeAgent`, `VegaHedgeAgent` in `agent/agent.py`.
- Logging & evaluation
  - Acme `EnvironmentLoop` with CSV loggers; see `run.py` and `greek_run.py`.

## Key flags (selected)

- Market and simulation: `--stochastic_process {GBM,Heston}`, `--init_vol`, `--kappa --theta --volvol --rho` (Heston), `--time_to_simulate`, `--frq`, `--TradingDaysPerYear`.
- Portfolio and costs: `--spread`, `--contract_size`, `--liab_ttms`, `--poisson_rate`, `--hed_type {European,American}`, `--hed_ttm`.
- DRL: `--obj_func {meanstd,var,cvar}`, `--threshold`, `--std_coef`, `--critic {c51,qr-*,iqn}`, `--n_step`, `--batch_size`, `--lr`, `--vega_obs`.
- Runs & perf: `--train_sim`, `--eval_sim`, `--train_seed`, `--eval_seed`, `--n_jobs`, `--numerical_accuracy {low,high}`, `--logger_prefix`, `--portfolio_folder`.

## Troubleshooting

- QuantLib errors on macOS/Heston/American
  - Prefer the Conda environment (`environment.yaml`), which includes `quantlib`.
- TensorFlow or Acme version conflicts
  - Use the pinned versions in the provided environments (TF 2.8, dm-acme 0.4.0, Sonnet 2.0, Reverb 2.x).
- Slow runs
  - Increase `--n_jobs`, reduce `--eval_sim`/`--train_sim`, or lower `--numerical_accuracy`.
- Reusing the same market/portfolio state across methods
  - Pass the same `--portfolio_folder` to both DRL and baseline runs for matched simulation paths and liability/hedging universes.

## Citation

If you use this code, please cite the corresponding research paper. You can also reference this repository in your experimental appendix for reproducibility.

## License

Please see the repository for licensing details. If no license file is present, contact the authors for usage permissions.
