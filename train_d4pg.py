#!/usr/bin/env python
"""Train and evaluate a D4PG-QR hedging agent (PyTorch).  Replaces run.py (acme/TF).

Three disjoint path sets:
  train  : --market_seed (default 1234), --n_train_paths paths (default = --train_episodes), shared by
           all agent seeds, as in the paper scripts (one saved portfolio folder for all runs)
  valid  : --val_seed  (default 2025), --n_val paths: the agent is evaluated every --val_every episodes
           and the checkpoint with the best validation objective is kept (agent selection)
  test   : --test_seed (default 654),  --n_test paths: only the selected checkpoint is reported
Seeds 2025 / 654 and 8,192 paths are the ones of the rule baselines (experiments/heston_rule_baselines.py).

Outputs in --out/<run_name>/:
  config.json, validation.csv, results.json (test metrics with bootstrap SEs, timing),
  best_actor.pt / final_actor.pt, policy_best.npz (numpy policy for tools/evaluate_policy_numpy.py),
  optionally market_test.npz (--export_test_market) for evaluating without this repository.

Example (Heston, CVaR, 1% option cost):
  python train_d4pg.py --process Heston --obj_func cvar --spread 0.01 --seed 11 --train_episodes 40000
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from environment.market import load_or_build_market, save_market_npz
from environment.utils import Utils
from torch_d4pg.agent import D4PGQR, export_actor_npz
from torch_d4pg.hedge_env import HedgingEnv
from torch_d4pg.metrics import objective_value, summarize


def str2bool(s):
    return str(s).lower() in ("1", "true", "yes", "y")


def int_list(s):
    return [int(x) for x in str(s).split(",") if x]


def get_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("market (defaults = paper setup, (EXP 1)/(EXP 2.1) scripts)")
    g.add_argument("--process", choices=["GBM", "Heston"], default="Heston")
    g.add_argument("--S0", type=float, default=10.0)
    g.add_argument("--K", type=float, default=10.0)
    g.add_argument("--init_ttm", type=int, default=30)
    g.add_argument("--r", type=float, default=0.0)
    g.add_argument("--mu", type=float, default=0.0)
    g.add_argument("--lambda_v", type=float, default=0.0,
                   help="variance risk premium for P-paths; --kappa/--theta are the Q (pricing) values")
    g.add_argument("--spread", type=float, default=0.01, help="option transaction cost")
    g.add_argument("--liab_ttms", type=int_list, default=[60])
    g.add_argument("--poisson_rate", type=float, default=1.0)
    g.add_argument("--moneyness_mean", type=float, default=1.0)
    g.add_argument("--moneyness_std", type=float, default=0.0)
    g.add_argument("--num_conts_to_add", type=int, default=-1)
    g.add_argument("--contract_size", type=int, default=100)
    g.add_argument("--hed_type", default="American")
    g.add_argument("--hed_ttm", type=int, default=30)
    g.add_argument("--init_vol", type=float, default=0.3)
    g.add_argument("--kappa", type=float, default=1.0)
    g.add_argument("--theta", type=float, default=0.09)
    g.add_argument("--volvol", type=float, default=0.3)
    g.add_argument("--rho", type=float, default=-0.7)
    g.add_argument("--time_to_simulate", type=int, default=30)
    g.add_argument("--TradingDaysPerYear", type=int, default=252)
    g.add_argument("--numerical_accuracy", default="low")
    g.add_argument("--pricing_backend", default="table", help="Heston only: table (v2) or quantlib")
    g.add_argument("--n_jobs", type=int, default=-1)
    g.add_argument("--market_dir", default="markets", help="cache of built markets (.npy, shared by runs)")

    g = ap.add_argument_group("paths / selection")
    g.add_argument("--train_episodes", type=int, default=40_000)
    g.add_argument("--n_train_paths", type=int, default=None, help="default = train_episodes")
    g.add_argument("--market_seed", type=int, default=1234)
    g.add_argument("--val_seed", type=int, default=2025)
    g.add_argument("--test_seed", type=int, default=654)
    g.add_argument("--n_val", type=int, default=8192)
    g.add_argument("--n_test", type=int, default=8192)
    g.add_argument("--val_every", type=int, default=1000)
    g.add_argument("--n_boot", type=int, default=1000)
    g.add_argument("--skip_test", action="store_true", help="smoke tests: no test set")

    g = ap.add_argument_group("agent (defaults = original acme agent in run.py / agent/)")
    g.add_argument("--seed", type=int, default=1, help="agent seed (network init, noise, replay)")
    g.add_argument("--obj_func", choices=["meanstd", "var", "cvar"], default="var")
    g.add_argument("--threshold", type=float, default=0.95)
    g.add_argument("--std_coef", type=float, default=1.645)
    g.add_argument("--critic", default="qr-huber", choices=["qr-huber"])
    g.add_argument("--vega_obs", type=str2bool, default=None, help="default: True for Heston, False for GBM")
    g.add_argument("--n_step", type=int, default=5)
    g.add_argument("--batch_size", type=int, default=256)
    g.add_argument("--lr", type=float, default=1e-4)
    g.add_argument("--sigma", type=float, default=0.3)
    g.add_argument("--target_update_period", type=int, default=100)
    g.add_argument("--samples_per_insert", type=float, default=32.0)
    g.add_argument("--min_replay_size", type=int, default=1000)
    g.add_argument("--actor_sizes", type=int_list, default=[256, 256, 256])
    g.add_argument("--critic_sizes", type=int_list, default=[512, 512, 256])
    g.add_argument("--torch_threads", type=int, default=1, help="1 is fastest for these small batches; "
                   "run several agents in parallel instead")

    g = ap.add_argument_group("output")
    g.add_argument("--out", default="logs_torch")
    g.add_argument("--run_name", default=None)
    g.add_argument("--export_test_market", action="store_true")
    a = ap.parse_args(argv)
    if a.vega_obs is None:
        a.vega_obs = a.process == "Heston"
    if a.n_train_paths is None:
        a.n_train_paths = a.train_episodes
    if a.run_name is None:
        a.run_name = f"{a.process}_{a.obj_func}_spread{a.spread}_seed{a.seed}"
    return a


def make_utils(a, seed, n_paths):
    return Utils(
        S0=a.S0, K=a.K, init_ttm=a.init_ttm, r=a.r, mu=a.mu, q=0.0, spread=a.spread,
        ttms=a.liab_ttms, poisson_rate=a.poisson_rate, moneyness_mean=a.moneyness_mean,
        moneyness_std=a.moneyness_std, num_conts_to_add=a.num_conts_to_add, contract_size=a.contract_size,
        hed_ttm=a.hed_ttm, hed_type=a.hed_type, init_vol=a.init_vol,
        kappa=a.kappa if a.process == "Heston" else None, theta=a.theta if a.process == "Heston" else None,
        volvol=a.volvol if a.process == "Heston" else None, rho=a.rho if a.process == "Heston" else None,
        stochastic_process=a.process, time_to_simulate=a.time_to_simulate, num_sim=n_paths, frq=1,
        TradingDaysPerYear=a.TradingDaysPerYear, numerical_accuracy=a.numerical_accuracy, n_jobs=a.n_jobs,
        np_seed=seed, action_low=0.0, action_high=1.0,
        pricing_backend=a.pricing_backend if a.process == "Heston" else "quantlib",
        lambda_v=a.lambda_v if a.process == "Heston" else 0.0,
    )


def get_market(a, seed, n):
    # spread does not change the market arrays; build with spread 0 so all costs share one cache
    u = make_utils(argparse.Namespace(**{**vars(a), "spread": 0.0}), seed, n)
    return load_or_build_market(u, a.market_dir)


def evaluate(agent_or_actor, market, spread, vega_obs, chunk=8192):
    """Greedy policy on all paths of a market. Returns episode P&L and option costs."""
    actor = agent_or_actor.actor if hasattr(agent_or_actor, "actor") else agent_or_actor
    env = HedgingEnv(market, spread, vega_obs)
    pnl, cost = [], []
    for s in range(0, env.P, chunk):
        idx = np.arange(s, min(env.P, s + chunk))
        obs = env.reset(idx)
        tot, c = np.zeros(len(idx)), np.zeros(len(idx))
        done = False
        while not done:
            with torch.inference_mode():
                u = actor(torch.as_tensor(obs, dtype=torch.float32)).numpy()
            x_before = env.t
            obs, r, done, x = env.step((u[:, 0] + 1.0) / 2.0)
            tot += r
            c += np.abs(spread * x * env.b["h_price"][:, x_before, x_before])
        pnl.append(tot)
        cost.append(c)
    return np.concatenate(pnl), np.concatenate(cost)


def main(argv=None):
    a = get_args(argv)
    torch.set_num_threads(a.torch_threads)
    out = Path(a.out) / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(a), open(out / "config.json", "w"), indent=2)

    t0 = time.time()
    train_m = get_market(a, a.market_seed, a.n_train_paths)
    val_m = get_market(a, a.val_seed, a.n_val)
    test_m = None if a.skip_test else get_market(a, a.test_seed, a.n_test)
    t_market = time.time() - t0
    print(f"markets ready in {t_market:.1f}s")
    # market building uses all cores; training uses --torch_threads (also for the numba loss kernel)
    import numba
    numba.set_num_threads(max(1, min(a.torch_threads, numba.config.NUMBA_NUM_THREADS)))

    env = HedgingEnv(train_m, a.spread, a.vega_obs)
    agent = D4PGQR(env.obs_dim, obj_func=a.obj_func, threshold=a.threshold, std_coef=a.std_coef, seed=a.seed,
                   actor_sizes=a.actor_sizes, critic_sizes=a.critic_sizes, lr=a.lr, batch_size=a.batch_size,
                   n_step=a.n_step, sigma=a.sigma, target_update_period=a.target_update_period,
                   min_replay_size=a.min_replay_size, samples_per_insert=a.samples_per_insert)

    rows, best = [], dict(score=np.inf)
    t_train, t_val, ep_rewards = 0.0, 0.0, []
    tic = time.time()
    for ep in range(a.train_episodes):
        obs = env.reset([ep % env.P])
        done, tot = False, 0.0
        while not done:
            u = agent.policy(obs, noise=True)
            next_obs, r, done, _ = env.step(agent.to_env_action(u))
            agent.observe(obs[0], u[0], r[0], next_obs[0], done)
            obs, tot = next_obs, tot + r[0]
        ep_rewards.append(tot)
        if (ep + 1) % a.val_every == 0 or ep + 1 == a.train_episodes:
            t_train += time.time() - tic
            tv = time.time()
            pnl, cost = evaluate(agent, val_m, a.spread, a.vega_obs)
            score = float(objective_value(pnl, a.obj_func, a.threshold, a.std_coef))
            s = summarize(pnl, cost, n_boot=200)
            t_val += time.time() - tv
            row = dict(episode=ep + 1, train_seconds=t_train, val_score=score,
                       train_mean_pnl_last=float(np.mean(ep_rewards[-a.val_every:])),
                       learner_steps=agent.learner_steps, critic_loss=agent.last_losses[0],
                       actor_loss=agent.last_losses[1],
                       **{f"val_{k}": s[k] for k in ["mean_pnl", "std_pnl", "VaR95", "CVaR95", "mean_cost"]})
            rows.append(row)
            if score < best["score"]:
                best = dict(score=score, episode=ep + 1, actor=agent.actor_state())
            print(f"ep {ep + 1:6d} | {t_train:7.1f}s train | val {a.obj_func} score {score:8.3f} "
                  f"(VaR95 {s['VaR95']:.2f}, CVaR95 {s['CVaR95']:.2f}, mean {s['mean_pnl']:.2f}) "
                  f"| best ep {best['episode']}", flush=True)
            pd.DataFrame(rows).to_csv(out / "validation.csv", index=False)
            tic = time.time()

    agent.save(out / "final_actor.pt")
    agent.save(out / "best_actor.pt", actor_state=best["actor"])
    agent.actor.load_state_dict(best["actor"])
    meta = dict(process=a.process, obj_func=a.obj_func, spread=a.spread, vega_obs=a.vega_obs,
                seed=a.seed, selected_episode=best["episode"], val_score=best["score"],
                obs=["S", "portfolio_gamma", "hedge_gamma"] + (["portfolio_vega", "hedge_vega"] if a.vega_obs else []))
    export_actor_npz(agent.actor, out / "policy_best.npz", meta)

    res = dict(run_name=a.run_name, selected_episode=best["episode"], val_score=best["score"],
               seconds_market=t_market, seconds_train=t_train, seconds_validation=t_val,
               seconds_per_1000_episodes=1000 * t_train / a.train_episodes,
               learner_steps=agent.learner_steps, torch_threads=a.torch_threads, cpu_count=os.cpu_count())
    if test_m is not None:
        pnl, cost = evaluate(agent, test_m, a.spread, a.vega_obs)
        res["test"] = summarize(pnl, cost, n_boot=a.n_boot)
        np.save(out / "test_pnl.npy", np.c_[pnl, cost])
        print("TEST (selected checkpoint):", {k: round(v, 3) for k, v in res["test"].items()})
        if a.export_test_market:
            save_market_npz(test_m, out / "market_test.npz")
    json.dump(res, open(out / "results.json", "w"), indent=2)
    print(f"done: {1000 * t_train / a.train_episodes:.1f}s per 1,000 episodes (training only)")
    return res


if __name__ == "__main__":
    main()
