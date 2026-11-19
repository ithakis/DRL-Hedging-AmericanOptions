#!/usr/bin/env python
"""Train the hedging D4PG-QR with the swing-options v64 recipe (torch_d4pg/agent_swing.py).

Same markets, train / validation / test split, selection rule and test metrics as train_d4pg.py:
validation every --val_every episodes on 8,192 paths (seed 2025), keep the best checkpoint
(of the EMA actor), report only that checkpoint on the 8,192 test paths (seed 654).

  python train_swing.py --process Heston --r 0.02 --mu 0.03 --lambda_v 0 --spread 0.01 --seed 1 \
      --out logs_torch/swing --run_name HestonP_swing_spread0.01_seed1
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import train_d4pg as base
from torch_d4pg.agent_swing import D4PGQRSwing
from torch_d4pg.hedge_env import HedgingEnv
from torch_d4pg.metrics import objective_value, summarize


def evaluate(actor, market, spread, vega_obs, chunk=8192, actions=None):
    env = HedgingEnv(market, spread, vega_obs)
    pnl, cost = [], []
    for s in range(0, env.P, chunk):
        idx = np.arange(s, min(env.P, s + chunk))
        obs = env.reset(idx)
        tot, c = np.zeros(len(idx)), np.zeros(len(idx))
        done = False
        while not done:
            with torch.inference_mode():
                frac = actor(torch.as_tensor(obs, dtype=torch.float32)).numpy()[:, 0]
            t0 = env.t
            if actions is not None:
                actions.append(frac)
            obs, r, done, x = env.step(frac)
            tot += r
            c += np.abs(spread * x * env.b["h_price"][:, t0, t0])
        pnl.append(tot)
        cost.append(c)
    return np.concatenate(pnl), np.concatenate(cost)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    a = base.get_args(argv)
    if "--run_name" not in argv:
        a.run_name = f"{a.process}_swing_{a.obj_func}_spread{a.spread}_seed{a.seed}"
    torch.set_num_threads(a.torch_threads)
    out = Path(a.out) / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    json.dump({**vars(a), "recipe": "swing_v64"}, open(out / "config.json", "w"), indent=2)

    t0 = time.time()
    train_m = base.get_market(a, a.market_seed, a.n_train_paths)
    val_m = base.get_market(a, a.val_seed, a.n_val)
    test_m = None if a.skip_test else base.get_market(a, a.test_seed, a.n_test)
    t_market = time.time() - t0
    import numba
    numba.set_num_threads(max(1, min(a.torch_threads, numba.config.NUMBA_NUM_THREADS)))

    env = HedgingEnv(train_m, a.spread, a.vega_obs)
    center, scale = base.obs_stats(train_m, a.spread, a.vega_obs, seed=a.market_seed)
    agent = D4PGQRSwing(env.obs_dim, obj_func=a.obj_func, threshold=a.threshold, std_coef=a.std_coef,
                        seed=a.seed, total_episodes=a.train_episodes, obs_center=center, obs_scale=scale)
    json.dump(agent.cfg, open(out / "agent_cfg.json", "w"), indent=2, default=str)

    rows, best = [], dict(score=np.inf, episode=0)
    t_train, t_val, ep_rewards = 0.0, 0.0, []
    tic = time.time()
    for ep in range(a.train_episodes):
        agent.start_episode(ep)
        obs = env.reset([ep % env.P])
        done, tot = False, 0.0
        while not done:
            act = agent.policy(obs, noise=True)
            next_obs, r, done, _ = env.step(agent.to_env_action(act))
            agent.observe(obs[0], act[0], r[0], next_obs[0], done)
            obs, tot = next_obs, tot + r[0]
        ep_rewards.append(tot)
        if (ep + 1) % a.val_every == 0 or ep + 1 == a.train_episodes:
            t_train += time.time() - tic
            tv = time.time()
            acts = []
            pnl, cost = evaluate(agent.eval_actor(), val_m, a.spread, a.vega_obs, actions=acts)
            acts = np.concatenate(acts)
            score = float(objective_value(pnl, a.obj_func, a.threshold, a.std_coef))
            s = summarize(pnl, cost, n_boot=200)
            t_val += time.time() - tv
            rows.append(dict(episode=ep + 1, train_seconds=t_train, val_score=score, noise_sigma=agent.noise_sigma(),
                             train_mean_pnl_last=float(np.mean(ep_rewards[-a.val_every:])),
                             learner_steps=agent.learner_steps, critic_loss=agent.last_losses[0],
                             actor_loss=agent.last_losses[1], act_mean=float(acts.mean()), act_std=float(acts.std()),
                             act_share_0=float((acts < 0.01).mean()), act_share_1=float((acts > 0.99).mean()),
                             **{f"val_{k}": s[k] for k in ["mean_pnl", "std_pnl", "VaR95", "CVaR95", "mean_cost"]}))
            # warm-up checkpoints have an untrained actor: select only after the critic warm-up
            if score < best["score"] and ep + 1 > agent.cfg["critic_warmup_episodes"]:
                best = dict(score=score, episode=ep + 1, actor=agent.actor_state())
            print(f"ep {ep + 1:6d} | {t_train:7.1f}s train | val VaR95 {s['VaR95']:.2f} CVaR95 {s['CVaR95']:.2f} "
                  f"mean {s['mean_pnl']:.2f} | act mean {acts.mean():.3f} | best ep {best['episode']}", flush=True)
            pd.DataFrame(rows).to_csv(out / "validation.csv", index=False)
            tic = time.time()

    agent.save(out / "final_actor.pt")
    agent.save(out / "best_actor.pt", actor_state=best["actor"])
    agent.actor_ema.load_state_dict(best["actor"])
    res = dict(run_name=a.run_name, recipe="swing_v64", selected_episode=best["episode"], val_score=best["score"],
               seconds_market=t_market, seconds_train=t_train, seconds_validation=t_val,
               seconds_per_1000_episodes=1000 * t_train / a.train_episodes, learner_steps=agent.learner_steps,
               torch_threads=a.torch_threads, cpu_count=os.cpu_count())
    if test_m is not None:
        pnl, cost = evaluate(agent.actor_ema, test_m, a.spread, a.vega_obs)
        res["test"] = summarize(pnl, cost, n_boot=a.n_boot)
        np.save(out / "test_pnl.npy", np.c_[pnl, cost])
        print("TEST (selected checkpoint):", {k: round(v, 3) for k, v in res["test"].items()})
    json.dump(res, open(out / "results.json", "w"), indent=2)
    return res


if __name__ == "__main__":
    main()
