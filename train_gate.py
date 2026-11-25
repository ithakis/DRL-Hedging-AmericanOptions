#!/usr/bin/env python
"""Train and evaluate a risk-sensitive gate on top of rule (b) (torch_d4pg/gate_agent.py, GateRuleEnv).

Same markets, split and selection as train_d4pg.py: train paths --market_seed, checkpoint selection on
validation paths (--val_seed), report the selected checkpoint on test paths (--test_seed).
Uses train_d4pg's arguments plus --warm_rule a,b,w (required), --gate_warmup_episodes, --epsilon, --margin.

    python train_gate.py <train_d4pg market/path flags> --warm_rule 0.3,0.3,0.5 --gate_warmup_episodes 2000
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import train_d4pg as T
from torch_d4pg.gate_agent import GateQR
from torch_d4pg.hedge_env import GateRuleEnv
from torch_d4pg.metrics import objective_value, summarize


def make_env(m, a):
    return GateRuleEnv(m, a.spread, a.vega_obs, [float(x) for x in a.warm_rule.split(",")])


def evaluate(agent, m, a, gates=None):
    env = make_env(m, a)
    idx = np.arange(env.P)
    obs = env.reset(idx)
    tot, cost, done = np.zeros(env.P), np.zeros(env.P), False
    while not done:
        g = agent.act(obs, env.rule_gate())
        if gates is not None:
            gates.append((g, env.rule_gate()))
        t = env.t
        obs, r, done, x = env.step(g)
        tot += r
        cost += np.abs(a.spread * x * env.b["h_price"][:, t, t])
    return tot, cost


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate_warmup_episodes", type=int, default=2000)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--margin", type=float, default=0.0)
    g, rest = ap.parse_known_args(argv)
    a = T.get_args(rest)
    assert a.warm_rule, "--warm_rule a,b,w is required"
    torch.set_num_threads(a.torch_threads)
    out = Path(a.out) / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    json.dump({**vars(a), **vars(g)}, open(out / "config.json", "w"), indent=2)
    train_m = T.get_market(a, a.market_seed, a.n_train_paths)
    val_m = T.get_market(a, a.val_seed, a.n_val)
    test_m = T.get_market(a, a.test_seed, a.n_test)
    import numba
    numba.set_num_threads(1)
    env = make_env(train_m, a)
    center, scale = T.obs_stats(train_m, a.spread, a.vega_obs, seed=a.market_seed)
    agent = GateQR(env.obs_dim, obj_func=a.obj_func, threshold=a.threshold, seed=a.seed, critic_sizes=a.critic_sizes,
                   lr=a.lr, batch_size=a.batch_size, n_step=a.n_step, target_update_period=a.target_update_period,
                   min_replay_size=a.min_replay_size, samples_per_insert=a.samples_per_insert,
                   obs_center=center, obs_scale=scale, epsilon=g.epsilon, margin=g.margin)

    def validate(ep, t_train):
        gates = []
        pnl, cost = evaluate(agent, val_m, a, gates)
        gg = np.concatenate([x for x, _ in gates]); gr = np.concatenate([y for _, y in gates])
        s = summarize(pnl, cost, n_boot=200)
        row = dict(episode=ep, train_seconds=t_train, val_score=float(objective_value(pnl, a.obj_func, a.threshold)),
                   trade_share=float(gg.mean()), rule_trade_share=float(gr.mean()), agree_rule=float((gg == gr).mean()),
                   critic_loss=agent.last_losses[0],
                   **{f"val_{k}": s[k] for k in ["mean_pnl", "std_pnl", "VaR95", "CVaR95", "mean_cost"]})
        print(f"ep {ep:6d} | {t_train:7.1f}s | val VaR95 {s['VaR95']:.2f} CVaR95 {s['CVaR95']:.2f} mean {s['mean_pnl']:.2f} "
              f"| trade {row['trade_share']:.2f} (rule {row['rule_trade_share']:.2f}, agree {row['agree_rule']:.2f})", flush=True)
        return row

    rows = [validate(0, 0.0)]  # = rule (b): not selectable
    best = dict(score=np.inf)
    t_train, tic = 0.0, time.time()
    for ep in range(a.train_episodes):
        agent.use_critic = ep >= g.gate_warmup_episodes
        obs = env.reset([ep % env.P])
        done = False
        while not done:
            gt = agent.act(obs, env.rule_gate(), explore=True)
            nxt, r, done, _ = env.step(gt)
            agent.observe(obs[0], gt[0], r[0], nxt[0], done)
            obs = nxt
        if (ep + 1) % a.val_every == 0 or ep + 1 == a.train_episodes:
            t_train += time.time() - tic
            use = agent.use_critic
            agent.use_critic = True  # validate the gate itself (during warm-up too, logged only)
            row = rows.append(validate(ep + 1, t_train)) or rows[-1]
            agent.use_critic = use
            if ep + 1 > g.gate_warmup_episodes and row["val_score"] < best["score"]:
                best = dict(score=row["val_score"], episode=ep + 1, critic=agent.state())
            pd.DataFrame(rows).to_csv(out / "validation.csv", index=False)
            tic = time.time()
    agent.critic.load_state_dict(best["critic"])
    agent.use_critic = True
    torch.save(dict(cfg=agent.cfg, critic=best["critic"]), out / "best_critic.pt")
    pnl, cost = evaluate(agent, test_m, a)
    res = dict(run_name=a.run_name, selected_episode=best["episode"], val_score=best["score"],
               seconds_train=t_train, seconds_per_1000_episodes=1000 * t_train / a.train_episodes,
               test=summarize(pnl, cost, n_boot=a.n_boot))
    np.save(out / "test_pnl.npy", np.c_[pnl, cost])
    json.dump(res, open(out / "results.json", "w"), indent=2)
    print("TEST (selected):", {k: round(v, 3) for k, v in res["test"].items()})


if __name__ == "__main__":
    main()
