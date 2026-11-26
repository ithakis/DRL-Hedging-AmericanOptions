"""Gamma / Vega hedge ratios (least-squares slope of the trade on n_G / n_V, as analysis.rollout) and Mean-Std of the
selected FINAL agents and rule (b) on the test paths (seed 4242)."""
import argparse, json
from pathlib import Path
import numpy as np, torch
from review.retrain import analysis as A
import train_d4pg as T
from torch_d4pg.agent import D4PGQR
from torch_d4pg.hedge_env import HedgingEnv, neutral_amounts

a = A.block_args("B", ["--market_dir", "../DRL-Hedging-retrain/markets", "--pricing_backend", "adi_trading",
                       "--table_dir", "../DRL-Hedging-retrain/pricing_tables"])
m = A.T.get_market(a, 4242, 8192)
out = {}
for c, s in {0.005: 4, 0.01: 1, 0.02: 8}.items():
    d = Path(f"../DRL-Hedging-AmericanOptions/logs_torch/final/HestonP_var_spread{c}_seed{s}_FINAL")
    cfg = json.load(open(d / "config.json"))
    actor, _ = D4PGQR.load_actor(d / "best_actor.pt")
    HedgingEnv.extra_obs, HedgingEnv.pnl_obs, HedgingEnv.band = True, True, float(cfg["band"])
    env = T.make_env(m, c, True, argparse.Namespace(**cfg))
    obs = env.reset(np.arange(env.P)); done = False
    sg = [0, 0]; sv = [0, 0]; tot = np.zeros(env.P)
    while not done:
        G, V = env._port(env.t)
        ng, nv = neutral_amounts(G, env.b["h_gamma"][:, env.t, env.t], V, env.b["h_vega"][:, env.t, env.t])
        with torch.inference_mode():
            u = actor(torch.as_tensor(obs, dtype=torch.float32)).numpy()[:, 0]
        obs, r, done, x = env.step((u + 1) / 2)
        tot += r
        sg[0] += np.sum(x * ng); sg[1] += np.sum(ng ** 2); sv[0] += np.sum(x * nv); sv[1] += np.sum(nv ** 2)
    HedgingEnv.extra_obs = HedgingEnv.pnl_obs = False; HedgingEnv.band = 0.0
    rb = json.load(open(f"review/results/warmstart/richer_rule_params_adi_trading_{c}.json"))[str(c)]["rule_b"]
    rr = A.rollout(m, A.rule_decider("rule", rb["a"], rb["b"], rb["w"]), True)
    pr = A.net(rr, c)
    out[c] = dict(drl_gamma_ratio=sg[0] / sg[1], drl_vega_ratio=sv[0] / sv[1], rule_gamma_ratio=rr["gamma_ratio"],
                  rule_vega_ratio=rr["vega_ratio"], drl_meanstd=float(-tot.mean() + 1.645 * tot.std()),
                  rule_meanstd=float(-pr.mean() + 1.645 * pr.std()))
    print(c, {k: round(v, 3) for k, v in out[c].items()}, flush=True)
json.dump({str(k): v for k, v in out.items()}, open("review/results/final/hedge_ratios.json", "w"), indent=2)
