"""Why is DRL's CVaR95 not better than rule (b)'s at 2%? Diagnosis only (frozen FINAL agents, no training).

Rolls out the FINAL agent (ResidualRuleEnv: traded fraction f = clip(f_rule + 0.5 u, 0, 1)) and rule (b) on the same
paths and records, per path and day: running P&L W_t before the trade, y_t = W_t + b (distance to the pre-commitment
level; y < 0 = the path is already past b), rule fraction f_rule(s_t), agent fraction f, option cost, P&L increment.
    PYTHONPATH=. python review/final/tail_diagnosis.py COST SEED_DIR... [--val]"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from review.retrain import analysis as A

ap = argparse.ArgumentParser()
ap.add_argument("cost", type=float)
ap.add_argument("runs", nargs="+")
ap.add_argument("--val", action="store_true", help="validation paths (seed 2025) instead of test 4242")
args = ap.parse_args()
import train_d4pg as T
from torch_d4pg.agent import D4PGQR
from torch_d4pg.hedge_env import HedgingEnv

a = A.block_args("B", ["--market_dir", "../DRL-Hedging-retrain/markets", "--pricing_backend", "adi_trading",
                       "--table_dir", "../DRL-Hedging-retrain/pricing_tables"])
m = A.T.get_market(a, 2025 if args.val else 4242, 8192)
c = args.cost


def rollout(cfg, actor):
    HedgingEnv.extra_obs, HedgingEnv.pnl_obs = True, True
    HedgingEnv.band = float(cfg["band"])
    env = T.make_env(m, c, True, argparse.Namespace(**cfg))
    obs = env.reset(np.arange(env.P))
    rec = dict(W=[], fr=[], f=[], cost=[], dpnl=[], absx=[])
    done = False
    while not done:
        t = env.t
        rec["W"].append(env.W.copy())
        fr = env.rule_fraction()
        if actor is None:
            u = np.zeros(env.P)
        else:
            with torch.inference_mode():
                u = actor(torch.as_tensor(obs, dtype=torch.float32)).numpy()[:, 0]
        obs, r, done, x = env.step((u + 1) / 2)
        rec["fr"].append(fr); rec["f"].append(env.last_frac.copy()); rec["dpnl"].append(r)
        rec["cost"].append(np.abs(c * x * env.b["h_price"][:, t, t])); rec["absx"].append(np.abs(x))
    HedgingEnv.extra_obs = HedgingEnv.pnl_obs = False
    HedgingEnv.band = 0.0
    return {k: np.stack(v, 1) for k, v in rec.items()}, env.W.copy()


cfg0 = json.load(open(Path(args.runs[0]) / "config.json"))
b = float(cfg0["precommit_b"])
R, W_rule = rollout(cfg0, None)            # residual 0 = rule (b) exactly
L_rule = -W_rule
print(f"cost {c}, b = {b}, paths {len(L_rule)}, {'validation' if args.val else 'test 4242'}")
print(f"rule (b): VaR95 {A.var_cvar(W_rule)[0]:.2f} CVaR95 {A.var_cvar(W_rule)[1]:.2f}")
out = []
for rd in args.runs:
    cfg = json.load(open(Path(rd) / "config.json"))
    actor, _ = D4PGQR.load_actor(Path(rd) / "best_actor.pt")
    D, W_drl = rollout(cfg, actor)
    L = -W_drl
    v, cv = A.var_cvar(W_drl)
    q = np.quantile(L, 0.95); qr = np.quantile(L_rule, 0.95)
    tail, tail_r = L >= q, L_rule >= qr
    print(f"\n== {Path(rd).name}: VaR95 {v:.2f} CVaR95 {cv:.2f}")
    print(f"training objective E[(L - b)^+]: DRL {np.maximum(L - b, 0).mean():.3f}, rule {np.maximum(L_rule - b, 0).mean():.3f}; "
          f"P(L > b): DRL {np.mean(L > b):.3f}, rule {np.mean(L_rule > b):.3f}")
    print(f"worst-5% overlap with rule's worst 5%: {np.mean(tail & tail_r) / 0.05:.0%}")
    print(f"on DRL's worst 5%: DRL mean loss {L[tail].mean():.2f}, rule on the same paths {L_rule[tail].mean():.2f}")
    print(f"on rule's worst 5%: rule {L_rule[tail_r].mean():.2f}, DRL on the same paths {L[tail_r].mean():.2f}")
    # loss quantiles beyond VaR
    for p in [0.95, 0.975, 0.99, 0.995]:
        print(f"  q{p}: DRL {np.quantile(L, p):6.2f}  rule {np.quantile(L_rule, p):6.2f}")
    # hedge deviation by distance to b and time
    y = D["W"] + b
    T_ = y.shape[1]
    tb = np.minimum((np.arange(T_) * 3) // T_, 2)[None, :].repeat(len(L), 0)
    ybins = [-np.inf, -10, -5, 0, 5, 10, np.inf]
    yb = np.digitize(y, ybins[1:-1])
    df = pd.DataFrame(dict(y=yb.ravel(), third=tb.ravel(), dev=(D["f"] - D["fr"]).ravel(), f=D["f"].ravel(),
                           fr=D["fr"].ravel(), cost=D["cost"].ravel(), dpnl=D["dpnl"].ravel(),
                           is_tail=np.repeat(tail[:, None], T_, 1).ravel()))
    lab = {i: f"[{ybins[i]}, {ybins[i + 1]})" for i in range(len(ybins) - 1)}
    g = df.groupby("y").agg(n=("dev", "size"), share=("dev", lambda s: len(s) / len(df)), dev=("dev", "mean"),
                            f=("f", "mean"), f_rule=("fr", "mean"), cost=("cost", "mean"), sd_dpnl=("dpnl", "std"))
    g.index = [lab[i] for i in g.index]
    print("by y = W_t + b (y < 0: already past b):")
    print(g.round(3).to_string())
    g2 = df.groupby(["third", "y"]).dev.mean().unstack().round(3)
    g2.columns = [lab[i] for i in g2.columns]
    print("mean f - f_rule by time third (rows) and y (cols):")
    print(g2.to_string())
    gt = df[df.is_tail].groupby("y").agg(n=("dev", "size"), dev=("dev", "mean"), f=("f", "mean"), f_rule=("fr", "mean"))
    gt.index = [lab[i] for i in gt.index]
    print("on DRL's worst-5% paths, by y:")
    print(gt.round(3).to_string())
    # extreme tail: worst 1% of DRL paths, paired with the rule on the same paths
    w1 = L >= np.quantile(L, 0.99)
    dW = np.cumsum(D["dpnl"] - R["dpnl"], 1)          # DRL - rule cumulative P&L (negative = DRL worse)
    dC = np.cumsum(D["cost"] - R["cost"], 1)
    print(f"worst 1% of DRL ({w1.sum()} paths): DRL {L[w1].mean():.2f}, rule same paths {L_rule[w1].mean():.2f}")
    print("  cumulative DRL - rule P&L on them, days 5/10/15/20/25/30:", np.round(dW[w1].mean(0)[[4, 9, 14, 19, 24, 29]], 2))
    print("  of which option cost difference (DRL - rule, >0 = DRL paid more):", np.round(dC[w1].mean(0)[[4, 9, 14, 19, 24, 29]], 2))
    print("  mean f - f_rule on them by day third:", np.round([(D['f'] - D['fr'])[w1][:, k * 10:(k + 1) * 10].mean() for k in range(3)], 3))
    print("  same, all paths:", np.round([(D['f'] - D['fr'])[:, k * 10:(k + 1) * 10].mean() for k in range(3)], 3))
    S, V = np.asarray(m["S"]), np.asarray(m["V"])
    lr = np.diff(np.log(S), axis=1)
    feats = dict(ST_S0=S[:, -1] / S[:, 0], max_abs_ret=np.abs(lr).max(1), vol_T=V[:, -1], vol_max=V.max(1))
    for k, f_ in feats.items():
        print(f"  {k}: worst-1% DRL mean {f_[w1].mean():.3f}, all paths {f_.mean():.3f}")
    # paths where DRL is much worse than the rule
    d = L - L_rule
    print(f"  paths DRL worse than rule by >5: {np.mean(d > 5):.2%}, better by >5: {np.mean(d < -5):.2%}; corr(d, rule loss) {np.corrcoef(d, L_rule)[0, 1]:.2f}")
    out.append(df.assign(run=Path(rd).name))
pd.concat(out).groupby(["run", "third", "y"]).dev.mean().to_csv(
    f"review/results/final/tail_diag_{c}_{'val' if args.val else 'test'}.csv")
