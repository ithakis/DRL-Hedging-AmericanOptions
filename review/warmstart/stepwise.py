"""Fair comparison on the step-wise (time-consistent) risk metric.

Step-wise metric (name: "step-wise VaR95", SW-VaR95):
    SW = (1/T) sum_{t=0}^{T-1} VaR95( L_{t:T} ),   L_{t:T} = -(P&L from day t to maturity), VaR over paths.
It is the unconditional counterpart of what the step-wise agent optimises (the 5% quantile of the
return-to-go at every state). The terminal metric is VaR95(L_{0:T}) (= the t = 0 term).

Steps: (1) re-tune rule (b)'s (a, b, w) on validation paths for SW-VaR95 (same grid as the terminal tuning);
(2) on test paths report SW-VaR95 and terminal VaR95 with bootstrap SEs for: delta, DG, rule (b) tuned on
terminal VaR, rule (b) tuned on SW-VaR, the PR #3 step-wise agents (10 seeds: seed selected on validation
SW-VaR, and the mean over seeds), and optional newer agents (--runs, torch checkpoints of train_d4pg.py).

    PYTHONPATH=. python review/warmstart/stepwise.py --markets ../DRL-Hedging-retrain/markets \
        --runs logs_torch/warmstart/*_E3a logs_torch/warmstart/*_P2
"""
import argparse
import glob
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from review.retrain import analysis as A
from review.retrain.settings import BLOCKS, COSTS, run_name

OUT = Path("review/results/warmstart")
N_BOOT = 500
WEIGHTS = Path("review/results/retrain/weights")  # PR #3 step-wise agents (policy_best.npz of each run)


def rollout_steps(m, decide, vega_obs=True):
    """As analysis.rollout, but returns per-day P&L before option cost (P, T) and per-day cost base (P, T)."""
    m = {k: np.asarray(v) for k, v in m.items()}
    S = m["S"]
    P, T1 = S.shape
    n = np.zeros((P, T1))
    pnl, ob = np.zeros((P, T1 - 1)), np.zeros((P, T1 - 1))
    for t in range(T1 - 1):
        act = ~m["h_inactive"][:, t, :]
        G = m["l_gamma"][:, t] + np.sum(n * m["h_gamma"][:, t, :] * act, axis=1)
        V = m["l_vega"][:, t] + np.sum(n * m["h_vega"][:, t, :] * act, axis=1)
        g_h, v_h = m["h_gamma"][:, t, t], m["h_vega"][:, t, t]
        with np.errstate(divide="ignore", invalid="ignore"):
            n_g = np.nan_to_num(-G / g_h, nan=0.0, posinf=np.inf, neginf=-np.inf)
            n_v = np.nan_to_num(-V / v_h, nan=0.0, posinf=np.inf, neginf=-np.inf)
        lo, hi = np.minimum(0.0, n_g), np.maximum(0.0, n_g)
        if vega_obs:
            lo, hi = np.minimum(lo, n_v), np.maximum(hi, n_v)
        obs = np.stack([S[:, t], G, g_h] + ([V, v_h] if vega_obs else []), axis=1)
        x = decide(t, obs, n_g, n_v, lo, hi)
        n[:, t] = x
        ob[:, t] = np.abs(x * m["h_price"][:, t, t])
        act_next = ~m["h_inactive"][:, t + 1, :]
        pos = -(m["l_delta"][:, t] + np.sum(n * m["h_delta"][:, t, :] * act, axis=1))
        p = m["l_next"][:, t] - m["l_price"][:, t]
        p += np.sum(n * (m["h_price"][:, t + 1, :] - m["h_price"][:, t, :]) * act_next, axis=1)
        p += pos * (S[:, t + 1] - S[:, t])
        pnl[:, t] = p
    return pnl, ob


def sw_var(step_net, idx=None):
    """step_net (P, T) net daily P&L -> (SW-VaR95, terminal VaR95); idx (B, P) bootstrap -> arrays (B,)."""
    L = -np.cumsum(step_net[:, ::-1], axis=1)[:, ::-1]  # L[:, t] = loss from t to maturity
    if idx is None:
        q = np.quantile(L, 0.95, axis=0)
        return float(q.mean()), float(q[0])
    q = np.stack([np.quantile(L[i], 0.95, axis=0) for i in idx])
    return q.mean(1), q[:, 0]


def daily_var(step_net, idx=None):
    """VaR95 of the one-day loss pooled over all days and paths (a desk's daily VaR)."""
    if idx is None:
        return float(np.quantile(-step_net, 0.95))
    return np.array([np.quantile(-step_net[i], 0.95) for i in idx])


def summary(step_net, idx, cost=None):
    sw, term = sw_var(step_net)
    bsw, bterm = sw_var(step_net, idx)
    bd = daily_var(step_net, idx[:100])
    out = dict(SW_VaR95=sw, SW_VaR95_se=float(bsw.std(ddof=1)), VaR95=term, VaR95_se=float(bterm.std(ddof=1)),
               daily_VaR95=daily_var(step_net), daily_VaR95_se=float(bd.std(ddof=1)),
               mean_pnl=float(step_net.sum(1).mean()))
    if cost is not None:
        out["mean_cost"] = float(cost.mean())
    return out, (bsw, bterm)


def torch_agent_steps(run_dir, m):
    """Per-day net P&L of a train_d4pg.py checkpoint (best_actor.pt) on market m, with its own env flags."""
    import train_d4pg as T
    from torch_d4pg.agent import D4PGQR
    from torch_d4pg.hedge_env import HedgingEnv
    cfg = json.load(open(Path(run_dir) / "config.json"))
    HedgingEnv.extra_obs = bool(cfg.get("extra_obs"))
    HedgingEnv.rule_obs = bool(cfg.get("rule_obs"))
    HedgingEnv.rule_frac = (tuple(float(x) for x in cfg["warm_rule"].split(",")) if cfg.get("rule_frac_obs") else None)
    HedgingEnv.band = float(cfg.get("band") or 0.0)
    HedgingEnv.pnl_obs = cfg.get("precommit_b") is not None
    HedgingEnv.pnl_offset = float(cfg["precommit_b"]) if cfg.get("precommit_b_range") else 0.0
    a = argparse.Namespace(**cfg)
    env = T.make_env(m, cfg["spread"], cfg["vega_obs"], a)
    actor, _ = D4PGQR.load_actor(Path(run_dir) / "best_actor.pt")
    obs = env.reset(np.arange(env.P))
    out, cost, done = [], np.zeros(env.P), False
    while not done:
        with torch.inference_mode():
            u = actor(torch.as_tensor(obs, dtype=torch.float32)).numpy()[:, 0]
        t = env.t
        obs, r, done, x = env.step((u + 1) / 2)
        cost += np.abs(cfg["spread"] * x * env.b["h_price"][:, t, t])
        out.append(r)
    HedgingEnv.extra_obs = HedgingEnv.rule_obs = HedgingEnv.pnl_obs = False
    HedgingEnv.pnl_offset = 0.0
    HedgingEnv.rule_frac = None
    HedgingEnv.band = 0.0
    return np.stack(out, 1), cost


def tune_sw(m_val):
    """Rule (b) (a, b, w) minimising validation SW-VaR95, per cost (one rollout per grid point)."""
    cache = {k: rollout_steps(m_val, A.rule_decider("rule", *k)) for k in itertools.product(A.A_GRID, A.A_GRID, A.W_GRID)}
    tuned = {}
    for c in COSTS:
        best = min(cache, key=lambda k: sw_var(cache[k][0] - c * cache[k][1])[0])
        tuned[c] = dict(a=float(best[0]), b=float(best[1]), w=float(best[2]),
                        val_SW=sw_var(cache[best][0] - c * cache[best][1])[0])
        print("SW-tuned rule", c, tuned[c], flush=True)
    return tuned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="../DRL-Hedging-retrain/markets")
    ap.add_argument("--runs", nargs="*", default=[])
    args = ap.parse_args()
    a = A.block_args("B", ["--market_dir", args.markets])
    mk = {w: A.T.get_market(a, *s) for w, s in dict(val=(a.val_seed, a.n_val), test=(a.test_seed, a.n_test)).items()}
    P = mk["test"]["S"].shape[0]
    idx = np.random.default_rng(7).integers(0, P, (N_BOOT, P))
    base = json.load(open("review/results/retrain/baselines_B.json"))
    # 1. rule (b) tuned on validation SW-VaR95 (one rollout per (a, b, w) serves all costs)
    tuned_path = OUT / "stepwise_rule_params.json"
    if tuned_path.exists():
        tuned = {float(k): v for k, v in json.load(open(tuned_path)).items()}
    else:
        tuned = tune_sw(mk["val"])
        OUT.mkdir(parents=True, exist_ok=True)
        json.dump({str(k): v for k, v in tuned.items()}, open(tuned_path, "w"), indent=2)
    rows = []
    for c in COSTS:
        rb = base[f"{c}_0"]["rule_b_VaR95"]
        st = {"delta": A.rule_decider("delta"), "DG": A.rule_decider("DG"), "DV": A.rule_decider("DV"),
              f"rule (b) terminal-tuned a={rb['a']},b={rb['b']},w={rb['w']}": A.rule_decider("rule", rb["a"], rb["b"], rb["w"]),
              f"rule (b) SW-tuned a={tuned[c]['a']},b={tuned[c]['b']},w={tuned[c]['w']}":
                  A.rule_decider("rule", tuned[c]["a"], tuned[c]["b"], tuned[c]["w"])}
        nets, costs = {}, {}
        for name, dec in st.items():
            p, o = rollout_steps(mk["test"], dec)
            nets[name], costs[name] = p - c * o, c * o.sum(1)
        # PR #3 step-wise agents (10 seeds): selected seed by validation SW-VaR95, and mean over seeds
        seeds = []
        for s in BLOCKS["B"]["seeds"]:
            npz = WEIGHTS / f"{run_name('B', c, s)}.npz"
            f, _ = A.policy_decider(npz)
            pv, ov = rollout_steps(mk["val"], f)
            pt, ot = rollout_steps(mk["test"], f)
            seeds.append((sw_var(pv - c * ov)[0], s, pt - c * ot, c * ot.sum(1)))
        seeds.sort(key=lambda x: x[0])
        nets[f"PR #3 agent (seed {seeds[0][1]}, best val SW)"] = seeds[0][2]
        costs[f"PR #3 agent (seed {seeds[0][1]}, best val SW)"] = seeds[0][3]
        med = seeds[(len(seeds) - 1) // 2]  # lower median of the validation SW-VaR95
        nets[f"PR #3 agent (seed {med[1]}, median val SW)"], costs[f"PR #3 agent (seed {med[1]}, median val SW)"] = med[2], med[3]
        sw_seeds = [sw_var(x[2]) for x in seeds]
        # newer agents (torch checkpoints)
        for pat in args.runs:
            for d in sorted(glob.glob(pat)):
                cfg = json.load(open(Path(d) / "config.json"))
                if abs(cfg["spread"] - c) < 1e-12 and (Path(d) / "best_actor.pt").exists():
                    nets[Path(d).name], costs[Path(d).name] = torch_agent_steps(d, mk["test"])
        sw_rule_b = sw_var(nets[[k for k in nets if k.startswith("rule (b) SW")][0]], idx)
        term_rule_b = sw_var(nets[[k for k in nets if k.startswith("rule (b) terminal")][0]], idx)
        for name, net in nets.items():
            s, (bsw, bterm) = summary(net, idx, costs.get(name))
            rows.append(dict(cost=c, policy=name, **s,
                             gain_SW_vs_SWrule=float(sw_rule_b[0].mean() * 0 + sw_var(nets[[k for k in nets if k.startswith('rule (b) SW')][0]])[0] - s["SW_VaR95"]),
                             gain_SW_vs_SWrule_se=float((sw_rule_b[0] - bsw).std(ddof=1)),
                             gain_term_vs_termrule=float(sw_var(nets[[k for k in nets if k.startswith('rule (b) terminal')][0]])[1] - s["VaR95"]),
                             gain_term_vs_termrule_se=float((term_rule_b[1] - bterm).std(ddof=1))))
        rows.append(dict(cost=c, policy="PR #3 agents, mean over 10 seeds", SW_VaR95=float(np.mean([x[0] for x in sw_seeds])),
                         SW_VaR95_se=float(np.std([x[0] for x in sw_seeds], ddof=1) / np.sqrt(len(sw_seeds))),
                         VaR95=float(np.mean([x[1] for x in sw_seeds])),
                         VaR95_se=float(np.std([x[1] for x in sw_seeds], ddof=1) / np.sqrt(len(sw_seeds)))))
        print(pd.DataFrame([r for r in rows if r["cost"] == c]).round(2).to_string(index=False), flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "stepwise.csv", index=False)



if __name__ == "__main__":
    main()
