"""Robustness grid and ex-post delta costs for the FINAL agents (adi_trading market family only).

Robustness: the selected and median FINAL agents (trained at the base Heston parameters, not retrained), rule (b) and
the richer rule (both tuned at the base parameters on validation), DG and delta, evaluated on new test paths
(seed 4242, 8,192 paths) of markets with one Heston parameter bumped (review/retrain/settings.py BUMPS); prices and
Greeks at the bumped parameters (adi_trading table built for each parameter set). init_vol 0.45 lies outside the
training range of the variance state (flagged in the table).
Delta costs: base test market, stock trading cost c_delta * |change in stock position| * S_t added after the fact
(0, 1, 2, 5 bp); the agents were trained without it.

    PYTHONPATH=. python review/warmstart/robustness_final.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from review.retrain import analysis as A
from review.retrain.settings import BLOCKS, BUMPS, COSTS, with_bump
from review.warmstart.richer_rule import decider as richer_decider

OUT = Path("review/results/final")
LOGS = Path("logs_torch/final")
DELTA_BP = [0, 1, 2, 5]
N_BOOT = 1000


def agent_rollout(run_dir, m):
    """FINAL agent on market m with its own env flags: per path P&L before option cost is not needed; returns
    dict(pnl_net, cost, db) with db = sum_t |stock position change| * S_t (as analysis.rollout)."""
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
    env = T.make_env(m, cfg["spread"], cfg["vega_obs"], argparse.Namespace(**cfg))
    actor, _ = D4PGQR.load_actor(Path(run_dir) / "best_actor.pt")
    obs = env.reset(np.arange(env.P))
    tot, cost, db, pos_prev = np.zeros(env.P), np.zeros(env.P), np.zeros(env.P), np.zeros(env.P)
    done = False
    while not done:
        with torch.inference_mode():
            u = actor(torch.as_tensor(obs, dtype=torch.float32)).numpy()[:, 0]
        t = env.t
        obs, r, done, x = env.step((u + 1) / 2)
        b = env.b
        act = ~b["h_inactive"][:, t, :]
        pos = -(b["l_delta"][:, t] + np.sum(env.n * b["h_delta"][:, t, :] * act, axis=1))
        db += np.abs(pos - pos_prev) * b["S"][:, t]
        pos_prev = pos
        tot += r
        cost += np.abs(cfg["spread"] * x * b["h_price"][:, t, t])
    HedgingEnv.extra_obs = HedgingEnv.rule_obs = HedgingEnv.pnl_obs = False
    HedgingEnv.pnl_offset = 0.0
    HedgingEnv.rule_frac = None
    HedgingEnv.band = 0.0
    return dict(net=tot, cost=cost, db=db)


def pick_agents(c):
    rows = []
    for d in sorted(LOGS.glob(f"HestonP_var_spread{c}_seed*_FINAL")):
        if (d / "results.json").exists():
            rows.append((json.load(open(d / "results.json"))["val_score"], d))
    rows.sort(key=lambda x: x[0])
    return {"DRL selected": rows[0][1], "DRL median": rows[(len(rows) - 1) // 2][1]}


def var_se(p, idx):
    v = A.var_cvar(p)[0]
    return float(v), float(A.var_cvar(p[idx])[0].std(ddof=1))


def gain(pb, p, idx):
    return (float(A.var_cvar(pb)[0] - A.var_cvar(p)[0]),
            float((A.var_cvar(pb[idx])[0] - A.var_cvar(p[idx])[0]).std(ddof=1)))


def main():
    extra = ["--market_dir", "../DRL-Hedging-retrain/markets", "--pricing_backend", "adi_trading",
             "--table_dir", "../DRL-Hedging-retrain/pricing_tables", "--obj_func", "var", "--spread", "0.0"]
    agents = {c: pick_agents(c) for c in COSTS}
    params = {c: json.load(open(f"review/results/warmstart/richer_rule_params_adi_trading_{c}.json"))[str(c)] for c in COSTS}
    rob, dcost = [], []
    for flag, v in [(None, None)] + BUMPS:
        flags = BLOCKS["B"]["flags"] if flag is None else with_bump(BLOCKS["B"]["flags"], flag, v)
        a = A.T.get_args(flags + extra)
        m = A.T.get_market(a, 4242, 8192)
        P = m["S"].shape[0]
        idx = np.random.default_rng(7).integers(0, P, (N_BOOT, P))
        for c in COSTS:
            rb, pr = params[c]["rule_b"], params[c]
            rich = [pr["early"][k] for k in "abw"] + [pr["late"][k] for k in "abw"]
            base = {}
            for name, dec in (("delta", A.rule_decider("delta")), ("DG", A.rule_decider("DG")),
                              ("rule_b", A.rule_decider("rule", rb["a"], rb["b"], rb["w"])),
                              ("richer_rule", richer_decider(rich))):
                r = A.rollout(m, dec, True)
                base[name] = dict(net=A.net(r, c), cost=c * r["ob"], db=r["db"])
            res = dict(base)
            for name, d in agents[c].items():
                res[name] = agent_rollout(d, m)
            row = dict(parameter="base" if flag is None else flag.lstrip("-"), value=v, cost=c,
                       outside_training_range=(flag == "--init_vol" and v == 0.45))
            for name, r in res.items():
                row[f"{name}_VaR95"], row[f"{name}_VaR95_se"] = var_se(r["net"], idx)
                row[f"{name}_mean_cost"] = float(r["cost"].mean())
            for name in agents[c]:
                for bname in ("rule_b", "richer_rule", "DG"):
                    row[f"{name}_gain_vs_{bname}"], row[f"{name}_gain_vs_{bname}_se"] = gain(res[bname]["net"], res[name]["net"], idx)
            rob.append(row)
            print(row["parameter"], v, c, {k: round(row[f'{k}_VaR95'], 2) for k in res}, flush=True)
            if flag is None:  # ex-post delta costs on the base market
                for bp in DELTA_BP:
                    dr = dict(cost=c, delta_bp=bp)
                    nets = {k: r["net"] - bp * 1e-4 * r["db"] for k, r in res.items()}
                    for k, p in nets.items():
                        dr[f"{k}_VaR95"], dr[f"{k}_VaR95_se"] = var_se(p, idx)
                    for name in agents[c]:
                        dr[f"{name}_gain_vs_rule_b"], dr[f"{name}_gain_vs_rule_b_se"] = gain(nets["rule_b"], nets[name], idx)
                    dcost.append(dr)
    OUT.mkdir(parents=True, exist_ok=True)
    rob, dcost = pd.DataFrame(rob), pd.DataFrame(dcost)
    rob.to_csv(OUT / "robustness.csv", index=False)
    dcost.to_csv(OUT / "deltacost.csv", index=False)
    f = lambda v, s: f"{v:.2f} ({s:.2f})"
    md = ["# Robustness (fresh test seed 4242, 8,192 paths per row, adi_trading pricing at the bumped parameters)", "",
          "Agents and rules tuned/trained at the base parameters only. VaR95 (SE); gain = rule (b) - DRL selected "
          "(paired SE), positive = DRL better. * = outside the training range.", "",
          "| Cost | Parameter | Value | DRL selected | DRL median | Rule (b) | Richer rule | DG | Delta | Gain sel. vs rule (b) | Gain sel. vs DG |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in rob.sort_values(["cost"], kind="stable").iterrows():
        md.append(f"| {r.cost:.1%} | {r.parameter}{'*' if r.outside_training_range else ''} | "
                  f"{'' if pd.isna(r.value) else r.value} | {f(r['DRL selected_VaR95'], r['DRL selected_VaR95_se'])} | "
                  f"{f(r['DRL median_VaR95'], r['DRL median_VaR95_se'])} | {f(r.rule_b_VaR95, r.rule_b_VaR95_se)} | "
                  f"{f(r.richer_rule_VaR95, r.richer_rule_VaR95_se)} | {f(r.DG_VaR95, r.DG_VaR95_se)} | "
                  f"{f(r.delta_VaR95, r.delta_VaR95_se)} | "
                  f"{f(r['DRL selected_gain_vs_rule_b'], r['DRL selected_gain_vs_rule_b_se'])} | "
                  f"{f(r['DRL selected_gain_vs_DG'], r['DRL selected_gain_vs_DG_se'])} |")
    md += ["", "# Ex-post delta (stock) trading costs, base test market", "",
           "| Cost | Delta cost (bp) | DRL selected | DRL median | Rule (b) | Richer rule | DG | Gain sel. vs rule (b) |",
           "|---|---|---|---|---|---|---|---|"]
    for _, r in dcost.iterrows():
        md.append(f"| {r.cost:.1%} | {r.delta_bp} | {f(r['DRL selected_VaR95'], r['DRL selected_VaR95_se'])} | "
                  f"{f(r['DRL median_VaR95'], r['DRL median_VaR95_se'])} | {f(r.rule_b_VaR95, r.rule_b_VaR95_se)} | "
                  f"{f(r.richer_rule_VaR95, r.richer_rule_VaR95_se)} | {f(r.DG_VaR95, r.DG_VaR95_se)} | "
                  f"{f(r['DRL selected_gain_vs_rule_b'], r['DRL selected_gain_vs_rule_b_se'])} |")
    (OUT / "robustness.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
