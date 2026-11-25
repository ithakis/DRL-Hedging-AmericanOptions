"""Final table (fresh test set, seed 4242, adi_trading market): pre-commitment agent vs baselines.

All numbers come from ONE market family (adi_trading pricing table, the paths' tau/252 clock):
training seed 1234, validation seed 2025 (all tuning and checkpoint/seed selection), test seed 4242 (used once here).
Baselines: delta, DG, DV, rule (b) re-tuned on adi_trading validation, richer rule (time-varying a, b, w) tuned on the
same validation paths with the same budget. Agents: logs_torch/final/<run>_FINAL (10 seeds per cost); "selected" =
best validation VaR95 over seeds, "median" = lower median; also mean over seeds.
Gains = baseline VaR95 - agent VaR95 (positive = agent better), paired bootstrap SE (1,000 resamples of test paths).

    PYTHONPATH=. python review/warmstart/final_table.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from review.retrain import analysis as A
from review.retrain.settings import COSTS, run_name
from review.warmstart.richer_rule import decider as richer_decider

OUT = Path("review/results/final")
LOGS = Path("logs_torch/final")
N_BOOT = 1000


def stats(p, cost, idx):
    v, c = A.var_cvar(p)
    bv, bc = A.var_cvar(p[idx])
    return dict(VaR95=float(v), VaR95_se=float(bv.std(ddof=1)), CVaR95=float(c), CVaR95_se=float(bc.std(ddof=1)),
                mean_pnl=float(p.mean()), mean_pnl_se=float(p.std(ddof=1) / np.sqrt(len(p))), mean_cost=float(cost.mean()))


def gain(p_base, p, idx):
    g = A.var_cvar(p_base)[0] - A.var_cvar(p)[0]
    gb = A.var_cvar(p_base[idx])[0] - A.var_cvar(p[idx])[0]
    return float(g), float(gb.std(ddof=1))


def main():
    a = A.block_args("B", ["--market_dir", "../DRL-Hedging-retrain/markets", "--pricing_backend", "adi_trading",
                           "--table_dir", "../DRL-Hedging-retrain/pricing_tables"])
    test = A.T.get_market(a, 4242, 8192)
    P = test["S"].shape[0]
    idx = np.random.default_rng(7).integers(0, P, (N_BOOT, P))
    rows, seed_rows = [], []
    for c in COSTS:
        prm = json.load(open(f"review/results/warmstart/richer_rule_params_adi_trading_{c}.json"))[str(c)]
        rb = prm["rule_b"]
        rich = [prm["early"][k] for k in "abw"] + [prm["late"][k] for k in "abw"]
        base = {"Delta": A.rule_decider("delta"), "DG": A.rule_decider("DG"), "DV": A.rule_decider("DV"),
                f"Rule (b) a={rb['a']}, b={rb['b']}, w={rb['w']}": A.rule_decider("rule", rb["a"], rb["b"], rb["w"]),
                "Richer rule (time-varying a, b, w)": richer_decider(rich)}
        pn, cs = {}, {}
        for name, dec in base.items():
            r = A.rollout(test, dec, True)
            pn[name], cs[name] = A.net(r, c), c * r["ob"]
        k_rule = [k for k in pn if k.startswith("Rule (b)")][0]
        k_rich = "Richer rule (time-varying a, b, w)"
        runs = []
        for d in sorted(LOGS.glob(f"HestonP_var_spread{c}_seed*_FINAL")):
            if not (d / "results.json").exists():
                continue
            r = json.load(open(d / "results.json"))
            tp = np.load(d / "test_pnl.npy")
            runs.append(dict(seed=int(json.load(open(d / "config.json"))["seed"]), val=r["val_score"],
                             sel_ep=r["selected_episode"], pnl=tp[:, 0], cost=tp[:, 1]))
        runs.sort(key=lambda x: x["val"])
        for rr in runs:
            g_rule = gain(pn[k_rule], rr["pnl"], idx)
            seed_rows.append(dict(cost=c, seed=rr["seed"], val_VaR95=rr["val"], selected_episode=rr["sel_ep"],
                                  **stats(rr["pnl"], rr["cost"], idx), gain_vs_rule_b=g_rule[0], gain_vs_rule_b_se=g_rule[1]))
        agents = {}
        if runs:
            agents[f"DRL pre-commitment, selected (seed {runs[0]['seed']})"] = runs[0]
            med = runs[(len(runs) - 1) // 2]
            agents[f"DRL pre-commitment, median (seed {med['seed']})"] = med
        for name in base:
            rows.append(dict(cost=c, policy=name, **stats(pn[name], cs[name], idx),
                             gain_vs_rule_b=gain(pn[k_rule], pn[name], idx)[0] if name != k_rule else 0.0))
        for name, rr in agents.items():
            g1, g2, g3 = gain(pn[k_rule], rr["pnl"], idx), gain(pn[k_rich], rr["pnl"], idx), gain(pn["DG"], rr["pnl"], idx)
            rows.append(dict(cost=c, policy=name, n_seeds=len(runs), **stats(rr["pnl"], rr["cost"], idx),
                             gain_vs_rule_b=g1[0], gain_vs_rule_b_se=g1[1], gain_vs_richer=g2[0], gain_vs_richer_se=g2[1],
                             gain_vs_DG=g3[0], gain_vs_DG_se=g3[1]))
        if runs:
            v = np.array([A.var_cvar(rr["pnl"])[0] for rr in runs])
            gr = np.array([gain(pn[k_rule], rr["pnl"], idx[:1])[0] for rr in runs])
            rows.append(dict(cost=c, policy=f"DRL pre-commitment, mean over {len(runs)} seeds", n_seeds=len(runs),
                             VaR95=float(v.mean()), VaR95_se=float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan,
                             gain_vs_rule_b=float(gr.mean()),
                             gain_vs_rule_b_se=float(gr.std(ddof=1) / np.sqrt(len(gr))) if len(gr) > 1 else np.nan,
                             seeds_beating_rule_b=int((gr > 0).sum())))
    OUT.mkdir(parents=True, exist_ok=True)
    df, ds = pd.DataFrame(rows), pd.DataFrame(seed_rows)
    df.to_csv(OUT / "final_table.csv", index=False)
    ds.to_csv(OUT / "final_seeds.csv", index=False)
    f = lambda v, s: "-" if pd.isna(v) else (f"{v:.2f}" if s is None or pd.isna(s) else f"{v:.2f} ({s:.2f})")
    md = ["# Final results: fresh test set (seed 4242, 8,192 paths), adi_trading market, VaR95 of the episode loss", "",
          "Tuning and selection on validation (seed 2025) only. Gains: baseline - DRL, paired bootstrap SE; positive = DRL better.", "",
          "| Cost | Policy | VaR95 | CVaR95 | Mean P&L | E[cost] | Gain vs rule (b) | Gain vs richer rule | Gain vs DG |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        md.append(f"| {r.cost:.1%} | {r.policy} | {f(r.VaR95, r.get('VaR95_se'))} | {f(r.get('CVaR95'), r.get('CVaR95_se'))} | "
                  f"{f(r.get('mean_pnl'), r.get('mean_pnl_se'))} | {f(r.get('mean_cost'), None)} | "
                  f"{f(r.get('gain_vs_rule_b'), r.get('gain_vs_rule_b_se'))} | {f(r.get('gain_vs_richer'), r.get('gain_vs_richer_se'))} | "
                  f"{f(r.get('gain_vs_DG'), r.get('gain_vs_DG_se'))} |")
    (OUT / "final_table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
