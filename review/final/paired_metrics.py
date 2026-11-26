"""Sven item 1: VaR95, CVaR95 and mean P&L of DG, rule (b), richer rule and the FINAL DRL agent (selected and median
seed) on the SAME test paths (seed 4242, 8,192 paths, adi_trading), with paired bootstrap SEs of the differences.
Difference = baseline - DRL for VaR95/CVaR95 (losses; positive = DRL better) and DRL - baseline for mean P&L
(positive = DRL better). Reads the stored test P&L of the FINAL runs (no retraining).

    PYTHONPATH=. python review/final/paired_metrics.py [FINAL_LOGS]"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from review.retrain import analysis as A
from review.retrain.settings import COSTS
from review.warmstart.richer_rule import decider as richer_decider

LOGS = Path(sys.argv[1] if len(sys.argv) > 1 else "../DRL-Hedging-AmericanOptions/logs_torch/final")
OUT = Path("review/results/final")
N_BOOT = 2000


def metrics(p):
    v, c = A.var_cvar(p)
    return np.asarray(v), np.asarray(c), p.mean(axis=-1)


def main():
    a = A.block_args("B", ["--market_dir", "../DRL-Hedging-retrain/markets", "--pricing_backend", "adi_trading",
                           "--table_dir", "../DRL-Hedging-retrain/pricing_tables"])
    test = A.T.get_market(a, 4242, 8192)
    P = test["S"].shape[0]
    idx = np.random.default_rng(11).integers(0, P, (N_BOOT, P))
    rows = []
    for c in COSTS:
        prm = json.load(open(f"review/results/warmstart/richer_rule_params_adi_trading_{c}.json"))[str(c)]
        rb = prm["rule_b"]
        rich = [prm["early"][k] for k in "abw"] + [prm["late"][k] for k in "abw"]
        pn = {}
        for name, dec in [("DG", A.rule_decider("DG")), ("Rule (b)", A.rule_decider("rule", rb["a"], rb["b"], rb["w"])),
                          ("Richer rule", richer_decider(rich))]:
            pn[name] = A.net(A.rollout(test, dec, True), c)
        runs = []
        for d in sorted(LOGS.glob(f"HestonP_var_spread{c}_seed*_FINAL")):
            r = json.load(open(d / "results.json"))
            runs.append((r["val_score"], int(json.load(open(d / "config.json"))["seed"]), np.load(d / "test_pnl.npy")[:, 0]))
        runs.sort(key=lambda x: x[0])
        sel, med = runs[0], runs[(len(runs) - 1) // 2]
        pn[f"DRL selected (seed {sel[1]})"], pn[f"DRL median (seed {med[1]})"] = sel[2], med[2]
        for name, p in pn.items():
            v, cv, m = metrics(p)
            bv, bc, bm = metrics(p[idx])
            rows.append(dict(cost=c, policy=name, VaR95=float(v), VaR95_se=float(bv.std(ddof=1)), CVaR95=float(cv),
                             CVaR95_se=float(bc.std(ddof=1)), mean_pnl=float(m), mean_pnl_se=float(bm.std(ddof=1))))
        for drl in [k for k in pn if k.startswith("DRL")]:
            pd_, pdb = pn[drl], pn[drl][idx]
            dv, dc, dm = metrics(pd_)
            bdv, bdc, bdm = metrics(pdb)
            for base in ["DG", "Rule (b)", "Richer rule"]:
                v, cv, m = metrics(pn[base])
                bv, bc, bm = metrics(pn[base][idx])
                rows.append(dict(cost=c, policy=f"{drl} vs {base}", dVaR95=float(v - dv), dVaR95_se=float((bv - bdv).std(ddof=1)),
                                 dCVaR95=float(cv - dc), dCVaR95_se=float((bc - bdc).std(ddof=1)),
                                 dmean_pnl=float(dm - m), dmean_pnl_se=float((bdm - bm).std(ddof=1))))
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "paired_metrics.csv", index=False)
    f = lambda v, s: "-" if pd.isna(v) else f"{v:.2f} ({s:.2f})"
    md = ["# VaR95, CVaR95 and mean P&L on the same test paths (seed 4242, 8,192 paths, adi_trading)", "",
          "Levels with bootstrap SE. Differences paired on the same paths (2,000 resamples): VaR/CVaR = baseline - DRL, "
          "mean P&L = DRL - baseline; positive = DRL better.", "",
          "| Cost | Policy | VaR95 | CVaR95 | Mean P&L |", "|---|---|---|---|---|"]
    for _, r in df[df.VaR95.notna()].iterrows():
        md.append(f"| {r.cost:.1%} | {r.policy} | {f(r.VaR95, r.VaR95_se)} | {f(r.CVaR95, r.CVaR95_se)} | {f(r.mean_pnl, r.mean_pnl_se)} |")
    md += ["", "| Cost | Comparison | dVaR95 | dCVaR95 | dMean P&L |", "|---|---|---|---|---|"]
    for _, r in df[df.VaR95.isna()].iterrows():
        md.append(f"| {r.cost:.1%} | {r.policy} | {f(r.dVaR95, r.dVaR95_se)} | {f(r.dCVaR95, r.dCVaR95_se)} | {f(r.dmean_pnl, r.dmean_pnl_se)} |")
    (OUT / "paired_metrics.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
