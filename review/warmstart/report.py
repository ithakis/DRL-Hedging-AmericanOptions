"""Warm-start pilot report: selected checkpoint (best validation VaR95) on test paths, paired vs rule (b).

    PYTHONPATH=. python review/warmstart/report.py [--markets ../DRL-Hedging-retrain/markets]

Same test paths (seed 654, 8,192) and same rule (b) parameters as the retrain main table (block B).
Gain = rule (b) VaR95 - RL VaR95 (positive = RL better), paired bootstrap SE (1,000 resamples).
Writes review/results/warmstart/pilot.md and pilot.csv.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from review.retrain import analysis as A

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "logs_torch" / "warmstart"
OUT = ROOT / "review" / "results" / "warmstart"
RL_PR3 = {0.005: 26.25, 0.01: 29.95, 0.02: 35.13}  # retrain block B, selected agent of 10 seeds (main_B.md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default=str(ROOT.parent / "DRL-Hedging-retrain" / "markets"))
    args = ap.parse_args()
    base = json.load(open(ROOT / "review/results/retrain/baselines_B.json"))
    a = A.block_args("B", ["--market_dir", args.markets])
    test_m = A.T.get_market(a, a.test_seed, a.n_test)
    rows = []
    for d in sorted(LOGS.glob("HestonP_*")):
        if not (d / "results.json").exists():
            continue
        r = json.load(open(d / "results.json"))
        cfg = json.load(open(d / "config.json"))
        val = pd.read_csv(d / "validation.csv")
        spread, seed = cfg["spread"], cfg["seed"]
        variant = re.sub(r".*_seed\d+_?", "", d.name) or "plain"
        rb = base[f"{spread}_0"]["rule_b_VaR95"]
        p_rule = A.net(A.rollout(test_m, A.rule_decider("rule", rb["a"], rb["b"], rb["w"]), True), spread)
        p = np.load(d / "test_pnl.npy")[:, 0]
        g, se = A.paired_gain(p_rule, p)
        v0 = val[val.episode == 0].val_VaR95
        rows.append(dict(variant=variant, spread=spread, seed=seed, selected_episode=r["selected_episode"],
                         val_VaR95_ep0=float(v0.iloc[0]) if len(v0) else np.nan, val_VaR95=r["val_score"],
                         test_VaR95=r["test"]["VaR95"], test_VaR95_se=r["test"]["VaR95_se"],
                         test_CVaR95=r["test"]["CVaR95"], test_cost=r["test"]["mean_cost"],
                         rule_b_test_VaR95=rb["VaR95"], gain_vs_rule_b=g, gain_vs_rule_b_se=se,
                         rl_pr3_test_VaR95=RL_PR3[spread], sec_per_1000=r["seconds_per_1000_episodes"]))
    df = pd.DataFrame(rows).sort_values(["variant", "spread", "seed"])
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "pilot.csv", index=False)
    md = ["# Warm-start pilot (Heston P, r = 2%, VaR95), test paths seed 654, 8,192 paths", "",
          "Selection: best validation VaR95 checkpoint (episode 0 = the rule itself is not selectable).",
          "Gain vs rule (b) = rule (b) - RL, paired bootstrap SE; positive = RL better.", "",
          "| Variant | Cost | Seed | Sel. ep | Val VaR95 (ep 0 = rule) | Val VaR95 (sel.) | Test VaR95 | "
          "Rule (b) test | Gain vs rule (b) | RL of PR #3 (test) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        md.append(f"| {r.variant} | {r.spread:.1%} | {r.seed} | {r.selected_episode} | {r.val_VaR95_ep0:.2f} | "
                  f"{r.val_VaR95:.2f} | {A.fmt(r.test_VaR95, r.test_VaR95_se)} | {r.rule_b_test_VaR95:.2f} | "
                  f"{A.fmt(r.gain_vs_rule_b, r.gain_vs_rule_b_se)} | {r.rl_pr3_test_VaR95:.2f} |")
    (OUT / "pilot.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
