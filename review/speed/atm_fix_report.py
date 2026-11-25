"""E3a seed 1: speed branch without / with --kernel_atm_fix, vs the tuning-thread E3a seed 1 (old code).

    PYTHONPATH=. python review/speed/atm_fix_report.py [--markets ../DRL-Hedging-retrain/markets] \
        [--old_logs ../DRL-Hedging-AmericanOptions/logs_torch/warmstart]

Test paths seed 654 (8,192), paired gain vs rule (b) as in review/warmstart/report.py.
Writes review/results/speed/atm_fix.md and atm_fix.csv.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from review.retrain import analysis as A

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "review" / "results" / "speed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default=str(ROOT.parent / "DRL-Hedging-retrain" / "markets"))
    ap.add_argument("--old_logs", default=str(ROOT.parent / "DRL-Hedging-AmericanOptions" / "logs_torch" / "warmstart"))
    args = ap.parse_args()
    base = json.load(open(ROOT / "review/results/retrain/baselines_B.json"))
    a = A.block_args("B", ["--market_dir", args.markets])
    test_m = A.T.get_market(a, a.test_seed, a.n_test)
    runs = [("E3a (tuning thread, old code)", Path(args.old_logs), "E3a"),
            ("E3a speed branch", ROOT / "logs_torch" / "speed", "E3a_speed"),
            ("E3a speed + atm_fix", ROOT / "logs_torch" / "speed", "E3a_atmfix")]
    rows = []
    for label, logs, suffix in runs:
        for c in (0.005, 0.01, 0.02):
            d = logs / f"HestonP_var_spread{c}_seed1_{suffix}"
            if not (d / "results.json").exists():
                continue
            r = json.load(open(d / "results.json"))
            rb = base[f"{c}_0"]["rule_b_VaR95"]
            p_rule = A.net(A.rollout(test_m, A.rule_decider("rule", rb["a"], rb["b"], rb["w"]), True), c)
            g, se = A.paired_gain(p_rule, np.load(d / "test_pnl.npy")[:, 0])
            rows.append(dict(run=label, spread=c, selected_episode=r["selected_episode"], val_VaR95=r["val_score"],
                             test_VaR95=r["test"]["VaR95"], test_VaR95_se=r["test"]["VaR95_se"],
                             rule_b_test_VaR95=rb["VaR95"], gain_vs_rule_b=g, gain_vs_rule_b_se=se,
                             sec_per_1000=r["seconds_per_1000_episodes"], timing=r.get("timing_seconds")))
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "atm_fix.csv", index=False)
    md = ["# E3a seed 1: speed branch, with and without --kernel_atm_fix", "",
          "Test paths seed 654 (8,192). Gain vs rule (b) = rule (b) - RL, paired bootstrap SE; positive = RL better.", "",
          "| Run | Cost | Sel. ep | Val VaR95 | Test VaR95 | Rule (b) test | Gain vs rule (b) | s / 1000 ep | timing (s) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        t = "" if not isinstance(r.timing, dict) else ", ".join(f"{k} {v:.0f}" for k, v in r.timing.items())
        md.append(f"| {r.run} | {r.spread:.1%} | {r.selected_episode} | {r.val_VaR95:.2f} | "
                  f"{A.fmt(r.test_VaR95, r.test_VaR95_se)} | {r.rule_b_test_VaR95:.2f} | "
                  f"{A.fmt(r.gain_vs_rule_b, r.gain_vs_rule_b_se)} | {r.sec_per_1000:.0f} | {t} |")
    (OUT / "atm_fix.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
