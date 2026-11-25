"""P3 (b = 0.6 x rule VaR): speed branch with / without --kernel_atm_fix, VALIDATION metrics only.

    PYTHONPATH=. python review/speed/p3_atm_fix_report.py
Best validation VaR95 (terminal, 8,192 validation paths seed 2025) of the selected checkpoint.
Writes review/results/speed/p3_atm_fix.md and p3_atm_fix.csv.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LOGS, OUT = ROOT / "logs_torch" / "speed", ROOT / "review" / "results" / "speed"
RULE_VAL = {0.005: 24.59, 0.01: 27.87, 0.02: 32.80}  # rule (b) validation VaR95 (run_pilot.sh PCB)

rows = []
for c in (0.005, 0.01, 0.02):
    for s in (1, 2, 3):
        r = {}
        for v in ("P3_speed", "P3_atmfix"):
            f = LOGS / f"HestonP_var_spread{c}_seed{s}_{v}" / "results.json"
            if f.exists():
                j = json.load(open(f))
                r[v] = (j["val_score"], j["selected_episode"], j["seconds_per_1000_episodes"])
        if len(r) == 2:
            rows.append(dict(cost=c, seed=s, rule_b_val=RULE_VAL[c],
                             val_nofix=r["P3_speed"][0], ep_nofix=r["P3_speed"][1],
                             val_fix=r["P3_atmfix"][0], ep_fix=r["P3_atmfix"][1],
                             fix_minus_nofix=r["P3_atmfix"][0] - r["P3_speed"][0],
                             sec_per_1000_nofix=r["P3_speed"][2], sec_per_1000_fix=r["P3_atmfix"][2]))
df = pd.DataFrame(rows)
df.to_csv(OUT / "p3_atm_fix.csv", index=False)
md = ["# P3 (pre-commitment, b = 0.6 x rule VaR): ATM fix on / off, validation only", "",
      "Speed branch (ad60894 + d17b0dd, mixture loss), 12,000 episodes, 6 runs in parallel.",
      "Numbers: best validation VaR95 (terminal) of the selected checkpoint, 8,192 validation paths (seed 2025).",
      "Lower is better. Fix - no fix < 0 means the ATM fix helps. No test-set numbers (exploratory test set).", "",
      "| Cost | Seed | Rule (b) val | No fix: val (sel. ep) | ATM fix: val (sel. ep) | Fix - no fix |",
      "|---|---|---|---|---|---|"]
for _, r in df.iterrows():
    md.append(f"| {r.cost:.1%} | {int(r.seed)} | {r.rule_b_val:.2f} | {r.val_nofix:.2f} ({int(r.ep_nofix)}) | "
              f"{r.val_fix:.2f} ({int(r.ep_fix)}) | {r.fix_minus_nofix:+.2f} |")
md += ["", "| Cost | Mean no fix | Mean fix | Mean fix - no fix | SD over seeds | n |", "|---|---|---|---|---|---|"]
for c, g in df.groupby("cost"):
    md.append(f"| {c:.1%} | {g.val_nofix.mean():.2f} | {g.val_fix.mean():.2f} | {g.fix_minus_nofix.mean():+.2f} | "
              f"{g.fix_minus_nofix.std(ddof=1):.2f} | {len(g)} |")
md += ["", f"Seconds per 1,000 episodes (6 parallel): no fix {df.sec_per_1000_nofix.mean():.0f}, "
           f"fix {df.sec_per_1000_fix.mean():.0f}."]
(OUT / "p3_atm_fix.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
