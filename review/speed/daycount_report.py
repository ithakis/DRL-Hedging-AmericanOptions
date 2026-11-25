"""Day-count check, VALIDATION only: P3 + ATM fix on the ADI table with maturity tau/252 (adi_trading, new markets)
vs P3 + ATM fix on the current QuantLib table (P3_atmfix).

    PYTHONPATH=. python review/speed/daycount_report.py
Rule (b) validation VaR95: adi_trading from the episode-0 line (untrained agent = rule (b), --val_at_start);
current table = the tuned rule (b) validation VaR95 (24.59 / 27.87 / 32.80, run_pilot.sh PCB).
Gain = own market's rule (b) val VaR95 - best val VaR95 (positive = RL better). Writes daycount.md / .csv.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LOGS, OUT = ROOT / "logs_torch" / "speed", ROOT / "review" / "results" / "speed"
RULE_VAL = {0.005: 24.59, 0.01: 27.87, 0.02: 32.80}

rows = []
for c in (0.005, 0.01, 0.02):
    for s in (1, 2, 3):
        d_dc = LOGS / f"HestonP_var_spread{c}_seed{s}_P3_daycount"
        d_fx = LOGS / f"HestonP_var_spread{c}_seed{s}_P3_atmfix"
        if not ((d_dc / "results.json").exists() and (d_fx / "results.json").exists()):
            continue
        dc, fx = json.load(open(d_dc / "results.json")), json.load(open(d_fx / "results.json"))
        v = pd.read_csv(d_dc / "validation.csv")
        rule_dc = float(v[v.episode == 0].val_VaR95.iloc[0])
        rows.append(dict(cost=c, seed=s, rule_val_table=RULE_VAL[c], rule_val_adi_trading=rule_dc,
                         best_val_atmfix=fx["val_score"], ep_atmfix=fx["selected_episode"],
                         best_val_daycount=dc["val_score"], ep_daycount=dc["selected_episode"],
                         gain_atmfix=RULE_VAL[c] - fx["val_score"], gain_daycount=rule_dc - dc["val_score"]))
df = pd.DataFrame(rows)
df.to_csv(OUT / "daycount.csv", index=False)
md = ["# Day-count check (validation only): P3 + ATM fix, QuantLib table vs ADI table with maturity tau/252", "",
      "Current table: QuantLib table (maturity round(tau*365/252) calendar days / 365), runs P3_atmfix.",
      "adi_trading: ADI table 400x200, maturity tau/252 (the paths' clock), new markets built with it, runs P3_daycount.",
      "Both: P3 config + --kernel_atm_fix, same absolute b (14.75 / 16.72 / 19.68), 12,000 episodes.",
      "Validation VaR95 on 8,192 validation paths (seed 2025; the adi_trading market has its own option prices).",
      "Gain = own market's rule (b) val VaR95 - best val VaR95; positive = RL better. No test-set numbers.", "",
      "| Cost | Seed | Rule (b) val, table | Rule (b) val, adi_trading | P3_atmfix best val (ep) | P3_daycount best val (ep) | Gain, table | Gain, adi_trading |",
      "|---|---|---|---|---|---|---|---|"]
for r in df.itertuples():
    md.append(f"| {r.cost:.1%} | {r.seed} | {r.rule_val_table:.2f} | {r.rule_val_adi_trading:.2f} | "
              f"{r.best_val_atmfix:.2f} ({r.ep_atmfix}) | {r.best_val_daycount:.2f} ({r.ep_daycount}) | "
              f"{r.gain_atmfix:+.2f} | {r.gain_daycount:+.2f} |")
md += ["", "| Cost | Mean gain, table | Mean gain, adi_trading | n |", "|---|---|---|---|"]
for c, g in df.groupby("cost"):
    md.append(f"| {c:.1%} | {g.gain_atmfix.mean():+.2f} ({g.gain_atmfix.std(ddof=1):.2f}) | "
              f"{g.gain_daycount.mean():+.2f} ({g.gain_daycount.std(ddof=1):.2f}) | {len(g)} |")
md += ["", "(SD over seeds in brackets.)"]
(OUT / "daycount.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
