"""Screen report (validation only). Per run: best validation VaR95 after the actor freeze, and the mean of the last 4
validation checks (less selection noise). Gains = ctrl - arm, paired by (seed, cost); positive = arm better.
    PYTHONPATH=. python review/improve/report.py [OUT_MD] [LOGS]"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUTMD = Path(sys.argv[1] if len(sys.argv) > 1 else "review/results/improve/screens.md")
LOGS = Path(sys.argv[2] if len(sys.argv) > 2 else "logs_torch/improve")

rows = []
for d in LOGS.glob("HestonP_var_spread*_seed*_*"):
    m = re.match(r"HestonP_var_spread([\d.]+)_seed(\d+)_(.+)", d.name)
    f = d / "validation.csv"
    if not m or not f.exists():
        continue
    v = pd.read_csv(f)
    v = v[v.episode > 1000]
    if len(v) == 0:
        continue
    rows.append(dict(cost=float(m[1]), seed=int(m[2]), arm=m[3], done=(d / "results.json").exists(),
                     episodes=int(v.episode.max()), best=v.val_VaR95.min(), last4=v.val_VaR95.tail(4).mean(),
                     cost_last4=v.val_mean_cost.tail(4).mean(), cvar_last4=v.val_CVaR95.tail(4).mean()))
df = pd.DataFrame(rows)
if df.empty:
    sys.exit("no runs")
import os
CTRL = os.environ.get("CTRL", "ctrl")
ctrl = df[df.arm == CTRL].set_index(["cost", "seed"])
out = []
for (arm, c), g in df.groupby(["arm", "cost"]):
    g = g.set_index(["cost", "seed"])
    common = g.index.intersection(ctrl.index)
    r = dict(arm=arm, cost=c, n=len(g), done=int(g.done.sum()), best=g.best.mean(), last4=g.last4.mean(),
             val_cost=g.cost_last4.mean(), cvar=g.cvar_last4.mean())
    if arm != CTRL and len(common):
        db = ctrl.loc[common, "best"] - g.loc[common, "best"]
        dl = ctrl.loc[common, "last4"] - g.loc[common, "last4"]
        r.update(paired=len(common), gain_best=db.mean(), gain_last4=dl.mean(),
                 se_last4=dl.std(ddof=1) / np.sqrt(len(dl)) if len(dl) > 1 else np.nan)
    out.append(r)
t = pd.DataFrame(out).sort_values(["cost", "arm"])
OUTMD.parent.mkdir(parents=True, exist_ok=True)
txt = ("# Improvement screens (validation seed 2025 only)\n\nbest = best val VaR95 after the freeze; last4 = mean of the "
       "last 4 validation checks. Gains = ctrl - arm, paired by seed (positive = arm better); se over seeds.\n\n"
       + t.round(3).to_markdown(index=False) + "\n")
OUTMD.write_text(txt)
print(txt)
