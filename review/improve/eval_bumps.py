"""Robustness screen (validation paths only, seed 2025): screen agents vs rule (b) on markets with init_vol / volvol
bumps that are NOT the test-grid values (0.18, 0.42 init vol; 0.38 vol-of-vol). Paired gain = rule VaR95 - agent VaR95.
    PYTHONPATH=. python review/improve/eval_bumps.py "ARM1 ARM2" [LOGS]"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from review.retrain import analysis as A
from review.retrain.settings import BLOCKS, with_bump
from review.warmstart.robustness_final import agent_rollout

ARMS = sys.argv[1].split()
LOGS = Path(sys.argv[2] if len(sys.argv) > 2 else "logs_torch/improve")
BUMPS = [(None, None), ("--init_vol", 0.18), ("--init_vol", 0.42), ("--volvol", 0.38)]
extra = ["--market_dir", "../DRL-Hedging-retrain/markets", "--pricing_backend", "adi_trading",
         "--table_dir", "../DRL-Hedging-retrain/pricing_tables", "--obj_func", "var", "--spread", "0.0"]
rows = []
for flag, v in BUMPS:
    flags = BLOCKS["B"]["flags"] if flag is None else with_bump(BLOCKS["B"]["flags"], flag, v)
    m = A.T.get_market(A.T.get_args(flags + extra), 2025, 8192)
    for d in sorted(LOGS.glob("HestonP_var_spread*_seed*_*")):
        arm = d.name.split("_")[-1]
        if arm not in ARMS or not (d / "best_actor.pt").exists():
            continue
        c = float(d.name.split("spread")[1].split("_")[0])
        rb = json.load(open(f"review/results/warmstart/richer_rule_params_adi_trading_{c}.json"))[str(c)]["rule_b"]
        rr = A.rollout(m, A.rule_decider("rule", rb["a"], rb["b"], rb["w"]), True)
        pr = A.net(rr, c)
        pa = agent_rollout(d, m)["net"]
        rows.append(dict(arm=arm, cost=c, seed=int(d.name.split("seed")[1].split("_")[0]),
                         bump="base" if flag is None else f"{flag.lstrip('-')}={v}",
                         agent=float(A.var_cvar(pa)[0]), rule=float(A.var_cvar(pr)[0])))
df = pd.DataFrame(rows)
df["gain"] = df.rule - df.agent
t = df.groupby(["bump", "cost", "arm"]).gain.agg(["mean", "std", "count"]).round(3)
print(t.to_string())
Path("review/results/improve").mkdir(parents=True, exist_ok=True)
df.to_csv(f"review/results/improve/bumps_{'_'.join(ARMS)}.csv", index=False)
